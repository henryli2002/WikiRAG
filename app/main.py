import os
import re
import sys
import time
import asyncio
import contextvars
import numpy as np
import asyncpg
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field
from FlagEmbedding import BGEM3FlagModel, FlagReranker
import jieba

# ─── 配置 ───────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMBEDDING_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bge-m3")
RERANKER_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bge-reranker-v2-m3")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "rag_db")
DB_USER = os.getenv("POSTGRES_USER", "rag_user")

VECTOR_RECALL_K = 50   # 向量召回数
BM25_RECALL_K = 50     # BM25 召回数
HYBRID_TOP_K = 30      # 混合排序取前 N
RERANK_TOP_K = 10      # 精排最多取前 N（实际数量还受 RERANK_MIN_HYBRID_SCORE 约束）
RERANK_MIN_HYBRID_SCORE = 0.010  # 低于此 RRF 分的文档跳过 rerank；约等于单路召回 top-40
FINAL_TOP_K = 5        # MMR 最终返回数
MMR_LAMBDA = 0.7       # MMR 多样性参数 (越大越相关，越小越多样)
SCORE_THRESHOLD = 0.8  # rerank 分数阈值，<=0.8 判定为不相似

VECTOR_TIMEOUT_S = 1.0
BM25_TIMEOUT_S = 1.0
MMR_TIMEOUT_S = 1.0

# ─── 全局模型 & 连接池 ────────────────────────────────────
embedding_model: BGEM3FlagModel | None = None
reranker_model: FlagReranker | None = None
thread_pool = ThreadPoolExecutor(max_workers=8)
db_pool: asyncpg.Pool | None = None

# 底层模型和分词器（供直接调用，绕过 FlagEmbedding 封装）
reranker_inner_model = None
reranker_tokenizer = None
embed_inner_model = None
embed_tokenizer = None

_embedding_col_type: str | None = None

# 请求并发控制
_search_sem: asyncio.Semaphore | None = None

# 模型调用串行锁（PyTorch 模型实例不支持多线程并发推理）
_embed_lock: asyncio.Lock | None = None
_rerank_lock: asyncio.Lock | None = None

# ─── Per-request 日志缓冲 ─────────────────────────────────
# create_task 和 run_in_executor 都会继承当前 context，
# 所以子任务/线程写入的也是同一个 list，最终统一输出，避免并发请求日志交错。
_log_buffer: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "log_buffer", default=None
)


def _rlog(*args: object, sep: str = " ") -> None:
    """向当前请求的日志缓冲追加一行；缓冲未设置时直接 print（启动日志等场景）。"""
    msg = sep.join(str(a) for a in args)
    buf = _log_buffer.get()
    if buf is not None:
        buf.append(msg)
    else:
        print(msg)


def _flush_log(buf: list[str]) -> None:
    """把缓冲内容原子写入 stdout，避免多请求行级交错。"""
    if buf:
        sys.stdout.write("\n".join(buf) + "\n")
        sys.stdout.flush()


def _get_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps:0"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, reranker_model, db_pool, _embedding_col_type
    global _search_sem, _embed_lock, _rerank_lock
    global reranker_inner_model, reranker_tokenizer, embed_inner_model, embed_tokenizer

    device = _get_device()
    print(f"推理设备: {device}")

    print("加载 embedding 模型...")
    embedding_model = BGEM3FlagModel(EMBEDDING_MODEL_PATH, use_fp16=True)
    embedding_model.model.to(device)
    embedding_model.model.model.eval()
    embed_inner_model = embedding_model.model.model  # 底层 AutoModel (XLM-RoBERTa)
    embed_tokenizer = embedding_model.tokenizer
    print(f"  embedding device: {next(embed_inner_model.parameters()).device}")

    print("加载 reranker 模型...")
    reranker_model = FlagReranker(RERANKER_MODEL_PATH, use_fp16=True)
    reranker_model.model.to(device)
    reranker_model.model.eval()
    reranker_inner_model = reranker_model.model   # 底层 AutoModelForSequenceClassification
    reranker_tokenizer = reranker_model.tokenizer
    print(f"  reranker device: {next(reranker_inner_model.parameters()).device}")

    print("✅ 模型加载完成（常驻内存, FP16 推理）")

    # MPS shader 预热：提前编译常用 sequence length 的 Metal kernel，
    # 避免第一次真实请求时触发 JIT 编译（几百 ms 延迟）
    import torch
    print("预热 MPS kernel...")
    with torch.no_grad():
        for seq_len in [32, 128, 256, 512]:
            dummy_text = "预热" * (seq_len // 2)
            d_emb = embed_tokenizer(
                [dummy_text], max_length=512, padding=True, truncation=True, return_tensors='pt'
            )
            embed_inner_model(**{k: v.to(device) for k, v in d_emb.items()})

            d_rnk = reranker_tokenizer(
                [["预热", dummy_text]], max_length=512, padding=True, truncation=True, return_tensors='pt'
            )
            reranker_inner_model(**{k: v.to(device) for k, v in d_rnk.items()})
    print("✅ MPS kernel 预热完成")

    db_pool = await asyncpg.create_pool(
        host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER,
        min_size=4, max_size=16,
    )
    print("✅ asyncpg 连接池就绪")

    row = await db_pool.fetchrow("""
        SELECT udt_name FROM information_schema.columns
        WHERE table_name = 'wiki_documents' AND column_name = 'embedding';
    """)
    _embedding_col_type = row["udt_name"]
    print(f"Embedding 列类型: {_embedding_col_type}")

    list(jieba.cut("预热分词"))
    print("✅ jieba 预热完成")

    _search_sem = asyncio.Semaphore(4)
    _embed_lock = asyncio.Lock()
    _rerank_lock = asyncio.Lock()
    print("✅ 搜索队列就绪 (concurrency=4, embed/rerank 串行锁已就绪)")

    yield

    await db_pool.close()
    thread_pool.shutdown(wait=False)


app = FastAPI(title="WikiRAG", lifespan=lifespan)


# ─── 请求/响应模型 ──────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=FINAL_TOP_K, ge=1, le=20)


class DocResult(BaseModel):
    id: int
    pageid: str
    title: str
    content: str
    score: float


class QueryResponse(BaseModel):
    query: str
    results: list[DocResult]


# ─── 核心函数 ────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    import torch
    device = next(embed_inner_model.parameters()).device
    inputs = embed_tokenizer(
        [query], max_length=512, padding=True, truncation=True, return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embed_inner_model(**inputs)
    # BGE-M3 dense 向量：CLS token 隐状态，L2 归一化
    cls_hidden = outputs.last_hidden_state[:, 0]
    embedding = torch.nn.functional.normalize(cls_hidden, p=2, dim=-1)
    return embedding[0].cpu().float().tolist()


async def vector_recall(query_vec: list[float], k: int) -> tuple[list[dict], float]:
    """pgvector 向量召回（asyncpg 原生异步）"""
    import json
    cast = f"::{_embedding_col_type}(1024)"
    vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"
    t0 = time.time()
    async with db_pool.acquire() as conn:
        await conn.execute(f"SET statement_timeout = '{int(VECTOR_TIMEOUT_S * 1000)}ms'")
        await conn.execute(f"SET hnsw.ef_search = {max(64, k)}")
        rows = await conn.fetch(
            f"""
            SELECT id, content, metadata,
                   1 - (embedding <=> $1{cast}) AS score
            FROM wiki_documents
            ORDER BY embedding <=> $1{cast}
            LIMIT $2;
            """,
            vec_str, k,
        )
    elapsed = (time.time() - t0) * 1000
    return [
        {"id": r["id"], "content": r["content"],
         "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
         "score": float(r["score"])}
        for r in rows
    ], elapsed


# ─── BM25 ────────────────────────────────────────────────

# 中文高频泛词停用词表，这类词选择性极低，放入 OR query 会命中大量无关行
_BM25_STOPWORDS = {
    "基本", "原理", "介绍", "方法", "作用", "什么", "怎么", "如何",
    "哪些", "一般", "通常", "主要", "相关", "以及", "所以", "因此",
    "但是", "然而", "还是", "还有", "这个", "那个", "这些", "那些",
    "可以", "需要", "进行", "使用", "通过", "对于", "关于", "由于",
    "根据", "包括", "其中", "之间", "之后", "之前", "以上", "以下",
    "就是", "一种", "一个", "我们", "他们", "它们", "这种", "那种",
    "方面", "问题", "情况", "过程", "系统", "技术", "应用", "研究",
    "分析", "实现", "结果", "影响", "发展", "目前", "已经", "非常",
}

# OR 回退最多保留的词数；pg_search 有 IDF，OR 查询不会再爆炸，但仍限制避免噪声
_BM25_OR_MAX_TERMS = 4


def _build_bm25_terms(query: str) -> list[str]:
    """分词 → 清理特殊字符 → 去重 → 去停用词"""
    raw = jieba.cut(query)
    seen: set[str] = set()
    terms: list[str] = []
    for t in raw:
        t = re.sub(r'[+\-&|!():{}\[\]^"~*?:\\]', "", t).strip()
        if len(t) < 2 or re.match(r'^[\s\W]+$', t):
            continue
        if t in seen or t in _BM25_STOPWORDS:
            continue
        seen.add(t)
        terms.append(t)
    return terms


async def bm25_recall(query: str, k: int) -> tuple[list[dict], float]:
    """真正的 BM25 召回（pg_search + Tantivy，有 IDF、TF 饱和、长度归一化）。
    AND 查询优先（所有词必须出现），不足时回退到 OR（IDF 自然抑制高频词噪声）。
    """
    import json

    terms = _build_bm25_terms(query)
    if not terms:
        return [], 0.0

    t0 = time.time()
    async with db_pool.acquire() as conn:
        await conn.execute(f"SET statement_timeout = '{int(BM25_TIMEOUT_S * 1000)}ms'")

        # AND：所有词必须出现（pg_search 查询语法：+field:term）
        if len(terms) >= 2:
            and_query = " ".join(f"+content_tokenized:{t}" for t in terms)
            rows = await conn.fetch(
                """
                SELECT id, content, metadata,
                       paradedb.score(id) AS score
                FROM wiki_documents
                WHERE wiki_documents @@@ $1
                ORDER BY score DESC
                LIMIT $2;
                """,
                and_query, k,
            )
            if len(rows) >= k:
                elapsed = (time.time() - t0) * 1000
                return [
                    {"id": r["id"], "content": r["content"],
                     "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
                     "score": float(r["score"])}
                    for r in rows
                ], elapsed

        # OR 回退：按词长降序取最多 _BM25_OR_MAX_TERMS 个词
        # IDF 负责抑制高频词，不再需要候选上限 hack
        or_terms = sorted(terms, key=len, reverse=True)[:_BM25_OR_MAX_TERMS]
        or_query = " ".join(f"content_tokenized:{t}" for t in or_terms)
        _rlog(f"  [bm25] AND 不足，OR fallback terms={or_terms}")
        rows = await conn.fetch(
            """
            SELECT id, content, metadata,
                   paradedb.score(id) AS score
            FROM wiki_documents
            WHERE wiki_documents @@@ $1
            ORDER BY score DESC
            LIMIT $2;
            """,
            or_query, k,
        )

    elapsed = (time.time() - t0) * 1000
    return [
        {"id": r["id"], "content": r["content"],
         "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
         "score": float(r["score"])}
        for r in rows
    ], elapsed


def hybrid_merge(vector_results: list[dict], bm25_results: list[dict], k: int) -> list[dict]:
    """并集 + RRF (Reciprocal Rank Fusion) 混合排分"""
    rrf_scores: dict[int, float] = {}
    doc_map: dict[int, dict] = {}

    for rank, doc in enumerate(vector_results):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rank + 60)
        doc_map[doc_id] = doc

    for rank, doc in enumerate(bm25_results):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rank + 60)
        doc_map[doc_id] = doc

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]
    return [
        {**doc_map[doc_id], "score": rrf_scores[doc_id]}
        for doc_id in sorted_ids
    ]


def rerank(query: str, docs: list[dict]) -> list[dict]:
    """Cross-encoder 精排（模型常驻内存，batch FP16 串行推理）。
    调用方须持有 _rerank_lock，确保同时只有一个推理任务在跑。
    """
    import torch
    if not docs:
        return []
    # 截断 content：cross-encoder 的 attention 是 O(L²)，
    # 一条长文档会把整批 50 条都 pad 到 512 token。
    pairs = [[query, doc["content"]] for doc in docs]
    device = next(reranker_inner_model.parameters()).device
    _rlog(f"  [rerank] {len(pairs)} pairs, device={device}")
    inputs = reranker_tokenizer(
        pairs, padding=True, truncation=True, max_length=512, return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = reranker_inner_model(**inputs)
    scores = torch.sigmoid(outputs.logits).squeeze(-1).cpu().float().tolist()
    if isinstance(scores, float):
        scores = [scores]
    for i, doc in enumerate(docs):
        doc["score"] = float(scores[i])
    return sorted(docs, key=lambda x: x["score"], reverse=True)[:RERANK_TOP_K]


async def mmr_select(query_vec: list[float], docs: list[dict], k: int, lam: float) -> list[dict]:
    """Maximal Marginal Relevance：平衡相关性与多样性"""
    if len(docs) <= k:
        return docs

    doc_ids = [doc["id"] for doc in docs]
    async with db_pool.acquire() as conn:
        await conn.execute(f"SET statement_timeout = '{int(MMR_TIMEOUT_S * 1000)}ms'")
        rows = await conn.fetch(
            "SELECT id, embedding::text FROM wiki_documents WHERE id = ANY($1);",
            doc_ids,
        )
    emb_map = {r["id"]: r["embedding"] for r in rows}

    doc_vecs = []
    for doc in docs:
        raw = emb_map.get(doc["id"])
        if raw:
            vec = np.array([float(x) for x in raw.strip("[]").split(",")], dtype=np.float32)
        else:
            vec = np.zeros(1024, dtype=np.float32)
        doc_vecs.append(vec)

    doc_norms = [v / (np.linalg.norm(v) + 1e-9) for v in doc_vecs]

    selected: list[int] = []
    candidates = list(range(len(docs)))

    for _ in range(k):
        best_idx = -1
        best_score = -float("inf")
        for i in candidates:
            relevance = docs[i]["score"]
            max_sim = max(
                (float(np.dot(doc_norms[i], doc_norms[j])) for j in selected),
                default=0.0,
            )
            mmr_score = lam * relevance - (1 - lam) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        if best_idx == -1:
            break
        selected.append(best_idx)
        candidates.remove(best_idx)

    return [docs[i] for i in selected]


# ─── 调试辅助 ──────────────────────────────────────────────

def _trunc(text: str, max_len: int = 10) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


def _log_docs(label: str, docs: list[dict], top_n: int = 3) -> None:
    _rlog(f"  [{label}] 共 {len(docs)} 条, top {min(top_n, len(docs))}:")
    for doc in docs[:top_n]:
        meta = doc.get("metadata", {})
        title = _trunc(str(meta.get("title", "")))
        content = _trunc(doc.get("content", ""))
        _rlog(f"    id={doc['id']} score={doc['score']:.4f} title={title} content={content}")


# ─── API 路由 ────────────────────────────────────────────

@app.post("/search", response_model=QueryResponse)
async def search(req: QueryRequest):
    async with _search_sem:
        return await _do_search(req)


async def _do_search(req: QueryRequest):
    query = req.query
    top_k = req.top_k
    loop = asyncio.get_event_loop()

    # 初始化 per-request 日志缓冲，子任务和 executor 线程通过 ContextVar 继承
    buf: list[str] = []
    token = _log_buffer.set(buf)

    try:
        _rlog(f"\n{'='*60}")
        _rlog(f"Query: {query}")

        # 1. Query Embedding（持锁串行，避免模型并发）
        t0 = time.time()
        async with _embed_lock:
            query_vec = await loop.run_in_executor(thread_pool, embed_query, query)
        t_embed = (time.time() - t0) * 1000
        _rlog(f"[1] Embedding: {t_embed:.1f}ms  vec[:5]={[round(v, 4) for v in query_vec[:5]]}")

        # 2. 多路召回（并发，任一超时/异常则取消，用已完成的继续）
        t0 = time.time()
        vec_task = asyncio.create_task(vector_recall(query_vec, VECTOR_RECALL_K))
        bm25_task = asyncio.create_task(bm25_recall(query, BM25_RECALL_K))

        done, pending = await asyncio.wait(
            [vec_task, bm25_task], return_when=asyncio.FIRST_EXCEPTION,
        )
        if pending:
            _, still_pending = await asyncio.wait(pending, timeout=0.5)
            for t in still_pending:
                t.cancel()

        vec_results, t_vec = [], 0.0
        bm25_results, t_bm25 = [], 0.0
        if vec_task.done() and not vec_task.cancelled() and vec_task.exception() is None:
            vec_results, t_vec = vec_task.result()
        if bm25_task.done() and not bm25_task.cancelled() and bm25_task.exception() is None:
            bm25_results, t_bm25 = bm25_task.result()

        t_recall = (time.time() - t0) * 1000
        _rlog(f"[2] Parallel recall: {t_recall:.1f}ms (wall) | "
              f"vector={t_vec:.1f}ms({len(vec_results)}条) | "
              f"bm25={t_bm25:.1f}ms({len(bm25_results)}条)")
        if not vec_results and vec_task.done() and vec_task.exception():
            _rlog(f"  ⚠ vector recall 超时/异常: {vec_task.exception()}")
        if not bm25_results and bm25_task.done() and bm25_task.exception():
            _rlog(f"  ⚠ bm25 recall 超时/异常: {bm25_task.exception()}")
        _log_docs("vector", vec_results)
        _log_docs("bm25", bm25_results)

        # 3. RRF 混合排分
        t0 = time.time()
        merged = hybrid_merge(vec_results, bm25_results, HYBRID_TOP_K)
        t_merge = (time.time() - t0) * 1000
        _rlog(f"[3] Hybrid merge: {t_merge:.1f}ms")
        _log_docs("merged", merged)

        # 4. Cross-encoder 精排（持锁串行 + 按分数阈值动态裁剪候选数）
        to_rerank = [d for d in merged if d["score"] >= RERANK_MIN_HYBRID_SCORE]
        if not to_rerank:
            # 全部低于阈值时兜底取 top-1，保证流水线不断
            to_rerank = merged[:1]
        _rlog(f"  [rerank] 送入 {len(to_rerank)}/{len(merged)} 条（score >= {RERANK_MIN_HYBRID_SCORE}）")

        t0 = time.time()
        async with _rerank_lock:
            reranked = await loop.run_in_executor(thread_pool, rerank, query, to_rerank)
        t_rerank = (time.time() - t0) * 1000
        _rlog(f"[4] Rerank: {t_rerank:.1f}ms")
        _log_docs("reranked", reranked)

        # 5. MMR 去重
        t0 = time.time()
        final = await mmr_select(query_vec, reranked, top_k, MMR_LAMBDA)
        t_mmr = (time.time() - t0) * 1000
        _rlog(f"[5] MMR: {t_mmr:.1f}ms")
        _log_docs("final", final)

        t_total = t_embed + t_recall + t_merge + t_rerank + t_mmr
        _rlog(f"[Total] {t_total:.1f}ms")
        _rlog(f"{'='*60}")

    finally:
        # 无论成功或异常，都原子输出本次请求的全部日志
        _log_buffer.reset(token)
        _flush_log(buf)

    # 格式化输出（过滤低分结果）
    results = []
    for doc in final:
        if doc["score"] <= SCORE_THRESHOLD:
            continue
        meta = doc.get("metadata", {})
        results.append(DocResult(
            id=doc["id"],
            pageid=str(meta.get("pageid", "")),
            title=str(meta.get("title", "")),
            content=doc["content"],
            score=round(doc["score"], 4),
        ))

    return QueryResponse(query=query, results=results)


@app.post("/search/debug")
async def search_debug(req: QueryRequest):
    """返回完整召回链路的排名信息，用于分析 top-K 设置是否合理。不做日志缓冲。"""
    async with _search_sem:
        return await _do_search_debug(req)


async def _do_search_debug(req: QueryRequest):
    query = req.query
    loop = asyncio.get_event_loop()

    async with _embed_lock:
        query_vec = await loop.run_in_executor(thread_pool, embed_query, query)

    vec_task = asyncio.create_task(vector_recall(query_vec, VECTOR_RECALL_K))
    bm25_task = asyncio.create_task(bm25_recall(query, BM25_RECALL_K))
    await asyncio.gather(vec_task, bm25_task, return_exceptions=True)

    vec_results, _ = vec_task.result() if (vec_task.done() and not vec_task.exception()) else ([], 0)
    bm25_results, _ = bm25_task.result() if (bm25_task.done() and not bm25_task.exception()) else ([], 0)

    vec_rank  = {doc["id"]: i + 1 for i, doc in enumerate(vec_results)}
    bm25_rank = {doc["id"]: i + 1 for i, doc in enumerate(bm25_results)}

    merged = hybrid_merge(vec_results, bm25_results, HYBRID_TOP_K)
    merged_rank = {doc["id"]: i + 1 for i, doc in enumerate(merged)}

    to_rerank = [d for d in merged if d["score"] >= RERANK_MIN_HYBRID_SCORE] or merged[:1]
    async with _rerank_lock:
        reranked = await loop.run_in_executor(thread_pool, rerank, query, to_rerank)
    reranked_rank = {doc["id"]: i + 1 for i, doc in enumerate(reranked)}

    final = await mmr_select(query_vec, reranked, req.top_k, MMR_LAMBDA)

    return {
        "query": query,
        "counts": {
            "vector": len(vec_results),
            "bm25": len(bm25_results),
            "merged": len(merged),
            "rerank_input": len(to_rerank),
            "reranked": len(reranked),
            "final": len(final),
        },
        "final_docs": [
            {
                "id": doc["id"],
                "title": doc.get("metadata", {}).get("title", ""),
                "rerank_score": round(doc["score"], 4),
                "vector_rank":   vec_rank.get(doc["id"]),    # None = 未被向量召回
                "bm25_rank":     bm25_rank.get(doc["id"]),   # None = 未被 BM25 召回
                "merged_rank":   merged_rank.get(doc["id"]),
                "reranked_rank": reranked_rank.get(doc["id"]),
            }
            for doc in final
        ],
        # 送入 rerank 的所有候选的 merged_rank 分布，用于画完整分布图
        "rerank_input_merged_ranks": [merged_rank[d["id"]] for d in to_rerank],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"service": "WikiRAG", "docs": "/docs", "health": "/health"}
