import os
import re
import time
import asyncio
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
BM25_RECALL_K = 10     # BM25 召回数
HYBRID_TOP_K = 32      # 混合排序取前 N
RERANK_TOP_K = 10      # 精排取前 N
FINAL_TOP_K = 5        # MMR 最终返回数
MMR_LAMBDA = 0.7       # MMR 多样性参数 (越大越相关，越小越多样)
SCORE_THRESHOLD = 0.8  # rerank 分数阈值，<=0.8 判定为不相似

VECTOR_TIMEOUT_S = 5.0   # 向量召回超时
BM25_TIMEOUT_S = 5.0     # BM25 召回超时
MMR_TIMEOUT_S = 100.0     # MMR 超时

# ─── 全局模型 & 连接池 ────────────────────────────────────
embedding_model = None
reranker_model = None
thread_pool = ThreadPoolExecutor(max_workers=8)
db_pool: asyncpg.Pool | None = None

# 缓存 embedding 列类型
_embedding_col_type: str | None = None

# 全局搜索信号量：同一时刻只处理 1 个搜索请求，其余排队
_search_sem: asyncio.Semaphore | None = None


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
    device = _get_device()
    print(f"推理设备: {device}")
    print("加载 embedding 模型...")
    embedding_model = BGEM3FlagModel(EMBEDDING_MODEL_PATH, use_fp16=True)
    embedding_model.model.to(device)
    print(f"  embedding device: {next(embedding_model.model.parameters()).device}")
    print("加载 reranker 模型...")
    reranker_model = FlagReranker(RERANKER_MODEL_PATH, use_fp16=True, batch_size=32)
    reranker_model.model.to(device)
    print(f"  reranker device: {next(reranker_model.model.parameters()).device}")
    print("✅ 模型加载完成（常驻内存, FP16 推理）")

    # 异步连接池
    db_pool = await asyncpg.create_pool(
        host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER,
        min_size=4, max_size=16,
    )
    print("✅ asyncpg 连接池就绪")

    # 预缓存 embedding 列类型
    row = await db_pool.fetchrow("""
        SELECT udt_name FROM information_schema.columns
        WHERE table_name = 'wiki_documents' AND column_name = 'embedding';
    """)
    _embedding_col_type = row["udt_name"]
    print(f"Embedding 列类型: {_embedding_col_type}")

    # 预热 jieba
    list(jieba.cut("预热分词"))
    print("✅ jieba 预热完成")

    global _search_sem
    _search_sem = asyncio.Semaphore(1)
    print("✅ 搜索队列就绪 (concurrency=1)")

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
    out = embedding_model.encode([query], max_length=512)
    return out["dense_vecs"][0].tolist()


async def vector_recall(query_vec: list[float], k: int) -> tuple[list[dict], float]:
    """pgvector 向量召回（asyncpg 原生异步）"""
    cast = f"::{_embedding_col_type}(1024)"
    vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"
    t0 = time.time()
    async with db_pool.acquire() as conn:
        await conn.execute(f"SET statement_timeout = '{int(VECTOR_TIMEOUT_S * 1000)}ms'")
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
    import json
    results = [
        {"id": r["id"], "content": r["content"],
         "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
         "score": float(r["score"])}
        for r in rows
    ]
    return results, elapsed


async def bm25_recall(query: str, k: int) -> tuple[list[dict], float]:
    """BM25 召回（asyncpg 原生异步）"""
    tokens = jieba.cut(query)
    terms = [re.sub(r"[&|!():'\\]", "", t).strip() for t in tokens]
    terms = [t for t in terms if len(t) >= 2 and not re.match(r'^[\s\W]+$', t)]
    if not terms:
        return [], 0.0

    t0 = time.time()
    async with db_pool.acquire() as conn:
        await conn.execute(f"SET statement_timeout = '{int(BM25_TIMEOUT_S * 1000)}ms'")
        # 先 AND，结果不够则 fallback 到 OR
        if len(terms) >= 2:
            tsquery_and = " & ".join(terms)
            rows = await conn.fetch(
                """
                SELECT id, content, metadata,
                       ts_rank(tsv, to_tsquery('simple', $1)) AS score
                FROM wiki_documents
                WHERE tsv @@ to_tsquery('simple', $1)
                ORDER BY score DESC
                LIMIT $2;
                """,
                tsquery_and, k,
            )
            if len(rows) >= k:
                elapsed = (time.time() - t0) * 1000
                import json
                return [
                    {"id": r["id"], "content": r["content"],
                     "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
                     "score": float(r["score"])}
                    for r in rows
                ], elapsed

        tsquery_or = " | ".join(terms)
        rows = await conn.fetch(
            """
            SELECT id, content, metadata,
                   ts_rank(tsv, to_tsquery('simple', $1)) AS score
            FROM wiki_documents
            WHERE tsv @@ to_tsquery('simple', $1)
            ORDER BY score DESC
            LIMIT $2;
            """,
            tsquery_or, k,
        )

    elapsed = (time.time() - t0) * 1000
    import json
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
    """cross-encoder 精排（模型常驻内存，batch FP16 并行推理）"""
    if not docs:
        return []
    pairs = [[query, doc["content"]] for doc in docs]
    print(f"  [rerank] {len(pairs)} pairs, batch_size={reranker_model.batch_size}, "
          f"device={next(reranker_model.model.parameters()).device}, "
          f"fp16={reranker_model.use_fp16}")
    scores = reranker_model.compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]
    for i, doc in enumerate(docs):
        doc["score"] = float(scores[i])
    return sorted(docs, key=lambda x: x["score"], reverse=True)[:RERANK_TOP_K]


async def mmr_select(query_vec: list[float], docs: list[dict], k: int, lam: float) -> list[dict]:
    """Maximal Marginal Relevance：平衡相关性与多样性"""
    if len(docs) <= k:
        return docs

    # 批量拉取 embedding（异步）
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

    query_np = np.array(query_vec, dtype=np.float32)
    query_norm = query_np / (np.linalg.norm(query_np) + 1e-9)
    doc_norms = [v / (np.linalg.norm(v) + 1e-9) for v in doc_vecs]

    selected = []
    candidates = list(range(len(docs)))

    for _ in range(k):
        best_idx = -1
        best_score = -float("inf")

        for i in candidates:
            relevance = docs[i]["score"]
            if selected:
                max_sim = max(float(np.dot(doc_norms[i], doc_norms[j])) for j in selected)
            else:
                max_sim = 0.0
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


def _log_docs(label: str, docs: list[dict], top_n: int = 3):
    print(f"  [{label}] 共 {len(docs)} 条, top {min(top_n, len(docs))}:")
    for doc in docs[:top_n]:
        meta = doc.get("metadata", {})
        title = _trunc(str(meta.get("title", "")))
        content = _trunc(doc.get("content", ""))
        print(f"    id={doc['id']} score={doc['score']:.4f} title={title} content={content}")


# ─── API 路由 ────────────────────────────────────────────

@app.post("/search", response_model=QueryResponse)
async def search(req: QueryRequest):
    async with _search_sem:
        return await _do_search(req)


async def _do_search(req: QueryRequest):
    query = req.query
    top_k = req.top_k
    loop = asyncio.get_event_loop()
    print(f"\n{'='*60}")
    print(f"Query: {query}")

    # 1. Query Embedding（线程池，避免阻塞事件循环）
    t0 = time.time()
    query_vec = await loop.run_in_executor(thread_pool, embed_query, query)
    t_embed = (time.time() - t0) * 1000
    print(f"[1] Embedding: {t_embed:.1f}ms  vec[:10]={[round(v, 4) for v in query_vec[:10]]}")

    # 2. 多路召回（并发，任一超时/异常则取消它，用已完成的继续）
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

    vec_results, t_vec = ([], 0.0)
    bm25_results, t_bm25 = ([], 0.0)
    if vec_task.done() and not vec_task.cancelled() and vec_task.exception() is None:
        vec_results, t_vec = vec_task.result()
    if bm25_task.done() and not bm25_task.cancelled() and bm25_task.exception() is None:
        bm25_results, t_bm25 = bm25_task.result()

    t_recall = (time.time() - t0) * 1000
    print(f"[2] Parallel recall: {t_recall:.1f}ms (wall) | vector={t_vec:.1f}ms({len(vec_results)}条) | bm25={t_bm25:.1f}ms({len(bm25_results)}条)")
    if not vec_results and vec_task.done() and vec_task.exception():
        print(f"  ⚠ vector recall 超时/异常: {vec_task.exception()}")
    if not bm25_results and bm25_task.done() and bm25_task.exception():
        print(f"  ⚠ bm25 recall 超时/异常: {bm25_task.exception()}")
    _log_docs("vector", vec_results)
    _log_docs("bm25", bm25_results)

    # 3. 并集 + RRF 混合排分 → top 50
    t0 = time.time()
    merged = hybrid_merge(vec_results, bm25_results, HYBRID_TOP_K)
    t_merge = (time.time() - t0) * 1000
    print(f"[3] Hybrid merge: {t_merge:.1f}ms")
    _log_docs("merged", merged)

    # 4. Cross-encoder 精排 → top 15（线程池，GPU 密集）
    t0 = time.time()
    reranked = await loop.run_in_executor(thread_pool, rerank, query, merged)
    t_rerank = (time.time() - t0) * 1000
    print(f"[4] Rerank: {t_rerank:.1f}ms")
    _log_docs("reranked", reranked)

    # 5. MMR 去重 → top K
    t0 = time.time()
    final = await mmr_select(query_vec, reranked, top_k, MMR_LAMBDA)
    t_mmr = (time.time() - t0) * 1000
    print(f"[5] MMR: {t_mmr:.1f}ms")
    _log_docs("final", final)

    t_total = t_embed + t_recall + t_merge + t_rerank + t_mmr
    print(f"[Total] {t_total:.1f}ms")
    print(f"{'='*60}\n")

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"service": "WikiRAG", "docs": "/docs", "health": "/health"}
