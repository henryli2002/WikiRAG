import os
import re
import time
import numpy as np
import psycopg2
import psycopg2.extras
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

VECTOR_RECALL_K = 40   # 向量召回数
BM25_RECALL_K = 30     # BM25 召回数
HYBRID_TOP_K = 50      # 混合排序取前 N
RERANK_TOP_K = 15      # 精排取前 N
FINAL_TOP_K = 5        # MMR 最终返回数
MMR_LAMBDA = 0.7       # MMR 多样性参数 (越大越相关，越小越多样)


# ─── 全局模型 & 数据库连接 ───────────────────────────────
embedding_model = None
reranker_model = None


def get_db():
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER)
    return conn


def _get_device() -> str:
    """优先 MPS (Apple Silicon)，其次 CUDA，最后 CPU"""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, reranker_model
    device = _get_device()
    print(f"推理设备: {device}")
    print("加载 embedding 模型...")
    embedding_model = BGEM3FlagModel(EMBEDDING_MODEL_PATH, use_fp16=True, devices=[device])
    print("加载 reranker 模型...")
    reranker_model = FlagReranker(RERANKER_MODEL_PATH, use_fp16=True, devices=[device])
    print("✅ 模型加载完成")
    yield


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
    """对 query 进行向量化 (use_fp16=True, 输出即 FP16)"""
    out = embedding_model.encode([query], max_length=512)
    return out["dense_vecs"][0].tolist()


def _get_embedding_type(cursor) -> str:
    """检测 embedding 列的存储类型"""
    cursor.execute("""
        SELECT udt_name FROM information_schema.columns
        WHERE table_name = 'wiki_documents' AND column_name = 'embedding';
    """)
    return cursor.fetchone()[0]  # "vector" 或 "halfvec"


def vector_recall(cursor, query_vec: list[float], k: int) -> list[dict]:
    """pgvector 向量召回，自动适配 embedding 列类型"""
    col_type = _get_embedding_type(cursor)
    cast = f"::{col_type}(1024)"

    cursor.execute(
        f"""
        SELECT id, content, metadata,
               1 - (embedding <=> %s{cast}) AS score
        FROM wiki_documents
        ORDER BY embedding <=> %s{cast}
        LIMIT %s;
        """,
        (query_vec, query_vec, k),
    )
    rows = cursor.fetchall()
    return [
        {"id": r[0], "content": r[1], "metadata": r[2], "score": float(r[3])}
        for r in rows
    ]


def bm25_recall(cursor, query: str, k: int) -> list[dict]:
    """
    BM25 召回：jieba 分词 → simple 字典 tsquery → GIN 索引检索 + ts_rank 排序。
    完全走 PostgreSQL 原生全文检索，无需 Python 端计算。
    """
    # jieba 分词，过滤标点和单字词
    tokens = jieba.cut(query)
    terms = [re.sub(r"[&|!():'\\]", "", t).strip() for t in tokens]
    terms = [t for t in terms if len(t) >= 2 and not re.match(r'^[\s\W]+$', t)]
    if not terms:
        return []

    def _do_query(tsquery_str):
        cursor.execute(
            """
            SELECT id, content, metadata,
                   ts_rank(tsv, to_tsquery('simple', %s)) AS score
            FROM wiki_documents
            WHERE tsv @@ to_tsquery('simple', %s)
            ORDER BY score DESC
            LIMIT %s;
            """,
            (tsquery_str, tsquery_str, k),
        )
        return cursor.fetchall()

    # 策略：先 AND，结果不够则 fallback 到 OR
    if len(terms) >= 2:
        rows = _do_query(" & ".join(terms))
        if len(rows) >= k:
            return [
                {"id": r[0], "content": r[1], "metadata": r[2], "score": float(r[3])}
                for r in rows
            ]

    rows = _do_query(" | ".join(terms))
    return [
        {"id": r[0], "content": r[1], "metadata": r[2], "score": float(r[3])}
        for r in rows
    ]


def hybrid_merge(vector_results: list[dict], bm25_results: list[dict], k: int) -> list[dict]:
    """
    并集 + RRF (Reciprocal Rank Fusion) 混合排分。
    RRF 公式：score = sum(1 / (rank + 60))
    """
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
    """使用 cross-encoder 精排"""
    if not docs:
        return []
    pairs = [[query, doc["content"]] for doc in docs]
    scores = reranker_model.compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]
    for i, doc in enumerate(docs):
        doc["score"] = float(scores[i])
    return sorted(docs, key=lambda x: x["score"], reverse=True)[:RERANK_TOP_K]


def mmr_select(query_vec: list[float], docs: list[dict], k: int, lam: float) -> list[dict]:
    """
    Maximal Marginal Relevance：平衡相关性与多样性。
    score = λ * relevance - (1-λ) * max_similarity_to_selected
    """
    if len(docs) <= k:
        return docs

    # 为每个 doc 拿 embedding
    doc_vecs = []
    conn = get_db()
    cursor = conn.cursor()
    for doc in docs:
        cursor.execute("SELECT embedding::text FROM wiki_documents WHERE id = %s;", (doc["id"],))
        row = cursor.fetchone()
        if row and row[0]:
            vec = np.array([float(x) for x in row[0].strip("[]").split(",")], dtype=np.float32)
        else:
            vec = np.zeros(1024, dtype=np.float32)
        doc_vecs.append(vec)
    cursor.close()
    conn.close()

    query_np = np.array(query_vec, dtype=np.float32)
    # 归一化
    query_norm = query_np / (np.linalg.norm(query_np) + 1e-9)
    doc_norms = [v / (np.linalg.norm(v) + 1e-9) for v in doc_vecs]

    selected = []
    candidates = list(range(len(docs)))

    for _ in range(k):
        best_idx = -1
        best_score = -float("inf")

        for i in candidates:
            relevance = docs[i]["score"]  # reranker 分数
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
def search(req: QueryRequest):
    query = req.query
    top_k = req.top_k
    print(f"\n{'='*60}")
    print(f"Query: {query}")

    # 1. Query Embedding
    t0 = time.time()
    query_vec = embed_query(query)
    t_embed = (time.time() - t0) * 1000
    print(f"[1] Embedding: {t_embed:.1f}ms  vec[:10]={[round(v, 4) for v in query_vec[:10]]}")

    conn = get_db()
    cursor = conn.cursor()

    # 2. 多路召回
    t0 = time.time()
    vec_results = vector_recall(cursor, query_vec, VECTOR_RECALL_K)
    t_vec = (time.time() - t0) * 1000
    print(f"[2a] Vector recall: {t_vec:.1f}ms")
    _log_docs("vector", vec_results)

    t0 = time.time()
    bm25_results = bm25_recall(cursor, query, BM25_RECALL_K)
    t_bm25 = (time.time() - t0) * 1000
    print(f"[2b] BM25 recall: {t_bm25:.1f}ms")
    _log_docs("bm25", bm25_results)

    cursor.close()
    conn.close()

    # 3. 并集 + RRF 混合排分 → top 50
    t0 = time.time()
    merged = hybrid_merge(vec_results, bm25_results, HYBRID_TOP_K)
    t_merge = (time.time() - t0) * 1000
    print(f"[3] Hybrid merge: {t_merge:.1f}ms")
    _log_docs("merged", merged)

    # 4. Cross-encoder 精排 → top 15
    t0 = time.time()
    reranked = rerank(query, merged)
    t_rerank = (time.time() - t0) * 1000
    print(f"[4] Rerank: {t_rerank:.1f}ms")
    _log_docs("reranked", reranked)

    # 5. MMR 去重 → top K
    t0 = time.time()
    final = mmr_select(query_vec, reranked, top_k, MMR_LAMBDA)
    t_mmr = (time.time() - t0) * 1000
    print(f"[5] MMR: {t_mmr:.1f}ms")
    _log_docs("final", final)

    t_total = t_embed + t_vec + t_bm25 + t_merge + t_rerank + t_mmr
    print(f"[Total] {t_total:.1f}ms")
    print(f"{'='*60}\n")

    # 格式化输出
    results = []
    for doc in final:
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
