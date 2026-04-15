"""
eval/retrieval_core.py
======================
评测专用检索核心引擎。

从 app/main.py 提取全部检索逻辑，剥离：
  - FastAPI / HTTP 层
  - asyncio.Semaphore / Lock（评测为顺序单线程，无需竞态保护）
  - RERANK_MIN_HYBRID_SCORE 过滤（评测需要完整候选，不能截断）
  - SCORE_THRESHOLD 过滤（评测保留所有原始分数）

增加：
  - PipelineResult：持有所有中间状态（各阶段结果 + rank 索引）
  - StageTimings：精确到毫秒的各阶段独立计时
  - PipelineMonitor：run_pipeline 期间的实时 verbose 输出（可开关）
  - RetrievalCore.run_pipeline()：一次调用跑完完整流水线并返回上述数据

超参默认值与 app/main.py 对齐，评测脚本可在构造函数中覆盖任意参数。
"""

from __future__ import annotations

import os
import re
import sys
import time
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import asyncpg
import jieba

# ─── 路径 ────────────────────────────────────────────────────────────
PROJECT_ROOT         = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMBEDDING_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bge-m3")
RERANKER_MODEL_PATH  = os.path.join(PROJECT_ROOT, "models", "bge-reranker-v2-m3")

# ─── 默认超参（与 main.py 保持一致） ─────────────────────────────────
DEFAULT_VECTOR_K        = 50
DEFAULT_BM25_K          = 50
DEFAULT_HYBRID_TOP_K    = 30
DEFAULT_RERANK_TOP_K    = 10
DEFAULT_FINAL_K         = 5
DEFAULT_MMR_LAMBDA      = 0.7

# BM25 停用词（与 main.py 完全一致）
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
_BM25_OR_MAX_TERMS = 4


# ══════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════

@dataclass
class StageTimings:
    """各阶段独立耗时（ms）。并发召回以各自的 DB 耗时计，wall-clock 取 max。"""
    embed_ms:  float = 0.0
    vector_ms: float = 0.0   # 向量召回 DB 耗时
    bm25_ms:   float = 0.0   # BM25 召回 DB 耗时
    merge_ms:  float = 0.0
    rerank_ms: float = 0.0
    mmr_ms:    float = 0.0

    @property
    def recall_wall_ms(self) -> float:
        """向量 + BM25 并发 wall-clock（取较大值）"""
        return max(self.vector_ms, self.bm25_ms)

    @property
    def total_ms(self) -> float:
        """完整流水线端到端耗时（embed + recall_wall + merge + rerank + mmr）"""
        return self.embed_ms + self.recall_wall_ms + self.merge_ms + self.rerank_ms + self.mmr_ms


@dataclass
class PipelineResult:
    """
    单次 run_pipeline 调用的完整快照。
    每个阶段都保留原始文档列表和 doc_id→rank 索引，
    供评测脚本直接查询任意 chunk 在任意阶段的排名。
    """
    query:     str
    query_vec: list[float]

    # ── 各阶段原始文档列表（list[dict]，每条含 id/content/metadata/score） ──
    vec_results:  list[dict] = field(default_factory=list)
    bm25_results: list[dict] = field(default_factory=list)
    merged:       list[dict] = field(default_factory=list)
    reranked:     list[dict] = field(default_factory=list)
    final:        list[dict] = field(default_factory=list)

    # ── doc_id → rank（1-based，不存在则该 stage 字典中无此 key） ────────
    vec_rank:      dict[int, int] = field(default_factory=dict)
    bm25_rank:     dict[int, int] = field(default_factory=dict)
    merged_rank:   dict[int, int] = field(default_factory=dict)
    reranked_rank: dict[int, int] = field(default_factory=dict)
    final_rank:    dict[int, int] = field(default_factory=dict)

    timings: StageTimings = field(default_factory=StageTimings)

    # ── 错误信息 ─────────────────────────────────────────────────────
    vec_error:  str | None = None
    bm25_error: str | None = None

    def rank_of(self, doc_id: int, stage: str) -> int | None:
        """查询 doc_id 在指定阶段的排名，未出现返回 None。"""
        mapping = {
            "vec":      self.vec_rank,
            "bm25":     self.bm25_rank,
            "merged":   self.merged_rank,
            "reranked": self.reranked_rank,
            "final":    self.final_rank,
        }
        return mapping[stage].get(doc_id)

    def rerank_inversion(self, doc_id: int, inversion_gap: int = 5) -> bool:
        """
        判断 doc_id 是否遭遇排序倒挂（死因 C）：
        进了 merged 前 10，但 reranked 排名倒退超过 inversion_gap，
        或直接被踢出 rerank_top_k（reranked_rank 为 None）。
        """
        mr = self.rank_of(doc_id, "merged")
        rr = self.rank_of(doc_id, "reranked")
        if mr is None or mr > 10:
            return False
        return rr is None or rr > mr + inversion_gap


# ══════════════════════════════════════════════════════════════════════
# 实时监控打印器
# ══════════════════════════════════════════════════════════════════════

class PipelineMonitor:
    """
    run_pipeline 期间的实时状态打印（verbose=True 时启用）。
    与 main.py 的 _rlog / _log_docs 等价，但直接写 stdout，无需 ContextVar。
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def header(self, query: str) -> None:
        if not self.verbose:
            return
        print(f"\n{'═' * 64}")
        print(f"  Query: {query}")
        print(f"{'─' * 64}")

    def stage(self, label: str, msg: str) -> None:
        if not self.verbose:
            return
        print(f"  [{label}] {msg}")

    def docs(self, label: str, docs: list[dict], top_n: int = 3) -> None:
        if not self.verbose:
            return
        print(f"  [{label}] 共 {len(docs)} 条，top {min(top_n, len(docs))}：")
        for doc in docs[:top_n]:
            meta    = doc.get("metadata", {})
            title   = str(meta.get("title", ""))[:12]
            content = doc.get("content", "")[:12] + "..."
            print(f"    id={doc['id']:6d}  score={doc['score']:.4f}"
                  f"  title={title}  content={content}")

    def timings(self, t: StageTimings) -> None:
        if not self.verbose:
            return
        print(f"{'─' * 64}")
        print(f"  embed={t.embed_ms:.1f}ms | "
              f"recall_wall={t.recall_wall_ms:.1f}ms "
              f"(vec={t.vector_ms:.1f} bm25={t.bm25_ms:.1f}) | "
              f"merge={t.merge_ms:.1f}ms | rerank={t.rerank_ms:.1f}ms | "
              f"mmr={t.mmr_ms:.1f}ms | total={t.total_ms:.1f}ms")
        print(f"{'═' * 64}\n")

    def warn(self, msg: str) -> None:
        print(f"  ⚠ {msg}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════
# 核心引擎
# ══════════════════════════════════════════════════════════════════════

class RetrievalCore:
    """
    评测专用检索引擎。

    与 app/main.py 的区别：
      - 无 FastAPI / HTTP 层
      - 无并发锁（评测为顺序调用，单 ThreadPoolExecutor(max_workers=1)）
      - run_pipeline 不施加 RERANK_MIN_HYBRID_SCORE / SCORE_THRESHOLD 过滤
      - 返回 PipelineResult，包含所有中间状态

    典型用法：
        core = RetrievalCore()
        core.load_models()
        await core.connect_db()
        result = await core.run_pipeline("量子计算的基本原理")
        await core.close()
    """

    def __init__(
        self,
        vector_k:      int   = DEFAULT_VECTOR_K,
        bm25_k:        int   = DEFAULT_BM25_K,
        hybrid_top_k:  int   = DEFAULT_HYBRID_TOP_K,
        rerank_top_k:  int   = DEFAULT_RERANK_TOP_K,
        final_k:       int   = DEFAULT_FINAL_K,
        mmr_lambda:    float = DEFAULT_MMR_LAMBDA,
        db_host: str | None = None,
        db_port: str | None = None,
        db_name: str | None = None,
        db_user: str | None = None,
        verbose: bool = False,
    ):
        self.vector_k     = vector_k
        self.bm25_k       = bm25_k
        self.hybrid_top_k = hybrid_top_k
        self.rerank_top_k = rerank_top_k
        self.final_k      = final_k
        self.mmr_lambda   = mmr_lambda

        self.db_host = db_host or os.getenv("DB_HOST", "localhost")
        self.db_port = db_port or os.getenv("DB_PORT", "5432")
        self.db_name = db_name or os.getenv("POSTGRES_DB", "rag_db")
        self.db_user = db_user or os.getenv("POSTGRES_USER", "rag_user")

        self._monitor = PipelineMonitor(verbose=verbose)

        # 模型相关（load_models() 后填充）
        self._embed_inner      = None
        self._embed_tokenizer  = None
        self._reranker_inner   = None
        self._reranker_tok     = None
        self._device: str | None = None

        # DB 相关（connect_db() 后填充）
        self._db_pool: asyncpg.Pool | None = None
        self._embedding_col_type: str | None = None

        # 评测为顺序单线程，max_workers=1 即可
        self._executor = ThreadPoolExecutor(max_workers=1)

    # ── 初始化 ────────────────────────────────────────────────────────

    def _detect_device(self) -> str:
        import torch
        if torch.backends.mps.is_available():
            return "mps:0"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_models(self, warmup: bool = True) -> None:
        """
        同步加载 embedding + reranker 模型并（可选）预热 kernel。
        调用一次即可，之后可反复调用 run_pipeline。
        """
        from FlagEmbedding import BGEM3FlagModel, FlagReranker
        import torch

        self._device = self._detect_device()
        print(f"[RetrievalCore] 推理设备: {self._device}")

        print("[RetrievalCore] 加载 embedding 模型...")
        embed_model = BGEM3FlagModel(EMBEDDING_MODEL_PATH, use_fp16=True)
        embed_model.model.to(self._device)
        embed_model.model.model.eval()
        self._embed_inner     = embed_model.model.model   # 底层 AutoModel (XLM-RoBERTa)
        self._embed_tokenizer = embed_model.tokenizer

        print("[RetrievalCore] 加载 reranker 模型...")
        reranker_model = FlagReranker(RERANKER_MODEL_PATH, use_fp16=True)
        reranker_model.model.to(self._device)
        reranker_model.model.eval()
        self._reranker_inner = reranker_model.model       # 底层 AutoModelForSequenceClassification
        self._reranker_tok   = reranker_model.tokenizer

        if warmup:
            print("[RetrievalCore] 预热 kernel...")
            with torch.no_grad():
                for seq_len in [32, 128, 256, 512]:
                    dummy = "预热" * (seq_len // 2)
                    d = self._embed_tokenizer(
                        [dummy], max_length=512, padding=True,
                        truncation=True, return_tensors="pt",
                    )
                    self._embed_inner(**{k: v.to(self._device) for k, v in d.items()})
                    d2 = self._reranker_tok(
                        [["预热", dummy]], max_length=512, padding=True,
                        truncation=True, return_tensors="pt",
                    )
                    self._reranker_inner(**{k: v.to(self._device) for k, v in d2.items()})
            print("[RetrievalCore] ✅ kernel 预热完成")

        list(jieba.cut("预热分词"))
        print("[RetrievalCore] ✅ 模型与分词器就绪")

    async def connect_db(self) -> None:
        """创建 asyncpg 连接池并探测 embedding 列类型。"""
        self._db_pool = await asyncpg.create_pool(
            host=self.db_host, port=self.db_port,
            database=self.db_name, user=self.db_user,
            min_size=2, max_size=4,
        )
        row = await self._db_pool.fetchrow("""
            SELECT udt_name FROM information_schema.columns
            WHERE table_name = 'wiki_documents' AND column_name = 'embedding';
        """)
        self._embedding_col_type = row["udt_name"]
        print(f"[RetrievalCore] ✅ DB 连接池就绪  embedding 列类型: {self._embedding_col_type}")

    async def close(self) -> None:
        if self._db_pool:
            await self._db_pool.close()
        self._executor.shutdown(wait=False)

    # ── 阶段函数 ─────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> list[float]:
        """同步推理：query → dense 向量（L2 归一化）。"""
        import torch
        device = next(self._embed_inner.parameters()).device
        inputs = self._embed_tokenizer(
            [query], max_length=512, padding=True,
            truncation=True, return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._embed_inner(**inputs)
        cls = out.last_hidden_state[:, 0]
        vec = torch.nn.functional.normalize(cls, p=2, dim=-1)
        return vec[0].cpu().float().tolist()

    async def _vector_recall(self, query_vec: list[float]) -> tuple[list[dict], float]:
        cast    = f"::{self._embedding_col_type}(1024)"
        vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"
        t0 = time.time()
        async with self._db_pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, content, metadata,
                       1 - (embedding <=> $1{cast}) AS score
                FROM wiki_documents
                ORDER BY embedding <=> $1{cast}
                LIMIT $2;
                """,
                vec_str, self.vector_k,
            )
        elapsed = (time.time() - t0) * 1000
        return [
            {
                "id":       r["id"],
                "content":  r["content"],
                "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
                "score":    float(r["score"]),
            }
            for r in rows
        ], elapsed

    async def _bm25_recall(self, query: str) -> tuple[list[dict], float]:
        terms = self._build_bm25_terms(query)
        if not terms:
            return [], 0.0
        t0 = time.time()
        async with self._db_pool.acquire() as conn:
            if len(terms) >= 2:
                and_q = " ".join(f"+content_tokenized:{t}" for t in terms)
                rows = await conn.fetch(
                    """
                    SELECT id, content, metadata, paradedb.score(id) AS score
                    FROM wiki_documents
                    WHERE wiki_documents @@@ $1
                    ORDER BY score DESC LIMIT $2;
                    """,
                    and_q, self.bm25_k,
                )
                if len(rows) >= self.bm25_k:
                    elapsed = (time.time() - t0) * 1000
                    return self._rows_to_docs(rows), elapsed

            or_terms = sorted(terms, key=len, reverse=True)[:_BM25_OR_MAX_TERMS]
            or_q = " ".join(f"content_tokenized:{t}" for t in or_terms)
            self._monitor.stage("bm25", f"AND 不足，OR fallback terms={or_terms}")
            rows = await conn.fetch(
                """
                SELECT id, content, metadata, paradedb.score(id) AS score
                FROM wiki_documents
                WHERE wiki_documents @@@ $1
                ORDER BY score DESC LIMIT $2;
                """,
                or_q, self.bm25_k,
            )
        return self._rows_to_docs(rows), (time.time() - t0) * 1000

    @staticmethod
    def _rows_to_docs(rows) -> list[dict]:
        return [
            {
                "id":       r["id"],
                "content":  r["content"],
                "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
                "score":    float(r["score"]),
            }
            for r in rows
        ]

    @staticmethod
    def _build_bm25_terms(query: str) -> list[str]:
        seen:  set[str]  = set()
        terms: list[str] = []
        for t in jieba.cut(query):
            t = re.sub(r'[+\-&|!():{}\[\]^"~*?:\\]', "", t).strip()
            if len(t) < 2 or re.match(r'^[\s\W]+$', t):
                continue
            if t in seen or t in _BM25_STOPWORDS:
                continue
            seen.add(t)
            terms.append(t)
        return terms

    @staticmethod
    def _hybrid_merge(
        vec_results: list[dict], bm25_results: list[dict], k: int
    ) -> list[dict]:
        """RRF（Reciprocal Rank Fusion）混合排分，与 main.py 完全一致。"""
        rrf:     dict[int, float] = {}
        doc_map: dict[int, dict]  = {}
        for rank, doc in enumerate(vec_results):
            did = doc["id"]
            rrf[did]     = rrf.get(did, 0) + 1.0 / (rank + 60)
            doc_map[did] = doc
        for rank, doc in enumerate(bm25_results):
            did = doc["id"]
            rrf[did]     = rrf.get(did, 0) + 1.0 / (rank + 60)
            doc_map[did] = doc
        sorted_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:k]
        return [{**doc_map[did], "score": rrf[did]} for did in sorted_ids]

    def _rerank(self, query: str, docs: list[dict]) -> list[dict]:
        """
        Cross-encoder 精排（同步，FP16）。
        评测版不施加 RERANK_MIN_HYBRID_SCORE 过滤：所有候选均参与精排。
        """
        import torch
        if not docs:
            return []
        pairs  = [[query, doc["content"]] for doc in docs]
        device = next(self._reranker_inner.parameters()).device
        inputs = self._reranker_tok(
            pairs, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._reranker_inner(**inputs)
        scores = torch.sigmoid(out.logits).squeeze(-1).cpu().float().tolist()
        if isinstance(scores, float):
            scores = [scores]
        for i, doc in enumerate(docs):
            doc["score"] = float(scores[i])
        return sorted(docs, key=lambda x: x["score"], reverse=True)[: self.rerank_top_k]

    async def _mmr_select(
        self, query_vec: list[float], docs: list[dict], k: int
    ) -> list[dict]:
        if len(docs) <= k:
            return docs
        doc_ids = [doc["id"] for doc in docs]
        async with self._db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, embedding::text FROM wiki_documents WHERE id = ANY($1);",
                doc_ids,
            )
        emb_map = {r["id"]: r["embedding"] for r in rows}
        doc_vecs = []
        for doc in docs:
            raw = emb_map.get(doc["id"])
            if raw:
                v = np.array(
                    [float(x) for x in raw.strip("[]").split(",")], dtype=np.float32
                )
            else:
                v = np.zeros(1024, dtype=np.float32)
            doc_vecs.append(v)
        norms = [v / (np.linalg.norm(v) + 1e-9) for v in doc_vecs]

        selected:   list[int] = []
        candidates: list[int] = list(range(len(docs)))
        for _ in range(k):
            best_idx, best_score = -1, -float("inf")
            for i in candidates:
                relevance = docs[i]["score"]
                max_sim   = max(
                    (float(np.dot(norms[i], norms[j])) for j in selected),
                    default=0.0,
                )
                s = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim
                if s > best_score:
                    best_score = s
                    best_idx   = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            candidates.remove(best_idx)
        return [docs[i] for i in selected]

    # ── 完整流水线 ────────────────────────────────────────────────────

    async def run_pipeline(
        self, query: str, final_k: int | None = None
    ) -> PipelineResult:
        """
        运行完整检索流水线，返回 PipelineResult。

        关键评测差异：
          - 不施加 RERANK_MIN_HYBRID_SCORE（所有 merged 文档均送入 reranker）
          - 不施加 SCORE_THRESHOLD（final 保留原始分数）
          - 无并发锁（评测为顺序调用）
        """
        loop    = asyncio.get_event_loop()
        final_k = final_k if final_k is not None else self.final_k
        t       = StageTimings()
        mon     = self._monitor

        mon.header(query)

        # ── 1. Embedding ─────────────────────────────────────────────
        t0 = time.time()
        query_vec = await loop.run_in_executor(self._executor, self._embed_query, query)
        t.embed_ms = (time.time() - t0) * 1000
        mon.stage("1 embed", f"{t.embed_ms:.1f}ms  vec[:3]={[round(v, 4) for v in query_vec[:3]]}")

        result = PipelineResult(query=query, query_vec=query_vec, timings=t)

        # ── 2. 并发多路召回 ──────────────────────────────────────────
        vec_task  = asyncio.create_task(self._vector_recall(query_vec))
        bm25_task = asyncio.create_task(self._bm25_recall(query))
        await asyncio.gather(vec_task, bm25_task, return_exceptions=True)

        if vec_task.exception() is None:
            result.vec_results, t.vector_ms = vec_task.result()
        else:
            result.vec_error = str(vec_task.exception())
            mon.warn(f"vector recall 异常: {result.vec_error}")

        if bm25_task.exception() is None:
            result.bm25_results, t.bm25_ms = bm25_task.result()
        else:
            result.bm25_error = str(bm25_task.exception())
            mon.warn(f"bm25 recall 异常: {result.bm25_error}")

        mon.stage(
            "2 recall",
            f"wall={t.recall_wall_ms:.1f}ms | "
            f"vector={t.vector_ms:.1f}ms({len(result.vec_results)}条) | "
            f"bm25={t.bm25_ms:.1f}ms({len(result.bm25_results)}条)",
        )
        mon.docs("vector", result.vec_results)
        mon.docs("bm25",   result.bm25_results)

        # ── 3. RRF 混合排分 ──────────────────────────────────────────
        t0 = time.time()
        result.merged = self._hybrid_merge(
            result.vec_results, result.bm25_results, self.hybrid_top_k
        )
        t.merge_ms = (time.time() - t0) * 1000
        mon.stage("3 merge", f"{t.merge_ms:.1f}ms  merged={len(result.merged)}条")
        mon.docs("merged", result.merged)

        # ── 4. Cross-encoder 精排（无分数截断） ──────────────────────
        # 评测版：hybrid_top_k 内全部候选送入 reranker，不跳过任何文档
        to_rerank = result.merged[: self.hybrid_top_k]
        mon.stage("4 rerank", f"送入 {len(to_rerank)} 条（无 RERANK_MIN_HYBRID_SCORE 过滤）")
        t0 = time.time()
        result.reranked = await loop.run_in_executor(
            self._executor, self._rerank, query, to_rerank
        )
        t.rerank_ms = (time.time() - t0) * 1000
        mon.stage("4 rerank", f"{t.rerank_ms:.1f}ms  reranked={len(result.reranked)}条")
        mon.docs("reranked", result.reranked)

        # ── 5. MMR 去重 ──────────────────────────────────────────────
        t0 = time.time()
        result.final = await self._mmr_select(query_vec, result.reranked, final_k)
        t.mmr_ms = (time.time() - t0) * 1000
        mon.stage("5 mmr", f"{t.mmr_ms:.1f}ms  final={len(result.final)}条")
        mon.docs("final", result.final)
        mon.timings(t)

        # ── 构建各阶段 rank 索引 ─────────────────────────────────────
        result.vec_rank      = {d["id"]: i + 1 for i, d in enumerate(result.vec_results)}
        result.bm25_rank     = {d["id"]: i + 1 for i, d in enumerate(result.bm25_results)}
        result.merged_rank   = {d["id"]: i + 1 for i, d in enumerate(result.merged)}
        result.reranked_rank = {d["id"]: i + 1 for i, d in enumerate(result.reranked)}
        result.final_rank    = {d["id"]: i + 1 for i, d in enumerate(result.final)}

        return result

    async def fetch_chunk_by_id(self, chunk_id: int) -> dict | None:
        """按 id 从 DB 获取单条 Chunk（供 dump_bad_cases 使用）。"""
        row = await self._db_pool.fetchrow(
            "SELECT id, content, metadata FROM wiki_documents WHERE id = $1;",
            chunk_id,
        )
        if row is None:
            return None
        meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        return {"id": row["id"], "content": row["content"], "metadata": meta}
