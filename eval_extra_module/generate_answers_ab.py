"""
eval_extra_module/generate_answers_ab.py
========================================
AB 消融实验的生成脚本。

在 eval_random_topics/generate_answers.py 基础上增加两个实验因子：
  Factor A：Query 重写（--rewrite）
    - 生成 N 条发散 query（默认 N=2）+ golden query 收束核心意图
    - (N+1) 条 query × 2 路召回 → 跨所有路 RRF 融合取 top-30
    - 用 golden query 做一次 reranker → MMR 去重取 top-k
  Factor B：Chunk 融合（--fusion）

三组消融实验条件：
  1. rewrite_only（A only）：多角度 query 重写 + 多路召回 RRF 融合
  2. fusion_only（B only）：原始 query + chunk 融合
  3. rewrite_fusion（A + B）：多角度 query 重写 + 多路召回 + chunk 融合

baseline 数据直接读取 eval_random_topics/rag_top3，无需重新生成。

检索流水线（统一）：
  1. embed golden query
  2. 对每条 query（golden + N 条发散）做 vector + bm25 双路召回
  3. 跨所有 (N+1)*2 路 RRF 融合，取 top --rrf-k（默认 30）
  4. 用 golden query 对 top-30 做一次 reranker
  5. MMR 去重取 top --topk（默认 3）

用法：
  # A only: query 重写（默认 2 条发散 query）
  python -m eval_extra_module.generate_answers_ab --rewrite

  # B only: chunk 融合
  python -m eval_extra_module.generate_answers_ab --fusion

  # A + B: 两者都用
  python -m eval_extra_module.generate_answers_ab --rewrite --fusion
"""

from __future__ import annotations

import os
import sys
import csv
import json
import asyncio
import argparse
import time
from dataclasses import dataclass

from openai import OpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval_random_topics.retrieval_core import RetrievalCore
from eval_extra_module.query_rewrite import rewrite_query
from eval_extra_module.chunk_fusion import fuse_chunks, FusionStats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ─── llama.cpp 配置（与 eval_random_topics/generate_answers.py 对齐） ───
LLAMACPP_BASE_URL = "http://msi.tailb813aa.ts.net:8080/v1"
LLAMACPP_API_KEY  = "sk-no-key-required"
LLAMACPP_MODEL    = "local-model"

_LLAMACPP_EXTRA = {
    "top_k":          40,
    "min_p":          0.05,
    "repeat_penalty": 1.1,
}

# ─── 生成 Prompt（与 eval_random_topics/generate_answers.py 完全一致） ───
_SYSTEM_PROMPT = """\
你是一个严谨的知识问答助手。请根据下方提供的参考资料，简洁地回答用户的问题。

规则：
1. 只能使用参考资料中明确提到的信息作答，不得引用或推断资料之外的内容
2. 如果参考资料中没有足够信息回答问题，直接回答"根据所提供的资料，无法回答该问题"
3. 答案控制在 3 句话以内，简洁清晰"""

_USER_TEMPLATE = """\
参考资料：
{context}

问题：{query}

请根据以上参考资料回答问题。"""


# ─── 工具函数 ─────────────────────────────────────────────────────────

def _format_context(docs: list[dict], max_chars_per_doc: int = 400) -> tuple[str, list[int]]:
    """与 eval_random_topics/generate_answers.py 完全一致。"""
    parts = []
    ids   = []
    for i, doc in enumerate(docs, 1):
        title   = doc.get("metadata", {}).get("title", "")
        content = doc["content"][:max_chars_per_doc]
        if len(doc["content"]) > max_chars_per_doc:
            content += "…"
        header = f"[{i}] {title}" if title else f"[{i}]"
        parts.append(f"{header}\n{content}")
        ids.append(doc["id"])
    return "\n\n".join(parts), ids


@dataclass
class GenStats:
    """与 eval_random_topics/generate_answers.py 完全一致。"""
    ttft_ms:          float
    gen_ms:           float
    prompt_tokens:    int
    completion_tokens:int
    total_tokens:     int


def generate_answer(llm: OpenAI, query: str, context: str) -> tuple[str, GenStats]:
    """与 eval_random_topics/generate_answers.py 完全一致：流式生成。"""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _USER_TEMPLATE.format(
            context=context, query=query
        )},
    ]
    extra = {**_LLAMACPP_EXTRA, "stream_options": {"include_usage": True}}

    t_start  = time.time()
    ttft_ms  = None
    parts: list[str] = []
    usage    = None

    stream = llm.chat.completions.create(
        model=LLAMACPP_MODEL,
        messages=messages,
        stream=True,
        temperature=0.1,
        top_p=0.7,
        extra_body=extra,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            if ttft_ms is None:
                ttft_ms = (time.time() - t_start) * 1000
            parts.append(delta)
        if getattr(chunk, "usage", None):
            usage = chunk.usage

    gen_ms = (time.time() - t_start) * 1000
    stats  = GenStats(
        ttft_ms=round(ttft_ms or gen_ms, 1),
        gen_ms=round(gen_ms, 1),
        prompt_tokens=usage.prompt_tokens     if usage else -1,
        completion_tokens=usage.completion_tokens if usage else -1,
        total_tokens=usage.total_tokens       if usage else -1,
    )
    return "".join(parts).strip(), stats


# ─── 主流程 ──────────────────────────────────────────────────────────

def _load_existing(path: str) -> tuple[dict[int, dict], list[str]]:
    if not os.path.exists(path):
        return {}, []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return {int(r["query_id"]): r for r in rows}, list(fieldnames)


def _is_failed(row: dict) -> bool:
    try:
        return float(row.get("gen_ms", -1)) < 0 or row.get("generated_answer", "").strip() == ""
    except (ValueError, TypeError):
        return True


async def run(args):
    golden_path = os.path.join(PROJECT_ROOT, "eval_random_topics", "golden_dataset.csv")
    if not os.path.exists(golden_path):
        print(f"❌ 找不到 {golden_path}", file=sys.stderr)
        sys.exit(1)

    with open(golden_path, encoding="utf-8") as f:
        golden = list(csv.DictReader(f))

    for row in golden:
        row["_query"] = row.get("query", "").strip()
    golden = [r for r in golden if r["_query"]]

    if args.limit:
        golden = golden[: args.limit]

    # ── 实验条件标识 ─────────────────────────────────────────────────
    cond_parts = []
    if args.rewrite:
        cond_parts.append("A(重写)")
    if args.fusion:
        cond_parts.append("B(融合)")
    cond_label = " + ".join(cond_parts) if cond_parts else "baseline"
    print(f"[AB 消融] 实验条件: {cond_label}")
    print(f"[AB 消融] --rewrite={args.rewrite}  --fusion={args.fusion}  --topk={args.topk}")

    # ── Resume 模式 ──────────────────────────────────────────────────
    existing_rows: dict[int, dict] = {}
    existing_fieldnames: list[str] = []

    if args.resume:
        existing_rows, existing_fieldnames = _load_existing(args.output)
        if not existing_rows:
            print(f"⚠ --resume 指定但未找到 {args.output}，将全量运行", file=sys.stderr)
        else:
            all_qids     = {int(g["query_id"]) for g in golden}
            failed_qids  = {qid for qid, r in existing_rows.items() if _is_failed(r)}
            missing_qids = all_qids - set(existing_rows.keys())
            retry_qids   = failed_qids | missing_qids
            golden = [g for g in golden if int(g["query_id"]) in retry_qids]
            print(f"[AB 消融] resume: 已有 {len(existing_rows)} 条，"
                  f"失败 {len(failed_qids)}，未处理 {len(missing_qids)}，"
                  f"本次补跑 {len(golden)}")
            if not golden:
                print("✅ 所有条目已完成，无需补跑")
                return
    else:
        print(f"[AB 消融] {len(golden)} 条查询")

    # ── 初始化 RetrievalCore ──────────────────────────────────────────
    core = RetrievalCore(verbose=False, final_k=args.topk)
    core.load_models(warmup=True)
    await core.connect_db()

    # ── 初始化 llama.cpp 客户端 ───────────────────────────────────────
    llm = OpenAI(base_url=LLAMACPP_BASE_URL, api_key=LLAMACPP_API_KEY)
    print(f"[AB 消融] llama.cpp @ {LLAMACPP_BASE_URL}")

    rows = []
    failed = 0

    try:
        for i, g in enumerate(golden, 1):
            query_id = int(g["query_id"])
            query    = g["_query"]

            print(f"\n  [{i:3d}/{len(golden)}] qid={query_id:3d}  q={query[:45]}")

            # ── Factor A: Query 重写（1 条 golden + N 条发散） ─────────
            rewrite_ms = 0.0
            original_query = query
            golden_query = query       # reranker/MMR 锚点
            golden_query_str = ""      # 存 CSV
            divergent_queries_str = "" # 用 "|" 分隔存 CSV
            divergent_queries: list[str] = []

            if args.rewrite:
                try:
                    rw = rewrite_query(llm, query, n=args.n_rewrite)
                    rewrite_ms = rw.rewrite_ms
                    golden_query = rw.golden_query
                    divergent_queries = rw.divergent_queries
                    golden_query_str = rw.golden_query
                    divergent_queries_str = "|".join(rw.divergent_queries)
                    print(f"           [A golden] {golden_query[:60]}")
                    for qi, rq in enumerate(rw.divergent_queries, 1):
                        print(f"           [A 发散 {qi}] {rq[:60]}")
                    print(f"           [A 重写] {rewrite_ms:.0f}ms  "
                          f"1+{len(rw.divergent_queries)} 条")
                except Exception as e:
                    print(f"           [A 重写] ⚠ 失败: {e}，仅使用原始 query")
                    golden_query = query
                    golden_query_str = ""
                    divergent_queries_str = ""
                    divergent_queries = []

            # ── 检索流水线 ────────────────────────────────────────────
            # golden query + N 条发散 query = N+1 条
            # 每条走 vector + bm25 双路 = (N+1)*2 路召回
            # → 跨所有路 RRF 融合取 top rrf-k
            # → 用 golden query 做一次 reranker（锚定核心意图）
            # → MMR 去重取 top-k
            all_queries = [golden_query] + divergent_queries
            loop = asyncio.get_event_loop()

            # 1. embed golden query（用于 reranker 和 MMR）
            t_embed_start = time.time()
            golden_vec = await loop.run_in_executor(
                core._executor, core._embed_query, golden_query)
            embed_ms = (time.time() - t_embed_start) * 1000

            # 2. 每条 query 做双路召回
            all_recall_lists: list[list[dict]] = []  # (N+1)*2 个排名列表
            total_vector_ms = 0.0
            total_bm25_ms = 0.0

            for qi, rq in enumerate(all_queries):
                # embed（golden query 已有，复用）
                if qi == 0:
                    q_vec = golden_vec
                else:
                    q_vec = await loop.run_in_executor(
                        core._executor, core._embed_query, rq)

                # vector + bm25 并发召回
                vec_task  = asyncio.create_task(core._vector_recall(q_vec))
                bm25_task = asyncio.create_task(core._bm25_recall(rq))
                await asyncio.gather(vec_task, bm25_task, return_exceptions=True)

                if vec_task.exception() is None:
                    vec_results, v_ms = vec_task.result()
                    all_recall_lists.append(vec_results)
                    total_vector_ms += v_ms
                if bm25_task.exception() is None:
                    bm25_results, b_ms = bm25_task.result()
                    all_recall_lists.append(bm25_results)
                    total_bm25_ms += b_ms

            n_queries = len(all_queries)
            n_recall_lists = len(all_recall_lists)
            print(f"           [召回] {n_queries}条query × 2路 = {n_recall_lists}路  "
                  f"vec={total_vector_ms:.0f}ms  bm25={total_bm25_ms:.0f}ms")

            # 3. 跨所有路 RRF 融合
            t_rrf_start = time.time()
            rrf_scores: dict[int, float] = {}
            doc_map: dict[int, dict] = {}
            for recall_list in all_recall_lists:
                for rank, doc in enumerate(recall_list):
                    did = doc["id"]
                    rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (rank + 60)
                    doc_map[did] = doc
            sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
            rrf_top = [{**doc_map[did], "score": rrf_scores[did]}
                       for did in sorted_ids[:args.rrf_k]]
            rrf_ms = (time.time() - t_rrf_start) * 1000
            print(f"           [RRF] {rrf_ms:.0f}ms  "
                  f"候选={len(rrf_scores)}  取top{args.rrf_k}={len(rrf_top)}")

            # 4. 用 golden query 做一次 reranker
            t_rerank_start = time.time()
            reranked = await loop.run_in_executor(
                core._executor, core._rerank, golden_query, rrf_top)
            rerank_ms = (time.time() - t_rerank_start) * 1000
            print(f"           [rerank] {rerank_ms:.0f}ms  "
                  f"输入={len(rrf_top)}  输出={len(reranked)}")

            # 5. MMR 去重取 top-k
            t_mmr_start = time.time()
            merged_docs = await core._mmr_select(golden_vec, reranked, args.topk)
            mmr_ms = (time.time() - t_mmr_start) * 1000
            print(f"           [MMR] {mmr_ms:.0f}ms  final={len(merged_docs)}")

            # 汇总延迟
            retrieval_ms = embed_ms + max(total_vector_ms, total_bm25_ms) + rrf_ms + rerank_ms + mmr_ms

            context, chunk_ids = _format_context(merged_docs)

            # ── Factor B: Chunk 融合 ─────────────────────────────────
            fusion_ms = 0.0
            fused_context = ""
            generation_context = context  # 用于生成的 context

            if args.fusion and merged_docs:
                try:
                    fs = fuse_chunks(llm, query, merged_docs)
                    fusion_ms = fs.fusion_ms
                    fused_context = fs.fused_context
                    generation_context = fs.fused_context
                    print(f"           [B 融合] {fusion_ms:.0f}ms  "
                          f"原始={fs.original_chars}字 → 融合={fs.fused_chars}字")
                except Exception as e:
                    print(f"           [B 融合] ⚠ 失败: {e}，使用原始 context")
                    fused_context = ""
                    generation_context = context

            # ── 生成答案（注意：问题用原始 query，context 视 B 而定） ──
            try:
                answer, stats = generate_answer(llm, query, generation_context)
                e2e_ms = round(rewrite_ms + retrieval_ms + fusion_ms + stats.gen_ms, 1)
                print(f"           → {answer[:50]}...  "
                      f"ret={retrieval_ms:.0f}ms  gen={stats.gen_ms:.0f}ms  "
                      f"e2e={e2e_ms:.0f}ms  tok={stats.prompt_tokens}+{stats.completion_tokens}")
            except Exception as e:
                print(f"           ⚠ 生成失败: {e}")
                answer = ""
                stats  = GenStats(ttft_ms=-1, gen_ms=-1, prompt_tokens=-1,
                                  completion_tokens=-1, total_tokens=-1)
                e2e_ms = -1
                failed += 1

            rows.append({
                "query_id":            query_id,
                "query":               query,
                "n_chunks":            len(chunk_ids),
                "context_chars":       len(generation_context),
                "retrieved_chunk_ids": json.dumps(chunk_ids, ensure_ascii=False),
                "retrieved_context":   generation_context,
                "generated_answer":    answer,
                # ── 检索各阶段延迟 ──
                "embed_ms":            round(embed_ms,        1),
                "vector_ms":           round(total_vector_ms, 1),
                "bm25_ms":             round(total_bm25_ms,   1),
                "rrf_ms":              round(rrf_ms,           1),
                "rerank_ms":           round(rerank_ms,        1),
                "mmr_ms":              round(mmr_ms,           1),
                "retrieval_ms":        round(retrieval_ms,     1),
                "n_queries":           n_queries,
                # ── 生成延迟 ──
                "ttft_ms":             stats.ttft_ms,
                "gen_ms":              stats.gen_ms,
                "e2e_ms":              e2e_ms,
                "prompt_tokens":       stats.prompt_tokens,
                "completion_tokens":   stats.completion_tokens,
                "total_tokens":        stats.total_tokens,
                # ── AB 额外列 ──
                "rewrite_ms":          rewrite_ms,
                "original_query":      original_query,
                "golden_query":        golden_query_str,
                "divergent_queries":   divergent_queries_str,
                "n_divergent":         len(divergent_queries),
                "fusion_ms":           fusion_ms,
                "fused_context":       fused_context,
            })

    finally:
        await core.close()

    # ── 写 CSV ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.resume and existing_rows:
        new_by_qid = {r["query_id"]: r for r in rows}
        existing_rows.update(new_by_qid)
        merged = [existing_rows[qid] for qid in sorted(existing_rows)]
        fieldnames = existing_fieldnames or list(rows[0].keys())
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged)
        print(f"\n✅ resume 完成：{len(rows)} 条补跑，合并后共 {len(merged)} 条 → {args.output}")
        rows = merged
    else:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # ── 汇总统计 ──────────────────────────────────────────────────────
    def _flt(r: dict, key: str) -> float:
        try:
            return float(r.get(key, -1))
        except (ValueError, TypeError):
            return -1.0

    valid = [r for r in rows if _flt(r, "gen_ms") >= 0]

    def _pct(vals, p):
        s = sorted(vals)
        return round(s[min(int(len(s) * p), len(s) - 1)], 1) if s else 0.0

    def _v(key):
        return [_flt(r, key) for r in valid if _flt(r, key) >= 0]

    print(f"\n✅ {len(rows)} 条写入 {args.output}，生成失败 {failed} 条")
    print(f"   实验条件: {cond_label}")
    print()

    # AB 额外延迟统计
    if args.rewrite:
        rw_vals = _v("rewrite_ms")
        if rw_vals:
            print(f"  [A 重写] P50={_pct(rw_vals,.50):>8}ms  P90={_pct(rw_vals,.90):>8}ms")
    if args.fusion:
        fu_vals = _v("fusion_ms")
        if fu_vals:
            print(f"  [B 融合] P50={_pct(fu_vals,.50):>8}ms  P90={_pct(fu_vals,.90):>8}ms")

    print()
    print("  ── 检索各阶段延迟 ─────────────────────────────────────────")
    for label, key in [
        ("embed",     "embed_ms"),
        ("vector",    "vector_ms"),
        ("bm25",      "bm25_ms"),
        ("rrf",       "rrf_ms"),
        ("rerank",    "rerank_ms"),
        ("mmr",       "mmr_ms"),
        ("retrieval", "retrieval_ms"),
    ]:
        vals = _v(key)
        if vals:
            print(f"  {label:<10} P50={_pct(vals,.50):>8}  "
                  f"P90={_pct(vals,.90):>8}  P99={_pct(vals,.99):>8}")
    print()
    print("  ── 生成延迟 + 端到端 ──────────────────────────────────────")
    for label, key in [
        ("ttft",      "ttft_ms"),
        ("gen",       "gen_ms"),
        ("e2e",       "e2e_ms"),
    ]:
        vals = _v(key)
        if vals:
            print(f"  {label:<10} P50={_pct(vals,.50):>8}  "
                  f"P90={_pct(vals,.90):>8}  P99={_pct(vals,.99):>8}")
    print()
    print(f"下一步：python -m eval_random_topics.judge_answers --answers {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="AB 消融实验：检索 + 生成（支持 query 重写 / chunk 融合）"
    )
    parser.add_argument("--golden", default=None,
                        help="黄金测试集路径（默认 eval_random_topics/golden_dataset.csv）")
    parser.add_argument("--output", default=None,
                        help="输出 CSV 路径（默认根据 --rewrite/--fusion 自动推导）")
    parser.add_argument("--rewrite", action="store_true",
                        help="启用 Factor A：Query 重写")
    parser.add_argument("--fusion", action="store_true",
                        help="启用 Factor B：Chunk 融合")
    parser.add_argument("--topk", type=int, default=3,
                        help="最终 top-k（默认 3，与前期实验对齐）")
    parser.add_argument("--n-rewrite", type=int, default=2,
                        help="发散 query 数量（默认 2，加上 golden 共 3 条）")
    parser.add_argument("--rrf-k", type=int, default=30,
                        help="RRF 融合后送入 reranker 的候选数（默认 30）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续传：只补跑失败/缺失条目")
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理前 N 条（调试用）")
    args = parser.parse_args()

    if not args.rewrite and not args.fusion:
        print("❌ 请指定 --rewrite 和/或 --fusion，或使用 run_ablation.sh 全跑",
              file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        if args.rewrite and args.fusion:
            tag = "rewrite_fusion"
        elif args.rewrite:
            tag = "rewrite_only"
        else:
            tag = "fusion_only"
        args.output = f"eval_extra_module/{tag}/eval_answers.csv"

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
