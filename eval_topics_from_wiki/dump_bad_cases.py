"""
eval/dump_bad_cases.py
======================
Bad Case 尸检与归因脚本（Human-in-the-Loop 核心工具）。

读取 eval_results.csv，对每条 Hit@5 失败的记录重新跑一次检索，
将以下三者并列打印供人工对比：
  ▶ QUERY        实际提问文本
  ▶ TARGET       期望命中的 Chunk 原文（系统漏掉了什么）
  ▶ NOISE TOP-K  reranker 错误提权的噪音 Chunk（系统认为更相关的是什么）

自动归因逻辑（供参考，人工最终确认）：
  死因 A  词汇鸿沟    目标 chunk 既未进向量 top-50 也未进 BM25 top-50
  死因 B  切块碎裂    目标 chunk 进了某召回路但 merged 后消失（或始终未出现）
  死因 C  Reranker倒挂  目标 chunk 进了 merged 前 10 但被 reranker 踢走（inversion=True）

死因 C 是本脚本的重点：只有并列展示 Query / Target / Noise 三者的文本，
人工才能判断是 RRF K 值问题、Reranker 对领域语境误判，还是需要微调 BGE-M3。

用法：
  # 打印所有死因的 bad case
  python eval/dump_bad_cases.py

  # 只看 Reranker 倒挂（死因 C），最多打印 10 条，每条展示前 3 条噪音 Chunk
  python eval/dump_bad_cases.py --cause C --max-cases 10 --top-noise 3

  # 打印死因 A（词汇鸿沟）
  python eval/dump_bad_cases.py --cause A

  # 输出到文件
  python eval/dump_bad_cases.py --cause C --output eval/bad_cases_C.txt
"""

from __future__ import annotations

import os
import sys
import csv
import asyncio
import argparse
from contextlib import redirect_stdout
from io import StringIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval_random_topics.retrieval_core import RetrievalCore, PipelineResult


# ══════════════════════════════════════════════════════════════════════
# 格式化输出
# ══════════════════════════════════════════════════════════════════════

SEP  = "─" * 72
SEP2 = "═" * 72

CAUSE_LABELS = {
    "A": "死因 A：词汇鸿沟",
    "B": "死因 B：切块碎裂 / Merged 后消失",
    "C": "死因 C：Reranker 排序倒挂",
}

CAUSE_DESC = {
    "A": "目标 Chunk 既未进向量召回 Top-50，也未进 BM25 Top-50。\n"
         "   Embedding 未能将 Query 映射到目标语义空间，或 BM25 词汇完全不重叠。",
    "B": "目标 Chunk 进了某召回路，但 RRF 合并或 Reranker 之后消失。\n"
         "   可能是相邻 Chunk 边界切割导致单 Chunk 匹配度不足，或 Overlap 不够。",
    "C": "目标 Chunk 进了 Merged Top-10，但被 Reranker 错误降权（倒退 >5 位或直接踢出）。\n"
         "   请对比 Query / Target / Noise 三者文本，判断 Reranker 的误判原因。",
}

# ── 工具函数（模块级，供 _classify 和主循环共用） ─────────────────────

def _int_or_none(v: str) -> int | None:
    return int(v) if v not in ("", "None", None) else None

CAUSE_ACTIONS = {
    "A": "建议决策：\n"
         "   1. 在前置链路引入 Query 重写（Query Rewriting）\n"
         "   2. 考虑换用对垂直领域更敏感的 Embedding 模型或进行微调",
    "B": "建议决策：\n"
         "   1. 扩大 Chunk Size 并增加 Overlap（如 512→768 tokens，overlap 128→256）\n"
         "   2. 规划引入父子块（Parent-Child）架构",
    "C": "建议决策（人工阅读文本后选择）：\n"
         "   1. 调整 RRF K 值（当前 K=60），降低高排名偏置\n"
         "   2. 降低 Reranker 的 merged_rank 阈值，让更多候选进入精排\n"
         "   3. 积累误判样本，微调 / 替换 BGE-M3 Reranker",
}


def _trunc(text: str, n: int = 350) -> str:
    return text[:n] + "…" if len(text) > n else text


def _rank_str(r: int | None, label: str) -> str:
    return f"{label}={r}" if r is not None else f"{label}=N/A"


def print_case_header(case_num: int, total: int, qid: int, cause: str) -> None:
    print(SEP2)
    print(f"  Bad Case [{case_num}/{total}]  query_id={qid}  {CAUSE_LABELS[cause]}")
    print(f"  {CAUSE_DESC[cause]}")
    print(SEP)


def print_query(query: str) -> None:
    print(f"  ▶ QUERY")
    print(f"    {query}")
    print()


def print_target(chunk_id: int, chunk: dict | None, ranks: dict) -> None:
    print(f"  ▶ TARGET CHUNK  (id={chunk_id})")
    if chunk:
        title = chunk["metadata"].get("title", "—")
        print(f"    title   : {title}")
        print(f"    content : {_trunc(chunk['content'])}")
    else:
        print(f"    ⚠ 无法从 DB 获取 chunk id={chunk_id}")
    print()
    print(f"  ▶ TARGET STAGE RANKS")
    parts = [
        _rank_str(ranks.get("vec"),      "vec"),
        _rank_str(ranks.get("bm25"),     "bm25"),
        _rank_str(ranks.get("merged"),   "merged"),
        _rank_str(ranks.get("reranked"), "reranked"),
        _rank_str(ranks.get("final"),    "final"),
    ]
    print(f"    {'  |  '.join(parts)}")
    print()


def print_noise_chunks(noise_docs: list[dict], top_n: int, pr: PipelineResult) -> None:
    if not noise_docs:
        return
    print(f"  ▶ NOISE CHUNKS  (Reranker 认为比 Target 更相关的前 {top_n} 条)")
    print(f"    {'─' * 60}")
    for j, doc in enumerate(noise_docs[:top_n], 1):
        did   = doc["id"]
        title = doc.get("metadata", {}).get("title", "—")
        rr    = pr.reranked_rank.get(did, "?")
        mr    = pr.merged_rank.get(did, "?")
        score = doc.get("score", 0.0)
        print(f"    [{j}] id={did:6d}  reranked_rank={rr}  merged_rank={mr}  score={score:.4f}")
        print(f"         title   : {title}")
        print(f"         content : {_trunc(doc['content'], 200)}")
    print()


def print_action(cause: str) -> None:
    print(f"  ▶ 人工决策参考")
    for line in CAUSE_ACTIONS[cause].split("\n"):
        print(f"  {line}")
    print()


def print_summary(cause_counts: dict[str, int], n_total_bad: int, n_total: int) -> None:
    print(SEP2)
    print(f"  ◆ 死因分布汇总  （Hit@5 失败 {n_total_bad}/{n_total} 条）")
    print(SEP)
    total_labeled = sum(cause_counts.values())
    for c in ("A", "B", "C"):
        cnt = cause_counts.get(c, 0)
        if cnt > 0:
            print(f"    {CAUSE_LABELS[c]:<24s}: {cnt:3d} 条  ({cnt/n_total_bad:.1%})")
    print(SEP2)


# ══════════════════════════════════════════════════════════════════════
# 归因逻辑（与 run_retrieval_eval.py 保持一致）
# ══════════════════════════════════════════════════════════════════════

def _classify(row: dict) -> str:
    vec_r     = _int_or_none(row.get("vec_rank", ""))
    bm25_r    = _int_or_none(row.get("bm25_rank", ""))
    inversion = str(row.get("reranker_inversion", "")).lower() == "true"
    merged_r  = _int_or_none(row.get("merged_rank", ""))

    if inversion and merged_r is not None and merged_r <= 10:
        return "C"
    if vec_r is None and bm25_r is None:
        return "A"
    return "B"


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

async def run(args):
    # ── 读取评测结果 ──────────────────────────────────────────────────
    if not os.path.exists(args.results):
        print(f"❌ 找不到 {args.results}，请先运行 run_retrieval_eval.py", file=sys.stderr)
        sys.exit(1)

    with open(args.results, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    n_total = len(all_rows)
    bad_rows = [
        r for r in all_rows
        if str(r.get("hit_at_5", "")).lower() in ("false", "0", "", "none")
    ]
    print(f"[dump_bad_cases] 共 {n_total} 条，Hit@5 失败: {len(bad_rows)} 条")

    # 归因
    for r in bad_rows:
        r["_cause"] = _classify(r)

    # 统计（全量）
    cause_counts: dict[str, int] = {}
    for r in bad_rows:
        cause_counts[r["_cause"]] = cause_counts.get(r["_cause"], 0) + 1

    # 筛选指定死因
    target_cause = args.cause.upper() if args.cause != "all" else "all"
    if target_cause != "all":
        bad_rows = [r for r in bad_rows if r["_cause"] == target_cause]
        print(f"  筛选死因 '{target_cause}': {len(bad_rows)} 条")
    else:
        print(f"  打印全部死因")

    if not bad_rows:
        print("  无符合条件的 Bad Case。")
        return

    if args.max_cases and len(bad_rows) > args.max_cases:
        bad_rows = bad_rows[: args.max_cases]
        print(f"  限制最多 {args.max_cases} 条（--max-cases）")

    # ── 初始化 RetrievalCore，重新跑检索获取噪音 Chunk ────────────────
    # 仅在有死因 C 时才需要加载模型（用于获取 reranked 阶段的噪音 chunks）
    need_rerun = any(r["_cause"] == "C" for r in bad_rows)
    core: RetrievalCore | None = None

    if need_rerun:
        print("\n  [死因 C 存在] 初始化 RetrievalCore，重跑检索以获取噪音 Chunk...")
        core = RetrievalCore(verbose=False)
        core.load_models(warmup=False)   # 尸检不需要预热，节省时间
        await core.connect_db()

    # ── 逐条打印 ─────────────────────────────────────────────────────
    output_lines = StringIO()

    def _do_print(*a, **kw):
        print(*a, **kw)
        print(*a, **kw, file=output_lines)

    # 临时重定向 print 到双通道（stdout + buffer）
    import builtins
    orig_print = builtins.print

    try:
        builtins.print = _do_print

        for i, row in enumerate(bad_rows, 1):
            cause    = row["_cause"]
            query_id = row.get("query_id", "?")
            chunk_id = int(row["chunk_id"])
            query    = row.get("query", "")

            target_ranks = {
                "vec":      _int_or_none(row.get("vec_rank", "")),
                "bm25":     _int_or_none(row.get("bm25_rank", "")),
                "merged":   _int_or_none(row.get("merged_rank", "")),
                "reranked": _int_or_none(row.get("reranked_rank", "")),
                "final":    _int_or_none(row.get("final_rank", "")),
            }

            print_case_header(i, len(bad_rows), query_id, cause)
            print_query(query)

            # 获取 target chunk 原文
            target_chunk = None
            if core:
                target_chunk = await core.fetch_chunk_by_id(chunk_id)
            print_target(chunk_id, target_chunk, target_ranks)

            # 死因 C：重跑检索，获取 reranked 阶段排在 target 前面的噪音 Chunk
            if cause == "C" and core:
                pr: PipelineResult = await core.run_pipeline(query)
                target_rr = pr.reranked_rank.get(chunk_id)
                # 取 reranked 中排名优于 target 的文档（噪音）
                noise_docs = [
                    d for d in pr.reranked
                    if d["id"] != chunk_id
                    and (target_rr is None or pr.reranked_rank.get(d["id"], 999) < target_rr)
                ]
                print_noise_chunks(noise_docs, args.top_noise, pr)
            elif cause == "C":
                print(f"  ▶ NOISE CHUNKS  ⚠ 需要 RetrievalCore（加载模型失败时跳过）")
                print()

            print_action(cause)

        print_summary(cause_counts, len([r for r in all_rows
                                         if str(r.get("hit_at_5","")).lower() in ("false","0","")]),
                      n_total)

    finally:
        builtins.print = orig_print
        if core:
            await core.close()

    # ── 写文件（可选） ────────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_lines.getvalue())
        print(f"\n  已写入 {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Bad Case 尸检与归因（Human-in-the-Loop）")
    parser.add_argument("--results",   default="eval/eval_results.csv",
                        help="评测结果 CSV（run_retrieval_eval.py 的输出）")
    parser.add_argument("--cause",     default="all",
                        choices=["all", "A", "B", "C"],
                        help="只打印指定死因（默认 all）")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="最多打印 N 条（默认不限）")
    parser.add_argument("--top-noise", type=int, default=3,
                        help="死因 C：展示前 N 条噪音 Chunk（默认 3）")
    parser.add_argument("--output",    default=None,
                        help="同时将结果写入指定文件（可选）")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
