"""
eval/run_retrieval_eval.py
==========================
阶段一核心评测脚本：静默检索 + 绝对算分。

直接实例化 RetrievalCore（不走任何 HTTP 接口），逐条运行检索 pipeline，
计算 Hit@1 / Hit@5 / Hit@10 / MRR@10，并输出完整的逐条结果供后续分析。

输入：
  eval/golden_dataset.csv  黄金测试集（人工精加工后）

输出：
  eval/eval_results.csv    逐条结果（含各阶段排名 + 延迟）
  eval/eval_summary.json   汇总指标

用法：
  python eval/run_retrieval_eval.py
  python eval/run_retrieval_eval.py --golden eval/golden_dataset.csv --limit 20
  python eval/run_retrieval_eval.py --verbose   # 开启 pipeline 实时监控输出
"""

from __future__ import annotations

import os
import sys
import csv
import json
import asyncio
import argparse
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval.retrieval_core import RetrievalCore, PipelineResult


# ══════════════════════════════════════════════════════════════════════
# 逐条结果数据结构
# ══════════════════════════════════════════════════════════════════════

@dataclass
class EvalRow:
    query_id:         int
    chunk_id:         int
    query:            str
    adversarial_type: str

    # ── 各阶段排名（None = 未出现，CSV 中写 ""） ────────────────────
    vec_rank:      int | None   # 向量召回中的排名
    bm25_rank:     int | None   # BM25 召回中的排名
    merged_rank:   int | None   # RRF 混合后的排名
    reranked_rank: int | None   # Cross-encoder 精排后的排名
    final_rank:    int | None   # MMR 选择后的排名

    # ── 命中标记 ────────────────────────────────────────────────────
    hit_at_1:  bool   # final_rank == 1
    hit_at_5:  bool   # final_rank <= 5（MMR 后的最终结果集）
    hit_at_10: bool   # reranked_rank <= 10（进了 reranker 输出 top-10）

    # ── 死因 C 自动标记 ─────────────────────────────────────────────
    # merged 前 10 但 reranker 倒退超过 5 位（或被踢出 rerank_top_k）
    reranker_inversion: bool

    # ── 各阶段独立延迟（ms） ─────────────────────────────────────────
    embed_ms:  float
    vector_ms: float
    bm25_ms:   float
    merge_ms:  float
    rerank_ms: float
    mmr_ms:    float
    total_ms:  float

    def to_csv_dict(self) -> dict:
        d = asdict(self)
        # None → "" 保证 CSV 可读
        for k in ("vec_rank", "bm25_rank", "merged_rank", "reranked_rank", "final_rank"):
            if d[k] is None:
                d[k] = ""
        return d


# ══════════════════════════════════════════════════════════════════════
# 指标计算
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(rows: list[EvalRow]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n_total": 0}

    hit1  = sum(r.hit_at_1  for r in rows) / n
    hit5  = sum(r.hit_at_5  for r in rows) / n
    hit10 = sum(r.hit_at_10 for r in rows) / n

    # MRR@10：基于 reranked_rank（反映精排质量，与最终 top_k 设置无关）
    mrr = sum(
        1.0 / r.reranked_rank
        for r in rows
        if r.reranked_rank is not None and r.reranked_rank <= 10
    ) / n

    # ── 延迟分位数 ──────────────────────────────────────────────────
    def pct(vals: list[float], p: float) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        return s[min(int(len(s) * p), len(s) - 1)]

    total_ms  = [r.total_ms  for r in rows]
    embed_ms  = [r.embed_ms  for r in rows]
    rerank_ms = [r.rerank_ms for r in rows]
    vector_ms = [r.vector_ms for r in rows]
    bm25_ms   = [r.bm25_ms   for r in rows]

    # ── 死因分布 ────────────────────────────────────────────────────
    n_inversion = sum(r.reranker_inversion for r in rows)

    # 对未命中 Hit@5 的条目自动归因
    cause_counts = {"A": 0, "B": 0, "C": 0}
    for r in rows:
        if r.hit_at_5:
            continue
        if r.reranker_inversion:
            cause_counts["C"] += 1
        elif r.vec_rank is None and r.bm25_rank is None:
            cause_counts["A"] += 1
        else:
            cause_counts["B"] += 1

    # ── 对抗样本分组 Hit@5 ─────────────────────────────────────────
    adv: dict[str, dict] = {}
    for r in rows:
        t = r.adversarial_type or "none"
        if t not in adv:
            adv[t] = {"n": 0, "hit5": 0}
        adv[t]["n"]    += 1
        adv[t]["hit5"] += int(r.hit_at_5)
    adv_hit5 = {
        t: {"n": v["n"], "hit5_rate": round(v["hit5"] / v["n"], 4)}
        for t, v in adv.items()
    }

    return {
        "n_total":                  n,
        "hit_at_1":                 round(hit1,  4),
        "hit_at_5":                 round(hit5,  4),
        "hit_at_10":                round(hit10, 4),
        "mrr_at_10":                round(mrr,   4),
        "reranker_inversion_count": n_inversion,
        "reranker_inversion_rate":  round(n_inversion / n, 4),
        "bad_case_cause_counts":    cause_counts,
        "latency_p50_ms":  round(pct(total_ms,  0.50), 1),
        "latency_p90_ms":  round(pct(total_ms,  0.90), 1),
        "latency_p99_ms":  round(pct(total_ms,  0.99), 1),
        "embed_p50_ms":    round(pct(embed_ms,  0.50), 1),
        "vector_p50_ms":   round(pct(vector_ms, 0.50), 1),
        "bm25_p50_ms":     round(pct(bm25_ms,   0.50), 1),
        "rerank_p50_ms":   round(pct(rerank_ms, 0.50), 1),
        "adversarial_hit5_by_type": adv_hit5,
    }


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def _print_summary(m: dict) -> None:
    SEP = "═" * 58
    print(f"\n{SEP}")
    print(f"  评测摘要  n={m['n_total']}")
    print(f"{'─' * 58}")
    print(f"  Hit@1   : {m['hit_at_1']:.1%}")
    print(f"  Hit@5   : {m['hit_at_5']:.1%}   ← 检索生死线（送给 LLM 的物理底线）")
    print(f"  Hit@10  : {m['hit_at_10']:.1%}   ← reranker 输出 top-10 覆盖率")
    print(f"  MRR@10  : {m['mrr_at_10']:.4f}  ← 精排精准度综合指标")
    print()
    print(f"  Reranker 倒挂 (死因C): {m['reranker_inversion_rate']:.1%}"
          f"  ({m['reranker_inversion_count']} 条)")
    print()
    print(f"  Hit@5 失败归因（自动）：")
    causes = m.get("bad_case_cause_counts", {})
    total_bad = sum(causes.values())
    for c, cnt in causes.items():
        labels = {"A": "词汇鸿沟", "B": "切块碎裂/合并失败", "C": "Reranker倒挂"}
        print(f"    死因 {c} ({labels.get(c,'')}): {cnt} 条  ({cnt/m['n_total']:.1%})")
    print()
    print(f"  端到端延迟 P50/P90/P99: "
          f"{m['latency_p50_ms']}/{m['latency_p90_ms']}/{m['latency_p99_ms']} ms")
    print(f"  各阶段 P50: embed={m['embed_p50_ms']}ms  "
          f"vector={m['vector_p50_ms']}ms  bm25={m['bm25_p50_ms']}ms  "
          f"rerank={m['rerank_p50_ms']}ms")
    print()
    if m.get("adversarial_hit5_by_type"):
        print(f"  对抗样本 Hit@5 分组：")
        for atype, stat in m["adversarial_hit5_by_type"].items():
            print(f"    {atype:<22s}: {stat['hit5_rate']:.1%}  (n={stat['n']})")
    print(f"{SEP}\n")


async def run_eval(args):
    # ── 读取黄金测试集 ────────────────────────────────────────────────
    if not os.path.exists(args.golden):
        print(f"❌ 找不到 {args.golden}，请先运行 build_golden_dataset.py", file=sys.stderr)
        sys.exit(1)

    with open(args.golden, encoding="utf-8") as f:
        golden = list(csv.DictReader(f))

    # human_query 优先，回退到 generated_query
    for row in golden:
        row["_query"] = (row.get("human_query") or row.get("generated_query", "")).strip()

    golden = [r for r in golden if r["_query"]]
    if not golden:
        print("❌ 黄金测试集中没有有效 query（human_query / generated_query 均为空）", file=sys.stderr)
        sys.exit(1)

    if args.limit:
        golden = golden[: args.limit]

    print(f"[run_retrieval_eval] 共 {len(golden)} 条有效记录")

    # ── 初始化检索核心 ────────────────────────────────────────────────
    core = RetrievalCore(verbose=args.verbose)
    core.load_models(warmup=True)
    await core.connect_db()

    eval_rows: list[EvalRow] = []

    try:
        for i, g in enumerate(golden, 1):
            query_id = int(g["query_id"])
            chunk_id = int(g["chunk_id"])
            query    = g["_query"]
            adv_type = g.get("adversarial_type", "").strip()

            if not args.verbose:
                print(f"  [{i:3d}/{len(golden)}] qid={query_id:3d}  "
                      f"chunk={chunk_id:6d}  q={query[:45]}")

            pr: PipelineResult = await core.run_pipeline(query)

            vec_r      = pr.rank_of(chunk_id, "vec")
            bm25_r     = pr.rank_of(chunk_id, "bm25")
            merged_r   = pr.rank_of(chunk_id, "merged")
            reranked_r = pr.rank_of(chunk_id, "reranked")
            final_r    = pr.rank_of(chunk_id, "final")

            # 命中标记
            hit1  = final_r is not None and final_r <= 1
            hit5  = final_r is not None and final_r <= 5
            hit10 = reranked_r is not None and reranked_r <= 10

            inversion = pr.rerank_inversion(chunk_id, inversion_gap=5)

            t = pr.timings
            eval_rows.append(EvalRow(
                query_id=query_id, chunk_id=chunk_id,
                query=query, adversarial_type=adv_type,
                vec_rank=vec_r, bm25_rank=bm25_r, merged_rank=merged_r,
                reranked_rank=reranked_r, final_rank=final_r,
                hit_at_1=hit1, hit_at_5=hit5, hit_at_10=hit10,
                reranker_inversion=inversion,
                embed_ms=round(t.embed_ms, 2),
                vector_ms=round(t.vector_ms, 2),
                bm25_ms=round(t.bm25_ms, 2),
                merge_ms=round(t.merge_ms, 2),
                rerank_ms=round(t.rerank_ms, 2),
                mmr_ms=round(t.mmr_ms, 2),
                total_ms=round(t.total_ms, 2),
            ))

            if not args.verbose:
                rank_str = (
                    f"vec={vec_r or '-':>4}  bm25={bm25_r or '-':>4}  "
                    f"merged={merged_r or '-':>4}  reranked={reranked_r or '-':>4}  "
                    f"final={final_r or '-':>3}  "
                    f"hit5={'✓' if hit5 else '✗'}  "
                    f"inv={'⚠' if inversion else ' '}"
                )
                print(f"         {rank_str}")

    finally:
        await core.close()

    # ── 写逐条结果 ────────────────────────────────────────────────────
    if not eval_rows:
        print("❌ 没有成功处理的 query，跳过写入", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)
    fieldnames = list(eval_rows[0].to_csv_dict().keys())
    with open(args.results, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in eval_rows:
            writer.writerow(row.to_csv_dict())

    # ── 写汇总指标 ────────────────────────────────────────────────────
    metrics = compute_metrics(eval_rows)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    _print_summary(metrics)
    print(f"  逐条结果 → {args.results}")
    print(f"  汇总指标 → {args.summary}")
    print()
    print("下一步：")
    print("  Hit@5 < 80%  → 运行 dump_bad_cases.py 进行人工尸检归因")
    print("  Hit@5 >= 80% → 进入阶段二：接入生成层 + LLM-as-Judge 评测")


def main():
    parser = argparse.ArgumentParser(description="RAG 检索质量评测（阶段一）")
    parser.add_argument("--golden",  default="eval/golden_dataset.csv",
                        help="黄金测试集 CSV（默认 eval/golden_dataset.csv）")
    parser.add_argument("--results", default="eval/eval_results.csv",
                        help="逐条结果输出路径")
    parser.add_argument("--summary", default="eval/eval_summary.json",
                        help="汇总指标输出路径")
    parser.add_argument("--limit",   type=int, default=None,
                        help="只跑前 N 条（调试用）")
    parser.add_argument("--verbose", action="store_true",
                        help="开启 RetrievalCore pipeline 实时监控输出")
    args = parser.parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
