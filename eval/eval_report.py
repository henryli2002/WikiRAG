"""
eval/eval_report.py
===================
全链路性能与质量统一报告。

读取同一目录下的三个 CSV：
  eval_results.csv       检索各阶段延迟 + 命中率
  eval_answers.csv       生成延迟 + Token 统计
  eval_judge_results.csv 生成质量（Faithfulness / Answer Relevance）

输出一张完整的分块延迟表（P50/P90/P99）+ 质量汇总，
方便后续用不同文件夹对比不同参数配置的结果。

用法：
  # 读取 eval/ 目录下的默认文件
  python eval/eval_report.py

  # 对比两个配置文件夹
  python eval/eval_report.py --dir eval/baseline
  python eval/eval_report.py --dir eval/topk7
"""

from __future__ import annotations

import os
import sys
import csv
import json
import argparse
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════════
# 通用工具
# ══════════════════════════════════════════════════════════════════════


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(v for v in vals if v is not None and v >= 0)
    if not s:
        return float("nan")
    return s[min(int(len(s) * p), len(s) - 1)]


def _avg(vals: list[float]) -> float:
    clean = [v for v in vals if v is not None and v >= 0]
    return sum(clean) / len(clean) if clean else float("nan")


def _fmt(v: float, unit: str = "") -> str:
    if v != v:  # nan
        return "  N/A   "
    return f"{v:>8.1f}{unit}"


def _read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _flt(row: dict, key: str) -> float:
    v = row.get(key, "")
    try:
        return float(v)
    except (ValueError, TypeError):
        return -1.0


def _int(row: dict, key: str) -> int:
    v = row.get(key, "")
    try:
        return int(v)
    except (ValueError, TypeError):
        return -1


# ══════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════


def load_retrieval(rows: list[dict]) -> dict:
    """从 eval_results.csv 提取各阶段延迟列表。"""
    d: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for col in (
            "embed_ms",
            "vector_ms",
            "bm25_ms",
            "merge_ms",
            "rerank_ms",
            "mmr_ms",
            "total_ms",
        ):
            v = _flt(r, col)
            if v >= 0:
                d[col].append(v)
        # recall wall-clock = max(vector, bm25)，逐行计算
        vec = _flt(r, "vector_ms")
        bm25 = _flt(r, "bm25_ms")
        if vec >= 0 and bm25 >= 0:
            d["recall_wall_ms"].append(max(vec, bm25))
    return dict(d)


def load_generation(rows: list[dict]) -> dict:
    """从 eval_answers.csv 提取生成延迟与 Token 列表。"""
    d: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for col in (
            "embed_ms",
            "vector_ms",
            "bm25_ms",
            "rerank_ms",
            "mmr_ms",
            "retrieval_ms",
            "ttft_ms",
            "gen_ms",
            "e2e_ms",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "n_chunks",
            "context_chars",
        ):
            v = _flt(r, col)
            if v >= 0:
                d[col].append(v)
    return dict(d)


def load_quality(ret_rows: list[dict], judge_rows: list[dict]) -> dict:
    """汇总检索命中率 + 生成质量分数。"""
    n = len(ret_rows)
    q: dict = {}

    if ret_rows:
        q["hit_at_1"] = (
            sum(
                r.get("hit_at_1", "").lower() not in ("false", "0", "")
                for r in ret_rows
            )
            / n
        )
        q["hit_at_5"] = (
            sum(
                r.get("hit_at_5", "").lower() not in ("false", "0", "")
                for r in ret_rows
            )
            / n
        )
        q["hit_at_10"] = (
            sum(
                r.get("hit_at_10", "").lower() not in ("false", "0", "")
                for r in ret_rows
            )
            / n
        )

        rr_vals = []
        for r in ret_rows:
            rk = _flt(r, "reranked_rank")
            if 0 < rk <= 10:
                rr_vals.append(1.0 / rk)
            else:
                rr_vals.append(0.0)
        q["mrr_at_10"] = sum(rr_vals) / n if rr_vals else 0.0

        # bad case 归因
        q["cause_A"] = sum(
            1
            for r in ret_rows
            if r.get("hit_at_5", "").lower() in ("false", "0")
            and _flt(r, "vec_rank") < 0
            and _flt(r, "bm25_rank") < 0
        )
        q["cause_B"] = sum(
            1
            for r in ret_rows
            if r.get("hit_at_5", "").lower() in ("false", "0")
            and r.get("reranker_inversion", "").lower() != "true"
            and not (_flt(r, "vec_rank") < 0 and _flt(r, "bm25_rank") < 0)
        )
        q["cause_C"] = sum(
            1 for r in ret_rows if r.get("reranker_inversion", "").lower() == "true"
        )

    if judge_rows:
        valid = [r for r in judge_rows if _flt(r, "faithfulness_score") >= 0]
        if valid:
            f_scores = [_flt(r, "faithfulness_score") for r in valid]
            ar_scores = [_flt(r, "answer_relevance_score") for r in valid]
            # 分数已经是 0-1 浮点，直接使用，无需除以 3.0
            q["faith_avg"] = _avg(f_scores)
            q["faith_ge067_pct"] = sum(s >= 0.67 for s in f_scores) / len(f_scores)
            q["rel_avg"] = _avg(ar_scores)
            q["rel_ge067_pct"] = sum(s >= 0.67 for s in ar_scores) / len(ar_scores)

            # Context Precision
            cp_valid = [
                r for r in judge_rows if _flt(r, "context_precision_score") >= 0
            ]
            if cp_valid:
                cp_scores = [_flt(r, "context_precision_score") for r in cp_valid]
                q["ctx_prec_avg"] = _avg(cp_scores)
                q["ctx_prec_ge067_pct"] = sum(s >= 0.67 for s in cp_scores) / len(
                    cp_scores
                )

            all_three = [q["faith_avg"], q["rel_avg"]] + (
                [q["ctx_prec_avg"]] if "ctx_prec_avg" in q else []
            )
            q["overall_avg"] = sum(all_three) / len(all_three)

            # 原子声明统计（来自新版 judge_answers.py）
            ct_vals = [
                _flt(r, "faithfulness_claims_total")
                for r in valid
                if _flt(r, "faithfulness_claims_total") > 0
            ]
            cs_vals = [
                _flt(r, "faithfulness_claims_supported")
                for r in valid
                if _flt(r, "faithfulness_claims_total") > 0
            ]
            if ct_vals:
                q["claims_total_avg"] = _avg(ct_vals)
                q["claims_supported_avg"] = _avg(cs_vals)

            # 命中 vs 未命中分层
            hit_j = [
                r
                for r in valid
                if str(r.get("hit_at_5", "")).lower() not in ("false", "0", "")
            ]
            miss_j = [
                r for r in valid if str(r.get("hit_at_5", "")).lower() in ("false", "0")
            ]
            for label, subset in [("hit", hit_j), ("miss", miss_j)]:
                if subset:
                    q[f"{label}_faith_avg"] = _avg(
                        [_flt(r, "faithfulness_score") for r in subset]
                    )
                    q[f"{label}_rel_avg"] = _avg(
                        [_flt(r, "answer_relevance_score") for r in subset]
                    )
                    q[f"{label}_cp_avg"] = _avg(
                        [
                            _flt(r, "context_precision_score")
                            for r in subset
                            if _flt(r, "context_precision_score") >= 0
                        ]
                    )
                    q[f"{label}_n"] = len(subset)
                    # 拒答率：由大模型在 judge 阶段判断为 is_refusal=True
                    refusals = sum(
                        1
                        for r in subset
                        if str(r.get("is_refusal", "False")).lower() == "true"
                    )
                    q[f"refusal_rate_{label}"] = refusals / len(subset)

    return q


# ══════════════════════════════════════════════════════════════════════
# 报告打印
# ══════════════════════════════════════════════════════════════════════

HDR = "═" * 68
SEP = "─" * 68
SCOL = 12  # 指标名列宽


def _latency_row(label: str, vals: list[float], unit: str = "ms") -> str:
    return (
        f"  {label:<{SCOL}}"
        f"  {_fmt(_pct(vals, 0.50), unit)}"
        f"  {_fmt(_pct(vals, 0.90), unit)}"
        f"  {_fmt(_pct(vals, 0.99), unit)}"
        f"    (n={len([v for v in vals if v >= 0])})"
    )


def print_report(
    ret_data: dict,
    gen_data: dict,
    quality: dict,
    eval_dir: str,
    n_ret: int,
    n_gen: int,
    n_judge: int,
) -> None:
    print(HDR)
    print(f"  WikiRAG 全链路性能与质量报告")
    print(f"  目录: {eval_dir}")
    print(f"  检索样本: {n_ret}  生成样本: {n_gen}  评分样本: {n_judge}")
    print(HDR)

    # ── 检索链路延迟 ──────────────────────────────────────────────────
    print(f"\n  ┌─ 检索链路延迟 {'─' * 37}┐")
    print(f"  │  {'指标':<{SCOL}}  {'P50':>9}  {'P90':>9}  {'P99':>9}         │")
    print(f"  │  {SEP[:58]}  │")
    sections = [
        (
            "Query 处理",
            [
                ("embed", ret_data.get("embed_ms", [])),
            ],
        ),
        (
            "并发召回",
            [
                ("vector recall", ret_data.get("vector_ms", [])),
                ("bm25 recall", ret_data.get("bm25_ms", [])),
                ("recall wall", ret_data.get("recall_wall_ms", [])),  # max(vec,bm25)
            ],
        ),
        (
            "精排",
            [
                ("rerank", ret_data.get("rerank_ms", [])),
            ],
        ),
        (
            "多样性",
            [
                ("mmr", ret_data.get("mmr_ms", [])),
            ],
        ),
        (
            "检索总计",
            [
                ("total", ret_data.get("total_ms", [])),
            ],
        ),
    ]
    for section_name, metrics in sections:
        print(f"  │  [{section_name}]")
        for label, vals in metrics:
            if vals:
                print(f"  │  {_latency_row(label, vals)}  │")
    print(f"  └{'─' * 64}┘")

    # ── 生成链路延迟 ──────────────────────────────────────────────────
    if gen_data:
        print(f"\n  ┌─ 生成链路延迟（Qwen 9B / llama.cpp）{'─' * 20}┐")
        print(f"  │  {'指标':<{SCOL}}  {'P50':>9}  {'P90':>9}  {'P99':>9}         │")
        print(f"  │  {SEP[:58]}  │")
        gen_metrics = [
            ("embed", gen_data.get("embed_ms", [])),
            ("vector", gen_data.get("vector_ms", [])),
            ("bm25", gen_data.get("bm25_ms", [])),
            ("rerank", gen_data.get("rerank_ms", [])),
            ("mmr", gen_data.get("mmr_ms", [])),
            ("retrieval", gen_data.get("retrieval_ms", [])),
            ("TTFT 首字", gen_data.get("ttft_ms", [])),
            ("generation", gen_data.get("gen_ms", [])),
            ("e2e", gen_data.get("e2e_ms", [])),
        ]
        for label, vals in gen_metrics:
            if vals:
                print(f"  │  {_latency_row(label, vals)}  │")
        print(f"  └{'─' * 64}┘")

        # ── Token 统计 ────────────────────────────────────────────────
        print(f"\n  ┌─ Token 统计（生成模型输入/输出）{'─' * 26}┐")
        print(f"  │  {'指标':<{SCOL}}  {'P50':>9}  {'P90':>9}  {'P99':>9}         │")
        print(f"  │  {SEP[:58]}  │")
        tok_metrics = [
            ("prompt tok", gen_data.get("prompt_tokens", [])),
            ("complete tok", gen_data.get("completion_tokens", [])),
            ("total tok", gen_data.get("total_tokens", [])),
            ("context chars", gen_data.get("context_chars", [])),
            ("n_chunks", gen_data.get("n_chunks", [])),
        ]
        for label, vals in tok_metrics:
            if vals:
                print(f"  │  {_latency_row(label, vals, '')}  │")
        print(f"  └{'─' * 64}┘")

    # ── 检索质量 ──────────────────────────────────────────────────────
    if quality:
        print(f"\n  ┌─ 检索质量 {'─' * 52}┐")
        for label, key, fmt in [
            ("Hit@1", "hit_at_1", ".1%"),
            ("Hit@5", "hit_at_5", ".1%"),
            ("Hit@10", "hit_at_10", ".1%"),
            ("MRR@10", "mrr_at_10", ".4f"),
        ]:
            v = quality.get(key)
            if v is not None:
                print(f"  │  {label:<12}  {v:{fmt}}")
        if "cause_A" in quality:
            print(
                f"  │  Bad case 归因: A(词汇鸿沟)={quality['cause_A']}  "
                f"B(切块碎裂)={quality['cause_B']}  "
                f"C(Reranker倒挂)={quality['cause_C']}"
            )
        print(f"  └{'─' * 64}┘")

        # ── 生成质量 ──────────────────────────────────────────────────
        if "faith_avg" in quality:
            print(
                f"\n  ┌─ 生成质量（Gemini 2.5 Flash judge，原子分解，RAGAS 3指标）{'─' * 5}┐"
            )
            print(
                f"  │  Faithfulness      avg={quality['faith_avg']:.3f}  "
                f"≥0.67达标率={quality['faith_ge067_pct']:.1%}"
            )
            if "claims_total_avg" in quality:
                print(
                    f"  │    原子声明  平均总数={quality['claims_total_avg']:.1f}  "
                    f"平均支持数={quality['claims_supported_avg']:.1f}"
                )
            print(
                f"  │  Answer Relevance  avg={quality['rel_avg']:.3f}  "
                f"≥0.67达标率={quality['rel_ge067_pct']:.1%}"
            )
            if "ctx_prec_avg" in quality:
                print(
                    f"  │  Ctx Precision     avg={quality['ctx_prec_avg']:.3f}  "
                    f"≥0.67达标率={quality['ctx_prec_ge067_pct']:.1%}"
                )
            print(f"  │  Overall           avg={quality['overall_avg']:.3f}")
            if "hit_n" in quality or "miss_n" in quality:
                print(f"  │  {'─' * 58}")
                print(f"  │  分层分析（检索命中 vs 未命中对生成质量的影响）")
                for key, label in [("hit", "命中"), ("miss", "未命中")]:
                    n = quality.get(f"{key}_n", 0)
                    fa = quality.get(f"{key}_faith_avg", float("nan"))
                    ra = quality.get(f"{key}_rel_avg", float("nan"))
                    cp = quality.get(f"{key}_cp_avg", float("nan"))
                    valid_scores = [s for s in [fa, ra, cp] if s == s]
                    ov = (
                        sum(valid_scores) / len(valid_scores)
                        if valid_scores
                        else float("nan")
                    )
                    if n:
                        print(
                            f"  │    {label}(n={n:2d})  "
                            f"Faith={fa:.3f}  Rel={ra:.3f}  "
                            f"CtxPrec={cp:.3f}  Overall={ov:.3f}"
                        )
            if "refusal_rate_hit" in quality or "refusal_rate_miss" in quality:
                print(f"  │  {'─' * 58}")
                print(f"  │  拒答率分析（区分 RAG 问题 vs 模型问题）")
                rh = quality.get("refusal_rate_hit", float("nan"))
                rm = quality.get("refusal_rate_miss", float("nan"))
                if rh == rh:
                    print(
                        f"  │    检索命中时拒答率  = {rh:.1%}  ← 高则为模型/Prompt 问题"
                    )
                if rm == rm:
                    print(
                        f"  │    检索未命中时拒答率 = {rm:.1%}  ← 高则正常（没有依据不乱编）"
                    )
            print(f"  └{'─' * 64}┘")

    print(f"\n{HDR}\n")


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="全链路性能与质量统一报告")
    parser.add_argument(
        "--dir", default="eval", help="包含三个 CSV 文件的目录（默认 eval/）"
    )
    parser.add_argument("--ret-csv", default=None, help="覆盖 eval_results.csv 路径")
    parser.add_argument("--gen-csv", default=None, help="覆盖 eval_answers.csv 路径")
    parser.add_argument(
        "--judge-csv", default=None, help="覆盖 eval_judge_results.csv 路径"
    )
    args = parser.parse_args()

    d = args.dir
    ret_path = args.ret_csv or os.path.join(d, "eval_results.csv")
    gen_path = args.gen_csv or os.path.join(d, "eval_answers.csv")
    judge_path = args.judge_csv or os.path.join(d, "eval_judge_results.csv")

    ret_rows = _read_csv(ret_path)
    gen_rows = _read_csv(gen_path)
    judge_rows = _read_csv(judge_path)

    if not ret_rows:
        print(f"⚠ 未找到 {ret_path}，检索部分将跳过", file=sys.stderr)
    if not gen_rows:
        print(f"⚠ 未找到 {gen_path}，生成部分将跳过", file=sys.stderr)
    if not judge_rows:
        print(f"⚠ 未找到 {judge_path}，质量评分部分将跳过", file=sys.stderr)

    ret_data = load_retrieval(ret_rows)
    gen_data = load_generation(gen_rows)
    quality = load_quality(ret_rows, judge_rows)

    print_report(
        ret_data=ret_data,
        gen_data=gen_data,
        quality=quality,
        eval_dir=os.path.abspath(d),
        n_ret=len(ret_rows),
        n_gen=len(gen_rows),
        n_judge=len(judge_rows),
    )


if __name__ == "__main__":
    main()
