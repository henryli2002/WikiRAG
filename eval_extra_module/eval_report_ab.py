"""
eval_extra_module/eval_report_ab.py
====================================
AB 消融实验对比报告。

读取 4 组实验的 eval_judge_summary.json 和 eval_answers.csv，
生成跨条件对比表，并写入 eval_extra_module/ablation_report.txt。

用法：
  python -m eval_extra_module.eval_report_ab
  python -m eval_extra_module.eval_report_ab --output eval_extra_module/ablation_report.txt
"""

from __future__ import annotations

import os
import sys
import csv
import json
import argparse
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# 四组实验条件
CONDITIONS = [
    ("baseline",        "无A 无B (baseline)"),
    ("rewrite_only",    "A only (Query重写)"),
    ("fusion_only",     "B only (Chunk融合)"),
    ("rewrite_fusion",  "A + B (重写+融合)"),
]


def _read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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


def _pct(vals: list[float], p: float) -> float:
    s = sorted(v for v in vals if v >= 0)
    if not s:
        return float("nan")
    return s[min(int(len(s) * p), len(s) - 1)]


def _avg(vals: list[float]) -> float:
    clean = [v for v in vals if v >= 0]
    return sum(clean) / len(clean) if clean else float("nan")


def _fmt(v: float, width: int = 8) -> str:
    if v != v:  # nan
        return " N/A".rjust(width)
    return f"{v:.3f}".rjust(width)


def _fmt_pct(v: float, width: int = 8) -> str:
    if v != v:
        return " N/A".rjust(width)
    return f"{v:.1%}".rjust(width)


def _fmt_ms(v: float, width: int = 8) -> str:
    if v != v:
        return " N/A".rjust(width)
    return f"{v:.0f}".rjust(width)


def generate_report(output_path: str) -> str:
    """生成消融对比报告，同时写入文件和返回字符串。"""
    lines: list[str] = []

    def w(s: str = ""):
        lines.append(s)

    w("=" * 76)
    w("  WikiRAG AB 消融实验对比报告")
    w("  Factor A = Query 重写 (llama.cpp / Qwen 9B)")
    w("  Factor B = Chunk 融合 (llama.cpp / Qwen 9B)")
    w("  Top-K = 3 | Golden Dataset = 50 条随机主题 query")
    w("=" * 76)

    # ── 加载数据 ──────────────────────────────────────────────────────
    summaries: dict[str, dict] = {}
    gen_data: dict[str, list[dict]] = {}

    for cond_key, cond_label in CONDITIONS:
        if cond_key == "baseline":
            # baseline 数据在 eval_random_topics/rag_top3，无需复制
            cond_dir = os.path.join(MODULE_DIR, "..", "eval_random_topics", "rag_top3")
        else:
            cond_dir = os.path.join(MODULE_DIR, cond_key)
        summary_path = os.path.join(cond_dir, "eval_judge_summary.json")
        answers_path = os.path.join(cond_dir, "eval_answers.csv")
        summaries[cond_key] = _read_json(summary_path)
        gen_data[cond_key] = _read_csv(answers_path)

    available = [k for k, _ in CONDITIONS if summaries[k]]
    if not available:
        w("\n  ❌ 未找到任何实验数据，请先运行 run_ablation.sh")
        w()
        report = "\n".join(lines)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        return report

    # ── 核心指标对比表 ────────────────────────────────────────────────
    w()
    w("  ┌─ 核心质量指标对比 " + "─" * 54 + "┐")

    # 表头
    header = f"  │ {'指标':<20}"
    for cond_key, cond_label in CONDITIONS:
        if cond_key in available:
            header += f"  {cond_label:>14}"
    header += "  │"
    w(header)
    w(f"  │ {'─' * 20}" + "".join(f"  {'─' * 14}" for k, _ in CONDITIONS if k in available) + "  │")

    # 指标行
    metrics = [
        ("Faithfulness",      "faithfulness_avg"),
        ("Faith ≥0.67 达标率", "faithfulness_ge067"),
        ("Answer Relevance",  "answer_relevance_avg"),
        ("Ctx Precision",     "context_precision_avg"),
        ("Ctx Prec ≥0.67",   "context_precision_ge067"),
        ("Overall",           "overall_avg"),
        ("拒答率",            "refusal_rate"),
        ("原子声明 avg",       "avg_claims_per_answer"),
    ]

    for label, key in metrics:
        row_str = f"  │ {label:<20}"
        is_pct = key in ("faithfulness_ge067", "context_precision_ge067",
                         "answer_relevance_ge067", "refusal_rate")
        for cond_key, _ in CONDITIONS:
            if cond_key not in available:
                continue
            val = summaries[cond_key].get(key, float("nan"))
            if val is None:
                val = float("nan")
            if is_pct:
                row_str += f"  {_fmt_pct(val, 14)}"
            elif key == "avg_claims_per_answer":
                row_str += f"  {_fmt(val, 14)}" if val == val else f"  {'N/A':>14}"
            else:
                row_str += f"  {_fmt(val, 14)}"
        row_str += "  │"
        w(row_str)

    w(f"  └{'─' * 74}┘")

    # ── 延迟对比表 ────────────────────────────────────────────────────
    w()
    w("  ┌─ 延迟对比（P50, ms）" + "─" * 52 + "┐")

    header = f"  │ {'阶段':<14}"
    for cond_key, cond_label in CONDITIONS:
        if cond_key in available:
            header += f"  {cond_label:>14}"
    header += "  │"
    w(header)
    w(f"  │ {'─' * 14}" + "".join(f"  {'─' * 14}" for k, _ in CONDITIONS if k in available) + "  │")

    latency_keys = [
        ("rewrite",    "rewrite_ms"),
        ("fusion",     "fusion_ms"),
        ("retrieval",  "retrieval_ms"),
        ("TTFT 首字",  "ttft_ms"),
        ("generation", "gen_ms"),
        ("e2e",        "e2e_ms"),
    ]

    for label, key in latency_keys:
        row_str = f"  │ {label:<14}"
        for cond_key, _ in CONDITIONS:
            if cond_key not in available:
                continue
            rows = gen_data[cond_key]
            vals = [_flt(r, key) for r in rows if _flt(r, key) >= 0]
            p50 = _pct(vals, 0.50) if vals else float("nan")
            row_str += f"  {_fmt_ms(p50, 14)}"
        row_str += "  │"
        w(row_str)

    w(f"  └{'─' * 74}┘")

    # ── Token 对比 ────────────────────────────────────────────────────
    w()
    w("  ┌─ Token & Context 对比（P50）" + "─" * 44 + "┐")

    header = f"  │ {'指标':<14}"
    for cond_key, cond_label in CONDITIONS:
        if cond_key in available:
            header += f"  {cond_label:>14}"
    header += "  │"
    w(header)
    w(f"  │ {'─' * 14}" + "".join(f"  {'─' * 14}" for k, _ in CONDITIONS if k in available) + "  │")

    tok_keys = [
        ("prompt_tok",    "prompt_tokens"),
        ("complete_tok",  "completion_tokens"),
        ("context_chars", "context_chars"),
        ("n_chunks",      "n_chunks"),
    ]

    for label, key in tok_keys:
        row_str = f"  │ {label:<14}"
        for cond_key, _ in CONDITIONS:
            if cond_key not in available:
                continue
            rows = gen_data[cond_key]
            vals = [_flt(r, key) for r in rows if _flt(r, key) >= 0]
            p50 = _pct(vals, 0.50) if vals else float("nan")
            row_str += f"  {_fmt_ms(p50, 14)}"
        row_str += "  │"
        w(row_str)

    w(f"  └{'─' * 74}┘")

    # ── 消融分析 ──────────────────────────────────────────────────────
    w()
    w("  ┌─ 消融效果分析（相对 baseline 的增量）" + "─" * 34 + "┐")

    baseline = summaries.get("baseline", {})
    if baseline:
        delta_metrics = [
            ("Faithfulness",    "faithfulness_avg"),
            ("Answer Relevance","answer_relevance_avg"),
            ("Ctx Precision",   "context_precision_avg"),
            ("Overall",         "overall_avg"),
            ("拒答率",          "refusal_rate"),
        ]

        header = f"  │ {'指标':<20}"
        for cond_key, cond_label in CONDITIONS[1:]:
            if cond_key in available:
                header += f"  {cond_label:>14}"
        header += "  │"
        w(header)
        w(f"  │ {'─' * 20}" + "".join(f"  {'─' * 14}" for k, _ in CONDITIONS[1:] if k in available) + "  │")

        for label, key in delta_metrics:
            row_str = f"  │ {label:<20}"
            base_val = baseline.get(key, 0) or 0
            for cond_key, _ in CONDITIONS[1:]:
                if cond_key not in available:
                    continue
                val = summaries[cond_key].get(key, float("nan"))
                if val is None or val != val:
                    row_str += f"  {'N/A':>14}"
                else:
                    delta = val - base_val
                    sign = "+" if delta >= 0 else ""
                    # 拒答率下降是好事，标记方向
                    row_str += f"  {sign}{delta:.3f}".rjust(16)
            row_str += "  │"
            w(row_str)

    w(f"  └{'─' * 74}┘")

    w()
    w("=" * 76)
    w()

    report = "\n".join(lines)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser(description="AB 消融实验对比报告")
    parser.add_argument(
        "--output",
        default=os.path.join(MODULE_DIR, "ablation_report.txt"),
        help="报告输出路径",
    )
    args = parser.parse_args()

    report = generate_report(args.output)
    print(report)
    print(f"  报告已写入 {args.output}")


if __name__ == "__main__":
    main()
