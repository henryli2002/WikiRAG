"""
eval/judge_answers.py
=====================
阶段二第二步：LLM-as-Judge（Gemini 裁判打分）。

读取 eval_answers.csv，对每条（Query + Context + Answer）调用 Gemini 打分：

  Faithfulness（忠实度，0-3）
    测试模型是否严格遵循 Context，不产生幻觉。
    3 = 答案完全源于 Context | 2 = 轻微引用外部知识 | 1 = 大量依赖外部知识 | 0 = 忽略/违背 Context

  Answer Relevance（相关性，0-3）
    测试模型是否真正回答了问题。
    3 = 完整直接回答 | 2 = 基本回答但有遗漏 | 1 = 仅与问题相关但未作答 | 0 = 未回答问题

输出：
  eval_judge_results.csv    逐条得分 + 裁判理由
  eval_judge_summary.json   汇总指标

人工抽检提示：
  NEXT.md 要求对裁判结果抽检 10% 以对齐判断标准。
  本脚本在摘要中自动输出"需人工复查"的边界 case（两项分数差异大或绝对分偏低）。

用法：
  python eval/judge_answers.py
  python eval/judge_answers.py --answers eval/eval_answers.csv --spot-check 5
"""

from __future__ import annotations

import os
import sys
import csv
import json
import time
import argparse
from dataclasses import dataclass, asdict

from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


# ══════════════════════════════════════════════════════════════════════
# 裁判 Prompt
# ══════════════════════════════════════════════════════════════════════

_JUDGE_PROMPT = """\
你是一个严格的 RAG 系统质量评估专家。你的任务是对一个问答系统的单条输出进行评分。

【输入信息】

问题：
{query}

系统检索到的参考资料（Context）：
{context}

系统生成的答案（Answer）：
{answer}

【评分标准】

请对以下两个维度分别打分（整数，0-3分）：

1. Faithfulness（忠实度）
   评估 Answer 是否严格来源于 Context，有无幻觉或自行补充外部知识。
   3分：Answer 的每一句话都能在 Context 中找到直接依据，无任何外部补充
   2分：Answer 基本忠实，但有 1-2 处轻微引用了 Context 未提及的细节
   1分：Answer 有大量内容依赖模型自身知识，Context 只起到部分支撑作用
   0分：Answer 与 Context 矛盾，或完全忽略了 Context 内容

2. Answer Relevance（相关性）
   评估 Answer 是否真正、完整地回答了 Question。
   3分：准确、完整地回答了问题，信息密度高
   2分：回答了问题的主要部分，但遗漏了某个关键方面
   1分：Answer 与问题相关，但实质上未作答（如仅重复问题或给出无意义的泛泛之词）
   0分：Answer 完全未回答问题，或答非所问

【输出格式】
请严格输出以下 JSON，不要输出任何其他内容：
{{
  "faithfulness": {{
    "score": <0-3的整数>,
    "reason": "<简要说明，1-2句，指出具体的支撑或扣分点>"
  }},
  "answer_relevance": {{
    "score": <0-3的整数>,
    "reason": "<简要说明，1-2句>"
  }}
}}"""


# ══════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════

@dataclass
class JudgeRow:
    query_id:          int
    query:             str
    adversarial_type:  str
    hit_at_5:          str
    generated_answer:  str

    faithfulness_score:      int   # 0-3
    faithfulness_reason:     str
    answer_relevance_score:  int   # 0-3
    answer_relevance_reason: str

    # 归一化到 0-1，方便汇总
    @property
    def faithfulness_norm(self) -> float:
        return self.faithfulness_score / 3.0

    @property
    def answer_relevance_norm(self) -> float:
        return self.answer_relevance_score / 3.0

    @property
    def overall_norm(self) -> float:
        return (self.faithfulness_norm + self.answer_relevance_norm) / 2.0

    def to_csv_dict(self) -> dict:
        return {
            "query_id":               self.query_id,
            "query":                  self.query,
            "adversarial_type":       self.adversarial_type,
            "hit_at_5":               self.hit_at_5,
            "generated_answer":       self.generated_answer,
            "faithfulness_score":     self.faithfulness_score,
            "faithfulness_norm":      round(self.faithfulness_norm, 4),
            "faithfulness_reason":    self.faithfulness_reason,
            "answer_relevance_score": self.answer_relevance_score,
            "answer_relevance_norm":  round(self.answer_relevance_norm, 4),
            "answer_relevance_reason":self.answer_relevance_reason,
            "overall_norm":           round(self.overall_norm, 4),
        }


# ══════════════════════════════════════════════════════════════════════
# Gemini 调用
# ══════════════════════════════════════════════════════════════════════

def _call_gemini(model, prompt: str, retries: int = 2) -> dict:
    """调用 Gemini，返回解析后的评分 dict。失败时重试，最终失败返回默认值。"""
    for attempt in range(retries + 1):
        try:
            generation_config = {"temperature": 0.1, "top_p": 0.7, "top_k": 1}
        response = model.generate_content(prompt, generation_config=generation_config)
            text = response.text.strip()
            # 去掉可能的 markdown 代码块包裹
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:-1])
            return json.loads(text)
        except Exception as e:
            if attempt < retries:
                time.sleep(1.5)
            else:
                print(f"    ⚠ Gemini 调用失败（{retries+1} 次）: {e}")
                return {
                    "faithfulness":    {"score": -1, "reason": f"调用失败: {e}"},
                    "answer_relevance":{"score": -1, "reason": f"调用失败: {e}"},
                }


# ══════════════════════════════════════════════════════════════════════
# 指标汇总
# ══════════════════════════════════════════════════════════════════════

def compute_summary(rows: list[JudgeRow]) -> dict:
    valid = [r for r in rows if r.faithfulness_score >= 0]
    n     = len(valid)
    if n == 0:
        return {"n_valid": 0}

    def avg(vals): return round(sum(vals) / len(vals), 4) if vals else 0.0
    def pct(vals, threshold): return round(sum(v >= threshold for v in vals) / len(vals), 4)

    f_norms  = [r.faithfulness_norm    for r in valid]
    ar_norms = [r.answer_relevance_norm for r in valid]
    ov_norms = [r.overall_norm          for r in valid]

    # 分层：命中 vs 未命中（检索质量对生成质量的影响）
    hit_rows  = [r for r in valid if str(r.hit_at_5).lower() not in ("false", "0", "")]
    miss_rows = [r for r in valid if str(r.hit_at_5).lower() in ("false", "0")]

    def layer_stats(subset):
        if not subset: return {}
        return {
            "n":                   len(subset),
            "faithfulness_avg":    avg([r.faithfulness_norm    for r in subset]),
            "answer_relevance_avg":avg([r.answer_relevance_norm for r in subset]),
            "overall_avg":         avg([r.overall_norm          for r in subset]),
        }

    # 对抗类型分层
    adv_groups: dict[str, list] = {}
    for r in valid:
        t = r.adversarial_type or "none"
        adv_groups.setdefault(t, []).append(r)
    adv_stats = {t: layer_stats(v) for t, v in adv_groups.items()}

    return {
        "n_total":              len(rows),
        "n_valid":              n,
        "n_failed":             len(rows) - n,
        "faithfulness_avg":     avg(f_norms),
        "faithfulness_ge2":     pct([r.faithfulness_score for r in valid], 2),  # ≥2/3
        "answer_relevance_avg": avg(ar_norms),
        "answer_relevance_ge2": pct([r.answer_relevance_score for r in valid], 2),
        "overall_avg":          avg(ov_norms),
        "by_retrieval_hit": {
            "hit":  layer_stats(hit_rows),
            "miss": layer_stats(miss_rows),
        },
        "by_adversarial_type": adv_stats,
    }


def _select_spot_check(rows: list[JudgeRow], n: int) -> list[JudgeRow]:
    """
    选取最值得人工抽检的 n 条：
      - 两项分数差距最大的（Gemini 自身评判不一致，怀疑 Criteria 偏差）
      - 分数处于中间区域（不是明显对也不是明显错，判断最模糊）
    """
    valid = [r for r in rows if r.faithfulness_score >= 0]
    scored = sorted(
        valid,
        key=lambda r: (
            abs(r.faithfulness_score - r.answer_relevance_score) * 10
            + (3 - r.overall_norm * 3)  # 偏低分优先
        ),
        reverse=True,
    )
    return scored[:n]


def _print_summary(m: dict) -> None:
    SEP = "═" * 60
    print(f"\n{SEP}")
    print(f"  生成质量评测摘要  (n={m['n_valid']}/{m['n_total']}, 失败={m['n_failed']})")
    print(f"{'─' * 60}")
    print(f"  Faithfulness    均值: {m['faithfulness_avg']:.3f}  "
          f"≥2/3 达标率: {m['faithfulness_ge2']:.1%}")
    print(f"  Answer Relevance 均值: {m['answer_relevance_avg']:.3f}  "
          f"≥2/3 达标率: {m['answer_relevance_ge2']:.1%}")
    print(f"  综合得分         均值: {m['overall_avg']:.3f}")

    hit  = m["by_retrieval_hit"].get("hit",  {})
    miss = m["by_retrieval_hit"].get("miss", {})
    if hit or miss:
        print(f"\n  检索命中 vs 未命中（生成质量对比）：")
        if hit:
            print(f"    命中 (n={hit.get('n',0)}):  "
                  f"Faith={hit.get('faithfulness_avg',0):.3f}  "
                  f"Rel={hit.get('answer_relevance_avg',0):.3f}  "
                  f"Overall={hit.get('overall_avg',0):.3f}")
        if miss:
            print(f"    未命中(n={miss.get('n',0)}):  "
                  f"Faith={miss.get('faithfulness_avg',0):.3f}  "
                  f"Rel={miss.get('answer_relevance_avg',0):.3f}  "
                  f"Overall={miss.get('overall_avg',0):.3f}")

    adv = m.get("by_adversarial_type", {})
    if adv:
        print(f"\n  对抗类型分层：")
        for t, stat in adv.items():
            if stat:
                print(f"    {t:<22s}: Faith={stat.get('faithfulness_avg',0):.3f}  "
                      f"Rel={stat.get('answer_relevance_avg',0):.3f}  "
                      f"(n={stat.get('n',0)})")
    print(f"{SEP}\n")


def _print_spot_check(cases: list[JudgeRow]) -> None:
    if not cases:
        return
    print("─" * 60)
    print(f"  ◆ 人工抽检候选（{len(cases)} 条，两项分差最大或分数最模糊）")
    print("─" * 60)
    for r in cases:
        print(f"\n  [qid={r.query_id}]  Faith={r.faithfulness_score}/3  "
              f"Rel={r.answer_relevance_score}/3  adv={r.adversarial_type or 'none'}")
        print(f"  Q: {r.query[:70]}")
        print(f"  A: {r.generated_answer[:100]}...")
        print(f"  Faith-reason : {r.faithfulness_reason}")
        print(f"  Rel-reason   : {r.answer_relevance_reason}")
    print()
    print("  ⚑ 如发现 Gemini 打分与人工判断不一致，请修改 _JUDGE_PROMPT 中的评分准则后重跑。")
    print()


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge 评分（阶段二 Step 2）")
    parser.add_argument("--answers",     default="eval/eval_answers.csv")
    parser.add_argument("--results",     default="eval/eval_judge_results.csv")
    parser.add_argument("--summary",     default="eval/eval_judge_summary.json")
    parser.add_argument("--model",       default="gemini-2.5-pro",
                        help="Gemini 裁判模型 ID")
    parser.add_argument("--delay",       type=float, default=0.5,
                        help="API 调用间隔（秒）")
    parser.add_argument("--spot-check",  type=int,   default=5,
                        help="打印 N 条人工抽检候选（默认 5）")
    parser.add_argument("--limit",       type=int,   default=None)
    args = parser.parse_args()

    # ── 读取答案 CSV ──────────────────────────────────────────────────
    if not os.path.exists(args.answers):
        print(f"❌ 找不到 {args.answers}，请先运行 generate_answers.py", file=sys.stderr)
        sys.exit(1)

    with open(args.answers, encoding="utf-8") as f:
        answer_rows = list(csv.DictReader(f))

    answer_rows = [r for r in answer_rows if r.get("generated_answer", "").strip()]
    if args.limit:
        answer_rows = answer_rows[: args.limit]

    print(f"[judge_answers] {len(answer_rows)} 条待评分")

    # ── 初始化 Gemini ─────────────────────────────────────────────────
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        judge_model = genai.GenerativeModel(args.model)
    except ImportError:
        print("❌ pip install -U google-generativeai", file=sys.stderr)
        sys.exit(1)

    # ── 逐条评分 ──────────────────────────────────────────────────────
    judge_rows: list[JudgeRow] = []

    for i, row in enumerate(answer_rows, 1):
        query_id = int(row["query_id"])
        query    = row["query"]
        context  = row["retrieved_context"]
        answer   = row["generated_answer"]

        print(f"  [{i:3d}/{len(answer_rows)}] qid={query_id:3d}  q={query[:45]}")

        prompt = _JUDGE_PROMPT.format(
            query=query, context=context[:1500], answer=answer
        )
        result = _call_gemini(judge_model, prompt)

        f_score  = result.get("faithfulness",    {}).get("score", -1)
        f_reason = result.get("faithfulness",    {}).get("reason", "")
        r_score  = result.get("answer_relevance",{}).get("score", -1)
        r_reason = result.get("answer_relevance",{}).get("reason", "")

        print(f"           Faith={f_score}/3  Rel={r_score}/3  "
              f"| {f_reason[:40]}...")

        judge_rows.append(JudgeRow(
            query_id=query_id,
            query=query,
            adversarial_type=row.get("adversarial_type", ""),
            hit_at_5=row.get("hit_at_5", ""),
            generated_answer=answer,
            faithfulness_score=int(f_score) if str(f_score).lstrip("-").isdigit() else -1,
            faithfulness_reason=f_reason,
            answer_relevance_score=int(r_score) if str(r_score).lstrip("-").isdigit() else -1,
            answer_relevance_reason=r_reason,
        ))

        if i < len(answer_rows) and args.delay > 0:
            time.sleep(args.delay)

    # ── 写结果 ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)

    with open(args.results, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(judge_rows[0].to_csv_dict().keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in judge_rows:
            writer.writerow(r.to_csv_dict())

    summary = compute_summary(judge_rows)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _print_summary(summary)

    # ── 人工抽检候选 ──────────────────────────────────────────────────
    spot = _select_spot_check(judge_rows, args.spot_check)
    _print_spot_check(spot)

    print(f"  逐条结果 → {args.results}")
    print(f"  汇总指标 → {args.summary}")
    print()
    print("  ⚠ 请人工阅读上方抽检候选，若 Gemini 评分逻辑与你的判断不一致，")
    print("    修改本文件中的 _JUDGE_PROMPT 并重新运行（无需重跑 generate_answers.py）。")


if __name__ == "__main__":
    main()
