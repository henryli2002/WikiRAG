"""
eval/judge_answers.py
=====================
阶段二第二步：原子级 LLM-as-Judge（Gemini 2.5 Flash 裁判打分）。

对每条（Query + Context + Answer）执行四次 Gemini 调用：

  Call 1  Faithfulness 原子分解
          将 Answer 拆成 N 条独立原子声明

  Call 2  Faithfulness 批量验证（一次调用处理所有声明）
          对每条声明判断：可否从 Context 直接推导？（YES/NO）
          Faithfulness = YES 数 / 总声明数  → 0-1 浮点

  Call 3  Answer Relevance 反向生成（Gemini） + 余弦相似度（BGE-M3）
          从 Answer 生成 3 个候选问题
          Answer Relevance = mean cosine_sim(original_query, generated_q)  → 0-1 浮点

  Call 4  Context Precision
          对每个召回 Chunk 判断：是否包含回答问题的有用信息？（YES/NO）
          Context Precision = 有用 Chunk 数 / 总 Chunk 数  → 0-1 浮点

  # TODO: Answer Correctness（需要 golden_dataset 提供人工标注的参考答案）
  #   实现方式：对比 generated_answer 与 reference_answer 的语义相似度 + 事实重叠
  #   成本高，适合上线前 A/B 测试阶段引入

用法：
  python eval/judge_answers.py
  python eval/judge_answers.py --answers eval/eval_answers.csv --spot-check 5
"""

from __future__ import annotations

import os
import re
import sys
import csv
import json
import time
import argparse
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

_GEN_CFG = {"temperature": 0.1, "top_p": 0.7, "top_k": 1}

# ══════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════

# ── Call 1：原子分解 ──────────────────────────────────────────────────
_DECOMPOSE_PROMPT = """\
将以下答案拆分成若干独立的原子声明，并判断该答案是否属于“拒答”。

要求：
- 如果答案明确表示无法从资料中得出结论、未提及、或者包含“无法回答”等拒答语义（即使附带了其他解释说明），请将 `is_refusal` 设为 true。
- 只有当答案给出了明确的、实质性的回答内容时，`is_refusal` 才为 false。
- 每条声明是一个最小化的、独立可验证的事实陈述。
- 不要将多个事实合并在一条声明中。
- 不要改写原文，保持原意。

答案：
{answer}

严格输出以下 JSON，不要包含任何其他内容或 Markdown 标记：
{{
  "is_refusal": true/false,
  "reason": "判断是否为拒答的简短理由",
  "claims": [
    "原子声明1",
    "原子声明2"
  ]
}}"""

# ── Call 2：批量验证 ──────────────────────────────────────────────────
_VERIFY_PROMPT = """\
判断以下每条声明能否从给定的 Context 中直接得出。
Context 中明确陈述、或可由 Context 直接推导的声明判定为 YES，否则为 NO。

规则：
- 按顺序输出 YES 或 NO，每行一个
- 输出行数必须与声明数量完全一致
- 不要输出任何其他内容

Context：
{context}

声明列表：
{claims_numbered}

判断结果（按顺序，每行 YES 或 NO）："""

# ── Call 4：Context Precision ────────────────────────────────────────
_CONTEXT_PRECISION_PROMPT = """\
判断以下每个检索到的 Chunk 是否包含有助于回答问题的信息。

有帮助 = Chunk 中存在能直接或间接支持回答该问题的内容
无帮助 = Chunk 内容与问题完全无关

问题：{query}

{chunks_numbered}

按顺序输出 YES 或 NO，每行一个，行数必须与 Chunk 数量完全一致，不要输出任何其他内容："""

# ── Call 3：反向生成（相似度由 BGE-M3 余弦距离计算，不依赖 LLM 打分） ────
_RELEVANCE_PROMPT = """\
基于下面的答案，生成 3 个不同角度的候选问题（即：如果有人问了这个问题，这个答案是合理的回答）。

答案：{answer}

严格输出以下 JSON，不要其他内容：
{{
  "questions": ["候选问题1", "候选问题2", "候选问题3"]
}}"""


# ══════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════


@dataclass
class JudgeRow:
    query_id: int
    query: str
    generated_answer: str

    # ── Faithfulness（0-1 浮点，原子级） ────────────────────────────
    is_refusal: bool
    faithfulness_score: float  # supported_claims / total_claims
    faithfulness_claims_total: int
    faithfulness_claims_supported: int
    # JSON 字符串：[{"claim": "...", "supported": true/false}, ...]
    faithfulness_details: str

    # ── Answer Relevance（0-1 浮点，反向生成 + BGE-M3 余弦相似度） ───
    answer_relevance_score: float  # mean cosine_sim(query, generated_q)
    # JSON 字符串：[{"q": "...", "sim": float}, ...]
    answer_relevance_questions: str

    # ── Context Precision（0-1 浮点，每个 Chunk 是否有用） ───────────
    context_precision_score: float  # useful_chunks / total_chunks
    # JSON 字符串：[{"chunk": "...", "useful": true/false}, ...]
    context_precision_details: str

    @property
    def overall_norm(self) -> float:
        scores = [
            self.faithfulness_score,
            self.answer_relevance_score,
            self.context_precision_score,
        ]
        valid = [s for s in scores if s >= 0]
        return round(sum(valid) / len(valid), 4) if valid else -1.0

    def to_csv_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "generated_answer": self.generated_answer,
            "is_refusal": self.is_refusal,
            # Faithfulness
            "faithfulness_score": round(self.faithfulness_score, 4)
            if self.faithfulness_score >= 0
            else "",
            "faithfulness_claims_total": self.faithfulness_claims_total,
            "faithfulness_claims_supported": self.faithfulness_claims_supported,
            "faithfulness_details": self.faithfulness_details,
            # Answer Relevance
            "answer_relevance_score": round(self.answer_relevance_score, 4)
            if self.answer_relevance_score >= 0
            else "",
            "answer_relevance_questions": self.answer_relevance_questions,
            # Context Precision
            "context_precision_score": round(self.context_precision_score, 4)
            if self.context_precision_score >= 0
            else "",
            "context_precision_details": self.context_precision_details,
            # Overall
            "overall_norm": round(self.overall_norm, 4)
            if self.overall_norm >= 0
            else "",
        }


# ══════════════════════════════════════════════════════════════════════
# Gemini 调用工具
# ══════════════════════════════════════════════════════════════════════


def _call_text(model, prompt: str, retries: int = 2) -> str:
    """调用 Gemini，返回纯文本。失败返回空字符串。"""
    for attempt in range(retries + 1):
        try:
            resp = model.generate_content(prompt, generation_config=_GEN_CFG)
            return resp.text.strip()
        except Exception as e:
            if attempt < retries:
                time.sleep(1.5)
            else:
                print(f"    ⚠ Gemini 文本调用失败: {e}")
    return ""


def _call_json(model, prompt: str, retries: int = 2) -> dict | None:
    """调用 Gemini，返回解析后的 dict。失败返回 None。"""
    for attempt in range(retries + 1):
        try:
            resp = model.generate_content(prompt, generation_config=_GEN_CFG)
            text = resp.text.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:-1])
            return json.loads(text)
        except Exception as e:
            if attempt < retries:
                time.sleep(1.5)
            else:
                print(f"    ⚠ Gemini JSON 调用失败: {e}")
    return None


# ══════════════════════════════════════════════════════════════════════
# 三步评分逻辑
# ══════════════════════════════════════════════════════════════════════


def score_faithfulness(
    model, context: str, answer: str
) -> tuple[bool, float, int, int, str]:
    """
    Call 1 + Call 2：原子分解 → 批量验证。
    返回 (is_refusal, score, total, supported, details_json)。
    score = -1 表示调用失败。
    """
    # ── Call 1：分解 ──────────────────────────────────────────────────
    data = _call_json(model, _DECOMPOSE_PROMPT.format(answer=answer))

    if not data:
        # LLM 调用失败或未返回合法 JSON
        return False, -1.0, 0, 0, "[]"

    is_refusal = bool(data.get("is_refusal", False))
    claims = data.get("claims", [])

    if not isinstance(claims, list):
        claims = []

    claims = [str(c).strip() for c in claims if str(c).strip()]

    if is_refusal or not claims:
        # 模型正确拒绝作答，或者确实没有任何实质性声明 = 无幻觉
        return is_refusal, 1.0, 0, 0, "[]"

    # ── Call 2：批量验证（一次调用） ──────────────────────────────────
    claims_numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    verify_raw = _call_text(
        model,
        _VERIFY_PROMPT.format(context=context[:3000], claims_numbered=claims_numbered),
    )

    if not verify_raw:
        return is_refusal, -1.0, len(claims), -1, "[]"

    # 解析 YES/NO 列表
    lines = [l.strip().upper() for l in verify_raw.splitlines() if l.strip()]
    # 容错：多一行或少一行时对齐到 claims 长度
    verdicts = (lines + ["NO"] * len(claims))[: len(claims)]

    details = []
    supported = 0
    for claim, verdict in zip(claims, verdicts):
        is_sup = verdict.startswith("Y")
        if is_sup:
            supported += 1
        details.append({"claim": claim, "supported": is_sup})

    score = supported / len(claims) if claims else 1.0
    return (
        is_refusal,
        round(score, 4),
        len(claims),
        supported,
        json.dumps(details, ensure_ascii=False),
    )


def score_context_precision(
    model, query: str, retrieved_context: str
) -> tuple[float, str]:
    """
    Call 4：对每个召回 Chunk 判断是否有助于回答问题。
    返回 (score, details_json)。
    score = useful_chunks / total_chunks，范围 0-1。
    score = -1 表示调用失败。
    """
    # 按 \n\n[N] 分割还原各 Chunk（generate_answers._format_context 的格式）
    chunks = re.split(r"\n\n(?=\[)", retrieved_context.strip())
    chunks = [c.strip() for c in chunks if c.strip()]
    if not chunks:
        return -1.0, "[]"

    chunks_numbered = "\n\n".join(
        f"Chunk {i + 1}:\n{c[:300]}" for i, c in enumerate(chunks)
    )
    raw = _call_text(
        model,
        _CONTEXT_PRECISION_PROMPT.format(query=query, chunks_numbered=chunks_numbered),
    )
    if not raw:
        return -1.0, "[]"

    lines = [l.strip().upper() for l in raw.splitlines() if l.strip()]
    verdicts = (lines + ["NO"] * len(chunks))[: len(chunks)]

    details = []
    useful = 0
    for chunk, verdict in zip(chunks, verdicts):
        is_useful = verdict.startswith("Y")
        if is_useful:
            useful += 1
        # 只存前 100 字符供 debug，避免 CSV 过大
        details.append({"chunk": chunk[:100], "useful": is_useful})

    score = round(useful / len(chunks), 4)
    return score, json.dumps(details, ensure_ascii=False)


def score_answer_relevance(
    gemini_model, embed_model, query: str, answer: str
) -> tuple[float, str]:
    """
    Call 3：反向生成候选问题（Gemini） + 余弦相似度计算（BGE-M3）。
    返回 (score, questions_json)。
    score = mean cosine_similarity(original_query, generated_q)，范围 0-1。
    score = -1 表示调用失败。
    """
    import numpy as np

    result = _call_json(gemini_model, _RELEVANCE_PROMPT.format(answer=answer))
    if result is None:
        return -1.0, "[]"

    questions = result.get("questions", [])
    if not questions or not isinstance(questions, list):
        return -1.0, "[]"
    # 兼容 Gemini 偶尔返回 [{"q": "..."}, ...] 的旧格式
    questions = [q["q"] if isinstance(q, dict) else str(q) for q in questions if q]
    if not questions:
        return -1.0, "[]"

    # 一次批量 encode：[query] + questions
    texts = [query] + questions
    vecs = embed_model.encode(
        texts, batch_size=len(texts), max_length=512, return_dense=True
    )["dense_vecs"]  # shape (n, 1024)，已 L2 归一化

    query_vec = vecs[0]
    details = []
    sims = []
    for q_text, q_vec in zip(questions, vecs[1:]):
        sim = float(np.dot(query_vec, q_vec))
        sim = max(0.0, min(1.0, sim))  # 余弦相似度理论上在 [-1,1]，截断到 [0,1]
        sims.append(sim)
        details.append({"q": q_text, "sim": round(sim, 4)})

    score = round(sum(sims) / len(sims), 4) if sims else -1.0
    return score, json.dumps(details, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════
# 指标汇总
# ══════════════════════════════════════════════════════════════════════


def compute_summary(rows: list[JudgeRow]) -> dict:
    valid_f = [r for r in rows if r.faithfulness_score >= 0]
    valid_ar = [r for r in rows if r.answer_relevance_score >= 0]
    valid_cp = [r for r in rows if r.context_precision_score >= 0]
    valid = [r for r in rows if r.overall_norm >= 0]
    n = len(rows)

    def avg(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def pct_ge(vals, threshold):
        return round(sum(v >= threshold for v in vals) / len(vals), 4) if vals else 0.0

    f_scores = [r.faithfulness_score for r in valid_f]
    ar_scores = [r.answer_relevance_score for r in valid_ar]
    cp_scores = [r.context_precision_score for r in valid_cp]
    ov_scores = [r.overall_norm for r in valid]

    # 拒答率
    refusals = sum(1 for r in rows if r.is_refusal)
    refusal_rate = round(refusals / n, 4) if n else 0.0

    return {
        "n_total": n,
        "n_faith_valid": len(valid_f),
        "n_relevance_valid": len(valid_ar),
        "n_ctx_precision_valid": len(valid_cp),
        "faithfulness_avg": avg(f_scores),
        "faithfulness_ge067": pct_ge(f_scores, 0.67),
        "answer_relevance_avg": avg(ar_scores),
        "answer_relevance_ge067": pct_ge(ar_scores, 0.67),
        "context_precision_avg": avg(cp_scores),
        "context_precision_ge067": pct_ge(cp_scores, 0.67),
        "overall_avg": avg(ov_scores),
        "avg_claims_per_answer": avg([r.faithfulness_claims_total for r in valid_f]),
        "refusal_rate": refusal_rate,
    }


def _select_spot_check(rows: list[JudgeRow], n: int) -> list[JudgeRow]:
    """
    选取最值得人工抽检的条目：
    - 两项分数差距大（Gemini 自身不一致，Criteria 可能有偏差）
    - 整体分偏低但不是明显失败（模糊地带，最难判断）
    """
    valid = [r for r in rows if r.overall_norm >= 0]
    scored = sorted(
        valid,
        key=lambda r: (
            abs(r.faithfulness_score - r.answer_relevance_score) * 5
            + max(0.0, 0.7 - r.overall_norm)
        ),
        reverse=True,
    )
    return scored[:n]


def _print_summary(m: dict) -> None:
    SEP = "═" * 62
    print(f"\n{SEP}")
    print(f"  生成质量评测摘要（原子级）")
    print(
        f"  n={m['n_total']}  "
        f"Faithfulness有效={m['n_faith_valid']}  "
        f"Relevance有效={m['n_relevance_valid']}  "
        f"CtxPrec有效={m.get('n_ctx_precision_valid', 0)}"
    )
    print(f"{'─' * 62}")
    print(
        f"  Faithfulness    avg={m['faithfulness_avg']:.3f}  "
        f"≥0.67达标率={m['faithfulness_ge067']:.1%}  "
        f"avg声明数={m['avg_claims_per_answer']:.1f}"
    )
    print(
        f"  AnswerRelevance avg={m['answer_relevance_avg']:.3f}  "
        f"≥0.67达标率={m['answer_relevance_ge067']:.1%}"
    )
    print(
        f"  CtxPrecision    avg={m.get('context_precision_avg', 0):.3f}  "
        f"≥0.67达标率={m.get('context_precision_ge067', 0):.1%}"
    )
    print(f"  Overall         avg={m['overall_avg']:.3f}")
    print(f"  拒答率          {m.get('refusal_rate', 0):.1%}")
    print(f"{SEP}\n")


def _print_spot_check(cases: list[JudgeRow]) -> None:
    if not cases:
        return
    print("─" * 62)
    print(f"  ◆ 人工抽检候选（{len(cases)} 条）")
    print("─" * 62)
    for r in cases:
        print(
            f"\n  [qid={r.query_id}]  "
            f"Faith={r.faithfulness_score:.2f}({r.faithfulness_claims_supported}/{r.faithfulness_claims_total})  "
            f"Rel={r.answer_relevance_score:.2f}"
        )
        print(f"  Q: {r.query[:70]}")
        print(f"  A: {r.generated_answer[:100]}...")
        # 打印未被支撑的原子声明（最直接的幻觉证据）
        try:
            details = json.loads(r.faithfulness_details)
            unsupported = [d["claim"] for d in details if not d.get("supported")]
            if unsupported:
                print(f"  未被支撑的声明：")
                for c in unsupported[:3]:
                    print(f"    ✗ {c}")
        except Exception:
            pass
        # 打印生成的候选问题
        try:
            questions = json.loads(r.answer_relevance_questions)
            if questions:
                print(f"  生成的候选问题：")
                for q in questions:
                    print(f"    sim={q.get('sim', 0)}  {q.get('q', '')[:60]}")
        except Exception:
            pass
    print()
    print("  ⚑ 若 Gemini 的原子声明拆分或 YES/NO 判断与人工判断不一致，")
    print("    修改本文件中对应的 Prompt 后重跑（无需重跑 generate_answers.py）。")
    print()


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════


def _load_existing_judge(path: str) -> tuple[dict[int, dict], list[str]]:
    """读取已有的 eval_judge_results.csv，返回 (rows_by_qid, fieldnames)。"""
    if not os.path.exists(path):
        return {}, []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return {int(r["query_id"]): r for r in rows}, list(fieldnames)


def _judge_row_failed(row: dict) -> bool:
    """判断已有打分结果是否属于失败（任一分数为空或 -1）。"""

    def _bad(key: str) -> bool:
        v = row.get(key, "")
        if v in ("", None):
            return True
        try:
            return float(v) < 0
        except (ValueError, TypeError):
            return True

    return (
        _bad("faithfulness_score")
        or _bad("answer_relevance_score")
        or _bad("context_precision_score")
    )


def main():
    parser = argparse.ArgumentParser(description="原子级 LLM-as-Judge（阶段二 Step 2）")
    parser.add_argument("--answers", default="eval/eval_answers.csv")
    parser.add_argument("--results", default="eval/eval_judge_results.csv")
    parser.add_argument("--summary", default="eval/eval_judge_summary.json")
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini 裁判模型（默认 gemini-2.5-flash）",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="每条记录处理完后的等待秒数（flash 限速更宽松）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="读取已有结果，只补跑 faithfulness 或 relevance 打分失败的条目",
    )
    parser.add_argument("--spot-check", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.answers):
        print(
            f"❌ 找不到 {args.answers}，请先运行 generate_answers.py", file=sys.stderr
        )
        sys.exit(1)

    with open(args.answers, encoding="utf-8") as f:
        answer_rows = list(csv.DictReader(f))
    answer_rows = [r for r in answer_rows if r.get("generated_answer", "").strip()]
    if args.limit:
        answer_rows = answer_rows[: args.limit]

    # ── Resume 模式：只补跑失败条目 ──────────────────────────────────
    existing_judge: dict[int, dict] = {}
    existing_judge_fields: list[str] = []

    if args.resume:
        existing_judge, existing_judge_fields = _load_existing_judge(args.results)
        if not existing_judge:
            print(
                f"⚠ --resume 指定但未找到 {args.results}，将全量运行", file=sys.stderr
            )
        else:
            all_qids = {int(r["query_id"]) for r in answer_rows}
            failed_qids = {
                qid for qid, r in existing_judge.items() if _judge_row_failed(r)
            }
            missing_qids = all_qids - set(existing_judge.keys())
            retry_qids = failed_qids | missing_qids
            answer_rows = [r for r in answer_rows if int(r["query_id"]) in retry_qids]
            print(
                f"[judge_answers] resume 模式：已有 {len(existing_judge)} 条，"
                f"失败 {len(failed_qids)} 条，未处理 {len(missing_qids)} 条，"
                f"本次补跑 {len(answer_rows)} 条"
            )
            if not answer_rows:
                print("✅ 所有条目已完成，无需补跑")
                return
    else:
        print(f"[judge_answers] {len(answer_rows)} 条  模型: {args.model}")

    print(f"  每条 4 次 Gemini 调用（分解 + 验证 + 反向生成 + 上下文精准率）")

    try:
        import google.generativeai as genai

        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        judge_model = genai.GenerativeModel(args.model)
    except ImportError:
        print("❌ pip install -U google-generativeai", file=sys.stderr)
        sys.exit(1)

    # BGE-M3 用于 Answer Relevance 余弦相似度计算
    try:
        from FlagEmbedding import BGEM3FlagModel
        from eval.retrieval_core import EMBEDDING_MODEL_PATH

        print("[judge_answers] 加载 BGE-M3 embedding 模型...")
        embed_model = BGEM3FlagModel(EMBEDDING_MODEL_PATH, use_fp16=True)
        print("[judge_answers] BGE-M3 加载完成")
    except Exception as e:
        print(f"❌ BGE-M3 加载失败: {e}", file=sys.stderr)
        sys.exit(1)

    # ── 增量写入：每条处理完立即落盘 ────────────────────────────────
    # fieldnames 固定来自 JudgeRow.to_csv_dict()，与 resume/新建模式一致
    _FIELDNAMES = [
        "query_id",
        "query",
        "generated_answer",
        "is_refusal",
        "faithfulness_score",
        "faithfulness_claims_total",
        "faithfulness_claims_supported",
        "faithfulness_details",
        "answer_relevance_score",
        "answer_relevance_questions",
        "context_precision_score",
        "context_precision_details",
        "overall_norm",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)

    # 打开文件：写入 header + 已有完成行（resume 模式），之后追加新行
    with open(args.results, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=_FIELDNAMES)
        writer.writeheader()

        # resume 模式：先把已经成功的旧行写入，保留在文件里
        completed_qids: set[int] = set()
        if args.resume and existing_judge:
            for qid in sorted(existing_judge):
                if not _judge_row_failed(existing_judge[qid]):
                    writer.writerow(existing_judge[qid])
                    completed_qids.add(qid)
            out_f.flush()

        judge_rows: list[JudgeRow] = []

        for i, row in enumerate(answer_rows, 1):
            query_id = int(row["query_id"])
            query = row["query"]
            context = row["retrieved_context"]
            answer = row["generated_answer"]

            print(f"  [{i:3d}/{len(answer_rows)}] qid={query_id:3d}  {query[:40]}")

            # Call 1 + 2：Faithfulness 原子分解 + 批量验证
            is_refusal, f_score, f_total, f_sup, f_details = score_faithfulness(
                judge_model, context, answer
            )
            print(
                f"           Faith={f_score:.2f}  ({f_sup}/{f_total} 声明有支撑) Refusal={is_refusal}"
            )

            # Call 3：Answer Relevance 反向生成
            ar_score, ar_questions = score_answer_relevance(
                judge_model, embed_model, query, answer
            )
            print(f"           Rel={ar_score:.2f}")

            # Call 4：Context Precision
            cp_score, cp_details = score_context_precision(judge_model, query, context)
            print(f"           CtxPrec={cp_score:.2f}")

            jr = JudgeRow(
                query_id=query_id,
                query=query,
                generated_answer=answer,
                is_refusal=is_refusal,
                faithfulness_score=f_score,
                faithfulness_claims_total=f_total,
                faithfulness_claims_supported=f_sup,
                faithfulness_details=f_details,
                answer_relevance_score=ar_score,
                answer_relevance_questions=ar_questions,
                context_precision_score=cp_score,
                context_precision_details=cp_details,
            )
            judge_rows.append(jr)

            # 立即写入并刷新，中断后不丢失已完成的行
            writer.writerow(jr.to_csv_dict())
            out_f.flush()

            if i < len(answer_rows) and args.delay > 0:
                time.sleep(args.delay)

    # resume 模式：把旧成功行也加入内存列表，用于汇总统计
    if args.resume and existing_judge:
        for qid in sorted(completed_qids):
            r = existing_judge[qid]
            try:
                judge_rows.append(
                    JudgeRow(
                        query_id=int(r["query_id"]),
                        query=r["query"],
                        generated_answer=r.get("generated_answer", ""),
                        is_refusal=str(r.get("is_refusal", "")).lower() == "true",
                        faithfulness_score=float(r.get("faithfulness_score") or -1),
                        faithfulness_claims_total=int(
                            r.get("faithfulness_claims_total") or 0
                        ),
                        faithfulness_claims_supported=int(
                            r.get("faithfulness_claims_supported") or 0
                        ),
                        faithfulness_details=r.get("faithfulness_details", "[]"),
                        answer_relevance_score=float(
                            r.get("answer_relevance_score") or -1
                        ),
                        answer_relevance_questions=r.get(
                            "answer_relevance_questions", "[]"
                        ),
                        context_precision_score=float(
                            r.get("context_precision_score") or -1
                        ),
                        context_precision_details=r.get(
                            "context_precision_details", "[]"
                        ),
                    )
                )
            except (ValueError, TypeError):
                pass
        print(
            f"\n✅ resume 完成：{len(answer_rows)} 条补跑，"
            f"合并后共 {len(judge_rows)} 条 → {args.results}"
        )

    summary = compute_summary(judge_rows)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _print_summary(summary)
    spot = _select_spot_check(judge_rows, args.spot_check)
    _print_spot_check(spot)

    print(f"  逐条结果 → {args.results}")
    print(f"  汇总指标 → {args.summary}")


if __name__ == "__main__":
    main()
