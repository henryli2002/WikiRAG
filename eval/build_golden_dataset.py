"""
eval/build_golden_dataset.py
============================
黄金测试集构建：随机主题 query 生成。

调用 Gemini 生成 N 个覆盖广泛维基百科主题的中文知识问答问题。
query 不绑定特定 chunk，用于测试 RAG 全链路端到端质量。

输出 golden_dataset.csv 列：
  query_id   唯一编号（1-based）
  query      问题文本
  topic      主题分类（仅供参考）

用法：
  python eval/build_golden_dataset.py
  python eval/build_golden_dataset.py --n 50 --batch-size 15
"""

import os
import sys
import csv
import json
import time
import argparse

from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


# ══════════════════════════════════════════════════════════════════════
# Prompt
# ══════════════════════════════════════════════════════════════════════

_TOPIC_QUERY_PROMPT = """\
你是一个知识问答测试集构建专家。请生成 {n} 个多样化的中文知识问答问题。

要求：
1. 问题必须涵盖广泛的维基百科主题，包括但不限于：
   历史事件、科学技术、地理地貌、文化艺术、人物传记、
   军事政治、体育赛事、建筑工程、自然生态、经济金融
2. 每个问题应是一个自然的用户提问，像真实用户在搜索框里会问的那样
3. 问题难度要有层次：
   - 简单事实题（"XX是什么？"、"XX在哪里？"）
   - 因果推理题（"为什么XX会导致YY？"）
   - 对比分析题（"XX和YY有什么区别？"）
   - 条件限定题（"在XX背景下，YY是如何发展的？"）
4. 问题之间不能重复或过于相似，每个问题聚焦不同的知识领域
5. 每个问题控制在一句话以内
6. 问题必须是中文维基百科大概率能回答的

{exclude_instruction}

严格输出以下 JSON，不要包含任何其他内容或 Markdown 标记：
{{
  "queries": [
    {{"query": "问题文本", "topic": "主题分类"}},
    {{"query": "问题文本", "topic": "主题分类"}}
  ]
}}"""


# ══════════════════════════════════════════════════════════════════════
# LLM 调用
# ══════════════════════════════════════════════════════════════════════

def _call_gemini_json(model, prompt: str, retries: int = 2) -> dict | None:
    for attempt in range(retries + 1):
        try:
            resp = model.generate_content(prompt)
            text = resp.text.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:-1])
            return json.loads(text)
        except Exception as e:
            if attempt < retries:
                print(f"    重试 ({attempt+1}/{retries}): {e}")
                time.sleep(2)
            else:
                print(f"    调用失败: {e}")
    return None


def generate_queries(model, n: int, existing_queries: list[str], delay: float) -> list[dict]:
    """
    分批生成 query，每批将已有 query 作为排除列表传入，避免重复。
    """
    results = []

    exclude_instruction = ""
    if existing_queries:
        sample = existing_queries[:20]  # 只传前 20 条避免 prompt 过长
        exclude_instruction = (
            "请避免与以下已有问题重复或相似：\n"
            + "\n".join(f"- {q}" for q in sample)
        )

    prompt = _TOPIC_QUERY_PROMPT.format(
        n=n,
        exclude_instruction=exclude_instruction,
    )

    data = _call_gemini_json(model, prompt)
    if not data:
        return []

    queries = data.get("queries", [])
    for q in queries:
        if isinstance(q, dict) and q.get("query", "").strip():
            results.append({
                "query": q["query"].strip(),
                "topic": q.get("topic", "").strip(),
            })

    if delay > 0:
        time.sleep(delay)

    return results


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="随机主题 query 生成（黄金测试集）")
    parser.add_argument("--n",          type=int,   default=50,
                        help="目标 query 数量（默认 50）")
    parser.add_argument("--batch-size", type=int,   default=15,
                        help="每批生成的 query 数量（默认 15）")
    parser.add_argument("--output",     default="eval/golden_dataset.csv")
    parser.add_argument("--model",      default="gemini-2.5-flash",
                        help="Gemini 模型 ID")
    parser.add_argument("--delay",      type=float, default=1.0,
                        help="每批 API 调用后的等待秒数")
    args = parser.parse_args()

    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(args.model)
    except ImportError:
        print("请先安装 Gemini SDK: pip install -U google-generativeai", file=sys.stderr)
        sys.exit(1)

    print(f"[build_golden_dataset] 目标 {args.n} 条，每批 {args.batch_size} 条")

    all_queries: list[dict] = []

    while len(all_queries) < args.n:
        remaining = args.n - len(all_queries)
        batch_n = min(args.batch_size, remaining)

        print(f"  已有 {len(all_queries)} 条，请求 {batch_n} 条...")
        existing_texts = [q["query"] for q in all_queries]
        batch = generate_queries(model, batch_n, existing_texts, args.delay)

        if not batch:
            print("  本批生成失败，重试...")
            time.sleep(2)
            continue

        # 去重
        existing_set = set(q["query"] for q in all_queries)
        for q in batch:
            if q["query"] not in existing_set:
                all_queries.append(q)
                existing_set.add(q["query"])

        print(f"  -> 累计 {len(all_queries)} 条")

    all_queries = all_queries[:args.n]

    # ── 写 CSV ────────────────────────────────────────────────────────
    rows = []
    for i, q in enumerate(all_queries, 1):
        rows.append({
            "query_id": i,
            "query":    q["query"],
            "topic":    q["topic"],
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "query", "topic"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} 条写入 {args.output}")

    # 统计主题分布
    topics: dict[str, int] = {}
    for r in rows:
        t = r["topic"] or "未分类"
        topics[t] = topics.get(t, 0) + 1
    print("\n主题分布：")
    for t, c in sorted(topics.items(), key=lambda x: -x[1]):
        print(f"  {t:<20s}: {c} 条")

    print(f"\n下一步：bash eval/run_experiments.sh --topk 1 3 5")


if __name__ == "__main__":
    main()
