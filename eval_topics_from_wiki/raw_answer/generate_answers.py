"""
eval/raw_answer/generate_answers.py
====================================
对照实验：纯 LLM 裸答（不提供任何检索 Context）。

与 eval/generate_answers.py 的唯一区别：
  调用 LLM 时 **不传入 retrieved_context**，让模型完全依靠自身参数知识作答。
  但仍然执行检索 pipeline 获取 top-5 chunks，以便后续 judge 阶段
  计算 Context Precision / Faithfulness 等指标，实现与 RAG 版本的公平对比。

目的：
  验证 RAG 是否真正提升了回答质量，还是模型本身就能答对。

输出 eval/raw_answer/eval_answers.csv，列定义与 eval/eval_answers.csv 一致。

用法：
  python eval/raw_answer/generate_answers.py
  python eval/raw_answer/generate_answers.py --limit 20
  python eval/raw_answer/generate_answers.py --resume
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from eval_random_topics.retrieval_core import RetrievalCore

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ─── llama.cpp 配置（与 eval/generate_answers.py 一致） ─────────────
LLAMACPP_BASE_URL = "http://msi.tailb813aa.ts.net:8080/v1"
LLAMACPP_API_KEY  = "sk-no-key-required"
LLAMACPP_MODEL    = "local-model"

_LLAMACPP_EXTRA = {
    "top_k":          40,
    "min_p":          0.05,
    "repeat_penalty": 1.1,
}

# ─── 裸答 Prompt（无 Context） ───────────────────────────────────────
_SYSTEM_PROMPT = """\
你是一个知识问答助手。请根据你自身的知识，简洁地回答用户的问题。

规则：
1. 如果你不确定答案，直接回答"我不确定该问题的答案"
2. 答案控制在 3 句话以内，简洁清晰"""

_USER_TEMPLATE = """\
问题：{query}

请回答问题。"""


# ─── 工具函数（复用 eval/generate_answers.py 的逻辑） ────────────────

def _hit_at_k(final_docs: list[dict], chunk_id: int, k: int) -> bool:
    return any(doc["id"] == chunk_id for doc in final_docs[:k])


def _format_context(docs: list[dict], max_chars_per_doc: int = 400) -> tuple[str, list[int]]:
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
    ttft_ms:          float
    gen_ms:           float
    prompt_tokens:    int
    completion_tokens:int
    total_tokens:     int


def generate_answer_raw(llm: OpenAI, query: str) -> tuple[str, GenStats]:
    """
    裸答模式：仅传入问题，不传入任何检索上下文。
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _USER_TEMPLATE.format(query=query)},
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
    if not os.path.exists(args.golden):
        print(f"找不到 {args.golden}", file=sys.stderr)
        sys.exit(1)

    with open(args.golden, encoding="utf-8") as f:
        golden = list(csv.DictReader(f))

    for row in golden:
        row["_query"] = (row.get("human_query") or row.get("generated_query", "")).strip()
    golden = [r for r in golden if r["_query"]]

    if args.limit:
        golden = golden[: args.limit]

    # ── Resume 模式 ──────────────────────────────────────────────────
    existing_rows: dict[int, dict] = {}
    existing_fieldnames: list[str] = []

    if args.resume:
        existing_rows, existing_fieldnames = _load_existing(args.output)
        if not existing_rows:
            print(f"--resume 指定但未找到 {args.output}，将全量运行", file=sys.stderr)
        else:
            all_qids     = {int(g["query_id"]) for g in golden}
            failed_qids  = {qid for qid, r in existing_rows.items() if _is_failed(r)}
            missing_qids = all_qids - set(existing_rows.keys())
            retry_qids   = failed_qids | missing_qids
            golden = [g for g in golden if int(g["query_id"]) in retry_qids]
            print(f"[raw_answer] resume 模式：已有 {len(existing_rows)} 条，"
                  f"失败 {len(failed_qids)} 条，未处理 {len(missing_qids)} 条，"
                  f"本次补跑 {len(golden)} 条")
            if not golden:
                print("所有条目已完成，无需补跑")
                return
    else:
        print(f"[raw_answer] {len(golden)} 条查询（纯 LLM 裸答，无 RAG Context）")

    # ── 初始化 RetrievalCore（仅用于获取 top-5 chunks 供 judge 使用） ─
    core = RetrievalCore(verbose=False)
    core.load_models(warmup=True)
    await core.connect_db()

    # ── 初始化 llama.cpp 客户端 ───────────────────────────────────────
    llm = OpenAI(base_url=LLAMACPP_BASE_URL, api_key=LLAMACPP_API_KEY)
    print(f"[raw_answer] llama.cpp @ {LLAMACPP_BASE_URL}")

    rows = []
    failed = 0

    try:
        for i, g in enumerate(golden, 1):
            query_id = int(g["query_id"])
            chunk_id = int(g["chunk_id"])
            query    = g["_query"]
            adv_type = g.get("adversarial_type", "").strip()

            print(f"  [{i:3d}/{len(golden)}] qid={query_id:3d}  q={query[:45]}")

            # 检索（仅为了获取 chunks 供 judge 计算指标）
            pr = await core.run_pipeline(query)
            t  = pr.timings
            context, chunk_ids = _format_context(pr.final)
            hit5 = _hit_at_k(pr.final, chunk_id, 5)

            # 裸答生成（不传入 context）
            try:
                answer, stats = generate_answer_raw(llm, query)
                print(f"           -> {answer[:50]}...  "
                      f"ttft={stats.ttft_ms:.0f}ms  gen={stats.gen_ms:.0f}ms  "
                      f"tok={stats.prompt_tokens}+{stats.completion_tokens}")
            except Exception as e:
                print(f"           -> 生成失败: {e}")
                answer = ""
                stats  = GenStats(ttft_ms=-1, gen_ms=-1, prompt_tokens=-1,
                                  completion_tokens=-1, total_tokens=-1)
                failed += 1

            rows.append({
                "query_id":            query_id,
                "chunk_id":            chunk_id,
                "query":               query,
                "adversarial_type":    adv_type,
                "hit_at_5":            str(hit5),
                "n_chunks":            len(chunk_ids),
                "context_chars":       len(context),
                "retrieved_chunk_ids": json.dumps(chunk_ids, ensure_ascii=False),
                "retrieved_context":   context,
                "generated_answer":    answer,
                # ── 检索各阶段延迟 ──
                "embed_ms":            round(t.embed_ms,  1),
                "vector_ms":           round(t.vector_ms, 1),
                "bm25_ms":             round(t.bm25_ms,   1),
                "rerank_ms":           round(t.rerank_ms, 1),
                "mmr_ms":              round(t.mmr_ms,    1),
                "retrieval_ms":        round(t.total_ms,  1),
                # ── 生成延迟（不含检索，因为裸答不依赖检索） ──
                "ttft_ms":             stats.ttft_ms,
                "gen_ms":              stats.gen_ms,
                "e2e_ms":              stats.gen_ms,  # 裸答模式下 e2e = gen
                "prompt_tokens":       stats.prompt_tokens,
                "completion_tokens":   stats.completion_tokens,
                "total_tokens":        stats.total_tokens,
            })

    finally:
        await core.close()

    # ── 写 CSV ───────────────────────────────────────────────────────
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
        print(f"\nresume 完成：{len(rows)} 条补跑，合并后共 {len(merged)} 条 -> {args.output}")
        rows = merged
    else:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n{len(rows)} 条写入 {args.output}，生成失败 {failed} 条")
    print(f"\n下一步：python eval/raw_answer/judge_answers.py")


def main():
    parser = argparse.ArgumentParser(
        description="纯 LLM 裸答（对照实验，不提供 RAG Context）"
    )
    parser.add_argument("--golden",  default="eval/golden_dataset.csv")
    parser.add_argument("--output",  default="eval/raw_answer/eval_answers.csv")
    parser.add_argument("--resume",  action="store_true")
    parser.add_argument("--limit",   type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
