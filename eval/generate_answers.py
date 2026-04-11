"""
eval/generate_answers.py
========================
阶段二第一步：检索 + 生成。

流程：
  1. 读取 golden_dataset.csv（human_query 优先）
  2. 通过 RetrievalCore 检索 top-K chunks（直接调用模型，不走 HTTP）
  3. 将检索结果组装为 Context，调用 llama.cpp（Qwen 9B）生成答案
  4. 写入 eval_answers.csv，供 judge_answers.py 打分

刻意选用 Qwen 9B 而非强模型：
  小模型更依赖 Context，不会凭自身参数知识"绕过"检索直接答对，
  从而让 Faithfulness / Answer Relevance 的测量更有区分度。

输出 eval_answers.csv 列：
  query_id            唯一编号
  chunk_id            ground truth chunk（用于与 eval_results.csv 对照）
  query               实际使用的问题文本
  adversarial_type    对抗类型（""=正常样本）
  retrieval_hit       检索是否命中 (基于当前 topk)（直接由本次 pipeline 结果计算，无需外部文件）
  retrieved_chunk_ids 实际召回的 chunk id（JSON 数组）
  retrieved_context   送入 LLM 的完整 Context 文本
  generated_answer    Qwen 9B 生成的答案

用法：
  python eval/generate_answers.py
  python eval/generate_answers.py --golden eval/golden_dataset.csv --limit 20
  python eval/generate_answers.py --resume   # 读取已有输出，只补跑失败条目
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
from eval.retrieval_core import RetrievalCore

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ─── llama.cpp 配置 ──────────────────────────────────────────────────
LLAMACPP_BASE_URL = "http://msi.tailb813aa.ts.net:8080/v1"
LLAMACPP_API_KEY  = "sk-no-key-required"
LLAMACPP_MODEL    = "local-model"

# llama.cpp 特有采样参数（通过 extra_body 传入）
_LLAMACPP_EXTRA = {
    "top_k":          40,
    "min_p":          0.05,
    "repeat_penalty": 1.1,
}

# ─── 生成 Prompt ─────────────────────────────────────────────────────
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

def _hit_at_k(final_docs: list[dict], chunk_id: int, k: int) -> bool:
    """检查目标 chunk 是否出现在 final 结果的前 k 位。"""
    return any(doc["id"] == chunk_id for doc in final_docs[:k])


def _format_context(docs: list[dict], max_chars_per_doc: int = 400) -> tuple[str, list[int]]:
    """
    将召回文档拼成 Context 字符串。
    每条 chunk 限 max_chars_per_doc 字符，避免超出模型上下文窗口。
    返回 (context_text, chunk_id_list)。
    """
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
    """llama.cpp 单次流式推理的延迟与 token 统计（不含 Gemini judge）。"""
    ttft_ms:          float  # 首字延迟：从发出请求到收到第一个有内容 token 的时间
    gen_ms:           float  # 完整生成墙钟时间（含 TTFT）
    prompt_tokens:    int    # 输入 token 数（system + context + query）
    completion_tokens:int    # 输出 token 数（生成的答案）
    total_tokens:     int    # 两者之和


def generate_answer(llm: OpenAI, query: str, context: str) -> tuple[str, GenStats]:
    """
    以流式模式调用 llama.cpp，同时捕获首字延迟（TTFT）和完整生成时间。
    extra_body 中传入 stream_options 以在流末尾获取 usage 统计。
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _USER_TEMPLATE.format(
            context=context, query=query
        )},
    ]
    extra = {**_LLAMACPP_EXTRA, "stream_options": {"include_usage": True}}

    t_start  = time.time()
    ttft_ms  = None
    parts:list[str] = []
    usage    = None

    stream = llm.chat.completions.create(  # type: ignore[call-overload]
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
        # 流末尾最后一个 chunk 携带 usage（需 stream_options.include_usage=true）
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
    """
    读取已有的 eval_answers.csv。
    返回 (rows_by_qid, fieldnames)。
    rows_by_qid: query_id -> row dict（保留原始字符串，原样回写）
    """
    if not os.path.exists(path):
        return {}, []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return {int(r["query_id"]): r for r in rows}, list(fieldnames)


def _is_failed(row: dict) -> bool:
    """判断某条已有结果是否属于生成失败（需要重跑）。"""
    try:
        return float(row.get("gen_ms", -1)) < 0 or row.get("generated_answer", "").strip() == ""
    except (ValueError, TypeError):
        return True


async def run(args):
    # ── 读取黄金测试集 ────────────────────────────────────────────────
    if not os.path.exists(args.golden):
        print(f"❌ 找不到 {args.golden}", file=sys.stderr)
        sys.exit(1)

    with open(args.golden, encoding="utf-8") as f:
        golden = list(csv.DictReader(f))

    for row in golden:
        row["_query"] = (row.get("human_query") or row.get("generated_query", "")).strip()
    golden = [r for r in golden if r["_query"]]

    if args.limit:
        golden = golden[: args.limit]

    # ── Resume 模式：只跑失败条目 ────────────────────────────────────
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
            print(f"[generate_answers] resume 模式：已有 {len(existing_rows)} 条，"
                  f"失败 {len(failed_qids)} 条，未处理 {len(missing_qids)} 条，"
                  f"本次补跑 {len(golden)} 条")
            if not golden:
                print("✅ 所有条目已完成，无需补跑")
                return
    else:
        print(f"[generate_answers] {len(golden)} 条查询")

    # ── 初始化 RetrievalCore ──────────────────────────────────────────
    core = RetrievalCore(verbose=False, final_k=args.topk)
    core.load_models(warmup=True)
    await core.connect_db()

    # ── 初始化 llama.cpp 客户端 ───────────────────────────────────────
    llm = OpenAI(base_url=LLAMACPP_BASE_URL, api_key=LLAMACPP_API_KEY)
    print(f"[generate_answers] llama.cpp @ {LLAMACPP_BASE_URL}")

    rows = []
    failed = 0

    try:
        for i, g in enumerate(golden, 1):
            query_id = int(g["query_id"])
            chunk_id = int(g["chunk_id"])
            query    = g["_query"]
            adv_type = g.get("adversarial_type", "").strip()

            print(f"  [{i:3d}/{len(golden)}] qid={query_id:3d}  q={query[:45]}")

            # 检索
            pr = await core.run_pipeline(query)
            t  = pr.timings
            context, chunk_ids = _format_context(pr.final)
            retrieval_hit = _hit_at_k(pr.final, chunk_id, args.topk)

            # 生成
            try:
                answer, stats = generate_answer(llm, query, context)
                e2e_ms = round(t.total_ms + stats.gen_ms, 1)
                print(f"           → {answer[:50]}...  "
                      f"ret={t.total_ms:.0f}ms  ttft={stats.ttft_ms:.0f}ms  "
                      f"gen={stats.gen_ms:.0f}ms  e2e={e2e_ms:.0f}ms  "
                      f"tok={stats.prompt_tokens}+{stats.completion_tokens}")
            except Exception as e:
                print(f"           ⚠ 生成失败: {e}")
                answer = ""
                stats  = GenStats(ttft_ms=-1, gen_ms=-1, prompt_tokens=-1,
                                  completion_tokens=-1, total_tokens=-1)
                e2e_ms = -1
                failed += 1

            rows.append({
                "query_id":            query_id,
                "chunk_id":            chunk_id,
                "query":               query,
                "adversarial_type":    adv_type,
                "retrieval_hit":     str(retrieval_hit),
                "n_chunks":            len(chunk_ids),
                "context_chars":       len(context),
                "retrieved_chunk_ids": json.dumps(chunk_ids, ensure_ascii=False),
                "retrieved_context":   context,
                "generated_answer":    answer,
                # ── 检索各阶段延迟（来自 pr.timings） ──
                "embed_ms":            round(t.embed_ms,  1),
                "vector_ms":           round(t.vector_ms, 1),
                "bm25_ms":             round(t.bm25_ms,   1),
                "rerank_ms":           round(t.rerank_ms, 1),
                "mmr_ms":              round(t.mmr_ms,    1),
                "retrieval_ms":        round(t.total_ms,  1),
                # ── 生成延迟 ──
                "ttft_ms":             stats.ttft_ms,
                "gen_ms":              stats.gen_ms,
                "e2e_ms":              e2e_ms,
                "prompt_tokens":       stats.prompt_tokens,
                "completion_tokens":   stats.completion_tokens,
                "total_tokens":        stats.total_tokens,
            })

    finally:
        await core.close()

    # ── 写 CSV（resume 模式下合并新旧结果） ──────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.resume and existing_rows:
        # 用本次结果覆盖对应 query_id 的旧行，其余保留
        new_by_qid = {r["query_id"]: r for r in rows}
        existing_rows.update(new_by_qid)
        # 按 query_id 排序还原原始顺序
        merged = [existing_rows[qid] for qid in sorted(existing_rows)]
        fieldnames = existing_fieldnames or list(rows[0].keys())
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged)
        print(f"\n✅ resume 完成：{len(rows)} 条补跑，合并后共 {len(merged)} 条 → {args.output}")
        # 汇总统计基于合并后全量数据
        rows = merged
    else:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # ── 生成延迟与 Token 汇总 ─────────────────────────────────────────
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
    print()
    print("  ── 检索各阶段延迟 ─────────────────────────────────────────")
    for label, key in [
        ("embed",     "embed_ms"),
        ("vector",    "vector_ms"),
        ("bm25",      "bm25_ms"),
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
    ptok_vals   = _v("prompt_tokens")
    ctok_vals   = _v("completion_tokens")
    nchunk_vals = _v("n_chunks")
    chars_vals  = _v("context_chars")
    if ptok_vals:
        print(f"  prompt     P50={_pct(ptok_vals,.50):>8.0f} tok  "
              f"P90={_pct(ptok_vals,.90):>8.0f} tok")
    if ctok_vals:
        print(f"  complete   P50={_pct(ctok_vals,.50):>8.0f} tok  "
              f"P90={_pct(ctok_vals,.90):>8.0f} tok")
    if nchunk_vals:
        print(f"  n_chunks   P50={_pct(nchunk_vals,.50):>8.0f}      "
              f"context_chars P50={_pct(chars_vals,.50):>8.0f}")
    print()
    print("  ── FINAL_TOP_K 调参参考 ───────────────────────────────────")
    by_n: dict[int, list] = {}
    for r in valid:
        n_chunks = int(_flt(r, "n_chunks"))
        by_n.setdefault(n_chunks, []).append(r)
    for n in sorted(by_n):
        g = [_flt(r, "gen_ms") for r in by_n[n]]
        p = [_flt(r, "prompt_tokens") for r in by_n[n] if _flt(r, "prompt_tokens") >= 0]
        print(f"  n_chunks={n}: gen_ms P50={_pct(g,.50):>7}  "
              f"prompt_tok P50={_pct(p,.50):>6.0f}  (n={len(g)}条)")
    print()
    print(f"下一步：python eval/judge_answers.py --answers {args.output}")


def main():
    parser = argparse.ArgumentParser(description="检索 + 生成答案（阶段二 Step 1）")
    parser.add_argument("--golden",  default="eval/golden_dataset.csv")
    parser.add_argument("--output",  default="eval/eval_answers.csv")
    parser.add_argument("--resume",  action="store_true",
                        help="读取已有 output，只补跑 gen_ms<0 或 answer 为空的失败条目")
    parser.add_argument("--topk",    type=int, default=5, help="检索传递给生成层的 top-k 数量")
    parser.add_argument("--limit",   type=int, default=None,
                        help="只处理前 N 条（调试用）")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
