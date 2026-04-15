"""
eval/build_golden_dataset.py
============================
黄金测试集全自动构建脚本（实验阶段三步一体化）。

Step 1  从 wiki_documents 随机采样 N 个 Chunk
Step 2  调用 Gemini API 生成 base query（4 种提问视角轮转）
Step 3  对 ~30% 的 Chunk 再次调用 Gemini 进行对抗性改造，
        自动填写 human_query / adversarial_type，无需人工介入

输出 golden_dataset.csv 列：
  query_id          唯一编号（1-based）
  chunk_id          wiki_documents.id
  chunk_title       来源文章标题
  chunk_content     Chunk 原文
  generated_type    base query 所属提问视角
  generated_query   Step 2 生成的原始问题
  human_query       Step 3 改造后的最终问题（未改造时 = generated_query）
  adversarial_type  synonym / missing_info / entity_ambiguity / ""（未改造）

用法：
  python eval/build_golden_dataset.py
  python eval/build_golden_dataset.py --n 200 --adv-ratio 0.3
"""

import os
import csv
import sys
import time
import json
import random
import asyncio
import argparse

import asyncpg
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


# ══════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════

# ── Step 2：基础问题生成（4 种提问视角） ─────────────────────────────

_QUERY_TYPES = [
    {
        "type": "细节提取 (Factoid)",
        "instruction": "针对文本中的某个具体时间、地点、数字或专有名词提问。问题要简短直接。",
    },
    {
        "type": "因果与逻辑推断 (Reasoning)",
        "instruction": "不要问表面的'是什么'。请提问文本中某件事发生的'原因'、'目的'或'导致的影响'。",
    },
    {
        "type": "概念对比或条件限制 (Conditional)",
        "instruction": "在问题中加入限制条件（如：'在XX背景下'，'除了XX之外'），"
                       "或者让模型区分文本中的两个概念。",
    },
    {
        "type": "暗含指代的口语化提问 (Implicit)",
        "instruction": "极度口语化，不要直接出现文本里的专业术语主语，"
                       "用'那个'、'这门技术'等代词指代，让模型必须结合上下文才能看懂提问。",
    },
]

_BASE_PROMPT = """\
你是一个知识问答数据集构建专家。下面是一段 Wiki 百科的文本片段（Chunk），\
请根据这段文字中包含的核心信息，逆向生成「恰好一个」自然的中文提问。

本次生成，你必须采用以下提问视角：
提问类型：{query_type}
具体要求：{instruction}

通用要求：
1. 问题必须能且仅能由这段文字回答，不能依赖段落以外的知识
2. 问题用日常口语表达，不要直接复制原文短语或句子
3. 只输出问题本身，不要加任何前缀、序号或解释

Chunk：
{chunk}

问题："""


# ── Step 3：对抗改造（3 种手法） ─────────────────────────────────────

_ADV_TYPES = [
    {
        "type": "synonym",
        "label": "同义词替换",
        "instruction": "将问题中出现的 1-2 个关键领域术语替换为语义相近但表述不同的词组（同义词或上位词）。"
                       "例如：'计算机视觉' → '图像识别技术'，'深度学习' → '神经网络训练'。"
                       "目的是测试 Embedding 模型能否跨越词汇差异做语义匹配。",
    },
    {
        "type": "missing_info",
        "label": "信息缺失",
        "instruction": "刻意删掉问题中 1 个关键限定语或定语短语，使问题变得更模糊、更口语化。"
                       "例如：'量子纠错码的纠错原理是什么？' → '纠错码是怎么工作的？'。"
                       "目的是测试系统面对模糊意图时的召回冗余度。",
    },
    {
        "type": "entity_ambiguity",
        "label": "实体歧义",
        "instruction": "将问题中的某个具体实体名替换为一个同名或形近的、易混淆的另一实体。"
                       "例如：将特定型号的产品名替换为同品牌的另一型号，或将人名替换为同名不同人。"
                       "目的是测试 Reranker 在精排阶段的精准消歧能力。",
    },
]

_ADV_PROMPT = """\
你是一个测试数据改造专家。下面有一个原始问题和对应的 Wiki 文本片段。
请将原始问题改造成指定的对抗类型，使其检索难度更高，但改造后的问题仍然可以被原文回答。

改造类型：{adv_label}
改造要求：{adv_instruction}

注意：
- 只能修改问题的措辞，不能改变问题的核心意图
- 改造后的问题必须仍然能且仅能被 Chunk 原文回答
- 只输出改造后的问题本身，不要加任何前缀或解释

Chunk：
{chunk}

原始问题：
{original_query}

改造后的问题："""


# ══════════════════════════════════════════════════════════════════════
# DB
# ══════════════════════════════════════════════════════════════════════

def _title_has_chinese(title: str) -> bool:
    """title 中只要含有至少一个中文字符，就视为中文词条。"""
    return any("\u4e00" <= c <= "\u9fff" for c in title)


async def fetch_chunks(args) -> list[dict]:
    pool = await asyncpg.create_pool(
        host=args.db_host, port=args.db_port,
        database=args.db_name, user=args.db_user,
        min_size=1, max_size=2,
    )
    # 多取 3 倍供 title 过滤后仍能凑够 n 条
    rows = await pool.fetch(
        """
        SELECT id, content, metadata
        FROM wiki_documents
        WHERE length(content) BETWEEN $1 AND $2
        ORDER BY random()
        LIMIT $3;
        """,
        args.min_len, args.max_len, args.n * 3,
    )
    await pool.close()
    result = []
    for r in rows:
        meta = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
        title = meta.get("title", "")
        if not _title_has_chinese(title):
            continue
        if "这是一个非中文标题的重定向" in r["content"]:
            continue
        result.append({"id": r["id"], "content": r["content"], "metadata": meta, "_title": title})
        if len(result) >= args.n:
            break
    return result


# ══════════════════════════════════════════════════════════════════════
# LLM 调用
# ══════════════════════════════════════════════════════════════════════

def _call_gemini(model, prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text.strip()


def generate_base_query(model, chunk_content: str, query_type: dict) -> str:
    prompt = _BASE_PROMPT.format(
        query_type=query_type["type"],
        instruction=query_type["instruction"],
        chunk=chunk_content,
    )
    return _call_gemini(model, prompt)


def generate_adversarial_query(
    model, chunk_content: str, original_query: str, adv_type: dict
) -> str:
    prompt = _ADV_PROMPT.format(
        adv_label=adv_type["label"],
        adv_instruction=adv_type["instruction"],
        chunk=chunk_content,
        original_query=original_query,
    )
    return _call_gemini(model, prompt)


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="黄金测试集全自动构建（三步一体）")
    parser.add_argument("--n",         type=int,   default=50,
                        help="采样 Chunk 数（默认 50）")
    parser.add_argument("--output",    default="eval/golden_dataset.csv")
    parser.add_argument("--model",     default="gemini-2.5-pro",
                        help="Gemini 模型 ID")
    parser.add_argument("--adv-ratio", type=float, default=0.3,
                        help="对抗样本比例（默认 0.3，即 30%%）")
    parser.add_argument("--min-len",   type=int,   default=80)
    parser.add_argument("--max-len",   type=int,   default=512)
    parser.add_argument("--delay",     type=float, default=0.3,
                        help="每次 API 调用后的等待秒数（避免限速）")
    parser.add_argument("--db-host",   default=os.getenv("DB_HOST", "localhost"))
    parser.add_argument("--db-port",   default=os.getenv("DB_PORT", "5432"))
    parser.add_argument("--db-name",   default=os.getenv("POSTGRES_DB", "rag_db"))
    parser.add_argument("--db-user",   default=os.getenv("POSTGRES_USER", "rag_user"))
    args = parser.parse_args()

    # ── Step 1：采样 Chunk ─────────────────────────────────────────────
    print(f"[Step 1] 从 DB 随机采样（{args.min_len}~{args.max_len} 字符，目标 {args.n} 条）...")
    chunks = asyncio.run(fetch_chunks(args))
    if not chunks:
        print("❌ 采样结果为空，请检查 DB 连接或过滤参数。", file=sys.stderr)
        sys.exit(1)
    print(f"         实际采样: {len(chunks)} 条")

    # 初始化 Gemini
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(args.model)
    except ImportError:
        print("❌ 请先安装 Gemini SDK: pip install -U google-generativeai", file=sys.stderr)
        sys.exit(1)

    # 预分配提问类型（顺序轮转，保证均匀分布）
    query_type_cycle = [_QUERY_TYPES[i % len(_QUERY_TYPES)] for i in range(len(chunks))]
    random.shuffle(query_type_cycle)

    # 预分配对抗类型（精确控制比例，非纯随机）
    n_adv   = max(1, round(len(chunks) * args.adv_ratio))
    adv_assignments: list[dict | None] = [None] * len(chunks)
    adv_indices = random.sample(range(len(chunks)), n_adv)
    for j, idx in enumerate(adv_indices):
        adv_assignments[idx] = _ADV_TYPES[j % len(_ADV_TYPES)]

    print(f"         提问类型分布: {len(chunks) // len(_QUERY_TYPES)} 条/类型（轮转）")
    print(f"         对抗样本: {n_adv} 条 ({args.adv_ratio:.0%})，"
          f"各类型 ~{n_adv // len(_ADV_TYPES)} 条")

    # ── Step 2 + 3：生成 base query，对指定 Chunk 进行对抗改造 ──────────
    print(f"\n[Step 2/3] 调用 {args.model} 生成 query + 对抗改造...")

    rows = []
    failed_base = 0
    failed_adv  = 0
    type_counts = {t["type"]: 0 for t in _QUERY_TYPES}

    for i, (chunk, qt, adv) in enumerate(
        zip(chunks, query_type_cycle, adv_assignments), 1
    ):
        title  = chunk["_title"]
        is_adv = adv is not None
        tag    = f"🎯 ADV:{adv['type']}" if is_adv else "base"
        print(f"  [{i:3d}/{len(chunks)}] chunk={chunk['id']:6d}  {tag:<30s}", end="")

        # Step 2：生成 base query
        generated_query = ""
        try:
            generated_query = generate_base_query(model, chunk["content"], qt)
            type_counts[qt["type"]] += 1
            print(f"  [{qt['type']}] {generated_query[:45]}...")
        except Exception as e:
            print(f"  ⚠ base 生成失败: {e}")
            failed_base += 1

        if args.delay > 0:
            time.sleep(args.delay)

        # Step 3：对抗改造（仅对预分配的 Chunk）
        human_query     = generated_query
        adversarial_type = ""

        if is_adv and generated_query:
            try:
                human_query = generate_adversarial_query(
                    model, chunk["content"], generated_query, adv
                )
                adversarial_type = adv["type"]
                print(f"           → [{adv['label']}] {human_query[:45]}...")
            except Exception as e:
                print(f"           ⚠ 对抗改造失败: {e}，使用原始 query")
                human_query = generated_query
                failed_adv += 1

            if args.delay > 0:
                time.sleep(args.delay)

        rows.append({
            "query_id":         i,
            "chunk_id":         chunk["id"],
            "chunk_title":      title,
            "chunk_content":    chunk["content"],
            "generated_type":   qt["type"],
            "generated_query":  generated_query,
            "human_query":      human_query,       # 最终评测使用此列
            "adversarial_type": adversarial_type,  # "" = 非对抗样本（对照组）
        })

    # ── 写 CSV ────────────────────────────────────────────────────────
    print(f"\n[写入] {args.output}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ── 汇总 ──────────────────────────────────────────────────────────
    n_adv_ok  = sum(1 for r in rows if r["adversarial_type"])
    n_base_ok = sum(1 for r in rows if r["generated_query"])

    print(f"\n✅ 完成。{len(rows)} 条写入 {args.output}")
    print(f"   base query 成功: {n_base_ok}/{len(rows)}  失败: {failed_base}")
    print(f"   对抗改造 成功:   {n_adv_ok}/{n_adv}  失败: {failed_adv}")
    print()
    print("   提问类型分布：")
    for t, c in type_counts.items():
        print(f"     {t:<35s}: {c} 条")
    print()
    print("   对抗类型分布：")
    for adv_t in _ADV_TYPES:
        cnt = sum(1 for r in rows if r["adversarial_type"] == adv_t["type"])
        print(f"     {adv_t['type']:<20s} ({adv_t['label']}): {cnt} 条")
    print()
    print(f"下一步：python eval/run_retrieval_eval.py --golden {args.output}")


if __name__ == "__main__":
    main()
