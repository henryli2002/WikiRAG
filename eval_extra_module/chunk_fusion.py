"""
eval_extra_module/chunk_fusion.py
=================================
Chunk 融合模块（消融实验 Factor B）。

通过 llama.cpp (Qwen 9B) 将 top-3 检索到的 chunks 融合为一段连贯的
参考资料，解决以下问题：
  - 切块碎裂（Death Cause B）：信息分散在多个 chunk 中
  - 噪声稀释：无关 chunk 拉低 Context Precision
  - 冗余重复：相似 chunk 浪费上下文窗口

融合策略：
  1. 将 top-3 chunks 送入 LLM
  2. LLM 提取与 query 相关的核心信息，去除冗余
  3. 输出一段紧凑、连贯的参考摘要
  4. 生成阶段使用融合后的 context 替代原始 chunks

用法：
    from eval_extra_module.chunk_fusion import fuse_chunks
    fused = fuse_chunks(llm_client, query, chunks)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from openai import OpenAI


_FUSION_SYSTEM_PROMPT = """\
你是一个信息整合助手。你的任务是将多段检索到的参考资料整合为一段简洁连贯的摘要。

规则：
1. 只保留与用户问题直接相关的信息，去除无关内容
2. 合并不同片段中的重复信息，消除冗余
3. 保持事实准确性，不得添加参考资料中没有的信息
4. 输出为一段连贯的文字，不要分点列举
5. 控制在 200 字以内"""

_FUSION_USER_TEMPLATE = """\
用户问题：{query}

检索到的参考资料片段：
{chunks_text}

请将以上参考资料整合为一段与问题相关的简洁摘要（直接输出摘要，不要其他内容）："""


@dataclass
class FusionStats:
    """Chunk 融合的延迟统计。"""
    fusion_ms: float           # 融合耗时（ms）
    original_context: str      # 原始拼接的 context
    fused_context: str         # 融合后的 context
    original_chars: int        # 原始字符数
    fused_chars: int           # 融合后字符数


def fuse_chunks(
    llm: OpenAI,
    query: str,
    chunks: list[dict],
    model: str = "local-model",
    max_chars_per_chunk: int = 400,
    extra_body: dict | None = None,
) -> FusionStats:
    """
    调用 llama.cpp 将多个 chunks 融合为紧凑的参考资料。

    Parameters
    ----------
    llm : OpenAI
        已配置好 base_url 的 OpenAI 客户端（指向 llama.cpp）。
    query : str
        用户 query（用于引导 LLM 提取相关信息）。
    chunks : list[dict]
        检索到的 chunks 列表，每个 dict 包含 id/content/metadata。
    model : str
        模型名称。
    max_chars_per_chunk : int
        每个 chunk 送入融合前的最大字符数。
    extra_body : dict, optional
        额外的采样参数。

    Returns
    -------
    FusionStats
        包含融合结果和统计信息。
    """
    if extra_body is None:
        extra_body = {
            "top_k": 40,
            "min_p": 0.05,
            "repeat_penalty": 1.1,
        }

    # 拼接原始 chunks（与 generate_answers._format_context 对齐）
    parts = []
    for i, doc in enumerate(chunks, 1):
        title = doc.get("metadata", {}).get("title", "")
        content = doc["content"][:max_chars_per_chunk]
        if len(doc["content"]) > max_chars_per_chunk:
            content += "…"
        header = f"[{i}] {title}" if title else f"[{i}]"
        parts.append(f"{header}\n{content}")
    original_context = "\n\n".join(parts)

    # 构造融合 prompt
    chunks_text = "\n\n".join(
        f"片段{i}：\n{doc['content'][:max_chars_per_chunk]}"
        for i, doc in enumerate(chunks, 1)
    )

    messages = [
        {"role": "system", "content": _FUSION_SYSTEM_PROMPT},
        {"role": "user", "content": _FUSION_USER_TEMPLATE.format(
            query=query, chunks_text=chunks_text
        )},
    ]

    t_start = time.time()

    resp = llm.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        top_p=0.7,
        max_tokens=512,
        extra_body=extra_body,
    )

    fusion_ms = (time.time() - t_start) * 1000
    fused = resp.choices[0].message.content.strip()

    # 如果融合结果为空，回退到原始 context
    if not fused:
        fused = original_context

    return FusionStats(
        fusion_ms=round(fusion_ms, 1),
        original_context=original_context,
        fused_context=fused,
        original_chars=len(original_context),
        fused_chars=len(fused),
    )
