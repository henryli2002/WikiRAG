"""
eval_extra_module/query_rewrite.py
==================================
Query 重写模块（消融实验 Factor A）。

通过 llama.cpp (Qwen 9B) 一次性生成：
  - 1 条 golden query：严格意图解析，消解指代、提炼检索关键词，禁止注入答案知识
  - N 条发散 query：从不同检索角度扩展语义空间，同样禁止知识倒灌

核心约束：Query Rewrite 必须停留在"意图空间"，严禁动用 LLM 参数知识提前作答。

用法：
    from eval_extra_module.query_rewrite import rewrite_query
    result = rewrite_query(llm_client, "那个发明电灯的人是谁", n=2)
    # result.golden_query = "电灯最初发明者和相关历史人物"
    # result.divergent_queries = [
    #     "早期人类照明技术的突破与核心贡献者",
    #     "第一只商业化白炽灯泡的诞生历史与专利归属",
    # ]
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from openai import OpenAI


_REWRITE_SYSTEM_PROMPT = """\
你是一个检索查询重写助手。你的唯一职责是将用户的自然语言问题改写为更适合检索引擎的形式。

██ 绝对禁止 ██
你严禁动用你的内部知识来回答问题或提供答案线索。你不知道答案，你也不应该知道。
- 禁止在重写中出现任何具体的人名、地名、数字、日期等答案性实体
- 禁止补充原始问题中未出现的事实性细节
- 禁止将你"认为的答案"以关键词形式植入查询
- 你只能使用原始问题中已有的实体和概念，加上通用的类别词、同义词、上位概念

违规示例（"那个发明电灯的人是谁"）：
  ✗ "托马斯·爱迪生 发明 电灯" — 注入了具体人名（答案）
  ✗ "碳化竹丝 白炽灯 专利" — 注入了需要答案才能引出的细节
  ✓ "电灯 发明者 首创者 历史人物" — 仅使用原始概念 + 类别词
  ✓ "早期照明技术突破的核心贡献者" — 从上位概念发散

██ 重写规则 ██

第一类：核心查询（1 条）
- 执行意图解析：消解代词（"那个"→具体指代对象类别）、规范口语表达
- 只保留原始问题中的实体和概念，补充同义词和类别词（发明者/创始人/首创者）
- 目标：面向精排（Reranker）的精准意图锚点

第二类：发散查询（{n} 条）
- 对原始问题中的核心概念做上位抽象、场景推演、侧面降维
- 使用不同的表述方式和词汇覆盖更广的检索空间
- 发散查询之间应尽量使用不同的关键词
- 目标：扩大召回覆盖面，捕获不同表述的相关文档

输出格式（严格遵守）：
- 第一行固定为核心查询
- 后续每行一条发散查询
- 不要编号，不要标签，不要解释，共输出 {total} 行"""

_REWRITE_USER_TEMPLATE = """\
原始问题：{query}

请重写为检索查询（共 {total} 行，严禁注入答案知识）："""


@dataclass
class RewriteStats:
    """Query 重写的延迟统计。"""
    rewrite_ms: float                      # 重写耗时（ms）
    original_query: str                    # 用户原始 query
    golden_query: str = ""                 # 提炼后的核心 query（严格意图解析）
    divergent_queries: list[str] = field(default_factory=list)  # N 条发散 query


def rewrite_query(
    llm: OpenAI,
    query: str,
    n: int = 2,
    model: str = "local-model",
    extra_body: dict | None = None,
) -> RewriteStats:
    """
    调用 llama.cpp 生成 1 条 golden query + N 条发散 query。

    核心约束：重写必须停留在"意图空间"，严禁 LLM 将参数知识注入查询。

    Parameters
    ----------
    llm : OpenAI
        已配置好 base_url 的 OpenAI 客户端（指向 llama.cpp）。
    query : str
        原始用户 query。
    n : int
        发散 query 数量（默认 2）。
    model : str
        模型名称（llama.cpp 固定为 "local-model"）。
    extra_body : dict, optional
        额外的采样参数。

    Returns
    -------
    RewriteStats
        包含 1 条 golden query + N 条发散 query 和耗时统计。
    """
    if extra_body is None:
        extra_body = {
            "top_k": 40,
            "min_p": 0.05,
            "repeat_penalty": 1.1,
        }

    total = n + 1
    messages = [
        {"role": "system", "content": _REWRITE_SYSTEM_PROMPT.format(n=n, total=total)},
        {"role": "user", "content": _REWRITE_USER_TEMPLATE.format(query=query, n=n, total=total)},
    ]

    t_start = time.time()

    resp = llm.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        top_p=0.8,
        max_tokens=512,
        extra_body=extra_body,
    )

    rewrite_ms = (time.time() - t_start) * 1000
    raw = resp.choices[0].message.content.strip()

    # 按行拆分，清理空行和编号前缀
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # 去除可能的编号前缀（1. / 1、/ A: / 核心查询: 等）
        line = re.sub(r"^[\d]+[.、)）:\s]+", "", line)
        line = re.sub(r"^[A-Ca-c][.、)）:\s]+", "", line)
        line = re.sub(r"^(核心查询|发散查询|核心|发散)[：:\s]*", "", line)
        line = line.strip()
        # 去除可能的引号包裹
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        if line.startswith("'") and line.endswith("'"):
            line = line[1:-1]
        if line:
            lines.append(line)

    # 第一行 = golden query，后续 = 发散 query
    if lines:
        golden = lines[0]
        divergent = lines[1:n + 1]
    else:
        # 全部解析失败，回退到原始 query
        golden = query
        divergent = []

    return RewriteStats(
        rewrite_ms=round(rewrite_ms, 1),
        original_query=query,
        golden_query=golden,
        divergent_queries=divergent,
    )
