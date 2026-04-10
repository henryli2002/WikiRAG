"""
recall_rank_analysis.py
=======================
对 /search/debug 端点发起多组查询，统计最终被选中文档在各召回阶段的排名分布，
画柱状图帮助判断 VECTOR_RECALL_K / BM25_RECALL_K / HYBRID_TOP_K 设多少合适。

用法：
    # 先启动服务
    uvicorn app.main:app --reload

    # 另开终端
    python scripts/recall_rank_analysis.py
"""

import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
from collections import defaultdict

# ─── 配置 ────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
TOP_K = 5          # 与服务端 FINAL_TOP_K 保持一致
RECALL_K = 50      # 与服务端 HYBRID_TOP_K 保持一致

TEST_QUERIES = [
    "量子计算的基本原理",
    "新冠病毒的起源和传播",
    "人工智能在医疗诊断中的应用",
    "气候变化对海平面的影响",
    "比特币区块链工作原理",
    "黑洞是如何形成的",
    "中国古代四大发明",
    "伪造川普视频",
    "深度学习神经网络训练",
    "爱因斯坦相对论",
    "新冠疫苗研发过程",
    "火星探测任务",
    "基因编辑CRISPR技术",
    "大语言模型的训练方法",
    "核聚变能源研究",
    "量子纠缠现象解释",
    "自动驾驶技术原理",
    "可再生能源太阳能发电",
    "抗生素耐药性问题",
    "宇宙暗物质暗能量",
]

# ─── 字体（macOS 上优先用系统中文字体）───────────────────
def _setup_font():
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
    ]
    for path in candidates:
        try:
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            matplotlib.rcParams["font.family"] = prop.get_name()
            return
        except Exception:
            continue
    # 回退：用英文 label 也能看懂
    print("[warn] 未找到中文字体，标签将显示为英文")

_setup_font()


# ─── 数据收集 ─────────────────────────────────────────────

def query_debug(query: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/search/debug",
            json={"query": query, "top_k": TOP_K},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [skip] '{query}': {e}")
        return None


def collect(queries: list[str]):
    """
    返回：
      final_merged_ranks   : 每个最终结果的 merged_rank（1-based）
      final_vector_ranks   : 每个最终结果的 vector_rank（None=未命中）
      final_bm25_ranks     : 每个最终结果的 bm25_rank（None=未命中）
      rerank_merged_ranks  : 送入 rerank 的全部候选的 merged_rank
      per_query            : 每条 query 的 final_docs 列表（调试用）
    """
    final_merged_ranks = []
    final_vector_ranks = []
    final_bm25_ranks   = []
    rerank_merged_ranks = []
    per_query = []

    for i, q in enumerate(queries, 1):
        print(f"[{i:2d}/{len(queries)}] {q}")
        data = query_debug(q)
        if data is None:
            continue

        for doc in data["final_docs"]:
            if doc["merged_rank"] is not None:
                final_merged_ranks.append(doc["merged_rank"])
            if doc["vector_rank"] is not None:
                final_vector_ranks.append(doc["vector_rank"])
            if doc["bm25_rank"] is not None:
                final_bm25_ranks.append(doc["bm25_rank"])

        rerank_merged_ranks.extend(data["rerank_input_merged_ranks"])
        per_query.append(data)

    return (
        final_merged_ranks, final_vector_ranks, final_bm25_ranks,
        rerank_merged_ranks, per_query,
    )


# ─── 绘图 ─────────────────────────────────────────────────

def plot(
    final_merged: list[int],
    final_vector: list[int],
    final_bm25: list[int],
    rerank_input: list[int],
    recall_k: int,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"召回 Top-{recall_k} 分析  "
        f"（共 {len(TEST_QUERIES)} 条查询，最终结果 {len(final_merged)} 条）",
        fontsize=14, fontweight="bold",
    )

    bins = np.arange(1, recall_k + 2) - 0.5  # 每个 rank 一个格

    # ── 图1：最终结果在 hybrid merged 列表中的排名分布 ──────
    ax = axes[0, 0]
    ax.hist(final_merged, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_title("最终结果在 Hybrid Merged 中的排名分布")
    ax.set_xlabel("Merged Rank（越小越靠前）")
    ax.set_ylabel("出现次数")
    ax.set_xlim(0.5, recall_k + 0.5)

    # 标出累计 80% / 90% 覆盖线
    sorted_ranks = sorted(final_merged)
    n = len(sorted_ranks)
    for pct, color, ls in [(0.8, "#E74C3C", "--"), (0.9, "#27AE60", "-.")]:
        idx = int(np.ceil(pct * n)) - 1
        if 0 <= idx < n:
            cutoff = sorted_ranks[idx]
            ax.axvline(cutoff, color=color, linestyle=ls, linewidth=1.5,
                       label=f"{int(pct*100)}% cutoff = rank {cutoff}")
    ax.legend(fontsize=8)

    # ── 图2：累计覆盖率曲线（核心决策图）─────────────────────
    ax = axes[0, 1]
    x_vals = np.arange(1, recall_k + 1)
    coverage = [sum(1 for r in final_merged if r <= k) / n * 100 for k in x_vals]
    ax.plot(x_vals, coverage, color="#4C72B0", linewidth=2)
    ax.axhline(80, color="#E74C3C", linestyle="--", linewidth=1, label="80%")
    ax.axhline(90, color="#27AE60", linestyle="-.", linewidth=1, label="90%")
    ax.axhline(95, color="#F39C12", linestyle=":", linewidth=1, label="95%")
    ax.fill_between(x_vals, coverage, alpha=0.15, color="#4C72B0")
    ax.set_title("取 Top-K 候选能覆盖多少最终结果（累计覆盖率）")
    ax.set_xlabel("K（Hybrid Merged 保留数量）")
    ax.set_ylabel("最终结果覆盖率 %")
    ax.set_xlim(1, recall_k)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

    # 标注 80/90/95% 对应的 K 值
    for pct, color in [(80, "#E74C3C"), (90, "#27AE60"), (95, "#F39C12")]:
        for k, cov in zip(x_vals, coverage):
            if cov >= pct:
                ax.annotate(f"K={k}", xy=(k, pct), xytext=(k + 1, pct - 6),
                            fontsize=8, color=color,
                            arrowprops=dict(arrowstyle="->", color=color, lw=1))
                break

    # ── 图3：向量召回 vs BM25 召回 各自贡献的排名分布 ───────
    ax = axes[1, 0]
    ax.hist(final_vector, bins=bins, alpha=0.6, color="#2196F3",
            edgecolor="white", linewidth=0.4, label=f"向量召回 (n={len(final_vector)})")
    ax.hist(final_bm25, bins=bins, alpha=0.6, color="#FF5722",
            edgecolor="white", linewidth=0.4, label=f"BM25召回 (n={len(final_bm25)})")
    ax.set_title("最终结果在各召回路中的排名（两路独立 1-50）")
    ax.set_xlabel("单路 Rank（1=该路第一名）")
    ax.set_ylabel("出现次数")
    ax.set_xlim(0.5, recall_k + 0.5)
    ax.legend(fontsize=8)

    # ── 图4：送入 rerank 的候选 merged_rank 分布 ─────────────
    ax = axes[1, 1]
    ax.hist(rerank_input, bins=bins, color="#9C27B0", edgecolor="white", linewidth=0.4,
            label=f"送入 rerank（n={len(rerank_input)}）")
    # 叠加最终结果的分布
    ax.hist(final_merged, bins=bins, color="#FF9800", edgecolor="white", linewidth=0.4,
            alpha=0.75, label=f"最终结果（n={len(final_merged)}）")
    ax.set_title("送入 Rerank 的候选 vs 最终结果（Merged Rank）")
    ax.set_xlabel("Merged Rank")
    ax.set_ylabel("出现次数")
    ax.set_xlim(0.5, recall_k + 0.5)
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = "recall_rank_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n图已保存到 {out}")
    plt.show()


# ─── 统计摘要 ─────────────────────────────────────────────

def print_summary(final_merged: list[int], recall_k: int):
    if not final_merged:
        print("没有收集到数据。")
        return
    n = len(final_merged)
    sorted_ranks = sorted(final_merged)
    print("\n" + "=" * 50)
    print(f"  最终结果 Merged Rank 分布摘要（共 {n} 条）")
    print("=" * 50)
    print(f"  最小 rank : {min(final_merged)}")
    print(f"  中位 rank : {np.median(final_merged):.1f}")
    print(f"  最大 rank : {max(final_merged)}")
    print(f"  均值 rank : {np.mean(final_merged):.1f}")
    print()
    for k in [10, 15, 20, 25, 30, 40, 50]:
        if k > recall_k:
            break
        cnt = sum(1 for r in final_merged if r <= k)
        print(f"  Top-{k:2d} 覆盖率 : {cnt/n*100:5.1f}%  ({cnt}/{n})")
    print("=" * 50)


# ─── 主流程 ──────────────────────────────────────────────

if __name__ == "__main__":
    print(f"开始分析，共 {len(TEST_QUERIES)} 条查询...\n")
    (
        final_merged, final_vector, final_bm25,
        rerank_input, per_query,
    ) = collect(TEST_QUERIES)

    print_summary(final_merged, RECALL_K)
    plot(final_merged, final_vector, final_bm25, rerank_input, RECALL_K)
