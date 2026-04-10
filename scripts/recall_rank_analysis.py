"""
recall_rank_analysis.py
=======================
对 /search/debug 端点发起多组查询，统计最终被选中文档在各召回阶段的排名分布，
画柱状图帮助判断 VECTOR_RECALL_K / BM25_RECALL_K / HYBRID_TOP_K 设多少合适。

用法：
    # 先启动服务
    uvicorn app.main:app

    # 另开终端
    python scripts/recall_rank_analysis.py
"""

import requests
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict

# ─── 配置 ────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
TOP_K    = 5

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

# ─── 中文字体 ─────────────────────────────────────────────
def _setup_font():
    for path in [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
    ]:
        try:
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            matplotlib.rcParams["font.family"] = prop.get_name()
            return
        except Exception:
            pass

_setup_font()
matplotlib.rcParams["axes.unicode_minus"] = False


# ─── 数据收集 ─────────────────────────────────────────────

def query_debug(query: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/search/debug",
            json={"query": query, "top_k": TOP_K},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [skip] '{query}': {e}")
        return None


def collect(queries: list[str]):
    """
    返回：
      final_merged_ranks      所有最终结果的 merged_rank（None 表示无法追溯）
      final_vector_ranks      各最终结果的 vector_rank（None=未被向量召回）
      final_bm25_ranks        各最终结果的 bm25_rank（None=未被BM25召回）
      rerank_input_merged_ranks  送入 rerank 的全部候选 merged_rank
      hybrid_top_k            服务端实际使用的 HYBRID_TOP_K（从数据推断）
    """
    final_merged_ranks      = []   # 含 None
    final_vector_ranks      = []   # 仅非 None
    final_bm25_ranks        = []   # 仅非 None
    rerank_input_merged_ranks = []
    max_merged_rank = 0

    for i, q in enumerate(queries, 1):
        print(f"[{i:2d}/{len(queries)}] {q}")
        data = query_debug(q)
        if data is None:
            continue

        for doc in data["final_docs"]:
            final_merged_ranks.append(doc["merged_rank"])   # 保留 None
            if doc["vector_rank"] is not None:
                final_vector_ranks.append(doc["vector_rank"])
            if doc["bm25_rank"] is not None:
                final_bm25_ranks.append(doc["bm25_rank"])
            if doc["merged_rank"] is not None:
                max_merged_rank = max(max_merged_rank, doc["merged_rank"])

        rerank_input_merged_ranks.extend(data["rerank_input_merged_ranks"])
        if data["rerank_input_merged_ranks"]:
            max_merged_rank = max(max_merged_rank, max(data["rerank_input_merged_ranks"]))

    return (
        final_merged_ranks,
        final_vector_ranks,
        final_bm25_ranks,
        rerank_input_merged_ranks,
        max_merged_rank or 50,
    )


# ─── 统计摘要 ─────────────────────────────────────────────

def print_summary(final_merged_ranks: list, hybrid_top_k: int):
    total = len(final_merged_ranks)
    known = [r for r in final_merged_ranks if r is not None]
    n_none = total - len(known)

    print("\n" + "=" * 56)
    print(f"  最终结果排名分布摘要  共 {total} 条（无法追溯: {n_none} 条）")
    print("=" * 56)
    if not known:
        print("  无有效数据")
        return

    print(f"  最小 merged_rank : {min(known)}")
    print(f"  中位 merged_rank : {np.median(known):.1f}")
    print(f"  均值 merged_rank : {np.mean(known):.1f}")
    print(f"  最大 merged_rank : {max(known)}")
    print()

    # 以 total 为分母（含 None），真实覆盖率
    cuts = [k for k in [5, 10, 15, 20, 25, 30, 40, 50] if k <= hybrid_top_k]
    if hybrid_top_k not in cuts:
        cuts.append(hybrid_top_k)
    for k in cuts:
        cnt = sum(1 for r in known if r <= k)
        print(f"  Top-{k:2d} 真实覆盖率 : {cnt/total*100:5.1f}%  ({cnt}/{total})")
    print("=" * 56)


# ─── 绘图 ─────────────────────────────────────────────────

def plot(
    final_merged_ranks: list,           # 含 None
    final_vector_ranks: list[int],
    final_bm25_ranks:   list[int],
    rerank_input_merged_ranks: list[int],
    hybrid_top_k: int,
):
    total   = len(final_merged_ranks)
    known   = [r for r in final_merged_ranks if r is not None]
    n_none  = total - len(known)
    n_queries = len(TEST_QUERIES)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"召回 Top-{hybrid_top_k} 分析"
        f"  （共 {n_queries} 条查询，最终结果 {total} 条"
        + (f"，{n_none} 条无法追溯" if n_none else "") + "）",
        fontsize=13, fontweight="bold",
    )

    bins = np.arange(0.5, hybrid_top_k + 1.5)   # 每个 rank 一格
    x_max = hybrid_top_k + 1

    # ── 图1（左上）：merged rank 分布柱状图 ─────────────────
    ax = axes[0, 0]
    ax.hist(known, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.set_title(f"最终结果在 Hybrid Merged 中的排名分布（越小越好）")
    ax.set_xlabel("Merged Rank")
    ax.set_ylabel("出现次数")
    ax.set_xlim(0.5, x_max)

    # 在图内标注累计 80%/90% 对应的竖线（不放 legend，直接文字标注）
    sorted_known = sorted(known)
    n_known = len(sorted_known)
    for pct, color, ls in [(0.80, "#E74C3C", "--"), (0.90, "#27AE60", "-.")]:
        # 以 total 为分母
        cutoff_rank = next(
            (r for r in range(1, hybrid_top_k + 1)
             if sum(1 for x in known if x <= r) / total >= pct),
            None,
        )
        if cutoff_rank is not None and cutoff_rank <= hybrid_top_k:
            ax.axvline(cutoff_rank, color=color, linestyle=ls, linewidth=1.5, alpha=0.8)
            ax.text(
                cutoff_rank + 0.3,
                ax.get_ylim()[1] * 0.92 if pct == 0.80 else ax.get_ylim()[1] * 0.80,
                f"{int(pct*100)}% cutoff\n= rank {cutoff_rank}",
                color=color, fontsize=8, va="top",
            )
        else:
            # 在 hybrid_top_k 内达不到该覆盖率，标注说明
            ax.text(
                x_max * 0.55,
                ax.get_ylim()[1] * (0.92 if pct == 0.80 else 0.75),
                f"{int(pct*100)}% cutoff > rank {hybrid_top_k}",
                color=color, fontsize=8, style="italic",
            )

    # ── 图2（右上）：累计覆盖率曲线 ────────────────────────
    ax = axes[0, 1]
    x_vals = np.arange(1, hybrid_top_k + 1)
    # 分母用 total（含 None），反映真实覆盖上限
    coverage = [
        sum(1 for r in known if r <= k) / total * 100
        for k in x_vals
    ]
    max_possible = len(known) / total * 100  # None 文档永远无法覆盖

    ax.plot(x_vals, coverage, color="#4C72B0", linewidth=2.2, zorder=3)
    ax.fill_between(x_vals, coverage, alpha=0.12, color="#4C72B0")

    # 上限虚线（有 None 时才画）
    if n_none > 0:
        ax.axhline(max_possible, color="#888", linestyle=":", linewidth=1,
                   label=f"理论上限 {max_possible:.1f}%（{n_none}条无法追溯）")
        ax.legend(fontsize=8, loc="lower right")

    # 80/90/95% 水平参考线 + 在曲线上标注对应 K 值
    for pct, color, ls in [(80, "#E74C3C", "--"), (90, "#27AE60", "-."), (95, "#F39C12", ":")]:
        ax.axhline(pct, color=color, linestyle=ls, linewidth=1, alpha=0.7)
        # 找到第一个超过该 pct 的 K
        k_hit = next((k for k, cov in zip(x_vals, coverage) if cov >= pct), None)
        if k_hit is not None:
            ax.annotate(
                f"K={k_hit}",
                xy=(k_hit, pct),
                xytext=(min(k_hit + 2, hybrid_top_k - 1), pct + 3),
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            )
        else:
            ax.text(
                hybrid_top_k * 0.75, pct - 4,
                f"{pct}% 未达到", fontsize=8, color=color, style="italic",
            )

    ax.set_title("取 Top-K 候选，能覆盖多少最终结果（真实覆盖率）")
    ax.set_xlabel("K（Hybrid Merged 保留数量）")
    ax.set_ylabel("最终结果覆盖率 %")
    ax.set_xlim(1, hybrid_top_k)
    ax.set_ylim(0, 105)

    # ── 图3（左下）：向量 vs BM25 各路排名分布 ─────────────
    ax = axes[1, 0]
    # 让两路 bars 并排
    width = 0.45
    rank_range = np.arange(1, hybrid_top_k + 1)
    vec_counts  = [final_vector_ranks.count(r) for r in rank_range]
    bm25_counts = [final_bm25_ranks.count(r)   for r in rank_range]
    ax.bar(rank_range - width/2, vec_counts,  width, color="#2196F3", alpha=0.8,
           label=f"向量召回 (n={len(final_vector_ranks)})")
    ax.bar(rank_range + width/2, bm25_counts, width, color="#FF5722", alpha=0.8,
           label=f"BM25召回 (n={len(final_bm25_ranks)})")
    ax.set_title("最终结果在各召回路中的排名（两路独立 1-50）")
    ax.set_xlabel("单路 Rank（1=该路第一名）")
    ax.set_ylabel("出现次数")
    ax.set_xlim(0.5, x_max)
    ax.legend(fontsize=8)

    # ── 图4（右下）：各 Merged Rank 的"晋升率" ─────────────
    # 指标：该 rank 的文档有多大比例最终进入了结果
    # = 该 rank 最终结果数 / 该 rank 的 rerank 候选数
    ax = axes[1, 1]

    rerank_counts = defaultdict(int)
    for r in rerank_input_merged_ranks:
        rerank_counts[r] += 1

    final_counts = defaultdict(int)
    for r in known:
        final_counts[r] += 1

    promo_x, promo_y, bar_colors = [], [], []
    for r in rank_range:
        rc = rerank_counts[r]
        fc = final_counts[r]
        if rc > 0:
            promo_x.append(r)
            promo_y.append(fc / rc * 100)
            bar_colors.append(
                "#2ECC71" if fc / rc >= 0.3 else
                "#F39C12" if fc / rc >= 0.1 else
                "#E74C3C"
            )

    ax.bar(promo_x, promo_y, color=bar_colors, edgecolor="white", linewidth=0.4)
    ax.axhline(100 / hybrid_top_k, color="#555", linestyle="--", linewidth=1,
               label=f"随机基线 {100/hybrid_top_k:.1f}%")
    ax.set_title("各 Merged Rank 的晋升率\n（该排名候选最终被选中的概率）")
    ax.set_xlabel("Merged Rank")
    ax.set_ylabel("晋升率 %")
    ax.set_xlim(0.5, x_max)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    # 颜色说明
    ax.text(x_max * 0.62, 95, "≥30% 高晋升", color="#2ECC71", fontsize=7)
    ax.text(x_max * 0.62, 87, "10~30% 中", color="#F39C12", fontsize=7)
    ax.text(x_max * 0.62, 79, "<10% 低晋升", color="#E74C3C", fontsize=7)

    plt.tight_layout()
    out = "recall_rank_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n图已保存到 {out}")
    plt.show()


# ─── 主流程 ──────────────────────────────────────────────

if __name__ == "__main__":
    print(f"开始分析，共 {len(TEST_QUERIES)} 条查询...\n")

    (
        final_merged, final_vector, final_bm25,
        rerank_input, hybrid_top_k,
    ) = collect(TEST_QUERIES)

    print_summary(final_merged, hybrid_top_k)
    plot(final_merged, final_vector, final_bm25, rerank_input, hybrid_top_k)
