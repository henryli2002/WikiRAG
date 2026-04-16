#!/bin/bash
set -e

# ══════════════════════════════════════════════════════════════════════
# AB 消融实验运行脚本
#
# 三组消融实验条件：
#   1. rewrite_only   — 多角度 Query 重写 (A) + 多路召回 RRF 融合
#   2. fusion_only    — 原始 query + Chunk 融合 (B)
#   3. rewrite_fusion — 多角度 Query 重写 (A) + Chunk 融合 (B)
#
# baseline 数据直接读取 eval_random_topics/rag_top3，无需复制或重新生成。
#
# 用法：
#   bash eval_extra_module/run_ablation.sh                          # 运行 3 组消融
#   bash eval_extra_module/run_ablation.sh --only rewrite_only      # 只跑某一组
#   bash eval_extra_module/run_ablation.sh --limit 5                # 调试模式，每组只跑 5 条
# ══════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 默认参数
ONLY=""
LIMIT_ARG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --only)
      ONLY="$2"
      shift 2
      ;;
    --limit)
      LIMIT_ARG="--limit $2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--only <condition>] [--limit N]"
      exit 1
      ;;
  esac
done

# ── 辅助函数 ──────────────────────────────────────────────────────────

run_condition() {
  local NAME=$1
  local FLAGS=$2

  echo ""
  echo "═══════════════════════════════════════════════════"
  echo "  实验条件: ${NAME}"
  echo "═══════════════════════════════════════════════════"

  # Step 1: 生成答案（--output 由 python 脚本自动推导）
  echo ">>> [${NAME}] Step 1: 生成答案 [支持断点续传]"
  ./.venv/bin/python -m eval_extra_module.generate_answers_ab \
    --resume \
    ${FLAGS} \
    ${LIMIT_ARG}

  local DIR="eval_extra_module/${NAME}"

  # Step 2: 裁判打分（复用 eval_random_topics 的 judge）
  echo ">>> [${NAME}] Step 2: 裁判打分 [支持断点续传]"
  ./.venv/bin/python -m eval_random_topics.judge_answers \
    --answers "${DIR}/eval_answers.csv" \
    --results "${DIR}/eval_judge_results.csv" \
    --summary "${DIR}/eval_judge_summary.json" \
    --resume

  # Step 3: 单组报告
  echo ">>> [${NAME}] Step 3: 生成实验报告"
  ./.venv/bin/python -m eval_random_topics.eval_report --dir "${DIR}"

  echo ""
  echo "  ✅ ${NAME} 完成"
  echo ""
}

# ── 主流程 ────────────────────────────────────────────────────────────

echo "╔═══════════════════════════════════════════════════╗"
echo "║      AB 消融实验 (Query重写 × Chunk融合)         ║"
echo "║                                                   ║"
echo "║   A = Query 重写 (1 golden + N 发散, Qwen 9B)    ║"
echo "║   B = Chunk 融合 (llama.cpp / Qwen 9B)           ║"
echo "║   baseline = eval_random_topics/rag_top3          ║"
echo "╚═══════════════════════════════════════════════════╝"

if [ -n "$ONLY" ]; then
  case $ONLY in
    rewrite_only)
      run_condition "rewrite_only" "--rewrite"
      ;;
    fusion_only)
      run_condition "fusion_only" "--fusion"
      ;;
    rewrite_fusion)
      run_condition "rewrite_fusion" "--rewrite --fusion"
      ;;
    *)
      echo "❌ 未知条件: $ONLY"
      echo "可选: rewrite_only, fusion_only, rewrite_fusion"
      exit 1
      ;;
  esac
else
  # 运行全部 3 组消融实验
  run_condition "rewrite_only" "--rewrite"
  run_condition "fusion_only" "--fusion"
  run_condition "rewrite_fusion" "--rewrite --fusion"
fi

# ── 汇总对比报告 ──────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  生成 AB 消融对比报告"
echo "═══════════════════════════════════════════════════"

./.venv/bin/python -m eval_extra_module.eval_report_ab

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║           所有消融实验已完成！                    ║"
echo "╚═══════════════════════════════════════════════════╝"