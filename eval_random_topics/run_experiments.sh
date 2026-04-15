#!/bin/bash
set -e

# 解析命令行参数
TOPK_VALS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --topk)
      shift
      while [[ $# -gt 0 && ! $1 == --* ]]; do
        TOPK_VALS+=("$1")
        shift
      done
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

if [ ${#TOPK_VALS[@]} -eq 0 ]; then
  echo "错误: 未提供 --topk 参数"
  echo "用法: $0 --topk <k1> [k2] [k3]..."
  echo "示例: $0 --topk 1 3 5"
  exit 1
fi

for K in "${TOPK_VALS[@]}"; do
  DIR="eval/rag_top${K}"
  mkdir -p "$DIR"

  echo "====================================="
  echo "        Running Top-${K} Experiment      "
  echo "====================================="

  # 1. 生成阶段
  echo ">>> [Top-${K}] Step 1: 生成答案 (generate_answers.py) [支持断点续传]"
  python eval/generate_answers.py --golden eval/golden_dataset.csv --output ${DIR}/eval_answers.csv --topk ${K} --resume

  # 2. 裁判打分
  echo ">>> [Top-${K}] Step 2: 裁判打分 (judge_answers.py) [支持断点续传]"
  python eval/judge_answers.py --answers ${DIR}/eval_answers.csv \
      --results ${DIR}/eval_judge_results.csv \
      --summary ${DIR}/eval_judge_summary.json \
      --resume

  # 3. 汇总报告
  echo ">>> [Top-${K}] Step 3: 生成实验报告 (eval_report.py)"
  python eval/eval_report.py --dir ${DIR}

  echo ""
done

echo "====================================="
echo "        所有指定的实验已完成！         "
echo "====================================="
