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
  echo "❌ 错误: 未提供 --topk 参数"
  echo "💡 用法: $0 --topk <k1> [k2] [k3]..."
  echo "💡 示例: $0 --topk 1 3 5"
  exit 1
fi

echo "====================================="
echo "   [全局] Step 1: 运行全量检索评测     "
echo "====================================="
if [ -f "eval/eval_results.csv" ] && [ -f "eval/eval_summary.json" ]; then
  echo "✅ 发现已存在的 eval/eval_results.csv，跳过全量检索评测 (如需重跑请手动删除该文件)。"
else
  # 跑一次全量检索评测，内部计算了 Hit@1, Hit@3, Hit@5, Hit@10
  python eval/run_retrieval_eval.py --golden eval/golden_dataset.csv --results eval/eval_results.csv --summary eval/eval_summary.json
fi

for K in "${TOPK_VALS[@]}"; do
  DIR="eval/rag_top${K}"
  mkdir -p "$DIR"

  echo "====================================="
  echo "        🚀 Running Top-${K} Experiment      "
  echo "====================================="
  
  # 2. 生成阶段
  echo ">>> [Top-${K}] Step 2: 生成答案 (generate_answers.py) [支持断点续传]"
  python eval/generate_answers.py --golden eval/golden_dataset.csv --output ${DIR}/eval_answers.csv --topk ${K} --resume

  # 3. 裁判打分
  echo ">>> [Top-${K}] Step 3: 裁判打分 (judge_answers.py) [支持断点续传]"
  python eval/judge_answers.py --answers ${DIR}/eval_answers.csv \
      --results ${DIR}/eval_judge_results.csv \
      --summary ${DIR}/eval_judge_summary.json \
      --resume

  # 4. 汇总报告
  echo ">>> [Top-${K}] Step 4: 生成实验报告 (eval_report.py)"
  python eval/eval_report.py --dir ${DIR} --ret-csv eval/eval_results.csv
  
  echo ""
done

echo "====================================="
echo "        ✅ 所有指定的实验已完成！         "
echo "====================================="
