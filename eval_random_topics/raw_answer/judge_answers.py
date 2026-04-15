"""
eval/raw_answer/judge_answers.py
================================
对裸答结果执行 LLM-as-Judge 打分。

直接复用 eval/judge_answers.py 的全部逻辑，仅修改默认路径指向 raw_answer 目录。

用法：
  python eval/raw_answer/judge_answers.py
  python eval/raw_answer/judge_answers.py --resume
  python eval/raw_answer/judge_answers.py --limit 10
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from eval_random_topics.judge_answers import main as _judge_main
import argparse


def main():
    # 替换默认参数后委托给 eval/judge_answers.py
    _defaults = {
        "--answers": "eval/raw_answer/eval_answers.csv",
        "--results": "eval/raw_answer/eval_judge_results.csv",
        "--summary": "eval/raw_answer/eval_judge_summary.json",
    }

    # 只在用户未显式指定时注入默认值
    injected = []
    for flag, default in _defaults.items():
        if flag not in sys.argv:
            injected += [flag, default]

    sys.argv = [sys.argv[0]] + injected + sys.argv[1:]
    _judge_main()


if __name__ == "__main__":
    main()
