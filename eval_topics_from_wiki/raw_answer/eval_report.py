"""
eval/raw_answer/eval_report.py
==============================
对裸答实验生成全链路报告。

复用 eval/eval_report.py 的全部逻辑，默认读取 raw_answer 目录下的 CSV。
检索部分复用 eval/eval_results.csv（检索 pipeline 相同，只是生成时不用 Context）。

用法：
  python eval/raw_answer/eval_report.py

对比两组结果：
  python eval/eval_report.py                   # RAG 版
  python eval/raw_answer/eval_report.py        # 裸答版
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from eval_random_topics.eval_report import main as _report_main


def main():
    _defaults = {
        "--dir": "eval/raw_answer",
        # 检索结果复用 eval/ 的（pipeline 完全一致）
        "--ret-csv": "eval/eval_results.csv",
    }

    injected = []
    for flag, default in _defaults.items():
        if flag not in sys.argv:
            injected += [flag, default]

    sys.argv = [sys.argv[0]] + injected + sys.argv[1:]
    _report_main()


if __name__ == "__main__":
    main()
