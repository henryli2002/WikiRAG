"""
WikiRAG 压力测试脚本
测量不同并发度下的端到端延迟和吞吐量。

依赖: pip install httpx
用法:
  python scripts/stress_test.py                        # 默认 http://localhost:8000
  python scripts/stress_test.py --url http://host:8000
  python scripts/stress_test.py --concurrency 1 2 4 8  # 指定并发梯度
  python scripts/stress_test.py --requests 50          # 每梯度请求数
"""
import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass, field


# ─── 测试查询集 ────────────────────────────────────────────
# 覆盖四类场景：
#   AND_OK  - 多特定词，AND 查询预期有足够结果
#   OR_FALL - 词少或有泛词，预期 OR fallback
#   SHORT   - 极短查询，边界测试
#   LONG    - 长查询，embedding 和 rerank 负担较重
QUERIES = [
    # AND_OK
    {"q": "量子纠缠实验验证贝尔不等式",      "tag": "AND_OK"},
    {"q": "北宋王安石变法历史背景",           "tag": "AND_OK"},
    {"q": "深度学习卷积神经网络图像分类",     "tag": "AND_OK"},
    {"q": "人类基因组计划测序技术发展",       "tag": "AND_OK"},
    {"q": "经济学博弈论纳什均衡应用",         "tag": "AND_OK"},
    # OR_FALL（含停用词 / 词短）
    {"q": "量子计算基本原理介绍",             "tag": "OR_FALL"},
    {"q": "机器学习方法分析",                 "tag": "OR_FALL"},
    {"q": "历史发展研究",                     "tag": "OR_FALL"},
    # SHORT
    {"q": "相对论",                           "tag": "SHORT"},
    {"q": "光合作用",                         "tag": "SHORT"},
    # LONG
    {"q": "请详细介绍量子力学中薛定谔方程的推导过程以及它在氢原子能级计算中的具体应用", "tag": "LONG"},
    {"q": "中国古代四大发明造纸术印刷术火药指南针对世界文明的深远影响和历史意义",     "tag": "LONG"},
]


@dataclass
class Result:
    query: str
    tag: str
    latency_ms: float
    status: int
    n_results: int
    error: str = ""


@dataclass
class Stats:
    concurrency: int
    total_requests: int
    errors: int
    duration_s: float
    latencies: list[float] = field(default_factory=list)

    @property
    def rps(self) -> float:
        return self.total_requests / self.duration_s if self.duration_s > 0 else 0

    def percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * p / 100)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    def report(self) -> str:
        if not self.latencies:
            return f"  concurrency={self.concurrency}  全部请求失败 (errors={self.errors})"
        return (
            f"  concurrency={self.concurrency:2d} | "
            f"RPS={self.rps:5.1f} | "
            f"p50={self.percentile(50):6.0f}ms  "
            f"p90={self.percentile(90):6.0f}ms  "
            f"p99={self.percentile(99):6.0f}ms  "
            f"max={max(self.latencies):6.0f}ms | "
            f"err={self.errors}/{self.total_requests}"
        )


async def single_request(client, base_url: str, query: str, tag: str) -> Result:
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{base_url}/search",
            json={"query": query, "top_k": 5},
            timeout=60.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            return Result(query=query, tag=tag, latency_ms=latency_ms,
                          status=200, n_results=len(data.get("results", [])))
        else:
            return Result(query=query, tag=tag, latency_ms=latency_ms,
                          status=resp.status_code, n_results=0,
                          error=resp.text[:120])
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return Result(query=query, tag=tag, latency_ms=latency_ms,
                      status=0, n_results=0, error=str(e)[:120])


async def run_concurrency_level(
    base_url: str,
    concurrency: int,
    n_requests: int,
    verbose: bool,
) -> Stats:
    try:
        import httpx
    except ImportError:
        print("缺少依赖: pip install httpx", file=sys.stderr)
        sys.exit(1)

    # 循环使用查询集
    jobs = [QUERIES[i % len(QUERIES)] for i in range(n_requests)]

    sem = asyncio.Semaphore(concurrency)
    results: list[Result] = []

    async def bounded(client, job):
        async with sem:
            r = await single_request(client, base_url, job["q"], job["tag"])
            results.append(r)
            if verbose:
                status = "✓" if r.status == 200 else "✗"
                print(f"    {status} [{r.tag:8s}] {r.latency_ms:6.0f}ms  "
                      f"hits={r.n_results}  {r.query[:30]}")

    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        await asyncio.gather(*[bounded(client, job) for job in jobs])
        duration = time.perf_counter() - t0

    errors = sum(1 for r in results if r.status != 200)
    latencies = [r.latency_ms for r in results if r.status == 200]
    return Stats(
        concurrency=concurrency,
        total_requests=n_requests,
        errors=errors,
        duration_s=duration,
        latencies=latencies,
    )


async def warmup(base_url: str):
    try:
        import httpx
    except ImportError:
        print("缺少依赖: pip install httpx", file=sys.stderr)
        sys.exit(1)

    print("预热（1 次请求）...", end=" ", flush=True)
    async with httpx.AsyncClient() as client:
        r = await single_request(client, base_url, "中国历史", "warmup")
    if r.status == 200:
        print(f"OK ({r.latency_ms:.0f}ms)")
    else:
        print(f"FAILED: {r.error}")
        sys.exit(1)


async def main():
    parser = argparse.ArgumentParser(description="WikiRAG 压力测试")
    parser.add_argument("--url", default="http://localhost:8000", help="服务地址")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8],
                        help="并发梯度列表 (默认: 1 2 4 8)")
    parser.add_argument("--requests", type=int, default=40,
                        help="每梯度发送的总请求数 (默认: 40)")
    parser.add_argument("--verbose", action="store_true",
                        help="打印每条请求的详细信息")
    parser.add_argument("--no-warmup", action="store_true",
                        help="跳过预热请求")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  WikiRAG 压力测试")
    print(f"  目标: {args.url}")
    print(f"  并发梯度: {args.concurrency}")
    print(f"  每梯度请求数: {args.requests}")
    print(f"{'='*65}\n")

    if not args.no_warmup:
        await warmup(args.url)
        print()

    all_stats: list[Stats] = []
    for c in args.concurrency:
        print(f"并发={c}, 共 {args.requests} 请求...")
        if args.verbose:
            print()
        stats = await run_concurrency_level(args.url, c, args.requests, args.verbose)
        all_stats.append(stats)
        if args.verbose:
            print()

    # ─── 汇总报告 ──────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  汇总（所有并发梯度）")
    print(f"{'─'*65}")
    for s in all_stats:
        print(s.report())

    # ─── 按 tag 分类统计（合并所有并发梯度） ──────────────
    # 需要重跑一次带 tag 的收集，这里只做各梯度合并的文字总结
    print(f"\n  注: 用 --verbose 查看每条请求的 tag 和命中数")
    print(f"{'─'*65}\n")


if __name__ == "__main__":
    asyncio.run(main())
