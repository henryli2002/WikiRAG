"""
bench_inference.py
==================
对比 PyTorch FP16 (MPS) 与 ONNX FP16 (CoreML CPUAndGPU EP) 的逐层推理延迟。

测试阶段：
  Embedding : tokenize / device_transfer / forward / postprocess
  Reranker  : tokenize / device_transfer / forward / postprocess
             × batch_size ∈ [5, 10, 20, 30]

用法：
    # 需先完成 FP16 转换：
    python scripts/convert_onnx_fp16.py

    # 运行基准测试：
    python scripts/bench_inference.py
"""

import os
import sys
import time
import statistics
import numpy as np

PROJECT_ROOT         = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMBED_ONNX_FP16      = os.path.join(PROJECT_ROOT, "models", "bge-m3-onnx",       "model.onnx")
RERANKER_ONNX_FP16   = os.path.join(PROJECT_ROOT, "models", "bge-reranker-onnx",  "model.onnx")
EMBEDDING_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bge-m3")
RERANKER_MODEL_PATH  = os.path.join(PROJECT_ROOT, "models", "bge-reranker-v2-m3")

N_WARMUP = 5
N_RUNS   = 30

# 测试文本
QUERY   = "量子计算的基本原理与应用前景"
DOCS = [
    "量子计算是一种利用量子力学现象（如叠加和纠缠）来处理信息的计算范式，与经典计算机有本质不同。",
    "比特币是一种去中心化的数字货币，使用区块链技术记录交易，由中本聪于2008年提出。",
    "气候变化导致全球平均温度上升，海平面升高，极端天气事件增多，对生态系统造成严重威胁。",
    "深度学习通过多层神经网络从数据中自动提取特征，广泛应用于图像识别、自然语言处理等领域。",
    "CRISPR-Cas9基因编辑技术允许科学家精确修改DNA序列，为遗传疾病治疗带来革命性突破。",
    "黑洞是时空中引力极强的区域，连光都无法逃脱，由质量极大的恒星坍缩形成。",
    "人工智能在医疗诊断中的应用日益广泛，包括医学影像分析、疾病预测和个性化治疗方案制定。",
    "可再生能源太阳能发电通过光伏电池将太阳光直接转换为电能，是清洁能源的重要组成部分。",
    "新冠疫苗的研发采用了mRNA、腺病毒载体等多种新技术，并创下最快获批的疫苗纪录。",
    "火星探测任务目标是了解火星的地质历史、气候演变，以及探索是否存在生命迹象。",
    "大语言模型通过在海量文本上进行预训练，学习语言的统计规律，再经过微调适应具体任务。",
    "核聚变能源研究致力于实现氢同位素在高温高压下融合，释放巨大能量，有望成为终极清洁能源。",
    "自动驾驶技术依赖激光雷达、摄像头、GPS等多种传感器，结合深度学习算法实现无人驾驶。",
    "抗生素耐药性问题日益严峻，细菌通过基因突变和水平基因转移迅速获得对抗生素的耐受能力。",
    "宇宙暗物质和暗能量占宇宙总质量能量的约95%，但目前仍无法直接探测或理解其本质。",
    "量子纠缠是两个或多个粒子之间的一种特殊关联，即使相距遥远，对其中一个的测量会影响另一个。",
    "5G通信技术具有低延迟、高带宽特点，将推动物联网、工业自动化、远程医疗等领域的发展。",
    "蛋白质折叠问题由AlphaFold人工智能系统得到突破性解决，对药物研发和生命科学意义重大。",
    "区块链技术提供去中心化的不可篡改账本，除加密货币外还用于供应链管理、数字版权保护。",
    "神经科学研究揭示人类大脑有约860亿个神经元，通过突触连接形成极其复杂的神经网络。",
    "太阳能电池板使用硅晶体等半导体材料，通过光电效应将光子能量转换为电子流产生电能。",
    "机器学习中的过拟合问题指模型对训练数据过度拟合，导致在新数据上泛化能力差的现象。",
    "基因组学通过高通量测序技术分析生物体的完整基因组，为个性化医疗和进化研究提供基础。",
    "量子密码学利用量子力学原理实现理论上无法被破解的加密通信，可探测任何窃听行为。",
    "人工智能伦理问题包括算法偏见、隐私保护、就业影响和自主武器等多个重要议题。",
    "飞秒激光技术可产生极短脉冲光，用于超精密材料加工、眼科手术和超快光谱学研究。",
    "纳米技术在医药领域的应用包括纳米粒子靶向给药系统，可精确将药物输送至癌细胞。",
    "空间引力波探测通过观测时空涟漪，为验证广义相对论和研究黑洞并合提供新窗口。",
    "脑机接口技术通过采集和解码神经信号，使瘫痪患者可以用思维控制外部设备或假肢。",
    "电动汽车的核心挑战包括电池能量密度、充电速度、续航里程和电网容量等多个技术问题。",
]

# ─── 工具函数 ────────────────────────────────────────────

def _stats(times_ms: list[float]) -> dict:
    return {
        "mean":  statistics.mean(times_ms),
        "p50":   statistics.median(times_ms),
        "p90":   sorted(times_ms)[int(len(times_ms) * 0.9)],
        "min":   min(times_ms),
        "max":   max(times_ms),
    }


def _fmt(stats: dict) -> str:
    return (f"mean={stats['mean']:6.1f}  p50={stats['p50']:6.1f}"
            f"  p90={stats['p90']:6.1f}  min={stats['min']:6.1f}  max={stats['max']:6.1f}  ms")


def _print_comparison(label: str, pt_times: list[float], onnx_times: list[float]) -> None:
    pt_s   = _stats(pt_times)
    onnx_s = _stats(onnx_times)
    speedup = pt_s["mean"] / onnx_s["mean"] if onnx_s["mean"] > 0 else float("inf")
    print(f"  {label}")
    print(f"    PyTorch  : {_fmt(pt_s)}")
    print(f"    ONNX FP16: {_fmt(onnx_s)}")
    sign = "faster" if speedup >= 1 else "slower"
    ratio = speedup if speedup >= 1 else 1 / speedup
    print(f"    加速比    : ONNX {ratio:.2f}x {sign} than PyTorch")


# ─── PyTorch 后端 ────────────────────────────────────────

def load_pytorch_models():
    import torch
    from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
    device = "mps:0" if torch.backends.mps.is_available() else "cpu"
    print(f"[PyTorch] device = {device}")

    # 正确的 FP16 加载顺序：FP32 加载 → eval() → to(device) → half()
    # 错误顺序 use_fp16=True + to(mps) 会触发 "Placeholder storage has not been allocated on MPS device!"
    embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH)
    embed_model.eval()
    embed_model = embed_model.to(device)
    if device != "cpu":
        embed_model = embed_model.half()
    embed_tok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)

    rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_PATH)
    rerank_model.eval()
    rerank_model = rerank_model.to(device)
    if device != "cpu":
        rerank_model = rerank_model.half()
    rerank_tok = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)

    # MPS 预热
    print("[PyTorch] 预热中...")
    with torch.no_grad():
        for seq_len in [32, 128, 256, 512]:
            dummy = "预热" * (seq_len // 2)
            inp = embed_tok([dummy], max_length=512, padding=True, truncation=True, return_tensors="pt")
            inp = {k: v.to(device) for k, v in inp.items()}
            _ = embed_model(**inp)

            pair = [[dummy, dummy]]
            inp2 = rerank_tok(pair, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inp2 = {k: v.to(device) for k, v in inp2.items()}
            _ = rerank_model(**inp2)
    if device == "mps:0":
        torch.mps.synchronize()
    print("[PyTorch] 预热完成\n")
    return embed_model, embed_tok, rerank_model, rerank_tok, device


def bench_pytorch_embed(embed_model, embed_tok, device, n_runs: int):
    import torch
    times = {"tokenize": [], "transfer": [], "forward": [], "postprocess": [], "total": []}

    for _ in range(n_runs):
        t1 = time.perf_counter()
        inputs = embed_tok([QUERY], max_length=512, padding=True, truncation=True, return_tensors="pt")
        t2 = time.perf_counter()

        inputs_dev = {k: v.to(device) for k, v in inputs.items()}
        if device == "mps:0":
            torch.mps.synchronize()
        t3 = time.perf_counter()

        with torch.no_grad():
            outputs = embed_model(**inputs_dev)
        if device == "mps:0":
            torch.mps.synchronize()
        t4 = time.perf_counter()

        cls_hidden = outputs.last_hidden_state[:, 0]
        embedding  = torch.nn.functional.normalize(cls_hidden, p=2, dim=-1)
        _ = embedding[0].cpu().float().tolist()
        if device == "mps:0":
            torch.mps.synchronize()
        t5 = time.perf_counter()

        times["tokenize"].append((t2 - t1) * 1000)
        times["transfer"].append((t3 - t2) * 1000)
        times["forward"].append((t4 - t3) * 1000)
        times["postprocess"].append((t5 - t4) * 1000)
        times["total"].append((t5 - t1) * 1000)

    return times


def bench_pytorch_rerank(rerank_model, rerank_tok, device, batch_size: int, n_runs: int):
    import torch
    pairs = [[QUERY, doc] for doc in DOCS[:batch_size]]
    times = {"tokenize": [], "transfer": [], "forward": [], "postprocess": [], "total": []}

    for _ in range(n_runs):
        t1 = time.perf_counter()
        inputs = rerank_tok(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        t2 = time.perf_counter()

        inputs_dev = {k: v.to(device) for k, v in inputs.items()}
        if device == "mps:0":
            torch.mps.synchronize()
        t3 = time.perf_counter()

        with torch.no_grad():
            outputs = rerank_model(**inputs_dev)
        if device == "mps:0":
            torch.mps.synchronize()
        t4 = time.perf_counter()

        scores = torch.sigmoid(outputs.logits).squeeze(-1).cpu().float().tolist()
        _ = scores
        if device == "mps:0":
            torch.mps.synchronize()
        t5 = time.perf_counter()

        times["tokenize"].append((t2 - t1) * 1000)
        times["transfer"].append((t3 - t2) * 1000)
        times["forward"].append((t4 - t3) * 1000)
        times["postprocess"].append((t5 - t4) * 1000)
        times["total"].append((t5 - t1) * 1000)

    return times


# ─── ONNX 后端 ───────────────────────────────────────────

def load_onnx_sessions():
    import onnxruntime as ort
    from transformers import AutoTokenizer

    for path in [EMBED_ONNX_FP16, RERANKER_ONNX_FP16]:
        if not os.path.exists(path):
            print(f"[ERROR] ONNX 模型不存在: {path}")
            sys.exit(1)

    providers = [
        ("CoreMLExecutionProvider", {"MLComputeUnits": "CPUAndGPU"}),
        "CPUExecutionProvider",
    ]
    available = ort.get_available_providers()
    if "CoreMLExecutionProvider" not in available:
        print("[ONNX] CoreML EP 不可用，回退到 CPU EP")
        providers = ["CPUExecutionProvider"]
    else:
        print("[ONNX] 使用 CoreML EP (CPUAndGPU)")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    embed_sess   = ort.InferenceSession(EMBED_ONNX_FP16,    sess_options=sess_opts, providers=providers)
    rerank_sess  = ort.InferenceSession(RERANKER_ONNX_FP16, sess_options=sess_opts, providers=providers)
    embed_tok    = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
    rerank_tok   = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)

    # 打印 embed 模型输出信息，确认 shape
    _inp = embed_tok([QUERY], max_length=32, padding=True, truncation=True, return_tensors="np")
    _out = embed_sess.run(None, dict(_inp))
    out_names = [o.name for o in embed_sess.get_outputs()]
    print(f"[ONNX] embed 输出: {[f'{n}{o.shape}' for n, o in zip(out_names, _out)]}")

    # ONNX 预热
    print("[ONNX] 预热中...")
    for _ in range(3):
        inp = embed_tok([QUERY], max_length=512, padding=True, truncation=True, return_tensors="np")
        _ = embed_sess.run(None, dict(inp))
        pair = [[QUERY, DOCS[0]]]
        inp2 = rerank_tok(pair, padding=True, truncation=True, max_length=512, return_tensors="np")
        _ = rerank_sess.run(None, dict(inp2))
    print("[ONNX] 预热完成\n")
    return embed_sess, embed_tok, rerank_sess, rerank_tok


def bench_onnx_embed(embed_sess, embed_tok, n_runs: int):
    times = {"tokenize": [], "transfer": [], "forward": [], "postprocess": [], "total": []}

    for _ in range(n_runs):
        t1 = time.perf_counter()
        inputs = embed_tok([QUERY], max_length=512, padding=True, truncation=True, return_tensors="np")
        t2 = time.perf_counter()

        feed = {k: np.ascontiguousarray(v) for k, v in inputs.items()}
        t3 = time.perf_counter()

        outputs = embed_sess.run(None, feed)
        t4 = time.perf_counter()

        # optimum export 输出 last_hidden_state: (1, seq_len, 1024)，取 CLS token
        # 若模型直接输出 sentence_embedding: (1, 1024)，则无需 [:, 0, :]
        raw = outputs[0]
        emb = raw[:, 0, :] if raw.ndim == 3 else raw
        norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        _ = norm[0].tolist()
        t5 = time.perf_counter()

        times["tokenize"].append((t2 - t1) * 1000)
        times["transfer"].append((t3 - t2) * 1000)
        times["forward"].append((t4 - t3) * 1000)
        times["postprocess"].append((t5 - t4) * 1000)
        times["total"].append((t5 - t1) * 1000)

    return times


def bench_onnx_rerank(rerank_sess, rerank_tok, batch_size: int, n_runs: int):
    pairs = [[QUERY, doc] for doc in DOCS[:batch_size]]
    times = {"tokenize": [], "transfer": [], "forward": [], "postprocess": [], "total": []}

    for _ in range(n_runs):
        t1 = time.perf_counter()
        inputs = rerank_tok(pairs, padding=True, truncation=True, max_length=512, return_tensors="np")
        t2 = time.perf_counter()

        feed = {k: np.ascontiguousarray(v) for k, v in inputs.items()}
        t3 = time.perf_counter()

        outputs = rerank_sess.run(None, feed)
        t4 = time.perf_counter()

        logits = outputs[0]
        scores = (1 / (1 + np.exp(-logits.squeeze(-1)))).tolist()
        _ = scores
        t5 = time.perf_counter()

        times["tokenize"].append((t2 - t1) * 1000)
        times["transfer"].append((t3 - t2) * 1000)
        times["forward"].append((t4 - t3) * 1000)
        times["postprocess"].append((t5 - t4) * 1000)
        times["total"].append((t5 - t1) * 1000)

    return times


# ─── 主流程 ─────────────────────────────────────────────

BATCH_SIZES = [5, 10, 20, 30]
STAGES      = ["tokenize", "transfer", "forward", "postprocess", "total"]


def main():
    print("=" * 70)
    print("  WikiRAG 推理延迟基准测试（严格串行：PyTorch 全部跑完再跑 ONNX）")
    print(f"  N_WARMUP={N_WARMUP}  N_RUNS={N_RUNS}")
    print("=" * 70)

    # ═══════════════════════════════════════════════════
    # 阶段一：PyTorch FP16 (MPS)
    # ═══════════════════════════════════════════════════
    print("\n[阶段 1/2] 加载 PyTorch 模型并完成所有测试...")
    pt_embed, pt_embed_tok, pt_rerank, pt_rerank_tok, device = load_pytorch_models()

    # embed 预热 + 测试
    for _ in range(N_WARMUP):
        bench_pytorch_embed(pt_embed, pt_embed_tok, device, n_runs=1)
    pt_et = bench_pytorch_embed(pt_embed, pt_embed_tok, device, N_RUNS)
    print(f"  embed done  (mean total={statistics.mean(pt_et['total']):.1f} ms)")

    # rerank 各 batch 预热 + 测试
    pt_rts = {}
    for bs in BATCH_SIZES:
        if bs > len(DOCS):
            continue
        for _ in range(N_WARMUP):
            bench_pytorch_rerank(pt_rerank, pt_rerank_tok, device, bs, n_runs=1)
        pt_rts[bs] = bench_pytorch_rerank(pt_rerank, pt_rerank_tok, device, bs, N_RUNS)
        print(f"  rerank b={bs} done  (mean total={statistics.mean(pt_rts[bs]['total']):.1f} ms)")

    # 显式释放 PyTorch 模型，腾出 GPU 显存
    import torch
    del pt_embed, pt_rerank
    if device == "mps:0":
        torch.mps.empty_cache()
    print("  PyTorch 模型已释放\n")

    # ═══════════════════════════════════════════════════
    # 阶段二：ONNX FP16 (CoreML CPUAndGPU EP)
    # ═══════════════════════════════════════════════════
    print("[阶段 2/2] 加载 ONNX 模型并完成所有测试...")
    onnx_embed_sess, onnx_embed_tok, onnx_rerank_sess, onnx_rerank_tok = load_onnx_sessions()

    # embed 预热 + 测试
    for _ in range(N_WARMUP):
        bench_onnx_embed(onnx_embed_sess, onnx_embed_tok, n_runs=1)
    onnx_et = bench_onnx_embed(onnx_embed_sess, onnx_embed_tok, N_RUNS)
    print(f"  embed done  (mean total={statistics.mean(onnx_et['total']):.1f} ms)")

    # rerank 各 batch 预热 + 测试
    onnx_rts = {}
    for bs in BATCH_SIZES:
        if bs > len(DOCS):
            continue
        for _ in range(N_WARMUP):
            bench_onnx_rerank(onnx_rerank_sess, onnx_rerank_tok, bs, n_runs=1)
        onnx_rts[bs] = bench_onnx_rerank(onnx_rerank_sess, onnx_rerank_tok, bs, N_RUNS)
        print(f"  rerank b={bs} done  (mean total={statistics.mean(onnx_rts[bs]['total']):.1f} ms)")

    del onnx_embed_sess, onnx_rerank_sess
    print("  ONNX 模型已释放\n")

    # ═══════════════════════════════════════════════════
    # 逐阶段详细对比
    # ═══════════════════════════════════════════════════
    print("═" * 70)
    print("  EMBEDDING（batch=1）逐阶段对比")
    print("═" * 70)
    for stage in STAGES:
        if stage == "total":
            print(f"  {'─'*68}")
        _print_comparison(f"embed.{stage}", pt_et[stage], onnx_et[stage])

    for bs in BATCH_SIZES:
        if bs not in pt_rts:
            continue
        print(f"\n{'═'*70}")
        print(f"  RERANKER（batch={bs}，content）逐阶段对比")
        print("═" * 70)
        for stage in STAGES:
            if stage == "total":
                print(f"  {'─'*68}")
            _print_comparison(f"rerank[{bs}].{stage}", pt_rts[bs][stage], onnx_rts[bs][stage])

    # ═══════════════════════════════════════════════════
    # 汇总表格
    # ═══════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print("  汇总（mean ms，N=%d）" % N_RUNS)
    print("═" * 70)
    print(f"  {'任务':<32} {'PyTorch FP16':>13} {'ONNX':>10} {'加速比':>10}")
    print(f"  {'-'*32} {'-'*13} {'-'*10} {'-'*10}")

    def _row(label, pt_t, onnx_t):
        pt_m   = statistics.mean(pt_t)
        onnx_m = statistics.mean(onnx_t)
        ratio  = pt_m / onnx_m if onnx_m > 0 else float("inf")
        sign   = "↑" if ratio >= 1 else "↓"
        r_disp = ratio if ratio >= 1 else 1 / ratio
        print(f"  {label:<32} {pt_m:>12.1f}  {onnx_m:>9.1f}  {sign}{r_disp:.2f}x")

    _row("embed (total)",        pt_et["total"],   onnx_et["total"])
    _row("embed (forward only)", pt_et["forward"], onnx_et["forward"])
    for bs in BATCH_SIZES:
        if bs not in pt_rts:
            continue
        _row(f"rerank b={bs} (total)",   pt_rts[bs]["total"],   onnx_rts[bs]["total"])
        _row(f"rerank b={bs} (forward)", pt_rts[bs]["forward"], onnx_rts[bs]["forward"])

    print()


if __name__ == "__main__":
    main()
