# WikiRAG 性能测试报告

## 测试概况

| 项目 | 4月9日（旧版） | 4月10日（新版） |
|------|--------------|----------------|
| 全文检索方案 | PostgreSQL tsvector + GIN + ts_rank | pg_search (ParadeDB / Tantivy) BM25 |
| 数据规模 | 2,787,678 行 | 2,784,657 行 |
| 测试工具 | scripts/stress_test.py | 同左 |
| 并发梯度 | 1 / 2 / 4 / 8 | 1 / 2 / 4 / 8 |
| 每梯度请求数 | 40 | 40 |
| 硬件 | MBP M4 Pro，推理设备 MPS | 同左 |

---

## 汇总数据对比

### p50 延迟（中位数）

| 并发 | 旧版 | 新版 | 变化 |
|------|------|------|------|
| 1 | 4803 ms | 4496 ms | **-6%** |
| 2 | 9273 ms | 10157 ms | +10% |
| 4 | 16238 ms | 19176 ms | +18% |
| 8 | 35589 ms | 39008 ms | +10% |

### p90 延迟

| 并发 | 旧版 | 新版 | 变化 |
|------|------|------|------|
| 1 | 9355 ms | 5960 ms | **-36%** |
| 2 | 15668 ms | 11997 ms | **-23%** |
| 4 | 22359 ms | 21901 ms | -2% |
| 8 | 42551 ms | 41206 ms | -3% |

### 最大延迟 / 错误率

| 并发 | 旧版 max | 新版 max | 旧版 err | 新版 err |
|------|---------|---------|---------|---------|
| 1 | 9725 ms | 6186 ms | 0/40 | 0/40 |
| 2 | 16620 ms | 13398 ms | 0/40 | 0/40 |
| 4 | 28180 ms | 22939 ms | **1/40** | 0/40 |
| 8 | 47514 ms | 41657 ms | 0/40 | 0/40 |

---

## 分析

### 1. 召回层：新版显著更稳定

旧版单并发下 **p50=4.8s vs p90=9.4s**，抖动接近 2 倍。根因是 tsvector 方案在某些查询上会走大范围 OR 扫描并对每行做 ts_rank 计算，代价不可控。

新版单并发 **p50=4.5s vs p90=6.0s**，抖动缩小到 1.3 倍，分布非常集中。这是 Tantivy BM25 索引的优势：IDF 在索引阶段已经计算好，查询只需做倒排合并，不需要对候选行逐行评分。

最直观的对比是 `深度学习卷积神经网络图像分类` 这条查询：

| 版本 | 耗时 | 命中数 |
|------|------|--------|
| 旧版（tsvector） | ~9300 ms | **0** |
| 新版（pg_search） | ~4200 ms | **5** |

旧版不仅慢，而且返回 0 条结果（BM25 路完全失效），导致后续 RRF 只靠向量单路。新版两路召回均正常，且速度快一倍。

### 2. 高并发层：reranker 是纯串行瓶颈

从 p50 延迟可以看到近乎完美的线性关系：

```
c=1 → ~4.5s
c=2 → ~10s  (≈ 2×)
c=4 → ~19s  (≈ 4×)
c=8 → ~39s  (≈ 8×)
```

这与 `_rerank_lock`（asyncio.Lock）的行为完全吻合：每个请求持锁期间阻塞其余所有请求，吞吐量被锁死在约 **0.2 RPS**，无论并发多高都不增加。

新版在 c=2/4/8 的 p50 略高于旧版，原因是新版单次 rerank 前的召回更完整（旧版部分查询 BM25 返回 0 条，候选更少，rerank 更快），所以总耗时反而略微上涨——这是质量提升的附带代价，属于正常现象。

### 3. RPS 上限

两个版本在所有并发梯度下 RPS 均为 **0.2**，即约 5 秒/请求。召回层（向量 + BM25 并发）总耗时 < 500ms，embedding ~200ms，剩余 ~4s 全部被 reranker 消耗。**召回层已不是瓶颈，reranker 占端到端延迟的 80%+。**

---

## Reranker 优化方向

### 方向 A：减少候选数（收益立竿见影）

当前流水线送入 reranker 的候选最多 32 条（`HYBRID_TOP_K`），过滤后实际约 20~30 条。BGE-Reranker-v2-M3 对每对 (query, doc) 做一次前向，batch 越大耗时越长。

```python
# main.py
HYBRID_TOP_K = 32   # → 可以降到 16
RERANK_TOP_K = 10   # → 可以降到 6~8
```

在召回质量足够好（新版 BM25 已大幅改善）的前提下，top-16 送 reranker 基本不影响最终结果，但 rerank 耗时可降 30~50%。

### 方向 B：换更小的 reranker 模型

BGE-Reranker-v2-M3 参数量约 560M，推理较重。可替换方案：

| 模型 | 参数量 | 特点 |
|------|--------|------|
| BAAI/bge-reranker-v2-m3（当前） | ~560M | 精度最高 |
| BAAI/bge-reranker-base | ~278M | 速度约快 2× |
| BAAI/bge-reranker-v2-minicpm-layerwise | ~2.4B 但支持 early exit | 可配置推理层数，灵活权衡 |

对于中文维基问答场景，`bge-reranker-base` 的精度损失通常在可接受范围内，但 rerank 耗时可减半。

### 方向 C：量化推理

> **注意（Apple Silicon 用户）**：`bitsandbytes` INT8 量化**不支持 MPS 后端**，在 Mac 上运行会强制 fallback 到 CPU，反而更慢。M4 Pro 的 GPU 对 FP16 已有原生加速，直接量化 PyTorch 模型在 MPS 上几乎没有收益。
>
> 在 Apple Silicon 上可行的量化路径是 **CoreML + ONNX Runtime**，在导出阶段完成 INT8/FP16 权重压缩，由 ANE 执行（见方向 A）。

### 方向 D：请求合并批处理（高并发场景）

当前每个请求独立持锁推理。在并发>=2 时，队列中的请求在等待期间什么都不做。可以改为将排队请求的 pairs **合并成一个大 batch** 一起推理：

```
当前:  [req1: 20 pairs] → rerank → [req2: 25 pairs] → rerank  (串行)
合并:  [req1+req2: 45 pairs] → 单次 rerank → 拆分结果         (吞吐提升)
```

这需要改造锁机制（用 asyncio.Queue + 后台 worker），实现较复杂，但在 c>=4 时吞吐量理论上可提升 2~3×。

### 方向 E：缓存（适合部分场景）

如果查询有重复率（如接入 LLM agent 的检索、固定问题集），可在 reranker 前加 LRU 缓存：

```python
from functools import lru_cache

@lru_cache(maxsize=256)
def rerank_cached(query: str, doc_ids_tuple: tuple) -> list[float]:
    ...
```

对于 Wikipedia QA 这类开放域场景重复率低，收益有限，更适合垂直领域或 FAQ 场景。

---

## 建议优先级

| 优先级 | 方向 | 难度 | 预期收益 |
|--------|------|------|---------|
| ★★★ | A：减少候选数 | 改 2 行常量 | rerank 耗时 -30~50% |
| ★★★ | B：换 bge-reranker-base | 替换模型文件 | 单次 rerank -50% |
| ★★ | C：INT8 量化 | 改模型加载参数 | 单次 rerank -30~40% |
| ★ | D：批处理合并 | 需重构锁机制 | 高并发吞吐 2~3× |
| ☆ | E：结果缓存 | 中等 | 仅重复查询有效 |

**最快的起手式**：先降 `HYBRID_TOP_K` 到 16，同时换 `bge-reranker-base` 跑一轮对比测试，预计单并发 p50 可从 4.5s 降到 2s 以内。

---

## 4月10日 推理层重构实验

### 变更摘要

| 变更项 | 改动前 | 改动后 |
|--------|--------|--------|
| Embedding 推理 | `BGEM3FlagModel.encode()` 封装 | 直接调用底层 `AutoModel`（XLM-RoBERTa）+ tokenizer |
| Reranker 推理 | `FlagReranker.compute_score()` 封装 | 直接调用底层 `AutoModelForSequenceClassification` + tokenizer |
| Reranker content 截断 | 无（全文送入，tokenizer 自动截断至 512 token） | 显式截断至 **300 字符**后再 tokenize |
| MPS kernel 预热 | 无 | lifespan 内用 4 种 seq_len（32/128/256/512）预跑 embed+rerank |
| `HYBRID_TOP_K` | 50 | **30**（用户根据 recall_rank_analysis 调整） |
| 各阶段 timeout | 5s / 5s / 100s | **1s / 1s / 1s** |

### 单条查询实测数据（Query: "伪造川普视频"）

| 阶段 | 耗时 | 说明 |
|------|------|------|
| Embedding | **361 ms** | MPS cold start；预热后预期降至 ~60 ms |
| 向量召回 | 29 ms | 50条 |
| BM25 召回 | 45 ms | 50条（OR fallback） |
| Hybrid merge | 0 ms | RRF 纯 CPU |
| Rerank | **3125 ms** | 50 对，全文未截断时 padding 到 512 token |
| MMR | 19 ms | |
| **总计** | **3553 ms** | |

### content 截断对 Rerank 耗时的理论影响

Cross-encoder attention 复杂度为 O(L²)。中文字符约 1字≈1 token：

| content 截断长度 | 有效序列长度（估算） | 相对于 512 的算力比 | 预期耗时（50对） |
|-----------------|-------------------|-------------------|----------------|
| 无截断（当前实测） | ~512 token | 100% | ~3125 ms |
| 300 字 | ~320 token | 39% | ~1220 ms |
| 200 字 | ~220 token | 18% | ~560 ms |
| 150 字 | ~170 token | 11% | ~340 ms |

实际收益还取决于 batch 内最长文档的长度（padding 机制）。

### 主要结论

1. **Embedding 361ms 的根因是 MPS shader cold start**，而非模型本身慢。预热脚本在 lifespan 内以 4 种长度分别跑一次 forward，可消除首次请求的编译延迟。

2. **Rerank 是绝对瓶颈**：50 对 × ~512 token padding = 3.1s，占端到端 88%。截断 content 至 300 字后预期降至 ~1.2s；再将 `HYBRID_TOP_K` 降至 30 后送入 rerank 的候选数也随之减少。

3. **直接调用底层模型与封装方法在速度上无差异**（Python 线程共享内存，无跨线程拷贝），主要收益是推理过程透明可控（可自定义截断、精度、batch 策略）。

4. **采用父子数据块的结构可能可以改善次问题** 要在数据嵌入之前就设计好，目前成本较高推迟此计划了。

---

## 4月10日 PyTorch FP16 (MPS) vs ONNX (CPU EP) 推理基准测试

### 测试环境

| 项目 | 值 |
|------|-----|
| 硬件 | MBP M4 Pro |
| PyTorch 后端 | MPS (Metal GPU)，FP16 |
| ONNX 后端 | CPU EP（CoreML EP 因 FP32 模型触发 SystemError:20 自动回退） |
| 模型格式 | ONNX FP32（`model.onnx`，optimum 导出） |
| Reranker 输入 | **全文 content**（未截断，tokenizer 自动截断至 512 token） |
| 测试方法 | 严格串行：PyTorch 全部完成 → 释放 GPU 显存 → 加载 ONNX |
| N_WARMUP / N_RUNS | 5 / 30 |

> **注**：CoreML EP 对 FP32 ONNX 模型触发 `SystemError: 20`（ANE 仅支持量化模型，Metal GPU 通路也不兼容当前图结构），自动回退至 CPU EP。下表 ONNX 列实为 CPU 单线程推理，并非 GPU 加速路径。

### Embedding（batch=1）

| 阶段 | PyTorch FP16 MPS | ONNX CPU EP | 对比 |
|------|-----------------|-------------|------|
| tokenize | 0.1 ms | 0.1 ms | 持平 |
| device transfer | 0.3 ms | 0.0 ms | — |
| **forward** | **8.3 ms** | **17.4 ms** | PyTorch 快 **2.1×** |
| postprocess | 0.3 ms | 0.0 ms | — |
| **total** | **9.1 ms** | **17.5 ms** | PyTorch 快 **1.9×** |

### Reranker（各 batch size，全文 content，tokenizer 截断至 512 token）

| batch | PyTorch FP16 MPS (total) | ONNX CPU EP (total) | 对比 |
|-------|--------------------------|---------------------|------|
| 5 | 38.4 ms | 128.4 ms | PyTorch 快 **3.3×** |
| 10 | 66.1 ms | 253.6 ms | PyTorch 快 **3.8×** |
| 20 | 119.0 ms | 588.5 ms | PyTorch 快 **5.0×** |
| 30 | 176.7 ms | 833.2 ms | PyTorch 快 **4.7×** |

Reranker forward 单独对比（去除 tokenize/transfer 开销）：

| batch | PyTorch forward | ONNX forward | 对比 |
|-------|----------------|--------------|------|
| 5 | 36.9 ms | 127.9 ms | PyTorch 快 **3.5×** |
| 10 | 64.9 ms | 252.9 ms | PyTorch 快 **3.9×** |
| 20 | 117.7 ms | 587.5 ms | PyTorch 快 **5.0×** |
| 30 | 175.3 ms | 831.7 ms | PyTorch 快 **4.7×** |

> 注：测试文档为约 30~80 字的中文短句，全文送入 tokenizer 后序列长度较短（远低于 512），实际 batch padding 长度偏低。生产场景下 content 更长（向 512 token 逼近），reranker 耗时会显著更高，两者差距也可能进一步拉大。

### 结论

1. **PyTorch FP16 (MPS) 全面碾压 ONNX CPU EP**：forward 阶段快 2~5×，这是 Metal GPU vs CPU 单线程的硬件差距，与 ONNX 格式本身无关。

2. **CoreML EP 无法使用的根因**：导出的 ONNX 模型为 FP32，CoreML EP 在 `CPUAndGPU` 模式下尝试将部分算子调度到 Metal GPU 时失败（`SystemError: 20`）。正确路径是先完成 FP16 转换（`convert_onnx_fp16.py`），再测试 CoreML EP。

3. **当前最优方案仍是 PyTorch FP16 (MPS)**：预热后 embed ~9ms、rerank(30) ~177ms，已是当前硬件的实际极限。

4. **下一步**：放弃对于模型的优化，延迟来源于硬件物理限制和python通信延迟，mlx可以解决后者但是开发成本较高



## 4月11日 RAG 召回效果测试

---

### 整体指标

| 指标 | 数值 | 判断 |
|------|------|------|
| Hit@5 | 82% | 超过 80% 生死线，可以进阶段二 |
| Hit@1 | 70% | 精排排第一的能力还有提升空间 |
| MRR@10 | 0.764 | 健康 |
| Reranker 倒挂 | 0 条 | BGE-M3 Reranker 本身没问题，不需要动 |
| Rerank P50 延迟 | 3180ms | 严重问题，几乎占满全链路耗时 |

---

### 9 条 Bad Case 归因

#### 死因 A（6 条）—— 词汇鸿沟，两路都没召回

全是 Implicit 类问题，用了代词或间接指代：

- "也不干的千户叫什么名字"
- "那个敲钟仪式"
- "弹劾魏忠贤被罢官的那个人"

Embedding 对这类口语化指代完全失效，BM25 也找不到关键词。

#### 死因 B（3 条）—— 切到了但被截断

- **qid=22 和 qid=26**：目标 chunk 进了 reranked（排名 5 和 7），但 final_k=5 + MMR 把它踢出去了，不是召回的问题，是 MMR 把它挤掉了。

- **qid=32**：BM25 rank=36，刚好超出 HYBRID_TOP_K=30 的截断线。
  
