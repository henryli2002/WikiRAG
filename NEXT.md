# WikiRAG 下一步优化方向

> 写于 2026-04-10。基于当前状态：BGE-M3 向量召回 + ParadeDB BM25 召回 + BGE-Reranker-v2-M3 精排，
> 单并发 p50 ~3.5s，rerank 占 88%。

---

## 方向 A：ONNX + Apple CoreML 原生加速

### 动机

PyTorch MPS 后端虽然支持苹果芯片，但：

- Metal shader 在首次运行新 shape 时需要 JIT 编译（cold start 数百 ms）
- MPS 后端未完整支持所有算子，部分操作可能 fallback 到 CPU
- 无法利用 Apple Neural Engine（ANE）——M4 Pro 的 ANE 理论 INT8 峰值约 38 TOPS，完全闲置

> **关于 INT8 量化**：`bitsandbytes` / PyTorch 原生 INT8 量化**不支持 MPS**，在 Mac 上强制 fallback 到 CPU，反而更慢。M4 Pro GPU 对 FP16 已有原生硬件加速，直接量化 PyTorch 模型无收益。正确做法是在 **ONNX → CoreML 导出阶段**完成权重压缩，由 ANE 执行 INT8/FP16 混合精度推理。

ONNX Runtime + CoreML Execution Provider（EP）可以：

1. 将模型静态编译为 CoreML `.mlpackage`，一次性编译，不再有 JIT
2. 自动调度 ANE / GPU Metal，由 CoreML 自行决策最优路径
3. FP16 权重在 CoreML 导出阶段完成，充分利用 M4 Pro 的 FP16 加速单元

### 技术路径

```
HuggingFace 权重
      │
      ▼  optimum-cli export onnx
  model.onnx  (FP32)
      │
      ▼  onnxruntime.quantization
  model_int8.onnx
      │
      ▼  coremltools.convert
  model.mlpackage  ← Apple 原生格式
      │
      ▼  onnxruntime (CoreML EP)
  推理（ANE/GPU 自动调度）
```

#### 步骤 1：导出 ONNX

```bash
pip install optimum[exporters] onnxruntime coremltools

# 导出 embedding 模型（XLM-RoBERTa backbone）
optimum-cli export onnx \
  --model models/bge-m3 \
  --task feature-extraction \
  --opset 17 \
  models/bge-m3-onnx/

# 导出 reranker（序列分类）
optimum-cli export onnx \
  --model models/bge-reranker-v2-m3 \
  --task text-classification \
  --opset 17 \
  models/bge-reranker-onnx/
```

#### 步骤 2：转换为 CoreML FP16（Apple Silicon 的正确量化路径）

不走 ONNX INT8，直接用 `coremltools` 将 ONNX 模型转为 CoreML FP16，让 Metal GPU / ANE 处理：

```python
import coremltools as ct

# 从 ONNX 转为 CoreML，权重压缩到 FP16
mlmodel = ct.convert(
    "models/bge-reranker-onnx/model.onnx",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS14,
)
mlmodel.save("models/bge-reranker-onnx/model.mlpackage")
```

> **为什么不用 INT8**：CoreML 的 LLM 量化（`ct.optimize.coreml.linear_quantize_weights`）针对 LLM 权重优化，但 Transformer encoder 在 M 系芯片上 FP16 已经是硬件最优路径，INT8 在 Metal 上不一定更快，且精度损失更大。

#### 步骤 3：推理替换（main.py）

```python
import onnxruntime as ort

# 优先使用 CoreML EP（ANE），回退到 CPU
providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

embed_session = ort.InferenceSession(
    "models/bge-m3-onnx/model.onnx", providers=providers
)
rerank_session = ort.InferenceSession(
    "models/bge-reranker-onnx/model_int8.onnx", providers=providers
)

def embed_query_onnx(query: str) -> list[float]:
    inputs = embed_tokenizer([query], max_length=512, padding=True,
                              truncation=True, return_tensors='np')
    outputs = embed_session.run(None, dict(inputs))
    cls_hidden = outputs[0][:, 0, :]  # last_hidden_state CLS token
    norm = cls_hidden / (np.linalg.norm(cls_hidden, axis=1, keepdims=True) + 1e-9)
    return norm[0].tolist()

def rerank_onnx(query: str, docs: list[dict]) -> list[dict]:
    pairs = [[query, doc["content"][:300]] for doc in docs]
    inputs = rerank_tokenizer(pairs, padding=True, truncation=True,
                               max_length=512, return_tensors='np')
    logits = rerank_session.run(None, dict(inputs))[0]  # shape (n, 1)
    scores = 1 / (1 + np.exp(-logits.squeeze(-1)))      # sigmoid
    for i, doc in enumerate(docs):
        doc["score"] = float(scores[i])
    return sorted(docs, key=lambda x: x["score"], reverse=True)[:RERANK_TOP_K]
```

### 已知挑战

| 挑战 | 说明 | 应对 |
|------|------|------|
| 动态 shape | ONNX 默认静态，transformer 输入长度可变 | `optimum` 导出时加 `--no-post-process`，或用动态轴 |
| CoreML 算子覆盖 | attention mask 某些写法不支持 | 用 `optimum` 导出后跑 `ort.check_model` 验证 |
| BGE-M3 多输出头 | 原模型有 dense/sparse/colbert 三路输出 | 导出时只保留 `last_hidden_state`，CLS 即 dense vec |
| ANE 实际调度 | CoreML 不保证一定用 ANE | `instruments` 工具可查看 ANE 利用率 |

### 预期收益

| 模型 | PyTorch MPS（当前） | ONNX + CoreML | 加速比 |
|------|-------------------|---------------|--------|
| Embedding（单条） | ~60 ms（预热后） | ~15 ms | **4×** |
| Reranker（30对 × 300字） | ~750 ms（估算） | ~150 ms | **5×** |

> 数据基于同类模型在 M 系芯片上的公开 benchmark，实测结果可能不同。

---

## 方向 B：级联排序（Cascaded Retrieval）

### 动机

当前架构是**单级精排**：50 个 hybrid 候选 → 全部送入重型 cross-encoder。
问题在于：

- cross-encoder 的计算是 O(n) in candidates，候选数加倍则延迟加倍
- 召回 500+500=1000 个候选理论上能找到更好的结果，但直接送入重排器太贵

级联排序用**轻量模型快速剪枝**，再用**重型模型精挑**：

```
向量召回 500  ──┐
                ├──> 1000 候选  →  轻量排序  →  top-30  →  精排  →  top-5
BM25  召回 500 ──┘
```

### 轻量排序层的候选方案

#### 方案 B-1：BGE-M3 Late Interaction（ColBERT 风格）

BGE-M3 本身支持三种向量：dense / sparse / **colbert_vecs**（token 级别的多向量表示）。

ColBERT 的 MaxSim 评分：

```
score(q, d) = Σᵢ max_j cos(qᵢ, dⱼ)
```

- 查询和文档**分别**编码，不需要联合前向传播（比 cross-encoder 快几十倍）
- 比单纯 dense cosine 更细粒度（term-level 交互）

**关键问题：文档 colbert_vecs 的存储**

| 参数 | 估算 |
|------|------|
| 文档数 | 2,784,657 |
| 平均 token 数（截断后） | 256 |
| ColBERT 维度 | 1024（BGE-M3 原始）或 128（投影后） |
| FP16 存储（原始维度） | 2.7M × 256 × 1024 × 2B ≈ **1.4 TB** |
| FP16 存储（128 维投影） | 2.7M × 256 × 128 × 2B ≈ **177 GB** |

全量预存不现实。实用化路径：

1. **仅对召回的 1000 个候选做在线推理**：embed_model 对 1000 docs 做 batch forward，取 colbert_vecs，内存约 1000 × 256 × 128 × 2B = **65 MB**，可接受
2. 在线推理 1000 docs 的耗时：BGE-M3 batch=32 × 31 批次，估计 ~2s（MPS），需要 ONNX 加速配合

#### 方案 B-2：bge-reranker-base 作轻量精排

用 278M 参数的 `bge-reranker-base`（约为当前模型的 1/2 参数量）作第一阶段排序：

```
1000 候选  →  bge-reranker-base（截断至 100字）  →  top-30  →  bge-reranker-v2-m3（全文 300字）  →  top-5
```

理论耗时估算（ONNX INT8 后）：

| 阶段 | 候选数 | 截断 | 预期耗时 |
|------|--------|------|---------|
| bge-reranker-base | 1000 | 100字 | ~500 ms |
| bge-reranker-v2-m3 | 30 | 300字 | ~150 ms |
| **合计** | | | **~650 ms** |

#### 方案 B-3：Dense Re-scoring（最简单，零额外推理）

1000 候选的 dense embedding 已存在数据库中。直接从 DB 批量取回，与 query_vec 做 cosine sim，选 top-30 送精排。

```
1000 候选  →  DB 取 embedding（1000 × 1024 FP16）  →  cosine sim（CPU numpy）  →  top-30  →  精排
```

- 无需额外模型，纯向量运算 < 5ms
- 质量不如 ColBERT / cross-encoder，但比 RRF 更准（用 query 语义重新对 1000 个排序）
- 实现最简单，可作为基线验证大召回量的价值

### 推荐实施顺序

```
阶段 0（已完成）：50 召回 + 重型精排
阶段 1（方向 A）：ONNX 加速，降低精排延迟到 ~150ms
阶段 2（方向 B-3）：扩召回到 500+500，Dense Re-scoring 快速剪枝到 30
阶段 3（方向 B-1/2）：加入 ColBERT 或 bge-reranker-base 提升剪枝质量
```

### 级联架构代码草图

```python
# 配置
VECTOR_RECALL_K  = 500
BM25_RECALL_K    = 500
CASCADE_TOP_K    = 30    # 轻量排序后保留数
RERANK_TOP_K     = 10
FINAL_TOP_K      = 5

async def _do_search_cascaded(req):
    query = req.query
    loop = asyncio.get_event_loop()

    # 1. Embedding
    async with _embed_lock:
        query_vec = await loop.run_in_executor(thread_pool, embed_query, query)

    # 2. 大召回（并发）
    vec_results, bm25_results = await asyncio.gather(
        vector_recall(query_vec, VECTOR_RECALL_K),
        bm25_recall(query, BM25_RECALL_K),
    )
    vec_results, _ = vec_results
    bm25_results, _ = bm25_results

    # 3. RRF 合并 → 1000 候选
    merged = hybrid_merge(vec_results, bm25_results, k=1000)

    # 4. 轻量剪枝（Dense Re-scoring，无额外推理）
    doc_ids = [d["id"] for d in merged]
    rows = await db_pool.fetch(
        "SELECT id, embedding::text FROM wiki_documents WHERE id = ANY($1)", doc_ids
    )
    emb_map = {r["id"]: np.fromstring(r["embedding"].strip("[]"), sep=",") for r in rows}
    q = np.array(query_vec, dtype=np.float32)
    for doc in merged:
        v = emb_map.get(doc["id"], np.zeros(1024))
        doc["cascade_score"] = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
    cascade_top = sorted(merged, key=lambda x: x["cascade_score"], reverse=True)[:CASCADE_TOP_K]

    # 5. 重型精排
    async with _rerank_lock:
        reranked = await loop.run_in_executor(thread_pool, rerank, query, cascade_top)

    # 6. MMR
    final = await mmr_select(query_vec, reranked, req.top_k, MMR_LAMBDA)
    return final
```

---

## 对比总结

| 维度 | 当前 | 方向 A（ONNX） | 方向 B（级联） | A+B 组合 |
|------|------|--------------|--------------|---------|
| 单并发 p50 | ~3.5s | ~0.5s | ~2.0s | **~0.3s** |
| 召回候选数 | 50+50 | 50+50 | 500+500 | 500+500 |
| 实现复杂度 | 当前基线 | 中（模型导出） | 中（新 pipeline） | 高 |
| 主要风险 | — | CoreML 算子兼容 | 剪枝质量 | 两者叠加 |

**建议先做方向 A**：ONNX 导出不改变召回逻辑，风险低，且方向 B 的大召回也需要快速推理支撑，两者天然互补。
