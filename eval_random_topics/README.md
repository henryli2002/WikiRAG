# WikiRAG 评测框架

半自动化 RAG 评测框架，分三个递进阶段验证系统质量。所有脚本直接实例化底层检索模型，不经过 HTTP 接口，确保延迟数据反映纯算法耗时。

---

## 目录结构

```
eval/
├── retrieval_core.py        检索核心引擎（剥离 FastAPI 的独立版本）
│
├── build_golden_dataset.py  阶段一 Step 1：构建黄金测试集
├── run_retrieval_eval.py    阶段一 Step 2：检索评测，输出 Hit@K / MRR
├── dump_bad_cases.py        阶段一 Step 3：Bad Case 尸检（Human-in-the-Loop）
│
├── generate_answers.py      阶段二 Step 1：调用生成模型产出答案
├── judge_answers.py         阶段二 Step 2：Gemini 裁判打分
│
├── eval_report.py           全链路统一报告（随时可运行）
│
├── golden_dataset.csv       黄金测试集（Step 1 输出，人工精加工后作为基线标尺）
├── eval_results.csv         检索逐条结果
├── eval_summary.json        检索汇总指标
├── eval_answers.csv         生成答案 + 延迟统计
├── eval_judge_results.csv   Gemini 逐条评分
└── eval_judge_summary.json  生成质量汇总指标
```

---

## 依赖

```bash
pip install asyncpg FlagEmbedding openai google-generativeai python-dotenv
```

`.env` 文件需包含：

```
GOOGLE_API_KEY=...
DB_HOST=localhost
DB_PORT=5432
POSTGRES_DB=rag_db
POSTGRES_USER=rag_user
```

---

## 阶段一：检索质量验证（Retrieval Baseline）

> 目标：在完全剔除生成模型干扰的前提下，验证多路召回 + 重排的纯物理寻址能力。

### Step 1 — 构建黄金测试集

```bash
python eval/build_golden_dataset.py --n 200
```

**做了什么**

1. 从 `wiki_documents` 随机采样 200 个 Chunk（过滤纯英文 title 和重定向页）
2. 对每个 Chunk，随机分配 4 种提问视角之一（细节提取 / 因果推断 / 条件限制 / 口语化指代），调用 Gemini 生成 `generated_query`
3. 对约 30% 的 Chunk 再次调用 Gemini 进行对抗改造，生成难度更高的 `human_query`，并标记 `adversarial_type`

**输出：`golden_dataset.csv`**

| 列 | 含义 |
|----|------|
| `query_id` | 唯一编号 |
| `chunk_id` | 期望被召回的目标 Chunk（wiki_documents.id） |
| `chunk_title` | 来源文章标题（供人工快速定位） |
| `chunk_content` | Chunk 原文（供人工参考） |
| `generated_type` | 提问视角类型 |
| `generated_query` | LLM 生成的原始问题 |
| `human_query` | 经对抗改造的最终问题（未改造时 = generated_query） |
| `adversarial_type` | `synonym` / `missing_info` / `entity_ambiguity` / 空（对照组） |

**如何分析这个文件**

- `adversarial_type` 为空的条目是对照组，分布应占 ~70%
- 检查 `adversarial_type = entity_ambiguity` 的条目：改造后的问题必须仍然可以被 `chunk_content` 回答，若 LLM 把语义完全迁移到另一实体，需手动修正或删除
- 若 `human_query` 为空，评测时自动回退使用 `generated_query`

---

### Step 2 — 静默检索 + 绝对算分

```bash
python eval/run_retrieval_eval.py
```

**做了什么**

对每条 query 运行完整检索流水线（Embed → Vector Recall → BM25 Recall → RRF Merge → Rerank → MMR），记录目标 Chunk 在每个阶段的排名，计算 Hit@K 和 MRR。

与生产服务的关键差异：

- **不施加 `RERANK_MIN_HYBRID_SCORE` 过滤**，所有 merged 候选均进入精排，避免评测数据被提前截断
- **不施加 `SCORE_THRESHOLD` 过滤**，final 结果保留原始分数
- **无并发锁**，延迟数字反映纯算法耗时

**输出：`eval_results.csv`**

| 列 | 含义 |
|----|------|
| `vec_rank` | 目标 Chunk 在向量召回中的排名（空 = 未召回） |
| `bm25_rank` | 目标 Chunk 在 BM25 召回中的排名 |
| `merged_rank` | RRF 混合后的排名 |
| `reranked_rank` | Cross-encoder 精排后的排名 |
| `final_rank` | MMR 选择后的最终排名 |
| `hit_at_1/5/10` | 命中标记 |
| `reranker_inversion` | True = 死因 C 嫌疑（merged 前 10 但精排倒退 >5 位） |
| `embed/vector/bm25/merge/rerank/mmr_ms` | 各阶段独立延迟（ms） |
| `total_ms` | 端到端检索总延迟 |

**如何分析这个文件**

- `hit_at_5 = False` 的行是 Bad Case，需要进一步尸检（Step 3）
- 对比 `vec_rank` 和 `bm25_rank`：若某条 query 仅在 BM25 中召回（`vec_rank` 为空），说明该 query 词汇对 Embedding 不友好
- `reranked_rank` 与 `merged_rank` 差值过大的行，是 Reranker 倒挂嫌疑（死因 C）
- 按 `adversarial_type` 分组看 `hit_at_5` 均值，可定位哪类对抗手法对系统破坏最大

**输出：`eval_summary.json`**

```json
{
  "hit_at_1": 0.70,
  "hit_at_5": 0.82,         // 检索生死线，目标 ≥ 80%
  "hit_at_10": 0.86,
  "mrr_at_10": 0.7635,
  "reranker_inversion_count": 0,
  "bad_case_cause_counts": {"A": 6, "B": 3, "C": 0},
  "latency_p50_ms": 3249.1,
  "adversarial_hit5_by_type": { ... }
}
```

**决策阈值**

| Hit@5 | 结论 |
|-------|------|
| < 70% | 检索地基不稳，禁止进入阶段二，先做架构调整 |
| 70–80% | 有明显缺陷，运行 Step 3 尸检后按归因分类处理 |
| ≥ 80% | 通过生死线，可进入阶段二，同时按需处理剩余 Bad Case |

---

### Step 3 — Bad Case 尸检（Human-in-the-Loop）

```bash
# 打印全部死因
python eval/dump_bad_cases.py

# 只看 Reranker 倒挂（最需要人工对比文本）
python eval/dump_bad_cases.py --cause C --top-noise 3

# 输出到文件
python eval/dump_bad_cases.py --cause C --output eval/bad_cases_C.txt
```

**三种死因与对应决策**

| 死因 | 自动判定依据 | 建议行动 |
|------|------------|---------|
| **A 词汇鸿沟** | `vec_rank` 和 `bm25_rank` 均为空 | 引入 Query 重写；或更换对垂直领域更敏感的 Embedding 模型 |
| **B 切块碎裂** | 进了某召回路但 merged 或 final 后消失 | 扩大 Chunk Size + Overlap；或引入父子块架构 |
| **C Reranker 倒挂** | `reranker_inversion = True` | 对比 Query / Target / Noise 三者文本后决策：调 RRF K 值 / 降 Reranker 权重 / 微调 BGE-M3 |

**死因 C 的输出格式**

```
▶ QUERY        量子计算中的纠错码是如何工作的？
▶ TARGET       id=1042  量子纠错码通过冗余量子比特...
▶ NOISE [1]    id=0839  经典纠错码最早由香农...     ← Reranker 认为更相关
▶ NOISE [2]    id=2107  量子计算机的基本门操作...
```

脚本只能告诉你"发生了倒挂"。**只有人工阅读并对比三列文本**，才能判断是 Reranker 的领域误判还是 RRF 融合参数问题。

---

## 阶段二：生成质量验证（Generation Quality）

> 目标：在确认检索正确的前提下，验证生成模型是否严格遵循 Context 输出无幻觉答案。

### Step 1 — 生成答案

```bash
python eval/generate_answers.py
```

**为什么用 Qwen 9B 而不是强模型**

强模型（GPT-4、Claude Opus）本身已记忆了大量 Wiki 知识，即使检索失败也能靠参数知识答对，导致 Faithfulness 的测量失去区分度。Qwen 9B 更依赖 Context，幻觉风险更真实。

**输出：`eval_answers.csv`**

| 列 | 含义 |
|----|------|
| `n_chunks` | 实际送入 LLM 的 Chunk 数（= `FINAL_TOP_K`） |
| `context_chars` | Context 总字符数 |
| `retrieved_context` | 完整 Context 文本（供人工核查） |
| `generated_answer` | Qwen 9B 的回答 |
| `ttft_ms` | **首字延迟**：从发出请求到收到第一个 Token 的时间 |
| `gen_ms` | 完整生成墙钟时间（含 TTFT） |
| `prompt_tokens` | 输入 Token 数（system + context + query） |
| `completion_tokens` | 输出 Token 数 |
| `total_tokens` | 两者之和 |

**如何分析这个文件**

- `ttft_ms` 反映 llama.cpp 服务的响应性，是用户感知延迟的关键；`gen_ms - ttft_ms` 是后续 Token 流速耗时
- `prompt_tokens` 与 `n_chunks` 正相关，用于评估调大 `FINAL_TOP_K` 的成本
- 对比 `hit_at_5 = True` 和 `False` 的两组：若未命中时 `generated_answer` 仍然"听起来像正确答案"，说明模型在幻觉而非如实说"无法回答"
- `context_chars` 过大（> 2000）时检查是否触发 `max_chars_per_doc` 截断，截断会导致关键信息丢失

---

### Step 2 — Gemini 裁判打分

```bash
python eval/judge_answers.py
```

**评分方法（工业级原子分解，0–1 浮点）**

| 维度 | 方法 | 分值 |
|------|------|------|
| **Faithfulness** | 将答案拆解为原子声明（每个事实点 1 条），对照 Context 逐条 YES/NO 验证；分数 = 支持声明数 / 总声明数 | 0–1，越高越少幻觉 |
| **Answer Relevance** | 基于答案反向生成 3 个候选问题，每题与原始 query 相似度打 0–2 分；分数 = 平均相似度 / 2 | 0–1，越高越切题 |

每条答案对 Gemini 发出 **3 次 API 调用**：① 原子分解、② 批量验证（单次调用验证所有声明）、③ 反向生成 + 相似度打分。

**输出：`eval_judge_results.csv`**

| 列 | 含义 |
|----|------|
| `faithfulness_score` | 0–1 浮点（supported / total） |
| `faithfulness_claims_total` | 原子声明总数 |
| `faithfulness_claims_supported` | 被 Context 支持的声明数 |
| `faithfulness_details` | JSON，每条声明的 YES/NO 结果（人工抽检的核心依据） |
| `answer_relevance_score` | 0–1 浮点（avg sim / 2） |
| `answer_relevance_questions` | JSON，Gemini 生成的 3 个候选问题 + 相似度评分 |
| `overall_norm` | 两维度均值 |

**如何分析这个文件**

**最重要的操作：人工抽检 10%，对齐裁判标准。** 脚本末尾会自动列出 `faithfulness_score < 0.67` 且有未支持声明的条目。

对齐流程：
1. 阅读 `faithfulness_details` JSON — 找到 `YES: false` 的声明，判断 Gemini 是否误判
2. 若 Gemini 把 Context 中有隐含依据的信息标为 NO，修改 `judge_answers.py` 中的 `_VERIFY_PROMPT` 验证标准
3. 重新运行 `judge_answers.py`（不需要重跑 `generate_answers.py`，输入文件未变）
4. 重复直到 Gemini 打分逻辑与你的判断基本一致

**典型问题信号**

| 现象 | 可能原因 |
|------|---------|
| Faithfulness 低但 Relevance 高 | 模型用自身知识补充了 Context 没说的细节（幻觉，`faithfulness_details` 可定位具体声明） |
| Faithfulness 高但 Relevance 低 | 模型找到了 Context 中的相关句子但答非所问 |
| 检索未命中（`hit_at_5=False`）时分数却高 | 模型在凭参数知识答题，检索完全没起作用 |
| `faithfulness_claims_total = 0` | 答案为拒答（"根据所提供的资料，无法回答"），无原子声明 = 无幻觉，score=1.0，正常参与统计 |

**输出：`eval_judge_summary.json`**

```json
{
  "faithfulness_avg": 0.85,
  "faithfulness_ge067_pct": 0.90,    // ≥0.67 达标率（对应原子声明 2/3 支持），目标 > 85%
  "answer_relevance_avg": 0.78,
  "claims_total_avg": 4.2,           // 平均每条答案分解出的原子声明数
  "claims_supported_avg": 3.7,
  "by_retrieval_hit": {
    "hit":  { "faithfulness_avg": 0.91, "answer_relevance_avg": 0.83 },
    "miss": { "faithfulness_avg": 0.62, "answer_relevance_avg": 0.51 }
    // 命中 vs 未命中的分层对比，反映检索质量对生成质量的传导效应
  }
}
```

---

## 全链路报告

```bash
python eval/eval_report.py
```

读取同目录下的三个 CSV，输出统一的性能与质量表格，无需所有文件都存在（缺失部分自动跳过）。

**延迟表结构**

```
检索链路
  embed             → Query 向量化耗时
  vector recall     → pgvector HNSW 查询耗时
  bm25 recall       → pg_search / Tantivy 查询耗时
  recall wall       → max(vector, bm25)，并发时的实际等待时间
  rerank            → BGE-M3 Cross-encoder 批量推理耗时  ← 通常是瓶颈
  mmr               → Maximal Marginal Relevance 耗时
  total             → 以上之和

生成链路
  TTFT 首字         → 用户感知到"开始输出"的延迟
  generation        → 完整答案生成时间

Token 统计
  prompt tokens     → 随 n_chunks 和 context 长度线性增长
  completion tokens → 答案长度分布
  n_chunks          → 实际送入 LLM 的 Chunk 数

全链路端到端
  e2e total         → retrieval + generation，用户等待全部答案的总时间
```

---

## 多参数对比实验

不同配置用独立目录隔离，直接对比报告数字：

```bash
# baseline: FINAL_TOP_K=5, HYBRID_TOP_K=30
mkdir eval/baseline
cp eval/eval_results.csv eval/eval_answers.csv eval/eval_judge_results.csv eval/baseline/

# 调整参数后重跑
# 修改 app/main.py 中的 FINAL_TOP_K=7, HYBRID_TOP_K=35
mkdir eval/topk7
python eval/run_retrieval_eval.py --results eval/topk7/eval_results.csv ...

# 对比两份报告
python eval/eval_report.py --dir eval/baseline
python eval/eval_report.py --dir eval/topk7
```

**报告中值得重点对比的指标**

| 调参方向 | 关注指标 |
|---------|---------|
| 增大 `FINAL_TOP_K` | `prompt_tokens` P50、`gen_ms` P50、`hit_at_5` 变化 |
| 增大 `HYBRID_TOP_K` | `hit_at_5`（死因 B 是否减少）、`rerank_ms` 变化 |
| 调整 `MMR_LAMBDA` | `hit_at_5`、生成答案多样性（需人工判断） |
| 更换 Embedding 模型 | 死因 A 数量、`vec_rank` 分布 |
| 引入 Query 重写 | 死因 A 数量、`adversarial_hit5_by_type["synonym"]` |
