# WikiRAG: 中文维基百科检索增强生成系统

基于中文维基百科全量数据构建的 RAG (Retrieval-Augmented Generation) 检索系统。从原始 Wikipedia dump 出发，经过数据清洗、向量化、入库，最终提供多路召回 + 精排 + 去重的高质量检索 API。

## 系统架构

<div align="center">

```
┌─────────────────────────────────────────────────────────┐
│                     数据处理流水线                         │
│                                                         │
│  Wikipedia XML/SQL                                      │
│        │                                                │
│        ▼                                                │
│  auto_wiki_parser.py ── XML 解析 + 繁简转换 + 分类提取    │
│        │                                                │
│        ▼                                                │
│  clean_brackets.py ──── 噪音清洗 + 标记去除               │
│        │                                                │
│        ▼                                                │
│  embed_factory.py ───── 文本分块 + BGE-M3 向量化 (FP16)   │
│        │                                                │
│        ▼                                                │
│  ingest_to_postgres.py ─ 多进程分词 + COPY 批量入库        │
│        │                                                │
│        ▼                                                │
│  build_index.py ──────── tsvector + GIN + 向量索引构建     │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     检索 API 流水线                       │
│                                                         │
│                      用户 Query                          │
│                         │                               │
│                         ▼                               │
│              BGE-M3 Embedding (1024 维)                  │
│                         │                               │
│             ┌───────────┴───────────┐                   │
│             ▼                       ▼                   │
│      向量召回 (40 条)         BM25 召回 (30 条)           │
│      pgvector 余弦距离        simple + ts_rank           │
│             │                       │                   │
│             └───────────┬───────────┘                   │
│                         ▼                               │
│          RRF 混合排分 (Reciprocal Rank Fusion)           │
│                      → top 50                           │
│                         │                               │
│                         ▼                               │
│            Cross-Encoder 精排 (BGE-Reranker)             │
│                      → top 15                           │
│                         │                               │
│                         ▼                               │
│              MMR 去重 (多样性筛选)                        │
│                      → top 5                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

</div>

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 向量模型 | BAAI/bge-m3 | 1024 维, FP16 推理 |
| 精排模型 | BAAI/bge-reranker-v2-m3 | Cross-Encoder |
| 向量数据库 | PostgreSQL + pgvector | 支持 HNSW / IVFFlat 索引 |
| 全文检索 | PostgreSQL ts_rank + GIN | jieba 分词 + simple 字典，标题 A 权重 / 内容 B 权重 |
| 文本分块 | RecursiveCharacterTextSplitter | 512 字符, 60 字符重叠 |
| 中文分词 | jieba | 多进程并行分词 |
| API 框架 | FastAPI + Uvicorn | |
| 繁简转换 | OpenCC | |

## 目录结构

```
WikiRAG/
├── app/
│   └── main.py                  # FastAPI 检索 API
├── scripts/
│   ├── auto_wiki_parser.py      # 步骤 2: XML 解析
│   ├── clean_brackets.py        # 步骤 3: 噪音清洗
│   ├── embed_factory.py         # 步骤 5: 向量化
│   ├── ingest_to_postgres.py    # 步骤 6: 批量入库
│   └── build_index.py           # 步骤 7: 索引构建
├── postgres/
│   └── init.sql                 # 数据库初始化 DDL
├── models/                      # 模型存放目录 (git ignored)
│   ├── bge-m3/
│   └── bge-reranker-v2-m3/
├── data/                        # 数据目录 (git ignored)
│   ├── raw/                     # 原始 & 清洗后的 JSONL
│   ├── parquet_output/          # 向量化后的 Parquet 文件
│   └── pgdata/                  # PostgreSQL 数据卷挂载
├── tests/                       # 单元测试
├── docker-compose.yml
├── download_embedding_model.py
├── requirements.txt
└── .env
```

## 快速开始

### 环境准备

```bash
# 克隆项目
git clone <repo-url> && cd WikiRAG

# 安装依赖
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 启动数据库
docker-compose up -d

# 下载模型 (BGE-M3 + BGE-Reranker)
python download_embedding_model.py
```

### 数据处理流水线

详细步骤见 [`scripts/README.md`](scripts/README.md)。

```bash
# 1. 准备原始数据: 将 zhwiki dump 放入 data/raw/

# 2. XML 解析 → JSONL
python scripts/auto_wiki_parser.py

# 3. 噪音清洗
python scripts/clean_brackets.py

# 4. 向量化 (耗时较长，支持断点续传)
python scripts/embed_factory.py

# 5. 批量入库 (多进程分词 + COPY, MBP M4 Pro / Colima 6GB 约 30 分钟)
python scripts/ingest_to_postgres.py

# 6. 构建索引
python scripts/build_index.py
```

### 启动检索 API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 服务使用指南

#### 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/search` | 检索接口 |
| GET | `/health` | 健康检查 |

#### 搜索请求

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "量子计算的基本原理", "top_k": 5}'
```

**请求参数**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| query | string | 是 | - | 检索查询文本 |
| top_k | int | 否 | 5 | 返回结果数 (1~20) |

**响应示例**

```json
{
  "query": "量子计算的基本原理",
  "results": [
    {
      "id": 2586222,
      "pageid": "12345",
      "title": "量子计算",
      "content": "条目：量子计算\n内容：量子计算是一种利用量子力学现象...",
      "score": 0.9231
    }
  ]
}
```

**响应字段**

| 字段 | 说明 |
|------|------|
| id | 数据库主键 |
| pageid | 维基百科页面 ID |
| title | 条目标题 |
| content | 检索到的文本块 |
| score | 精排分数 (0~1, 越高越相关) |


## 实验细节

### 1. 数据处理

| 指标 | 数值 |
|------|------|
| 原始数据 | 中文维基百科全量 dump |
| 清洗后条目数 | 1,415,455 |
| 文本分块 | 512 字符 / 60 字符重叠 |
| 分块语义增强 | 每个 chunk 前拼接 `条目：{title}\n内容：` |
| 向量化耗时 | ~18 小时 42 分钟 (RTX 3070 Laptop 140W, 8GB VRAM) |
| 向量化速度 | 21.02 条目/秒 |
| 最大显存占用 | ~7.66 GB |
| 输出 Parquet | 218 个文件 |
| 向量精度 | float16 (存储 & 推理) |
| 数据库导入耗时 | ~30 分钟 |

> 以下索引构建耗时基于 **MBP M4 Pro, Colima 6GB 内存 / 4 CPU** 环境测得。

| 索引构建阶段 | 耗时 |
|-------------|------|
| 构建 tsvector (title=A, content=B) | 7 分 7 秒 |
| 清理临时分词列 | < 1 秒 |
| metadata GIN 索引 | 1 分 10 秒 |
| tsv GIN 索引 | 4 分 0 秒 |
| HNSW 向量索引 (fp16, m=16, ef=64) | 待补充 |

### 2. 数据库表结构

```sql
CREATE TABLE wiki_documents (
    id         BIGSERIAL PRIMARY KEY,
    content    TEXT NOT NULL,          -- 分块后的文本 (含标题前缀)
    metadata   JSONB NOT NULL,         -- {"pageid": ..., "title": ...}
    embedding  VECTOR(1024),           -- BGE-M3 稠密向量
    tsv        TSVECTOR                -- 全文检索向量 (title=A, content=B)
);
```

### 3. 全文检索方案: "瞒天过海"法

无需安装 zhparser 等中文分词扩展。核心思路：**Python 端分词，PostgreSQL 端当英文处理**。

- **入库时**: jieba 分词 → 空格连接 → 存入临时列 → `to_tsvector('simple', ...)` 生成 tsvector
- **标题权重 A** (1.0), **内容权重 B** (0.4): `setweight(..., 'A') || setweight(..., 'B')`
- **查询时**: jieba 分词 → OR 连接 → `to_tsquery('simple', ...)` → GIN 索引检索 + `ts_rank` 排序
- **优势**: 白嫖 PostgreSQL 底层 C 语言 BM25 检索和 GIN 索引，零 Docker 镜像修改

### 4. 向量索引实验

#### 量化精度对比

以原始向量 `[0.1234, -0.5678, 0.9012, -0.0034, 0.4567]` 为例，各精度的映射方式：

| 量化方式 | 编码格式 | 每维占用 | 索引大小 (相对) | 精度损失 | 适用场景 |
|---------|---------|---------|----------------|---------|---------|
| **fp32** | 1 符号 + 8 指数 + 23 尾数 | 4 bytes | 1x (基准) | 无 | 精度要求极高 |
| **fp16** (默认) | 1 符号 + 5 指数 + 10 尾数 | 2 bytes | 0.5x | 几乎无损 | **推荐，本项目嵌入原始精度即为 FP16** |
| int8 | 线性映射到 [-128, 127] | 1 byte | 0.25x | 轻微 | pgvector 不支持 |
| **bit** | 正→1, 负→0 | 1/8 byte | 0.03x | 较大 | 粗筛/候选集预过滤 |

```
原始 FP16 向量:  [ 0.1234,  -0.5678,   0.9012,  -0.0034,   0.4567]
                     │         │          │         │          │
fp32 映射:       [ 0.12340,  -0.56780,  0.90120, -0.00340,  0.45670]
                 精确到小数点后 7 位，IEEE 754 单精度浮点数

fp16 映射:       [ 0.1234,   -0.5678,   0.9012,  -0.0034,   0.4567]
                 精确到小数点后 3~4 位，与存储一致，零损失

int8 映射:       [   89,      -128,       127,     -87,        83  ]
(pgvector        线性缩放: val → round((val - min) / (max - min) * 255) - 128
 不支持)         min=-0.5678, max=0.9012, scale=0.00576

bit  映射:       [    1,         0,         1,        0,          1 ]
                 只保留符号位: 正数→1, 负数→0, 用汉明距离计算相似度
```

> **注**: pgvector (0.8.x) 不支持 INT8 标量量化。如需 INT8 精度，可考虑 Qdrant、Milvus 等专用向量数据库。本项目嵌入本身为 FP16，使用 fp16 索引即为零损失。

#### 索引配置命令

`build_index.py` 支持多种索引配置，可共存对比：

| 方案 | 命令 | 特点 |
|------|------|------|
| HNSW + FP16 (默认) | `python scripts/build_index.py` | 推荐，精度无损 |
| HNSW + FP32 | `--quantize fp32` | 升精度存索引，无实际收益 |
| HNSW + 二值量化 | `--quantize bit` | 索引体积极小，精度有损 |
| HNSW 高精度 | `--m 32 --ef-construction 64` | 更高召回率，构建更慢 |
| IVFFlat + FP16 | `--method ivfflat` | 构建快，查询稍慢 |
| IVFFlat 细分 | `--method ivfflat --lists 2000` | 更多聚类中心 |
| 降维 | `--dim 512` | 截取前 N 维，索引更小 |

索引名自动编码配置参数（如 `idx_emb_hnsw_fp16_1024_m16_ef32`），不同配置可共存。检索 API 会自动检测 embedding 列类型，查询时自动匹配。

查看已建索引及大小：
```sql
SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass))
FROM pg_indexes WHERE tablename = 'wiki_documents' AND indexname LIKE 'idx_emb_%';
```

### 5. 检索流水线参数

| 阶段 | 参数 | 说明 |
|------|------|------|
| Query Embedding | BGE-M3 FP16 | 与存储精度一致，查询时自动适配索引类型 |
| 向量召回 | top 40 | pgvector 余弦距离，自动匹配 vector/halfvec/bit |
| BM25 召回 | top 30 | PostgreSQL ts_rank + GIN 索引 |
| 混合排分 | RRF, k=60 | `score = Σ 1/(rank + 60)`, 取并集 top 50 |
| 精排 | BGE-Reranker-v2-M3 | Cross-Encoder 逐对打分, 取 top 15 |
| MMR 去重 | λ=0.7, top 5 | `score = λ·relevance - (1-λ)·max_sim` |

### 6. 安全机制

- **文件锁**: `ingest_to_postgres.py` 成功后写入 `data/.ingest.lock`，防止误操作删库重建。需手动 `rm data/.ingest.lock` 才能重新导入
- **断点续传**: `embed_factory.py` 通过 checkpoint 文件支持中断恢复
- **LOGGED 表**: 入库直接使用 LOGGED 表，确保数据持久化，不因容器异常重启丢失数据

## 测试

```bash
# 全部测试
pytest tests/ -v

# 按模块测试
pytest tests/test_db_connection.py -v    # 数据库连接与数据完整性
pytest tests/test_vector_recall.py -v    # pgvector 向量召回
pytest tests/test_bm25.py -v             # 全文检索 (simple + jieba)
pytest tests/test_hybrid_merge.py -v     # RRF 混合排分 (纯逻辑)
pytest tests/test_mmr.py -v              # MMR 多样性选择
pytest tests/test_index.py -v            # 索引状态检查
```

## 部署配置

### Docker (PostgreSQL + pgvector)

```yaml
# docker-compose.yml 关键配置
shared_buffers: 1GB
work_mem: 32MB
maintenance_work_mem: 256MB
max_wal_size: 4GB
shm_size: 4gb
```

> 以上为 Colima 6GB 内存虚拟机的保守配置。如服务器内存充裕，可适当调大 `shared_buffers` 和 `maintenance_work_mem` 以加速索引构建。

### 环境变量 (.env)

```bash
VECTOR_DIM=1024                          # BGE-M3 向量维度
BATCH_SIZE=1280                          # 入库批大小
MODEL_PATH=/app/models/bge-m3            # 容器内模型路径
RERANKER_MODEL_PATH=/app/models/bge-reranker-v2-m3
```

## 版本依赖

### 基础设施

| 组件 | 最低版本 | 当前使用 | 说明 |
|------|---------|---------|------|
| Python | 3.10+ | 3.12 | f-string、类型注解等语法依赖 |
| PostgreSQL | 15+ | 17 | JSONB、UNLOGGED 表等特性 |
| pgvector | **0.7.0+** | 0.8.2 | halfvec、binary_quantize、bit_hamming_ops 需要 0.7+ |
| Docker | 20.10+ | - | docker-compose v2 |

> **pgvector 版本关键**：halfvec 量化和二值量化需要 pgvector >= 0.7.0。推荐使用 `pgvector/pgvector:pg17` 官方镜像（自带 0.8+）。旧版 `ankane/pgvector:latest` 可能只有 0.5.x，不支持量化特性。

### Python 依赖

| 包 | 最低版本 | 用途 |
|---|---------|------|
| FlagEmbedding | 1.3.4+ | BGE-M3 embedding + BGE-Reranker 精排 |
| transformers | 4.38+, <5 | HuggingFace 模型加载 (v5 与 FlagEmbedding 不兼容) |
| torch | 2.0+ | 推理框架, MPS/CUDA 加速 |
| fastapi | 0.110+ | API 框架 |
| uvicorn | 0.29+ | ASGI 服务器 |
| psycopg2-binary | 2.9+ | PostgreSQL 驱动 (数据导入) |
| asyncpg | 0.29+ | 异步 PostgreSQL 驱动 (API 服务) |
| jieba | 0.42+ | 中文分词 |
| pandas | 2.0+ | Parquet 读取 |
| pyarrow | 15.0+ | Parquet 序列化 |
| numpy | 1.24+ | 向量运算 |
| langchain-text-splitters | 0.2+ | 文本分块 |
| pytest | 8.0+ | 测试 |
