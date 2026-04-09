# WikiRAG: 中文维基百科检索系统

基于中文维基百科全量数据构建的检索系统。从原始 Wikipedia dump 出发，经过数据清洗、向量化、入库，提供向量 + 全文混合召回、精排、MMR 去重的检索 API。

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
│      向量召回 (top 50)         BM25 召回 (top 10)         │
│      pgvector 余弦距离        simple + ts_rank           │
│             │                       │                   │
│             └───────────┬───────────┘                   │
│                         ▼                               │
│          RRF 混合排分 (Reciprocal Rank Fusion)           │
│                      → top 32                           │
│                         │                               │
│                         ▼                               │
│            Cross-Encoder 精排 (BGE-Reranker)             │
│                      → top 10                           │
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
| 向量数据库 | PostgreSQL + pgvector | HNSW 索引，halfvec 存储 |
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
│   ├── build_index.py           # 步骤 7: 索引构建
│   └── stress_test.py           # 压力测试（需 pip install httpx）
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

# 6. 构建索引（含 ANALYZE）
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
| score | 精排分数 (0~1，越高越相关)；低于 0.8 的结果会被过滤 |

## 数据与索引

### 数据规模（实测）

| 指标 | 数值 |
|------|------|
| 入库行数（文本块） | 2,787,678 |
| 表堆大小 | 3.5 GB |
| 索引合计 | ~19 GB |
| 表 + 索引总计 | 22 GB |

### 索引大小（实测）

| 索引 | 类型 | 大小 |
|------|------|------|
| idx_emb_hnsw_fp16_1024_m16_ef32 | HNSW halfvec cosine | 7.25 GB |
| idx_wiki_documents_tsv | GIN tsvector | 1.04 GB |
| idx_wiki_documents_metadata | GIN jsonb | 185 MB |
| wiki_documents_pkey | B-tree bigserial | 60 MB |

### 数据处理耗时（参考）

| 阶段 | 耗时 | 环境 |
|------|------|------|
| 向量化 | ~18 小时 42 分钟 | RTX 3070 Laptop 140W, 8GB VRAM |
| 批量入库 | ~30 分钟 | MBP M4 Pro, Colima 6GB |
| 构建 tsvector | 7 分 7 秒 | MBP M4 Pro, Colima 6GB |
| tsv GIN 索引 | 4 分 0 秒 | MBP M4 Pro, Colima 6GB |
| metadata GIN 索引 | 1 分 10 秒 | MBP M4 Pro, Colima 6GB |
| HNSW 向量索引 (fp16, m=16, ef=32) | 待补充 | — |

## 实现细节

### 1. 数据库表结构

```sql
CREATE TABLE wiki_documents (
    id         BIGSERIAL PRIMARY KEY,
    content    TEXT NOT NULL,          -- 分块后的文本 (含标题前缀)
    metadata   JSONB NOT NULL,         -- {"pageid": ..., "title": ...}
    embedding  HALFVEC(1024),          -- BGE-M3 稠密向量，FP16 存储
    tsv        TSVECTOR                -- 全文检索向量 (title=A, content=B)
);
```

入库时还有两个临时列 `title_seg`、`content_seg`（存 jieba 分词结果），`build_index.py` 构建完 tsv 后会 DROP 掉。

### 2. 全文检索方案

无需安装 zhparser 等中文分词扩展。核心思路：**Python 端分词，PostgreSQL 端当英文处理**。

- **入库时**：jieba 分词 → 空格连接 → 存入临时列 → `to_tsvector('simple', ...)` 生成 tsvector
- **标题权重 A** (1.0)，**内容权重 B** (0.4)：`setweight(..., 'A') || setweight(..., 'B')`
- **查询时**：jieba 分词 → 去停用词、去重 → 先尝试 AND，结果不足则 OR 回退
- **OR 回退限制**：按词长降序取最多 4 个词，候选行数上限 500，避免对大量命中行做全量 ts_rank 排序

### 3. 向量索引

向量以 `halfvec(1024)` 存储（FP16），查询时按列存储类型 cast 后做余弦距离计算。

`build_index.py` 支持多种索引配置，可共存：

| 方案 | 命令 | 特点 |
|------|------|------|
| HNSW + FP16（默认） | `python scripts/build_index.py` | 推荐 |
| HNSW + FP32 | `--quantize fp32` | 无实际精度收益，索引更大 |
| HNSW + 二值量化 | `--quantize bit` | 索引极小，精度有损 |
| HNSW 高精度 | `--m 32 --ef-construction 64` | 召回率更高，构建更慢 |
| IVFFlat + FP16 | `--method ivfflat` | 构建快，查询稍慢 |
| 降维 | `--dim 512` | 截取前 N 维 |

索引名编码配置参数（如 `idx_emb_hnsw_fp16_1024_m16_ef32`），不同配置可共存。

查看已建索引：
```sql
SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass))
FROM pg_indexes WHERE tablename = 'wiki_documents' AND indexname LIKE 'idx_emb_%';
```

### 4. 检索流水线参数

| 阶段 | 参数 | 说明 |
|------|------|------|
| Query Embedding | BGE-M3 FP16 | 1024 维稠密向量 |
| 向量召回 | top 50 | pgvector 余弦距离 |
| BM25 召回 | top 10 | PostgreSQL ts_rank + GIN 索引 |
| 混合排分 | RRF k=60，取 top 32 | `score = Σ 1/(rank + 60)` |
| 精排 | BGE-Reranker-v2-M3，取 top 10 | Cross-Encoder 逐对打分 |
| MMR 去重 | λ=0.7，top 5 | `score = λ·relevance - (1-λ)·max_sim` |
| 结果过滤 | score > 0.8 | 低于阈值的精排结果不返回 |

### 5. 安全机制

- **文件锁**：`ingest_to_postgres.py` 成功后写入 `data/.ingest.lock`，防止误操作重建。需手动 `rm data/.ingest.lock` 才能重新导入
- **断点续传**：`embed_factory.py` 通过 checkpoint 文件支持中断恢复

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

# 压力测试（需服务运行中）
python scripts/stress_test.py
python scripts/stress_test.py --concurrency 1 2 4 8 --requests 40 --verbose
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

> 以上为 Colima 6GB 内存虚拟机的保守配置。内存充裕时可适当调大 `shared_buffers` 和 `maintenance_work_mem`。

### 环境变量 (.env)

```bash
POSTGRES_DB=rag_db
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=...
DB_HOST=localhost
DB_PORT=5432
```

## 版本依赖

### 基础设施

| 组件 | 最低版本 | 当前使用 | 说明 |
|------|---------|---------|------|
| Python | 3.10+ | 3.12 | f-string、类型注解等语法依赖 |
| PostgreSQL | 15+ | 17 | JSONB 等特性 |
| pgvector | 0.7.0+ | 0.8.2 | halfvec、binary_quantize 需要 0.7+ |
| Docker | 20.10+ | — | docker-compose v2 |

> **pgvector 版本**：halfvec 和二值量化需要 pgvector >= 0.7.0。推荐使用 `pgvector/pgvector:pg17` 官方镜像。

### Python 依赖

| 包 | 最低版本 | 用途 |
|---|---------|------|
| FlagEmbedding | 1.3.4+ | BGE-M3 embedding + BGE-Reranker 精排 |
| transformers | 4.38+, <5 | HuggingFace 模型加载 (v5 与 FlagEmbedding 不兼容) |
| torch | 2.0+ | 推理框架，MPS/CUDA 加速 |
| fastapi | 0.110+ | API 框架 |
| uvicorn | 0.29+ | ASGI 服务器 |
| psycopg2-binary | 2.9+ | PostgreSQL 驱动（数据导入） |
| asyncpg | 0.29+ | 异步 PostgreSQL 驱动（API 服务） |
| jieba | 0.42+ | 中文分词 |
| pandas | 2.0+ | Parquet 读取 |
| pyarrow | 15.0+ | Parquet 序列化 |
| numpy | 1.24+ | 向量运算 |
| langchain-text-splitters | 0.2+ | 文本分块 |
| pytest | 8.0+ | 测试 |
