# WikiRAG: 中文维基百科检索系统

基于中文维基百科全量数据构建的检索系统。从原始 Wikipedia dump 出发，经过数据清洗、向量化、入库，提供向量 + BM25 混合召回、精排、MMR 去重的检索 API。

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
│  ingest_to_postgres.py ─ 多进程 jieba 分词 + COPY 批量入库 │
│        │                                                │
│        ▼                                                │
│  build_index.py ──────── GIN + HNSW + BM25 索引构建       │
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
│      pgvector 余弦距离        pg_search / Tantivy        │
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
| 全文检索 | pg_search (ParadeDB / Tantivy) | 真正的 BM25：IDF + TF 饱和 + 长度归一化 |
| 中文分词 | jieba | 入库时多进程并行分词，写入 content_tokenized 列 |
| 文本分块 | RecursiveCharacterTextSplitter | 512 字符, 60 字符重叠 |
| API 框架 | FastAPI + asyncpg | 异步连接池，请求并发控制 |
| 繁简转换 | OpenCC | |

## 目录结构

```
WikiRAG/
├── app/
│   └── main.py                  # FastAPI 检索 API
├── scripts/
│   ├── auto_wiki_parser.py      # 步骤 1: XML 解析
│   ├── clean_brackets.py        # 步骤 2: 噪音清洗
│   ├── embed_factory.py         # 步骤 3: 向量化
│   ├── ingest_to_postgres.py    # 步骤 4: 批量入库（含 jieba 分词）
│   ├── build_index.py           # 步骤 5: 索引构建
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

# 启动数据库（paradedb 镜像，内置 pgvector + pg_search）
mkdir -p data/pgdata && chmod 777 data/pgdata
docker compose up -d

# 下载模型 (BGE-M3 + BGE-Reranker)
python download_embedding_model.py
```

### 数据处理流水线

```bash
# 1. 将 zhwiki dump 放入 data/raw/

# 2. XML 解析 → JSONL
python scripts/auto_wiki_parser.py

# 3. 噪音清洗
python scripts/clean_brackets.py

# 4. 向量化（耗时较长，支持断点续传）
python scripts/embed_factory.py

# 5. 批量入库（多进程 jieba 分词直接写入 content_tokenized，约 30 分钟）
python scripts/ingest_to_postgres.py

# 6. 构建索引（HNSW 约 3 小时，BM25 约 1.5 分钟）
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
| 入库行数（文本块） | 2,784,657 |
| 表堆大小 | ~3.5 GB |

### 索引大小（实测）

| 索引 | 类型 | 大小 |
|------|------|------|
| idx_emb_hnsw_fp16_1024_m16_ef32 | HNSW halfvec cosine | 7.25 GB |
| idx_wiki_bm25 | BM25 (pg_search / Tantivy) | — |
| idx_wiki_documents_metadata | GIN jsonb | ~185 MB |
| wiki_documents_pkey | B-tree bigserial | ~60 MB |

### 数据处理耗时（参考）

| 阶段 | 耗时 | 环境 |
|------|------|------|
| 向量化 | ~18 小时 42 分钟 | RTX 3070 Laptop 140W, 8GB VRAM |
| 批量入库（含 jieba 分词） | ~30 分钟 | MBP M4 Pro, Docker Desktop |
| metadata GIN 索引 | 12 秒 | MBP M4 Pro, Docker Desktop |
| HNSW 向量索引 (fp16, m=16, ef=32) | 184 分 53 秒 | MBP M4 Pro, Docker Desktop |
| BM25 索引 (pg_search) | 1 分 30 秒 | MBP M4 Pro, Docker Desktop |

## 实现细节

### 1. 数据库表结构

```sql
CREATE TABLE wiki_documents (
    id                BIGSERIAL PRIMARY KEY,
    content           TEXT NOT NULL,          -- 分块后的文本（含标题前缀）
    metadata          JSONB NOT NULL,          -- {"pageid": ..., "title": ...}
    embedding         HALFVEC(1024),           -- BGE-M3 稠密向量，FP16 存储
    content_tokenized TEXT NOT NULL DEFAULT '' -- jieba 空格分词结果，供 BM25 索引
);
```

`content_tokenized` 在 `ingest_to_postgres.py` 入库时由多进程 jieba 直接填充，无需后续迁移步骤。

### 2. BM25 检索方案

使用 ParadeDB 的 `pg_search` 扩展（基于 Tantivy），在 `content_tokenized` 列上建 BM25 索引，使用 whitespace tokenizer。

- **入库时**：jieba 分词 → 空格连接 → 写入 `content_tokenized`
- **查询时**：jieba 分词 → 去停用词/去重 → 先尝试 AND（所有词必须出现），结果不足则 OR 回退（按词长降序取最多 4 词）
- **评分**：Tantivy BM25 原生评分（IDF + TF 饱和 + 长度归一化），由 `paradedb.score(id)` 返回

与旧方案（PostgreSQL ts_rank + GIN tsvector）相比：
- 有真正的 IDF，高频词自动降权，不再需要手工停用词 hack
- 长文档不会因 term 频次绝对值高而虚高

### 3. 向量索引

向量以 `halfvec(1024)` 存储（FP16），查询时做余弦距离计算。

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
| BM25 召回 | top 10 | pg_search BM25，AND 优先，OR 回退 |
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
pytest tests/test_bm25.py -v             # BM25 检索 (pg_search)
pytest tests/test_index.py -v            # 索引状态检查

# 压力测试（需服务运行中）
python scripts/stress_test.py
python scripts/stress_test.py --concurrency 1 2 4 8 --requests 40 --verbose
```

## 部署配置

### Docker (ParadeDB = PostgreSQL + pgvector + pg_search)

```yaml
# docker-compose.yml 关键配置
image: paradedb/paradedb:latest-pg17
shared_buffers: 1GB
work_mem: 32MB
maintenance_work_mem: 256MB
max_wal_size: 4GB
shm_size: 4gb
```

> bind mount 在 macOS 上需提前 `mkdir -p data/pgdata && chmod 777 data/pgdata`，否则容器内 postgres 进程写堆文件时会报 Permission denied。

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
| pg_search | — | ParadeDB latest | Tantivy BM25 |
| Docker | 20.10+ | — | docker-compose v2 |

### Python 依赖

| 包 | 最低版本 | 用途 |
|---|---------|------|
| FlagEmbedding | 1.3.4+ | BGE-M3 embedding + BGE-Reranker 精排 |
| transformers | 4.38+, <5 | HuggingFace 模型加载 |
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

---

## Legacy

<details>
<summary>旧版方案（tsvector + ts_rank，已废弃）</summary>

旧版 BM25 使用 PostgreSQL 原生 `tsvector` + `ts_rank` + GIN 索引，无真正的 IDF 和长度归一化。表结构中包含 `tsv TSVECTOR` 列，入库时生成两个临时列 `title_seg`/`content_seg`，`build_index.py` 构建完后再 DROP。

### 旧版表结构

```sql
CREATE TABLE wiki_documents (
    id         BIGSERIAL PRIMARY KEY,
    content    TEXT NOT NULL,
    metadata   JSONB NOT NULL,
    embedding  HALFVEC(1024),
    tsv        TSVECTOR   -- title=A, content=B 加权全文向量
);
```

### 旧版流程

```
ingest_to_postgres.py  →  migrate_to_bm25.py  →  build_index.py
（生成 title_seg/content_seg）  （填充 content_tokenized）  （删临时列，建索引）
```

废弃原因：
- `ts_rank` 没有 IDF，高频词无法自动降权，依赖手工维护停用词表
- 标题/内容双权重方案复杂但效果提升有限
- 迁移脚本（`migrate_to_bm25.py`）使流程割裂，新建库时不再需要

</details>
