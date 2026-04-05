# 数据库与向量索引指南 (RAG 专用)

## 1. 数据资产规格 (基于 Wikipedia + BGE-M3)
- **原始数据规模**: 约 6GB (Parquet 格式)
- **向量模型**: BGE-M3
- **向量维度**: 1024 维 (float32, 占用 4096 bytes/行)
- **表结构 (`wiki_documents`)**:
  - `id`: BIGSERIAL (主键)
  - `content`: TEXT (文本切片)
  - `metadata`: JSONB (包含 title, pageid 等元信息)
  - `embedding`: VECTOR(1024) 

## 2. 向量索引构建策略 (建表 & 导数据后执行)

在完成 6GB 数据的导入后，根据业务对精度和性能的需求，连接数据库执行以下 SQL 之一：

### 方案 A：标准 HNSW 索引 (推荐：高精度，高内存占用)
查询最快，精度最高，但生成的索引文件较大（约占原始向量体积的 1-1.5 倍）。
```sql
-- 建立基于余弦距离的 HNSW 索引
CREATE INDEX ON wiki_documents USING hnsw (embedding vector_cosine_ops);
```

### 方案 B：FP16 半精度量化 HNSW (推荐：内存节省 50%)
利用 pgvector 的 halfvec 技术，将 32 位浮点数砍半，肉眼几乎无法感知精度损失，M4 Pro 内存福音。
```sql
-- 将向量强转为 halfvec(1024) 并建索引，大幅节约内存
CREATE INDEX ON wiki_documents USING hnsw ((embedding::halfvec(1024)) halfvec_cosine_ops);
```

### 方案 C：IVFFlat 索引 (构建极快，查询稍慢)
如果你发现 HNSW 构建时间太长，可以使用 IVFFlat。必须在数据完全导入后构建，以便 K-means 找准聚类中心。
*(注：lists 数量建议设置为 总行数的平方根)*
```sql
-- 假设你有大约 1,000,000 行数据，平方根是 1000
CREATE INDEX ON wiki_documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1000);
```

### 方案 D：标量量化 (Scalar Quantization - pgvector 0.7+ 新特性)
如果想极致省内存，可以使用 bit 操作或者更高级的量化（视 pgvector 具体版本支持情况）。最实用的是结合 `halfvec` 使用。

## 3. 免密连接方式
本机 Python 代码连接方式（无需密码）：
```python
import psycopg2
# Host auth trust 已经开启
conn = psycopg2.connect(
    host="localhost", port=5432, user="rag_user", dbname="rag_db"
)
```
