# WikiRAG 数据清洗与入库流水线

本目录包含了将原始 Wikipedia 数据转换为向量并存入 PostgreSQL (pgvector) 的全套脚本。

## ⚙️ 环境准备
确保你已经安装了所需依赖：
```bash
pip install -r ../requirements.txt
```

## 🚀 完整执行步骤 (从 0 到 1)

### 步骤 1: 准备原始数据
请从维基百科 dump 下载以下两个文件，并放入项目根目录下的 `data/raw/` 目录：
1. `zhwiki-latest-pages-articles.xml.bz2` (正文数据)
2. `zhwiki-latest-categorylinks.sql.gz` (分类映射数据)

### 步骤 2: XML 提取与格式化
将 XML 转为初步的 JSONL 格式。此脚本会自动调用多进程加速。
```bash
python scripts/auto_wiki_parser.py
```
*输出: `data/raw/zhwiki.jsonl`*

### 步骤 3: 噪音清洗
清理维基标记、HTML 标签、空括号等，提纯文本质量。
```bash
python scripts/clean_brackets.py
```
*输出: `data/raw/zhwiki_clean.jsonl`*

### 步骤 4: 下载模型
此脚本会自动下载以下模型至项目根目录的 `models/` 目录中：
- **BAAI/bge-m3** — Embedding 向量模型 (用于向量化 & 检索)
- **BAAI/bge-reranker-v2-m3** — Cross-Encoder 精排模型 (用于 RAG 精排)
```bash
python download_embedding_model.py
```

### 步骤 5: 向量化处理 (Embedding)
加载本地的 BGE-M3 模型，将清洗后的文本进行分块 (Chunking) 并生成 1024 维向量。
```bash
python scripts/embed_factory.py
```
*输出: `data/parquet_output/*.parquet` (碎片化的高效列式存储文件)*

### 步骤 6: 启动数据库并全量导入
确保 Docker 环境已准备好。导入脚本会自动使用多进程 jieba 分词。
```bash
# 后台启动带向量支持的 PostgreSQL 容器
docker-compose up -d

# 全量数据导入（多进程分词 + COPY 写入）
python scripts/ingest_to_postgres.py
```

> **⚠️ 注意：`ingest_to_postgres.py` 每次运行都会 DROP 并重建 `wiki_documents` 表，已有数据将被全部清除。脚本内置文件锁保护，成功导入后需手动删除锁文件 `data/.ingest.lock` 才能再次运行。**

### 步骤 7: 构建索引
数据全量导入后，单独运行索引构建脚本。分离执行比边导边建快得多。
```bash
python scripts/build_index.py
```
该脚本会依次执行：
1. 从临时分词列构建 tsvector（title=A 权重, content=B 权重）
2. 清理临时分词列
3. 创建 metadata GIN 索引、tsv GIN 索引、向量 ivfflat 索引
4. 将 UNLOGGED 表转为 LOGGED（启用 WAL 持久化）

每一步都会打印耗时，方便写实验报告。

*完成！你的 RAG 数据库已就绪。接下来参考 `app/README.md` 启动检索 API。*
