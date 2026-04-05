# WikiRAG 检索 API

基于 FastAPI 的 RAG 检索服务，接收自然语言 query，经过多路召回、混合排序、精排和去重，返回最相关的文档片段。

## 检索流水线

<div align="center">

```
                    Query ----> LLM重写、原子分割
                      │
                      ▼
          Embedding (BGE-M3, 1024 维)
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   向量召回 (40 条)         BM25 召回 (30 条)
          │                       │
          └───────────┬───────────┘
                      ▼
       并集 + RRF 混合排分 → top 50
                      │
                      ▼
  Cross-Encoder 精排 (BGE-Reranker-v2-M3) → top 15
                      │
                      ▼
         MMR 去重 (多样性筛选) → top 5
```

</div>

### 各阶段说明

| 阶段 | 方法 | 说明 |
|------|------|------|
| Query Embedding | BGE-M3 (FP16) | 将用户查询编码为 1024 维向量 |
| 向量召回 | pgvector 余弦距离 | 从数据库检索语义最近的 40 条文档 |
| BM25 召回 | 向量粗筛 500 条 + jieba 分词 + BM25 | 先缩小候选集避免全表分词，再做关键词匹配取 30 条 |
| 混合排分 | RRF (Reciprocal Rank Fusion) | 对两路结果取并集，按 `1/(rank+60)` 融合排分，取 top 50 |
| 精排 | BGE-Reranker-v2-M3 (Cross-Encoder) | 对 query-doc 对逐一打分，取 top 15 |
| MMR | Maximal Marginal Relevance (λ=0.7) | 平衡相关性与多样性，去除内容高度重复的文档，返回 top 5 |

## 启动服务

```bash
# 1. 确保数据库已启动且数据已导入
docker-compose up -d

# 2. 确保模型已下载至 models/ 目录
python download_embedding_model.py

# 3. 启动 API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> 首次启动需要加载两个模型，耗时约 30 秒，请等待 `✅ 模型加载完成` 日志后再发请求。

## API 接口

### POST /search

检索接口，接收 query 返回最相关文档。

**请求：**

```json
{
  "query": "量子计算的基本原理",
  "top_k": 5
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| query | string | 是 | 查询文本 |
| top_k | int | 否 | 返回条数，默认 5，范围 1-20 |

**响应：**

```json
{
  "query": "量子计算的基本原理",
  "results": [
    {
      "id": 123456,
      "pageid": "54321",
      "title": "量子计算",
      "content": "条目：量子计算\n内容：量子计算是一种利用量子力学现象...",
      "score": 0.8732
    }
  ]
}
```

**示例：**

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "量子计算的基本原理"}'
```

### GET /health

健康检查。

```bash
curl http://localhost:8000/health
```
