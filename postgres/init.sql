-- 1. 扩展
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;

-- 2. 文档表
CREATE TABLE IF NOT EXISTS wiki_documents (
    id                 BIGSERIAL PRIMARY KEY,
    content            TEXT NOT NULL,
    metadata           JSONB NOT NULL DEFAULT '{}',
    embedding          HALFVEC(1024),
    content_tokenized  TEXT NOT NULL DEFAULT ''  -- jieba 分词结果，供 pg_search BM25 索引
);

-- 3. 基础索引（向量和全文索引在 build_index.py 里按需构建）
CREATE INDEX IF NOT EXISTS idx_wiki_documents_metadata
    ON wiki_documents USING gin (metadata);
