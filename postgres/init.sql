-- 1. 开启 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. 创建文档数据表
CREATE TABLE IF NOT EXISTS wiki_documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    embedding VECTOR(1024),
    tsv TSVECTOR
);

-- 3. 索引
CREATE INDEX IF NOT EXISTS idx_wiki_documents_metadata ON wiki_documents USING gin (metadata);
CREATE INDEX IF NOT EXISTS idx_wiki_documents_tsv ON wiki_documents USING gin (tsv);
