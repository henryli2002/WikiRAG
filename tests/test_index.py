"""测试数据库索引状态"""


def test_metadata_gin_index_exists(cursor):
    """GIN 索引是否存在"""
    cursor.execute("""
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'wiki_documents'
          AND indexname = 'idx_wiki_documents_metadata';
    """)
    assert cursor.fetchone() is not None, "metadata GIN 索引不存在"


def test_embedding_index_exists(cursor):
    """向量索引是否存在（idx_emb_* 命名规则）"""
    cursor.execute("""
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'wiki_documents'
          AND indexname LIKE 'idx_emb_%';
    """)
    rows = cursor.fetchall()
    assert len(rows) > 0, (
        "embedding 向量索引不存在，请运行 build_index.py 建立索引。"
    )


def test_bm25_index_exists(cursor):
    """pg_search BM25 索引是否存在"""
    cursor.execute("""
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'wiki_documents'
          AND indexname = 'idx_wiki_bm25';
    """)
    assert cursor.fetchone() is not None, "BM25 索引不存在，请运行 migrate_to_bm25.py 后再执行 build_index.py"


def test_primary_key_exists(cursor):
    """主键索引存在"""
    cursor.execute("""
        SELECT constraint_name FROM information_schema.table_constraints
        WHERE table_name = 'wiki_documents'
          AND constraint_type = 'PRIMARY KEY';
    """)
    assert cursor.fetchone() is not None, "wiki_documents 缺少主键"
