"""测试数据库连接与基础数据完整性"""


def test_connection_alive(db_conn):
    """数据库连接是否正常"""
    assert db_conn.closed == 0


def test_pgvector_extension(cursor):
    """pgvector 扩展是否已安装"""
    cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
    row = cursor.fetchone()
    assert row is not None, "pgvector 扩展未安装"


def test_pgvector_version(cursor):
    """pgvector 版本 >= 0.7.0 (halfvec/bit 量化需要)"""
    cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
    version = cursor.fetchone()[0]
    major, minor = [int(x) for x in version.split(".")[:2]]
    assert (major, minor) >= (0, 7), (
        f"pgvector 版本 {version} 过低，需要 >= 0.7.0 以支持 halfvec 量化。"
        "请使用 pgvector/pgvector:pg17 镜像。"
    )


def test_table_exists(cursor):
    """wiki_documents 表是否存在"""
    cursor.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'wiki_documents'
        );
    """)
    assert cursor.fetchone()[0] is True


def test_table_has_data(cursor):
    """表中是否有数据"""
    cursor.execute("SELECT COUNT(*) FROM wiki_documents;")
    count = cursor.fetchone()[0]
    assert count > 0, f"wiki_documents 表为空 (count={count})"


def test_table_schema(cursor):
    """表结构是否符合预期"""
    cursor.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'wiki_documents'
        ORDER BY ordinal_position;
    """)
    columns = {row[0]: row[1] for row in cursor.fetchall()}
    assert "id" in columns
    assert "content" in columns
    assert "metadata" in columns
    assert "embedding" in columns
    assert "tsv" in columns


def test_embedding_not_null(cursor):
    """抽查 embedding 是否非空"""
    cursor.execute("""
        SELECT COUNT(*) FROM wiki_documents
        WHERE embedding IS NULL
        LIMIT 1;
    """)
    null_count = cursor.fetchone()[0]
    assert null_count == 0, f"存在 {null_count} 条 embedding 为 NULL 的记录"


def test_embedding_dimension(cursor):
    """向量维度是否为 1024"""
    cursor.execute("SELECT vector_dims(embedding) FROM wiki_documents LIMIT 1;")
    dim = cursor.fetchone()[0]
    assert dim == 1024, f"向量维度应为 1024，实际为 {dim}"


def test_metadata_has_fields(cursor):
    """metadata 是否包含 pageid 和 title"""
    cursor.execute("""
        SELECT metadata FROM wiki_documents LIMIT 5;
    """)
    for row in cursor.fetchall():
        meta = row[0]
        assert "pageid" in meta, f"metadata 缺少 pageid: {meta}"
        assert "title" in meta, f"metadata 缺少 title: {meta}"
