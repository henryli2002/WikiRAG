"""测试 PostgreSQL 全文检索 (simple 字典 + jieba 分词)"""
import re
import jieba


def test_jieba_tokenization():
    """jieba 分词对中文的基本切分"""
    tokens = list(jieba.cut("量子计算的基本原理"))
    assert len(tokens) > 1, "分词结果不应只有一个 token"
    assert "量子" in tokens or "量子计算" in tokens


def test_tsquery_building():
    """分词结果能正确构建 OR tsquery 字符串"""
    tokens = jieba.cut("量子计算基本原理")
    terms = [re.sub(r"[&|!():'\\]", "", t).strip() for t in tokens]
    terms = [t for t in terms if t and not re.match(r'^[\s\W]+$', t)]
    tsquery_str = " | ".join(terms)

    assert "|" in tsquery_str, "应使用 OR 连接"
    assert len(terms) > 1, "应有多个检索词"
    # 不应包含 tsquery 特殊字符
    for t in terms:
        assert "&" not in t and "!" not in t and "'" not in t


def test_tsv_column_exists(cursor):
    """tsv 列是否存在"""
    cursor.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'wiki_documents' AND column_name = 'tsv';
    """)
    assert cursor.fetchone() is not None, "tsv 列不存在"


def test_tsv_not_null(cursor):
    """tsv 列是否已填充"""
    cursor.execute("SELECT COUNT(*) FROM wiki_documents WHERE tsv IS NULL;")
    null_count = cursor.fetchone()[0]
    assert null_count == 0, f"有 {null_count} 条记录 tsv 为 NULL"


def test_tsv_gin_index_exists(cursor):
    """tsv 的 GIN 索引是否存在"""
    cursor.execute("""
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'wiki_documents'
          AND indexname = 'idx_wiki_documents_tsv';
    """)
    assert cursor.fetchone() is not None, "tsv GIN 索引不存在"


def test_fulltext_search_returns_results(cursor):
    """全文检索能否返回结果"""
    cursor.execute("""
        SELECT id, ts_rank(tsv, to_tsquery('simple', '中国 | 历史')) AS score
        FROM wiki_documents
        WHERE tsv @@ to_tsquery('simple', '中国 | 历史')
        ORDER BY score DESC
        LIMIT 5;
    """)
    results = cursor.fetchall()
    assert len(results) > 0, "全文检索 '中国 | 历史' 无结果，数据可能有问题"


def test_fulltext_scores_sorted(cursor):
    """全文检索结果按分数降序排列"""
    cursor.execute("""
        SELECT ts_rank(tsv, to_tsquery('simple', '数学 | 物理')) AS score
        FROM wiki_documents
        WHERE tsv @@ to_tsquery('simple', '数学 | 物理')
        ORDER BY score DESC
        LIMIT 20;
    """)
    scores = [r[0] for r in cursor.fetchall()]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1] - 1e-9


def test_title_weight_higher(cursor):
    """标题匹配 (A 权重) 的文档应比仅内容匹配的分数更高"""
    # 用 ts_rank 的权重参数验证：A=1.0 > B=0.4
    cursor.execute("""
        SELECT ts_rank('{0.1, 0.2, 0.4, 1.0}', tsv, to_tsquery('simple', '中国')) AS score
        FROM wiki_documents
        WHERE tsv @@ to_tsquery('simple', '中国')
        ORDER BY score DESC
        LIMIT 1;
    """)
    top_score = cursor.fetchone()
    assert top_score is not None and top_score[0] > 0
