"""测试 pg_search 真 BM25 检索（有 IDF + TF 饱和 + 长度归一化）"""
import re
import jieba


# ─── 分词逻辑（与 main.py 保持一致）────────────────────────

_STOPWORDS = {
    "基本", "原理", "介绍", "方法", "作用", "什么", "怎么", "如何",
    "哪些", "一般", "通常", "主要", "相关", "以及", "所以", "因此",
    "但是", "然而", "还是", "还有", "这个", "那个", "这些", "那些",
    "可以", "需要", "进行", "使用", "通过", "对于", "关于", "由于",
    "根据", "包括", "其中", "之间", "之后", "之前", "以上", "以下",
}


def _build_terms(query: str) -> list[str]:
    raw = jieba.cut(query)
    seen: set[str] = set()
    terms: list[str] = []
    for t in raw:
        t = re.sub(r'[+\-&|!():{}\[\]^"~*?:\\]', "", t).strip()
        if len(t) < 2 or re.match(r'^[\s\W]+$', t):
            continue
        if t in seen or t in _STOPWORDS:
            continue
        seen.add(t)
        terms.append(t)
    return terms


# ─── 扩展 & 结构检查 ──────────────────────────────────────

def test_pg_search_extension_installed(cursor):
    """pg_search 扩展是否已安装"""
    cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'pg_search';")
    assert cursor.fetchone() is not None, "pg_search 未安装，请切换到 paradedb/paradedb:latest-pg17 镜像"


def test_bm25_index_exists(cursor):
    """BM25 索引是否存在"""
    cursor.execute("""
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'wiki_documents' AND indexname = 'idx_wiki_bm25';
    """)
    assert cursor.fetchone() is not None, "BM25 索引不存在，请运行 migrate_to_bm25.py 后再执行 build_index.py"


def test_content_tokenized_filled(cursor):
    """content_tokenized 是否已填充"""
    cursor.execute("SELECT COUNT(*) FROM wiki_documents WHERE content_tokenized = '';")
    empty = cursor.fetchone()[0]
    assert empty == 0, f"{empty:,} 行 content_tokenized 为空，请运行 migrate_to_bm25.py"


def test_tsv_column_removed(cursor):
    """tsv 列（旧假 BM25）应已删除"""
    cursor.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'wiki_documents' AND column_name = 'tsv';
    """)
    assert cursor.fetchone() is None, "tsv 列尚未删除，运行: ALTER TABLE wiki_documents DROP COLUMN tsv;"


# ─── 分词逻辑检查 ────────────────────────────────────────

def test_jieba_tokenization():
    """jieba 分词基本切分"""
    tokens = list(jieba.cut("量子计算的基本原理"))
    assert len(tokens) > 1

def test_stopword_filtering():
    """高频泛词应被过滤"""
    terms = _build_terms("量子计算基本原理介绍")
    assert "基本" not in terms
    assert "原理" not in terms
    assert "介绍" not in terms
    assert len(terms) >= 1

def test_deduplication():
    """重复词应去重"""
    terms = _build_terms("中国中国历史")
    assert terms.count("中国") <= 1

def test_special_char_removal():
    """tsquery 特殊字符应被清除"""
    terms = _build_terms("量子+计算&原理")
    for t in terms:
        assert "+" not in t and "&" not in t


# ─── BM25 检索功能测试 ───────────────────────────────────

def test_bm25_and_query_returns_results(cursor):
    """AND 查询（所有词必须出现）能返回结果"""
    cursor.execute("""
        SELECT id, paradedb.score(id) as score
        FROM wiki_documents
        WHERE wiki_documents @@@ '+content_tokenized:中国 +content_tokenized:历史'
        ORDER BY score DESC
        LIMIT 5;
    """)
    rows = cursor.fetchall()
    assert len(rows) > 0, "BM25 AND 查询无结果"


def test_bm25_or_query_returns_results(cursor):
    """OR 查询能返回结果"""
    cursor.execute("""
        SELECT id, paradedb.score(id) as score
        FROM wiki_documents
        WHERE wiki_documents @@@ 'content_tokenized:数学 content_tokenized:物理'
        ORDER BY score DESC
        LIMIT 10;
    """)
    rows = cursor.fetchall()
    assert len(rows) > 0, "BM25 OR 查询无结果"


def test_bm25_scores_descending(cursor):
    """BM25 分数应降序排列"""
    cursor.execute("""
        SELECT paradedb.score(id) as score
        FROM wiki_documents
        WHERE wiki_documents @@@ 'content_tokenized:量子 content_tokenized:计算'
        ORDER BY score DESC
        LIMIT 20;
    """)
    scores = [r[0] for r in cursor.fetchall()]
    assert len(scores) > 0
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1] - 1e-9, "BM25 分数未降序"


def test_bm25_rare_term_scores_higher(cursor):
    """稀有词的 BM25 分（IDF 效应）应高于高频词"""
    # 找一个极具体的词（高 IDF）和一个常见词（低 IDF），比较各自最高分
    cursor.execute("""
        SELECT paradedb.score(id) as score
        FROM wiki_documents
        WHERE wiki_documents @@@ 'content_tokenized:海德格尔'
        ORDER BY score DESC LIMIT 1;
    """)
    row_rare = cursor.fetchone()

    cursor.execute("""
        SELECT paradedb.score(id) as score
        FROM wiki_documents
        WHERE wiki_documents @@@ 'content_tokenized:中国'
        ORDER BY score DESC LIMIT 1;
    """)
    row_common = cursor.fetchone()

    if row_rare and row_common:
        assert row_rare[0] > row_common[0], (
            f"稀有词 '海德格尔' 分数 ({row_rare[0]:.4f}) 应高于高频词 '中国' ({row_common[0]:.4f})，"
            "IDF 未生效"
        )
