"""测试 pgvector 向量召回功能（存储类型 halfvec）"""
import numpy as np


def test_cosine_distance_query(cursor):
    """基本的余弦距离查询是否能执行"""
    cursor.execute("SELECT id, embedding FROM wiki_documents LIMIT 1;")
    row = cursor.fetchone()
    doc_id, embedding = row[0], row[1]

    cursor.execute(
        """
        SELECT id, 1 - (embedding <=> %s::halfvec(1024)) AS score
        FROM wiki_documents
        ORDER BY embedding <=> %s::halfvec(1024)
        LIMIT 5;
        """,
        (embedding, embedding),
    )
    results = cursor.fetchall()
    assert len(results) > 0, "向量召回无结果"
    assert results[0][0] == doc_id, "用自身 embedding 查询，自身未排在第一位"
    assert results[0][1] > 0.99, f"自身余弦相似度应接近 1，实际为 {results[0][1]}"


def test_recall_returns_k_results(cursor):
    """向量召回是否返回指定数量的结果"""
    cursor.execute("SELECT embedding FROM wiki_documents LIMIT 1;")
    embedding = cursor.fetchone()[0]

    k = 40
    cursor.execute(
        """
        SELECT id FROM wiki_documents
        ORDER BY embedding <=> %s::halfvec(1024)
        LIMIT %s;
        """,
        (embedding, k),
    )
    results = cursor.fetchall()
    assert len(results) == k, f"期望返回 {k} 条，实际 {len(results)} 条"


def test_recall_scores_are_sorted(cursor):
    """返回结果的相似度是否从高到低排序"""
    cursor.execute("SELECT embedding FROM wiki_documents LIMIT 1;")
    embedding = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT id, 1 - (embedding <=> %s::halfvec(1024)) AS score
        FROM wiki_documents
        ORDER BY embedding <=> %s::halfvec(1024)
        LIMIT 20;
        """,
        (embedding, embedding),
    )
    results = cursor.fetchall()
    scores = [r[1] for r in results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1] - 1e-6, (
            f"结果未按相似度降序排列: scores[{i}]={scores[i]}, scores[{i+1}]={scores[i+1]}"
        )


def test_recall_no_duplicate_ids(cursor):
    """召回结果中不应有重复 id"""
    cursor.execute("SELECT embedding FROM wiki_documents LIMIT 1;")
    embedding = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT id FROM wiki_documents
        ORDER BY embedding <=> %s::halfvec(1024)
        LIMIT 40;
        """,
        (embedding,),
    )
    ids = [r[0] for r in cursor.fetchall()]
    assert len(ids) == len(set(ids)), "召回结果中存在重复 id"


def test_random_vector_recall(cursor):
    """用随机向量查询也应返回结果（不应报错）"""
    random_vec = np.random.randn(1024).astype(np.float16).tolist()
    cursor.execute(
        """
        SELECT id, 1 - (embedding <=> %s::halfvec(1024)) AS score
        FROM wiki_documents
        ORDER BY embedding <=> %s::halfvec(1024)
        LIMIT 5;
        """,
        (random_vec, random_vec),
    )
    results = cursor.fetchall()
    assert len(results) == 5, "随机向量查询应返回 5 条结果"
    for r in results:
        assert -1.0 <= r[1] <= 1.0, f"余弦相似度越界: {r[1]}"


def test_embedding_column_is_halfvec(cursor):
    """确认 embedding 列类型为 halfvec"""
    cursor.execute("""
        SELECT udt_name FROM information_schema.columns
        WHERE table_name = 'wiki_documents' AND column_name = 'embedding';
    """)
    udt = cursor.fetchone()[0]
    assert udt == "halfvec", f"embedding 列类型应为 halfvec，实际为 {udt}"
