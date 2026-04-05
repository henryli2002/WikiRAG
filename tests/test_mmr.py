"""测试 MMR 选择逻辑（纯函数，mock 数据库调用）"""
import sys
import os
from unittest.mock import patch, MagicMock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.main import mmr_select


def _make_doc(doc_id, score=0.5):
    return {"id": doc_id, "content": f"doc_{doc_id}", "metadata": {}, "score": score}


def _mock_db(doc_vecs: dict):
    """构造一个 mock 数据库，根据 doc id 返回对应向量"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    def fake_fetchone():
        call_args = mock_cursor.execute.call_args
        doc_id = call_args[0][1][0]
        vec = doc_vecs.get(doc_id, np.zeros(1024))
        return (vec.tolist(),)

    mock_cursor.fetchone = fake_fetchone
    return mock_conn


def test_mmr_returns_k(monkeypatch):
    """MMR 返回恰好 k 条结果"""
    docs = [_make_doc(i, score=1.0 - i * 0.05) for i in range(10)]
    vecs = {i: np.random.randn(1024).astype(np.float32) for i in range(10)}
    query_vec = np.random.randn(1024).tolist()

    mock_conn = _mock_db(vecs)
    with patch("app.main.get_db", return_value=mock_conn):
        result = mmr_select(query_vec, docs, k=5, lam=0.7)
    assert len(result) == 5


def test_mmr_fewer_than_k():
    """文档数不足 k 时直接返回全部"""
    docs = [_make_doc(1), _make_doc(2)]
    query_vec = np.random.randn(1024).tolist()
    result = mmr_select(query_vec, docs, k=5, lam=0.7)
    assert len(result) == 2


def test_mmr_diversity(monkeypatch):
    """高度相似的文档不应全部被选中"""
    # 构造 3 个几乎相同的向量 + 1 个差异向量
    base_vec = np.random.randn(1024).astype(np.float32)
    similar_vec = base_vec + np.random.randn(1024).astype(np.float32) * 0.01
    different_vec = -base_vec  # 完全相反

    docs = [
        _make_doc(0, score=0.9),  # similar to base
        _make_doc(1, score=0.85),  # similar to base
        _make_doc(2, score=0.80),  # similar to base
        _make_doc(3, score=0.70),  # different
    ]
    vecs = {
        0: base_vec,
        1: similar_vec,
        2: base_vec + np.random.randn(1024).astype(np.float32) * 0.01,
        3: different_vec,
    }
    query_vec = base_vec.tolist()

    mock_conn = _mock_db(vecs)
    with patch("app.main.get_db", return_value=mock_conn):
        result = mmr_select(query_vec, docs, k=3, lam=0.5)

    selected_ids = [d["id"] for d in result]
    # 差异向量（id=3）应该被选中以增加多样性
    assert 3 in selected_ids, f"MMR 应选中差异文档 id=3，实际选中: {selected_ids}"


def test_mmr_lambda_1_equals_greedy(monkeypatch):
    """lambda=1.0 时退化为纯贪心（按 score 排序）"""
    docs = [_make_doc(i, score=1.0 - i * 0.1) for i in range(6)]
    vecs = {i: np.random.randn(1024).astype(np.float32) for i in range(6)}
    query_vec = np.random.randn(1024).tolist()

    mock_conn = _mock_db(vecs)
    with patch("app.main.get_db", return_value=mock_conn):
        result = mmr_select(query_vec, docs, k=3, lam=1.0)

    selected_ids = [d["id"] for d in result]
    assert selected_ids == [0, 1, 2], f"lambda=1 应按分数贪心选择，实际: {selected_ids}"
