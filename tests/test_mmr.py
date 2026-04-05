"""测试 MMR 选择逻辑（纯函数，mock asyncpg 连接池）"""
import sys
import os
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.main import mmr_select


def _make_doc(doc_id, score=0.5):
    return {"id": doc_id, "content": f"doc_{doc_id}", "metadata": {}, "score": score}


def _mock_pool(doc_vecs: dict):
    """构造 mock asyncpg pool，根据 doc id 返回对应向量字符串"""
    mock_conn = AsyncMock()

    async def fake_fetch(sql, *args):
        doc_ids = args[0]
        results = []
        for did in doc_ids:
            vec = doc_vecs.get(did, np.zeros(1024))
            vec_str = "[" + ",".join(str(float(x)) for x in vec) + "]"
            results.append({"id": did, "embedding": vec_str})
        return results

    mock_conn.fetch = fake_fetch
    mock_conn.execute = AsyncMock()

    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_pool


@pytest.mark.asyncio
async def test_mmr_returns_k():
    """MMR 返回恰好 k 条结果"""
    docs = [_make_doc(i, score=1.0 - i * 0.05) for i in range(10)]
    vecs = {i: np.random.randn(1024).astype(np.float32) for i in range(10)}
    query_vec = np.random.randn(1024).tolist()

    with patch("app.main.db_pool", _mock_pool(vecs)):
        result = await mmr_select(query_vec, docs, k=5, lam=0.7)
    assert len(result) == 5


@pytest.mark.asyncio
async def test_mmr_fewer_than_k():
    """文档数不足 k 时直接返回全部"""
    docs = [_make_doc(1), _make_doc(2)]
    query_vec = np.random.randn(1024).tolist()
    result = await mmr_select(query_vec, docs, k=5, lam=0.7)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_mmr_diversity():
    """高度相似的文档不应全部被选中"""
    base_vec = np.random.randn(1024).astype(np.float32)
    similar_vec = base_vec + np.random.randn(1024).astype(np.float32) * 0.01
    different_vec = -base_vec

    docs = [
        _make_doc(0, score=0.9),
        _make_doc(1, score=0.85),
        _make_doc(2, score=0.80),
        _make_doc(3, score=0.70),
    ]
    vecs = {
        0: base_vec,
        1: similar_vec,
        2: base_vec + np.random.randn(1024).astype(np.float32) * 0.01,
        3: different_vec,
    }
    query_vec = base_vec.tolist()

    with patch("app.main.db_pool", _mock_pool(vecs)):
        result = await mmr_select(query_vec, docs, k=3, lam=0.5)

    selected_ids = [d["id"] for d in result]
    assert 3 in selected_ids, f"MMR 应选中差异文档 id=3，实际选中: {selected_ids}"


@pytest.mark.asyncio
async def test_mmr_lambda_1_equals_greedy():
    """lambda=1.0 时退化为纯贪心（按 score 排序）"""
    docs = [_make_doc(i, score=1.0 - i * 0.1) for i in range(6)]
    vecs = {i: np.random.randn(1024).astype(np.float32) for i in range(6)}
    query_vec = np.random.randn(1024).tolist()

    with patch("app.main.db_pool", _mock_pool(vecs)):
        result = await mmr_select(query_vec, docs, k=3, lam=1.0)

    selected_ids = [d["id"] for d in result]
    assert selected_ids == [0, 1, 2], f"lambda=1 应按分数贪心选择，实际: {selected_ids}"
