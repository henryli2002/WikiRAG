"""测试 RRF 混合排分逻辑（纯函数，不依赖数据库和模型）"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.main import hybrid_merge


def _make_doc(doc_id, score=0.0):
    return {"id": doc_id, "content": f"doc_{doc_id}", "metadata": {}, "score": score}


def test_union_of_two_lists():
    """两路结果的并集：不重不漏"""
    vec = [_make_doc(1), _make_doc(2), _make_doc(3)]
    bm25 = [_make_doc(3), _make_doc(4), _make_doc(5)]
    merged = hybrid_merge(vec, bm25, k=10)
    merged_ids = {d["id"] for d in merged}
    assert merged_ids == {1, 2, 3, 4, 5}


def test_overlap_gets_higher_score():
    """同时出现在两路中的文档应该得分更高"""
    vec = [_make_doc(1), _make_doc(2)]
    bm25 = [_make_doc(2), _make_doc(3)]
    merged = hybrid_merge(vec, bm25, k=10)
    score_map = {d["id"]: d["score"] for d in merged}
    # id=2 在两路中都出现，RRF 得分应最高
    assert score_map[2] > score_map[1]
    assert score_map[2] > score_map[3]


def test_respects_k_limit():
    """结果数不超过 k"""
    vec = [_make_doc(i) for i in range(20)]
    bm25 = [_make_doc(i + 20) for i in range(20)]
    merged = hybrid_merge(vec, bm25, k=5)
    assert len(merged) == 5


def test_rank_order_matters():
    """排名靠前的文档 RRF 得分更高"""
    vec = [_make_doc(1), _make_doc(2), _make_doc(3)]
    bm25 = []
    merged = hybrid_merge(vec, bm25, k=10)
    scores = [d["score"] for d in merged]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]


def test_empty_inputs():
    """两路都为空应返回空列表"""
    merged = hybrid_merge([], [], k=10)
    assert merged == []


def test_one_side_empty():
    """只有一路有结果也能正常工作"""
    vec = [_make_doc(1), _make_doc(2)]
    merged = hybrid_merge(vec, [], k=10)
    assert len(merged) == 2

    merged2 = hybrid_merge([], [_make_doc(3)], k=10)
    assert len(merged2) == 1
