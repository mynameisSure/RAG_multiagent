"""验证证据去重和评分过滤逻辑。"""

from RAG_multiagent.models import Evidence, SourceType
from RAG_multiagent.retrieval.evidence import deduplicate_evidence, grade_and_filter


def test_deduplicate_prefers_higher_score():
    """同一 URL 的证据重复时应保留分数更高的版本。"""
    low = Evidence(
        id="W1",
        source_type=SourceType.web,
        title="same",
        content="x",
        url="https://a.org",
        score=0.2,
    )
    high = Evidence(
        id="W2",
        source_type=SourceType.web,
        title="same",
        content="y",
        url="https://a.org",
        score=0.9,
    )
    result = deduplicate_evidence([low, high])
    assert len(result) == 1
    assert result[0].id == "W2"


def test_grade_and_filter_limits_results():
    """证据评分结果应按最终分排序，并遵守返回数量上限。"""
    items = [
        Evidence(
            id=f"L{i}",
            source_type=SourceType.local,
            title=f"t{i}",
            content="content",
            score=0.7,
        )
        for i in range(10)
    ]
    graded = grade_and_filter(items, min_score=0.1, limit=3)
    assert len(graded) == 3
    assert graded[0].final_score >= graded[-1].final_score
