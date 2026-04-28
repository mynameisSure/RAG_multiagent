"""验证核心 Pydantic 模型的默认值和字段规范化行为。"""

from RAG_multiagent.models import Evidence, ResearchRequest, SourceType


def test_research_request_defaults():
    """研究请求在未显式传参时应使用默认深度并启用网页检索。"""
    req = ResearchRequest(topic="多智能体 RAG 系统评估")
    assert req.depth.value == "detailed"
    assert req.require_web is True


def test_evidence_normalizes_content():
    """证据内容初始化时应折叠多余空白，并能生成引用标签。"""
    ev = Evidence(
        id="L1", source_type=SourceType.local, title="t", content="a\n\n b   c"
    )
    assert ev.content == "a b c"
    assert ev.citation_label() == "[L1]"
