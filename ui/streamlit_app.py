"""提供面向人工操作的 Streamlit 研究助手界面。"""

from __future__ import annotations

import streamlit as st

from RAG_multiagent.ingestion.pipeline import ingest_path
from RAG_multiagent.models import ReportFormat, ResearchDepth, ResearchRequest
from RAG_multiagent.services.research_service import ResearchService

st.set_page_config(page_title="Research Assistant Pro", layout="wide")
st.title("Research Assistant Pro")
st.caption("Production-oriented RAG + multi-agents research workflow")

with st.sidebar:
    st.header("Knowledge base")
    ingest_dir = st.text_input("Directory to ingest", "data/raw")
    force = st.checkbox("Force re-index", value=False)
    if st.button("Ingest documents"):
        stats = ingest_path(ingest_dir, force=force)
        st.json(stats.model_dump())

    st.header("Research options")
    depth = st.selectbox("Depth", [d.value for d in ResearchDepth], index=1)
    report_format = st.selectbox("Format", [f.value for f in ReportFormat], index=0)
    require_web = st.checkbox("Use web search", value=True)

query = st.text_area(
    "Research topic",
    height=120,
    placeholder="例如：多智能体 RAG 在企业知识管理中的应用",
)
if st.button("Run research", type="primary", disabled=not query.strip()):
    request = ResearchRequest(
        topic=query.strip(),
        depth=ResearchDepth(depth),
        report_format=ReportFormat(report_format),
        require_web=require_web,
    )
    with st.spinner("Running research..."):
        report = ResearchService().run(request)
    st.subheader(report.title)
    st.markdown(report.report_markdown)
    with st.expander("Evidence"):
        st.json([e.model_dump(mode="json") for e in report.evidence])
    with st.expander("Critique"):
        st.json(report.critique.model_dump() if report.critique else {})
    with st.expander("Agent Trace"):
        st.json(report.agent_trace)
