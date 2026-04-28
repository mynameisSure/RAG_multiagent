import os
from pathlib import Path

from dotenv import load_dotenv

from RAG_multiagent.agents.supervisor import AgenticResearchOrchestrator
from RAG_multiagent.models import ResearchReport, ResearchRequest

load_dotenv()


class ResearchService:
    def __init__(self):
        self.orchestrator = AgenticResearchOrchestrator()

    def run(self, request: ResearchRequest):
        return self.orchestrator.run(request)

    def save_report(self, report: ResearchReport, path: str | Path | None = None):
        output = Path(path) if path else os.getenv("OUTPUT_DIR") / f"{_slug(report.title).md}"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.report_markdown, encoding="utf-8")
        return output


def _slug(text: str) -> str:
    """把报告标题转换成适合文件系统使用的 Markdown 文件名。"""
    keep = "".join(ch if ch.isalnum() else "-" for ch in text.lower())
    return "-".join(part for part in keep.split("-") if part)[:120] or "report"
