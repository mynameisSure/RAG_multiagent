"""提供配置查看、文档入库和研究执行的 Typer 命令行入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from RAG_multiagent.config import get_settings
from RAG_multiagent.ingestion.pipeline import ingest_path
from RAG_multiagent.models import ReportFormat, ResearchDepth, ResearchRequest
from RAG_multiagent.services.research_service import ResearchService
from RAG_multiagent.utils.logging import configure_logging

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.callback()
def main(log_level: str = typer.Option("INFO", help="Logging level")) -> None:
    """在任何子命令执行前配置当前进程的日志输出。"""
    configure_logging(log_level)


@app.command()
def config() -> None:
    """打印当前生效的非敏感配置，便于排查运行环境。"""
    settings = get_settings()
    table = Table("Setting", "Value")
    for key, value in settings.model_dump(exclude={"tavily_api_key"}).items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def ingest(
    path: Path = typer.Argument(
        ..., help="File or directory containing txt/md/pdf/docx/html files"
    ),
    force: bool = typer.Option(
        False, help="Re-index files even if their hash is unchanged"
    ),
) -> None:
    """把本地支持格式的文档切块后写入向量库。"""
    stats = ingest_path(path, force=force)
    console.print_json(stats.model_dump_json())


@app.command()
def ask(
    topic: str = typer.Argument(..., help="Research topic"),
    depth: ResearchDepth = typer.Option(ResearchDepth.detailed, "--depth"),
    format: ReportFormat = typer.Option(ReportFormat.markdown, "--format"),  # noqa: A002 - 对外参数名保持为 format
    no_web: bool = typer.Option(False, help="Disable web search"),
    out: Optional[Path] = typer.Option(None, help="Optional Markdown output path"),
) -> None:
    """运行一次研究任务，并把报告打印到终端或保存到指定路径。"""
    request = ResearchRequest(
        topic=topic, depth=depth, report_format=format, require_web=not no_web
    )
    service = ResearchService()
    report = service.run(request)
    console.print(report.report_markdown)
    if out:
        saved = service.save_report(report, out)
        console.print(f"\nSaved: {saved}")


if __name__ == "__main__":
    app()
