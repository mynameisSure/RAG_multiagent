from datetime import datetime, timezone

from langchain_tavily import TavilySearch

from RAG_multiagent.models import Evidence, SourceType


def tavily_search(query: str, max_results: int = 5) -> list[Evidence]:
    tool = TavilySearch(max_results=max_results, include_answer=False, include_raw_content=False)

    result = tool.invoke({"query": query})

    rows = result.get("results", result if isinstance(result, list) else [])

    evidence: list[Evidence] = []

    for idx, row in enumerate(rows):
        title = row.get("title") or row.get("url") or query
        content = row.get("content") or row.get("snippet") or row.get("raw_content") or ""
        if not content:
            continue
        evidence.append(
            Evidence(
                id=f"W{abs(hash((query, idx, title))) % 10_000_000}",
                source_type=SourceType.web,
                title=title,
                content=content,
                url=row.get("url"),
                source=row.get("source"),
                retrieved_at=datetime.now(timezone.utc),
                score=row.get("score" or 0.5),
                metadata={"query": query},
            )
        )
    return evidence
