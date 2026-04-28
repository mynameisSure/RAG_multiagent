from collections import OrderedDict
from datetime import datetime
from urllib.parse import urlparse

from RAG_multiagent.models import Evidence, GradeEvidence, SourceType


def deduplicate_evidence(items: list[Evidence]) -> list[Evidence]:
    seen: OrderedDict[str, Evidence] = OrderedDict()
    for item in items:
        key = item.url or f"{item.title}:{item.content[:240]}"
        if key not in seen or item.score > seen[key].score:
            seen[key] = item
    return list(seen.values())


def credibility_score(item: Evidence) -> float:
    if item.source_type == SourceType.local:
        return 0.82
    if not item.url:
        return 0.45
    host = urlparse(item.url).netloc.lower()

    if host.endswith((".edu", ".gov", ".org")):
        return 0.86

    if any(domain in host for domain in ["arxiv.org", "nature.com", "science.org", "acm.org", "ieee.org"]):
        return 0.9
    return 0.62


def freshness_score(item: Evidence) -> float:
    if not item.published_at:
        return 0.55
    try:
        year = int(item.published_at[:4])
    except ValueError:
        return 0.55
    current_year = datetime.utcnow().year
    age = max(0, current_year - year)
    return max(0.25, 1.0 - age * 0.08)


def grade_and_filter(items: list[Evidence], min_score: float, limit: int) -> list[GradeEvidence]:
    graded = []
    for item in deduplicate_evidence(items):
        relevance = min(1.0, max(0.0, item.score))
        graded_item = GradeEvidence(
            evidence=item,
            relevance=relevance,
            credibility=credibility_score(item),
            freshness=freshness_score(item),
            reason="weighted relevance, source credibility and freshness",
        )
        if graded_item.final_score >= min_score:
            graded.append(graded_item)

    return sorted(graded, key=lambda g: g.final_score, reverse=True)[:limit]


def format_evidence_for_prompt(items: list[Evidence], max_chars: int = 18000) -> str:
    rows: list[str] = []
    total = 0
    for item in items:
        row = (
            f"source_type={item.source_type.value}; source={item.source or item.url or 'unknown'}; "
            f"score={item.score:.2f}\n"
            f"excerpt: {item.content[:1400]}"
            f"item_id:{item.id}"
        )
        if total + len(row) > max_chars:
            break
        rows.append(row)
        total += len(row)
    return "\n\n".join(rows)
