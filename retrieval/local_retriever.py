import os

from dotenv import load_dotenv

from RAG_multiagent.models import Evidence, SourceType
from RAG_multiagent.retrieval.vector_store import similarity_search

load_dotenv()


def retrieve_local(query: str, settings=None, k: int | None = None):
    docs = similarity_search(query, k=k or os.getenv("RA_RETRIEVAL_K"))
    evidence: list[Evidence] = []

    for idx, doc in enumerate(docs):
        source = str(doc.metadata.get("source") or doc.metadata.get("filename") or "local")
        title = str(doc.metadata.get("title") or doc.metadata.get("filename") or source)

        evidence.append(
            Evidence(
                id=f"L{abs(hash((query, source, idx, doc.page_content[:60]))) % 10_000_000}",
                source_type=SourceType.local,
                title=title,
                content=doc.page_content,
                source=source,
                score=max(0.4, 1.0 - idx * 0.6),
                metadata=dict(doc.metadata) | {"query": query},
            )
        )

    return evidence
