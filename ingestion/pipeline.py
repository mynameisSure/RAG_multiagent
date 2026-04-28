import os
from pathlib import Path

from dotenv import load_dotenv

from RAG_multiagent.ingestion.chunker import split_documents
from RAG_multiagent.ingestion.loaders import iter_supported_files, load_file
from RAG_multiagent.models import IngestStates
from RAG_multiagent.retrieval.vector_store import add_documents
from RAG_multiagent.storage.manifest import DocumentManifest

load_dotenv()


def ingest_path(path: str | Path, force: bool = False):
    root = Path(path)
    manifest = DocumentManifest(Path("data/vectorstore/manifest.json"))
    stats = IngestStates()
    for file_path in iter_supported_files(root):
        stats.files_seen += 1
        if not force and manifest.is_seen(file_path):
            stats.skipped.append(str(file_path))
            continue
        try:
            docs = load_file(file_path)
            for doc in docs:
                doc.metadata.setdefault("source", str(file_path))
                doc.metadata.setdefault("filename", file_path.name)

            chunks = split_documents(docs)

            added = add_documents(chunks)

            manifest.mark_seen(file_path, added)

            stats.files_indexed += 1
            stats.chunks_added += added
        except Exception as exc:  # noqa: BLE001 - 保持批量入库的容错能力
            stats.errors.append(f"{file_path}: {exc}")
            print(exc)
    return stats
