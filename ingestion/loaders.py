from pathlib import Path

from bs4 import BeautifulSoup

from langchain_core.documents import Document

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf", ".docx", ".html", ".htm"}


def iter_supported_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_SUFFIXES else []
    return sorted(
        p
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(path).load()
    if suffix == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        return Docx2txtLoader(str(path)).load()
    if suffix in {".html", ".htm"}:
        text = BeautifulSoup(
            path.read_text(encoding="utf-8", errors="ignore"), "html.parser"
        ).get_text(" ")
        return [
            Document(
                page_content=text, metadata={"source": str(path), "file_type": suffix}
            )
        ]
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [
        Document(page_content=text, metadata={"source": str(path), "file_type": suffix})
    ]
