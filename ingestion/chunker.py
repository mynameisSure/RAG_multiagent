import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from RAG_multiagent.config import Settings, get_settings

load_dotenv()


def split_documents(documents: list[Document], settings: Settings | None = None) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),
        separators=["\n\n", "\n", "。", "；", "，", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    return chunks
