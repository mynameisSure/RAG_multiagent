import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from RAG_multiagent.config import Settings, get_settings
from RAG_multiagent.llm import get_embeddings

load_dotenv()


def get_vector_store(
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
):
    settings = settings or get_settings()
    embeddings = embeddings or get_embeddings(settings)
    Path(os.getenv("RA_VECTOR_STORE_PATH")).mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=os.getenv("RA_COLLECTION_NAME"),
        embedding_function=embeddings,
        persist_directory=str(os.getenv("RA_VECTOR_STORE_PATH")),
    )


def add_documents(documents: list[Document]) -> int:
    if not documents:
        return 0

    store = get_vector_store()
    store.add_documents(documents)
    return len(documents)


def similarity_search(query, k):
    store = get_vector_store()
    return store.similarity_search(query, k=k or os.getenv("RA_RETRIEVAL_K"))
