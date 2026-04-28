import hashlib
import math
import os
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from RAG_multiagent.config import Settings, get_settings

load_dotenv()


def get_chat_model(settings):
    if settings == "dashscope":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("RA_CHAT_MODEL"),
            temperature=1.0,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
        )

        return Chat(model=settings.chat_model, temperature=0.8)
    raise ValueError(f"Unsupported LLM provider: {os.getenv('RA_LLM_PROVIDER')}")


def get_embeddings(settings: Settings):
    from langchain_community.embeddings import DashScopeEmbeddings


def compact_messages(items: Iterable[str], max_chars: int = 12000) -> str:
    """在字符预算内拼接提示词片段。"""
    output: list[str] = []
    total = 0

    for item in items:
        if total + len(item) > max_chars:
            break
        output.append(item)
        total += len(item)

    return "\n".join(output)
