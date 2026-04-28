from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从环境变量和 .env 文件加载的应用配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_prefix="RA_",
    )

    app_name: str = "research-assistant-pro"
    environment: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"

    llm_provider: Literal["dashscope", "ollama", "openai"] = "dashscope"
    chat_model: str = "qwen-plus"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    dashscope_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("DASHSCOPE_API_KEY", "RA_DASHSCOPE_API_KEY"),
    )
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    embedding_provider: Literal["dashscope", "huggingface", "hash", "openai"] = "dashscope"
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int | None = None

    vector_store_path: Path = Path("data/vectorstore")
    collection_name: str = "research_assistant"

    chunk_size: int = Field(default=900, ge=200, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    retrieval_k: int = Field(default=8, ge=1, le=50)

    web_search_enabled: bool = True

    tavily_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("TAVILY_API_KEY", "RA_TAVILY_API_KEY"),
    )

    max_iterations: int = Field(default=2, ge=1, le=5)
    quality_threshold: float = Field(default=0.78, ge=0.0, le=1.0)

    sqlite_checkpoint_path: Path = Path("data/checkpoints.sqlite")
    outputs_dir: Path = Path("data/outputs")

    request_timeout_seconds: float = Field(default=60.0, gt=0)
    max_evidence_items: int = Field(default=24, ge=4, le=80)
    min_evidence_score: float = Field(default=0.35, ge=0.0, le=1.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回缓存后的配置对象，并确保运行时需要的数据目录已经创建。"""
    settings = Settings()

    settings.vector_store_path.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    settings.sqlite_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    return settings
