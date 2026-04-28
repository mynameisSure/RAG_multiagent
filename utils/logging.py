"""为 CLI、API 和后台流程提供统一的结构化日志配置。"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    """配置标准库日志级别，并让 structlog 以 JSON 形式输出日志。"""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
