from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


class ResearchDepth(str, Enum):
    brief = "brief"
    detailed = "detailed"
    deep = "deep"


class ReportFormat(str, Enum):
    markdown = "markdown"
    academic = "academic"
    executive = "executive"


class SourceType(str, Enum):
    local = "local"
    web = "web"
    academic = "academic"
    user = "user"


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=100)
    depth: ResearchDepth = ResearchDepth.detailed
    report_format: ReportFormat = ReportFormat.markdown
    language: Literal["zh", "en"] = "zh"
    require_web: bool = True
    allowed_domains: list[str] = Field(default_factory=list)
    blacklist: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    title: str
    intent: str
    research_questions: list[str] = Field(min_length=2, max_length=10)
    local_queries: list[str] = Field(default_factory=list, max_length=12)
    web_queries: list[str] = Field(default_factory=list, max_length=12)
    expected_sections: list[str] = Field(default_factory=list, max_length=12)
    risk_notes: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    id: str
    source_type: SourceType
    title: str
    content: str
    url: str | None = None
    source: str | None = None
    author: str | None = None
    published_at: str | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        return " ".join(value.split())

    def citation_label(self):
        return f"{self.id}"


class GradeEvidence(BaseModel):
    evidence: Evidence
    relevance: float = Field(ge=0.0, le=1.0)
    credibility: float = Field(ge=0.0, le=1.0)
    freshness: float = Field(ge=0.0, le=1.0)
    reason: str = ""

    @property
    def final_score(self) -> float:
        return round(self.relevance * 0.5 + self.credibility * 0.3 + self.freshness * 0.2, 4)


class Claim(BaseModel):
    text: str
    evidence_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class Critique(BaseModel):
    passed: bool
    quality_score: float = Field(ge=0.0, le=1.0)
    missing_evidence: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    next_queries: list[str] = Field(default_factory=list)
    notes: str = ""


class ResearchReport(BaseModel):
    title: str
    executive_summary: str
    report_markdown: str
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)
    critique: Critique | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IngestStates(BaseModel):
    files_seen: int = 0
    files_indexed: int = 0
    chunks_added: int = 0
    skipped: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
