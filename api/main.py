import os

from dotenv import load_dotenv
from fastapi import FastAPI

from RAG_multiagent.ingestion.pipeline import ingest_path
from RAG_multiagent.models import IngestStates, ResearchReport, ResearchRequest
from RAG_multiagent.services.research_service import ResearchService

load_dotenv()

app = FastAPI(title="RAG Multiagent", version="2.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research", response_model=ResearchReport)
def research(request: ResearchRequest):
    return ResearchService().run(request)


@app.post("/ingest", response_model=IngestStates)
def ingest(path: str, force: bool = False):
    return ingest_path(path, force=force)
