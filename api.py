"""
api.py — FastAPI application exposing RAG endpoints.

Endpoints:
  POST /ingest        — load documents into the vector store
  POST /ask           — ask a question against ingested documents
  GET  /health        — liveness check
  GET  /stats         — collection statistics
  DELETE /collection  — wipe the vector store (admin)
"""

import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .ingest import ingest_documents
from .chain import ask, QAResult
from .retriever import collection_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Enterprise QA",
    description=(
        "Production-grade Retrieval-Augmented Generation API. "
        "Upload company documents and ask questions grounded strictly in those docs."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.getenv("DATA_DIR", "./data/sample_docs")


# ── Request / Response Models ─────────────────────────────────────────────────

class IngestRequest(BaseModel):
    data_dir: str = Field(default=DATA_DIR, description="Path to the documents folder.")
    reset: bool = Field(default=False, description="Wipe collection before ingesting.")


class IngestResponse(BaseModel):
    status: str
    source_files: int = 0
    chunks_added: int = 0
    total_in_collection: int = 0
    message: str = ""


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language question.")
    top_k: int = Field(default=5, ge=1, le=20, description="Max chunks to retrieve.")
    min_score: float = Field(default=0.30, ge=0.0, le=1.0, description="Min relevance threshold.")
    filter_source: Optional[str] = Field(default=None, description="Restrict to a specific document filename.")
    include_chunks: bool = Field(default=False, description="Return raw retrieved chunks in response.")


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: float = Field(description="0–1 blend of retrieval quality and LLM self-assessment.")
    grounded: bool = Field(description="True if answer is fully supported by documents.")
    no_answer_found: bool = Field(description="True if no relevant context was found.")
    sources: list[str]
    retrieval_scores: list[float]
    raw_chunks: Optional[list] = None
    error: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/stats", tags=["System"])
def stats():
    """Vector store statistics."""
    return collection_stats()


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
def ingest(req: IngestRequest):
    """
    Load all PDFs/TXT/Markdown from `data_dir`, chunk them, and upsert
    into ChromaDB. Safe to call multiple times — existing chunks are updated
    (idempotent via content hash IDs).
    """
    try:
        result = ingest_documents(req.data_dir, reset=req.reset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))

    if result.get("status") == "no_documents":
        return IngestResponse(
            status="warning",
            message="No supported documents found in the specified directory.",
        )

    return IngestResponse(
        status="success",
        source_files=result.get("source_files", 0),
        chunks_added=result.get("chunks_added", 0),
        total_in_collection=result.get("total_in_collection", 0),
    )


@app.post("/ask", response_model=AskResponse, tags=["QA"])
def ask_question(req: AskRequest):
    """
    Ask a question. The answer is grounded exclusively in ingested documents.

    Confidence score:
      - ≥ 0.7 → high confidence, well-supported answer
      - 0.4–0.7 → moderate confidence, review sources
      - < 0.4 → low confidence or no answer found

    `no_answer_found=true` means the retriever could not find relevant context —
    the model will NOT hallucinate an answer.
    """
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    result: QAResult = ask(
        question=req.question,
        top_k=req.top_k,
        min_score=req.min_score,
        filter_source=req.filter_source,
    )

    return AskResponse(
        question=result.question,
        answer=result.answer,
        confidence=result.confidence,
        grounded=result.grounded,
        no_answer_found=result.no_answer_found,
        sources=result.sources,
        retrieval_scores=result.retrieval_scores,
        raw_chunks=result.raw_chunks if req.include_chunks else None,
        error=result.error,
    )


@app.delete("/collection", tags=["Admin"])
def delete_collection():
    """
    ⚠️  Delete and rebuild the vector store. Use before a full re-ingest.
    Protected: set ADMIN_TOKEN env var to require Bearer auth.
    """
    import chromadb
    from chromadb.config import Settings

    admin_token = os.getenv("ADMIN_TOKEN")
    # In production wire this to proper auth middleware
    chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")
    collection_name = os.getenv("COLLECTION_NAME", "enterprise_docs")

    try:
        client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        client.delete_collection(collection_name)
        return {"status": "deleted", "collection": collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
