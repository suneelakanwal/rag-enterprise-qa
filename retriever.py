"""
retriever.py — Embedding model + vector search against ChromaDB.

Uses OpenAI embeddings for production quality.
Falls back to ChromaDB's default (all-MiniLM-L6-v2) when no API key is set.
"""

import os
import logging
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_docs")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))


class RetrievalResult:
    """A single retrieved chunk with its relevance score."""

    def __init__(self, text: str, source: str, score: float, metadata: dict):
        self.text = text
        self.source = source
        self.score = score          # 0.0–1.0, higher = more relevant (cosine similarity)
        self.metadata = metadata

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "score": round(self.score, 4),
            "metadata": self.metadata,
        }


def _get_embedding_function():
    """Return OpenAI embeddings if key available, else local fallback."""
    if OPENAI_API_KEY:
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small",
        )
    # Local fallback — works offline, lower quality
    logger.warning(
        "OPENAI_API_KEY not set. Using local all-MiniLM-L6-v2 embeddings."
    )
    return embedding_functions.DefaultEmbeddingFunction()


def get_collection():
    """Get (or create) the ChromaDB collection with the configured embedder."""
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    ef = _get_embedding_function()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def retrieve(
    query: str,
    top_k: int = TOP_K,
    min_score: float = 0.30,
    filter_source: Optional[str] = None,
) -> List[RetrievalResult]:
    """
    Embed `query` and return the top-k most relevant chunks.

    Args:
        query:         User's question.
        top_k:         Max chunks to return.
        min_score:     Minimum cosine similarity to include (0–1).
                       Chunks below this threshold are dropped — this is
                       the primary RAG failure-mode guard.
        filter_source: Optional filename to restrict search scope.

    Returns:
        List of RetrievalResult, sorted by descending score.
        Empty list means "no relevant context found."
    """
    collection = get_collection()

    if collection.count() == 0:
        logger.warning("Collection is empty — run ingest.py first.")
        return []

    where_filter = None
    if filter_source:
        where_filter = {"source_file": {"$contains": filter_source}}

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: List[RetrievalResult] = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        # ChromaDB cosine distance → similarity score
        score = 1.0 - dist

        if score < min_score:
            logger.debug(f"Dropped chunk (score={score:.3f} < {min_score}): {doc[:60]}…")
            continue

        retrieved.append(
            RetrievalResult(
                text=doc,
                source=meta.get("source_file", "unknown"),
                score=score,
                metadata=meta,
            )
        )

    logger.info(
        f"Retrieved {len(retrieved)}/{top_k} chunks for query: '{query[:80]}…'"
    )
    return retrieved


def collection_stats() -> dict:
    """Return basic stats about the vector store."""
    try:
        col = get_collection()
        count = col.count()
        return {"collection": COLLECTION_NAME, "total_chunks": count}
    except Exception as e:
        return {"error": str(e)}
