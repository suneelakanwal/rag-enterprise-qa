"""
ingest.py — Document loading, chunking, and embedding into ChromaDB.

Supports: PDF, TXT, Markdown
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_docs")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


def load_documents(data_dir: str) -> List[Document]:
    """Load all supported documents from a directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents: List[Document] = []

    # PDF files
    pdf_files = list(data_path.rglob("*.pdf"))
    for pdf_path in pdf_files:
        logger.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = str(pdf_path)
            doc.metadata["file_type"] = "pdf"
        documents.extend(docs)

    # TXT files
    txt_files = list(data_path.rglob("*.txt"))
    for txt_path in txt_files:
        logger.info(f"Loading TXT: {txt_path}")
        loader = TextLoader(str(txt_path), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = str(txt_path)
            doc.metadata["file_type"] = "txt"
        documents.extend(docs)

    # Markdown files
    md_files = list(data_path.rglob("*.md"))
    for md_path in md_files:
        logger.info(f"Loading Markdown: {md_path}")
        loader = TextLoader(str(md_path), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = str(md_path)
            doc.metadata["file_type"] = "markdown"
        documents.extend(docs)

    logger.info(f"Loaded {len(documents)} document pages from {data_dir}")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def _make_chunk_id(chunk: Document, index: int) -> str:
    """Deterministic ID so re-ingestion is idempotent."""
    content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
    source = chunk.metadata.get("source_file", "unknown")
    return f"{Path(source).stem}_{index}_{content_hash}"


def ingest_documents(
    data_dir: str,
    reset: bool = False,
) -> dict:
    """
    Full pipeline: load → chunk → embed → store.

    Args:
        data_dir: Path to folder containing source documents.
        reset:    If True, wipe and rebuild the collection.

    Returns:
        Summary dict with counts.
    """
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Load & chunk
    raw_docs = load_documents(data_dir)
    chunks = chunk_documents(raw_docs)

    if not chunks:
        return {"status": "no_documents", "chunks_added": 0}

    # Build lists for ChromaDB batch upsert
    ids, texts, metadatas = [], [], []
    for i, chunk in enumerate(chunks):
        chunk_id = _make_chunk_id(chunk, i)
        ids.append(chunk_id)
        texts.append(chunk.page_content)
        metadatas.append(
            {
                "source_file": str(chunk.metadata.get("source_file", "")),
                "file_type": str(chunk.metadata.get("file_type", "")),
                "page": str(chunk.metadata.get("page", "")),
                "chunk_index": str(i),
            }
        )

    # ChromaDB uses its own embedding function (default: all-MiniLM-L6-v2)
    # Swap to OpenAI embeddings via retriever.py for production
    BATCH = 100
    for start in range(0, len(ids), BATCH):
        collection.upsert(
            ids=ids[start : start + BATCH],
            documents=texts[start : start + BATCH],
            metadatas=metadatas[start : start + BATCH],
        )
        logger.info(f"Upserted batch {start // BATCH + 1}")

    count = collection.count()
    logger.info(f"Collection '{COLLECTION_NAME}' now has {count} chunks.")

    return {
        "status": "success",
        "source_files": len(set(c.metadata.get("source_file", "") for c in chunks)),
        "chunks_added": len(ids),
        "total_in_collection": count,
    }


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/sample_docs"
    reset = "--reset" in sys.argv
    result = ingest_documents(data_dir, reset=reset)
    print(result)
