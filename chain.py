"""
chain.py — LLM chain with strict grounding and confidence scoring.

Key design decisions:
  - Prompt explicitly forbids hallucination outside provided context
  - Returns structured output: answer + confidence + sources + fallback flag
  - Confidence is derived from retrieval scores + LLM self-assessment
"""

import os
import json
import logging
from typing import List, Optional
from dataclasses import dataclass, field, asdict

from openai import OpenAI

from .retriever import RetrievalResult, retrieve

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.30"))
NO_ANSWER_THRESHOLD = float(os.getenv("NO_ANSWER_THRESHOLD", "0.40"))


SYSTEM_PROMPT = """You are an enterprise knowledge assistant. Your job is to answer questions
using ONLY the context passages provided below. You must follow these rules without exception:

1. Base your answer exclusively on the provided context passages.
2. If the context does not contain enough information to answer the question, say exactly:
   "I don't have enough information in the provided documents to answer this question."
3. Never fabricate facts, statistics, names, dates, or policies not present in the context.
4. Always cite which source document(s) your answer comes from.
5. After your answer, output a JSON block in this exact format (no markdown fences):
   {"confidence": <float 0.0-1.0>, "grounded": <true|false>, "sources": [<source filenames>]}
   - confidence: your honest assessment of how well the context supports the answer
   - grounded: true if the answer is fully supported by context, false if partially or not at all
"""

CONTEXT_TEMPLATE = """--- CONTEXT PASSAGES ---
{passages}
--- END OF CONTEXT ---

Question: {question}

Answer (cite sources, then output the JSON block):"""


@dataclass
class QAResult:
    question: str
    answer: str
    confidence: float          # 0.0–1.0
    grounded: bool             # False = LLM signaled it couldn't ground the answer
    no_answer_found: bool      # True = retrieval returned nothing useful
    retrieval_scores: List[float] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    raw_chunks: List[dict] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _build_context_block(chunks: List[RetrievalResult]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source_name = chunk.source.split("/")[-1]
        parts.append(
            f"[{i}] Source: {source_name} (relevance: {chunk.score:.2f})\n{chunk.text}"
        )
    return "\n\n".join(parts)


def _parse_llm_output(raw: str):
    """
    Split LLM output into (answer_text, json_block).
    The model is instructed to append a JSON block — we find the last '{'.
    """
    last_brace = raw.rfind("{")
    if last_brace == -1:
        return raw.strip(), {}

    answer_part = raw[:last_brace].strip()
    json_part = raw[last_brace:].strip()

    try:
        parsed_json = json.loads(json_part)
    except json.JSONDecodeError:
        parsed_json = {}

    return answer_part, parsed_json


def ask(
    question: str,
    top_k: int = 5,
    min_score: float = MIN_RETRIEVAL_SCORE,
    filter_source: Optional[str] = None,
) -> QAResult:
    """
    Full RAG pipeline: retrieve → build prompt → call LLM → parse + validate.

    Failure modes handled:
      - Empty collection / no matching chunks → no_answer_found=True
      - Low retrieval scores → no_answer_found=True
      - LLM signals it can't answer → grounded=False, confidence penalised
      - OpenAI API error → error field set, safe fallback returned
    """
    if not OPENAI_API_KEY:
        return QAResult(
            question=question,
            answer="",
            confidence=0.0,
            grounded=False,
            no_answer_found=True,
            error="OPENAI_API_KEY not configured.",
        )

    # --- Retrieval ---
    chunks = retrieve(question, top_k=top_k, min_score=min_score, filter_source=filter_source)

    if not chunks:
        return QAResult(
            question=question,
            answer="I don't have enough information in the provided documents to answer this question.",
            confidence=0.0,
            grounded=False,
            no_answer_found=True,
        )

    avg_score = sum(c.score for c in chunks) / len(chunks)

    # Hard cutoff: if best chunk is still weak, don't bother calling LLM
    if chunks[0].score < NO_ANSWER_THRESHOLD:
        return QAResult(
            question=question,
            answer="I don't have enough information in the provided documents to answer this question.",
            confidence=round(avg_score, 3),
            grounded=False,
            no_answer_found=True,
            retrieval_scores=[round(c.score, 3) for c in chunks],
            sources=list({c.source for c in chunks}),
        )

    # --- LLM call ---
    context_block = _build_context_block(chunks)
    user_message = CONTEXT_TEMPLATE.format(
        passages=context_block,
        question=question,
    )

    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.1,   # low temp for factual, grounded output
        )
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return QAResult(
            question=question,
            answer="",
            confidence=0.0,
            grounded=False,
            no_answer_found=False,
            error=str(e),
        )

    raw_output = response.choices[0].message.content or ""
    answer_text, llm_json = _parse_llm_output(raw_output)

    # --- Parse LLM's self-assessment ---
    llm_confidence = float(llm_json.get("confidence", 0.5))
    llm_grounded = bool(llm_json.get("grounded", True))
    llm_sources = llm_json.get("sources", [])

    # Blend retrieval quality into confidence
    blended_confidence = round((llm_confidence * 0.6 + avg_score * 0.4), 3)

    no_answer_signal = (
        "don't have enough information" in answer_text.lower()
        or not llm_grounded
    )

    return QAResult(
        question=question,
        answer=answer_text,
        confidence=blended_confidence if not no_answer_signal else round(avg_score * 0.5, 3),
        grounded=llm_grounded and not no_answer_signal,
        no_answer_found=no_answer_signal,
        retrieval_scores=[round(c.score, 3) for c in chunks],
        sources=llm_sources or list({c.source.split("/")[-1] for c in chunks}),
        raw_chunks=[c.to_dict() for c in chunks],
    )
