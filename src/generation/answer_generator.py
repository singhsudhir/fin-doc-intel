from __future__ import annotations

import os
import re

import structlog
from dotenv import load_dotenv
from google import genai

from src.generation.gemini_utils import generate_with_fallback
from src.models.responses import AnswerResponse, Citation
from src.models.schemas import RetrievedChunk

load_dotenv()
log = structlog.get_logger()

SYSTEM_INSTRUCTION = """You are a financial document analyst. You answer \
questions based ONLY on the provided context documents.

Rules:
1. Every factual claim must cite its source: [Document: X, Page: Y]
2. If you cannot find the answer, say: "I cannot find this information \
in the provided documents."
3. For numerical data, quote the exact figures from the source
4. Never speculate or use knowledge outside the provided context
5. You can answer in Dutch or English — match the language of the question
"""

# Matches [Document: some_name.pdf, Page: 42] with flexible spacing
_CITATION_RE = re.compile(
    r"\[Document:\s*(?P<doc>[^\],]+?)\s*,\s*Page:\s*(?P<page>\d+)\s*\]",
    re.IGNORECASE,
)


class AnswerGenerator:
    """Generates a grounded answer from retrieved chunks using Gemini 2.5 Pro."""

    def __init__(self) -> None:
        self._gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> AnswerResponse:
        """Return a structured answer with citations for the given query and chunks."""
        if not chunks:
            return AnswerResponse(
                answer=(
                    "I cannot find this information in the provided documents."
                ),
                citations=[],
                model_used=_ANSWER_MODEL,
                chunks_used=0,
            )

        user_prompt = _build_prompt(query, chunks)

        config = genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.1,
        )
        response, model_used = generate_with_fallback(self._gemini, user_prompt, config)

        answer_text = response.text.strip()
        citations = _extract_citations(answer_text, chunks)

        log.info(
            "answer_generated",
            query_preview=query[:60],
            chunks_used=len(chunks),
            citations_found=len(citations),
            model=model_used,
        )

        return AnswerResponse(
            answer=answer_text,
            citations=citations,
            model_used=model_used,
            chunks_used=len(chunks),
        )


# ------------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------------


def _build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    lines = ["Context Documents:\n"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"[{i}] Document: {chunk.document_name or chunk.doc_id}, "
            f"Page: {chunk.page_number}\n"
            f"{chunk.text}\n"
        )
    lines.append(f"\nQuestion: {query}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Citation extraction
# ------------------------------------------------------------------


def _extract_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Parse [Document: X, Page: Y] tags from the answer and map to source chunks."""
    seen: set[tuple[str, int]] = set()
    citations: list[Citation] = []

    for match in _CITATION_RE.finditer(answer):
        doc_name = match.group("doc").strip()
        page_num = int(match.group("page"))
        key = (doc_name, page_num)
        if key in seen:
            continue
        seen.add(key)

        # Find the source chunk with the closest document/page match
        source_text = _find_source_text(doc_name, page_num, chunks)
        citations.append(
            Citation(
                document_name=doc_name,
                page_number=page_num,
                source_text=source_text,
            )
        )

    # Fallback: if the model answered but wrote no citation tags, surface the
    # top chunk so the caller always has at least one reference.
    if not citations and answer and "cannot find" not in answer.lower():
        top = chunks[0]
        citations.append(
            Citation(
                document_name=top.document_name or top.doc_id,
                page_number=top.page_number,
                source_text=top.text[:300],
            )
        )

    return citations


def _find_source_text(doc_name: str, page_num: int, chunks: list[RetrievedChunk]) -> str:
    """Return up to 300 chars of the chunk that best matches the citation."""
    doc_lower = doc_name.lower()
    # Exact page + document name substring match
    for chunk in chunks:
        chunk_doc = (chunk.document_name or "").lower()
        if chunk.page_number == page_num and (
            doc_lower in chunk_doc or chunk_doc in doc_lower
        ):
            return chunk.text[:300]
    # Page-only fallback
    for chunk in chunks:
        if chunk.page_number == page_num:
            return chunk.text[:300]
    return ""
