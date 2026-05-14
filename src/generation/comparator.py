from __future__ import annotations

import os
import re

import structlog
from dotenv import load_dotenv
from google import genai

from src.embedding.embedder import Embedder
from src.embedding.qdrant_store import QdrantStore
from src.generation.gemini_utils import generate_with_fallback
from src.models.responses import Citation, CompareResponse
from src.models.schemas import RetrievedChunk

load_dotenv()
log = structlog.get_logger()

_SYSTEM_INSTRUCTION = """You are a senior financial analyst comparing two financial documents.
Provide a structured, objective comparison covering each requested topic.

Rules:
1. For every claim, cite its source: [Document: X, Page: Y]
2. Use exact figures from the documents — do not round or paraphrase numbers
3. Be balanced: highlight both similarities and differences
4. If a topic is not addressed in one document, state that explicitly
5. You can answer in Dutch or English — match the language of the request
"""

_CITATION_RE = re.compile(
    r"\[Document:\s*(?P<doc>[^\],]+?)\s*,\s*Page:\s*(?P<page>\d+)\s*\]",
    re.IGNORECASE,
)


class Comparator:
    """Compares two financial documents across specified topics using Gemini."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: QdrantStore | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.store = store or QdrantStore()
        self._gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def compare(
        self,
        document_name_a: str,
        document_name_b: str,
        focus_topics: list[str],
    ) -> CompareResponse:
        """Retrieve relevant chunks from both documents and ask Gemini to compare."""
        combined_query = " ".join(focus_topics)
        query_vec = self.embedder.embed_query(combined_query)

        chunks_a = self.store.search(query_vec, top_k=8, document_filter=document_name_a)
        chunks_b = self.store.search(query_vec, top_k=8, document_filter=document_name_b)

        topics_md = "\n".join(f"- {t}" for t in focus_topics)
        user_prompt = (
            f"Compare the two financial documents on these topics:\n{topics_md}\n\n"
            f"--- Document A: {document_name_a} ---\n"
            f"{_build_context(chunks_a)}\n\n"
            f"--- Document B: {document_name_b} ---\n"
            f"{_build_context(chunks_b)}\n\n"
            "Provide a section for each topic above. "
            "Always cite [Document: X, Page: Y] for every claim."
        )

        response, model_used = generate_with_fallback(
            self._gemini,
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                temperature=0.1,
            ),
        )

        text = response.text.strip()
        log.info(
            "comparison_generated",
            doc_a=document_name_a,
            doc_b=document_name_b,
            model=model_used,
        )

        return CompareResponse(
            document_name_a=document_name_a,
            document_name_b=document_name_b,
            comparison=text,
            citations_a=_extract_citations_for(text, document_name_a, chunks_a),
            citations_b=_extract_citations_for(text, document_name_b, chunks_b),
            model_used=model_used,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "[No relevant content retrieved]"
    return "\n\n".join(
        f"[Page {c.page_number}]\n{c.text}" for c in chunks
    )


def _extract_citations_for(
    answer: str,
    doc_name: str,
    chunks: list[RetrievedChunk],
) -> list[Citation]:
    doc_lower = doc_name.lower()
    seen: set[int] = set()
    citations: list[Citation] = []

    for match in _CITATION_RE.finditer(answer):
        ref_doc = match.group("doc").strip().lower()
        page_num = int(match.group("page"))
        if doc_lower not in ref_doc and ref_doc not in doc_lower:
            continue
        if page_num in seen:
            continue
        seen.add(page_num)
        source = next(
            (c.text[:300] for c in chunks if c.page_number == page_num), ""
        )
        citations.append(
            Citation(document_name=doc_name, page_number=page_num, source_text=source)
        )

    return citations
