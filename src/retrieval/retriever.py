from __future__ import annotations

import json
import os

import structlog
from dotenv import load_dotenv
from google import genai

from src.embedding.embedder import Embedder
from src.embedding.qdrant_store import QdrantStore
from src.generation.gemini_utils import generate_with_fallback
from src.models.schemas import RetrievedChunk

load_dotenv()
log = structlog.get_logger()

_RERANK_MODELS = ["gemini-2.5-flash"]
# Truncate each chunk to this length when building the re-rank prompt so the
# 15-chunk prompt stays well under Gemini's context limit.
_RERANK_CHUNK_PREVIEW = 500


class Retriever:
    """Two-stage retrieval: dense vector search → Gemini re-ranking.

    Stage 1 — Qdrant cosine search returns `top_k_initial` candidates (default 15).
    Stage 2 — Gemini scores and reorders them, returning the best `top_k_final`
              (default 5) with updated relevance scores.
    If re-ranking fails for any reason the top-`top_k_final` Qdrant results
    are returned as a fallback.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: QdrantStore | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.store = store or QdrantStore()
        self._gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k_initial: int = 15,
        top_k_final: int = 5,
        document_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top `top_k_final` chunks most relevant to `query`."""
        query_vec = self.embedder.embed_query(query)
        candidates = self.store.search(
            query_vec,
            top_k=top_k_initial,
            document_filter=document_filter,
        )

        if not candidates:
            log.info("retriever_no_results", query_preview=query[:60])
            return []

        if len(candidates) <= top_k_final:
            return candidates

        try:
            reranked = self._rerank(query, candidates, top_k_final)
            log.info(
                "retrieval_complete",
                query_preview=query[:60],
                initial=len(candidates),
                final=len(reranked),
            )
            return reranked
        except Exception as exc:
            log.warning("rerank_failed_falling_back", error=str(exc))
            return candidates[:top_k_final]

    # ------------------------------------------------------------------
    # Gemini re-ranking
    # ------------------------------------------------------------------

    def _rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        chunk_block = "\n\n".join(
            f"Chunk {i} [Document: {c.document_name or 'unknown'}, Page: {c.page_number}]:\n"
            f"{c.text[:_RERANK_CHUNK_PREVIEW]}"
            for i, c in enumerate(chunks)
        )

        prompt = (
            f"Query: {query}\n\n"
            f"You are a relevance judge. Below are {len(chunks)} retrieved document chunks.\n"
            f"Select and rank the {top_k} most relevant chunks for answering the query.\n\n"
            f"{chunk_block}\n\n"
            f"Return ONLY a JSON object with this exact structure "
            f"(no markdown, no explanation):\n"
            f'{{"rankings": ['
            f'{{"index": <int>, "relevance_score": <float 0.0-1.0>}}, ...'
            f"]}}\n"
            f"List exactly {top_k} items, from most to least relevant."
        )

        response, _ = generate_with_fallback(
            self._gemini,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
            models=_RERANK_MODELS,
        )

        data = json.loads(response.text)
        rankings = data.get("rankings", [])

        reranked: list[RetrievedChunk] = []
        seen_ids: set[str] = set()

        for entry in rankings[:top_k]:
            idx = entry.get("index")
            score = float(entry.get("relevance_score", 0.0))
            if idx is None or not (0 <= idx < len(chunks)):
                continue
            chunk = chunks[idx]
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    document_name=chunk.document_name,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    token_count=chunk.token_count,
                    score=min(max(score, 0.0), 1.0),
                )
            )

        # Pad with highest-scoring Qdrant candidates if Gemini returned fewer than top_k
        if len(reranked) < top_k:
            for chunk in chunks:
                if chunk.chunk_id not in seen_ids:
                    reranked.append(chunk)
                    seen_ids.add(chunk.chunk_id)
                if len(reranked) >= top_k:
                    break

        return reranked
