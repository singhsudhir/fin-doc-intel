"""Tests for Embedder — no network required."""
from __future__ import annotations

import pytest

from src.embedding.embedder import Embedder
from src.models.schemas import Chunk, EmbeddedChunk

EMBEDDER = Embedder()  # shared instance; model loads once for the module


def _chunk(text: str = "Revenue grew 12% year-over-year.", page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=f"test_p{page}_c0",
        doc_id="test_doc",
        document_name="test.pdf",
        text=text,
        page_number=page,
        chunk_index=0,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def test_model_loads_on_first_access():
    e = Embedder()
    assert e._model is None
    _ = e.model
    assert e._model is not None


def test_model_reused_on_second_access():
    e = Embedder()
    m1 = e.model
    m2 = e.model
    assert m1 is m2


# ---------------------------------------------------------------------------
# embed_chunks
# ---------------------------------------------------------------------------


def test_embed_chunks_returns_embedded_chunk_objects():
    result = EMBEDDER.embed_chunks([_chunk()], show_progress=False)
    assert all(isinstance(r, EmbeddedChunk) for r in result)


def test_embed_chunks_count_matches_input():
    chunks = [_chunk(page=i) for i in range(1, 6)]
    result = EMBEDDER.embed_chunks(chunks, show_progress=False)
    assert len(result) == len(chunks)


def test_embed_chunks_vector_dim_is_384():
    result = EMBEDDER.embed_chunks([_chunk()], show_progress=False)
    assert len(result[0].embedding) == Embedder.VECTOR_DIM


def test_embed_chunks_all_vectors_are_384_dim():
    chunks = [_chunk(page=i) for i in range(1, 4)]
    result = EMBEDDER.embed_chunks(chunks, show_progress=False)
    assert all(len(r.embedding) == Embedder.VECTOR_DIM for r in result)


def test_embed_chunks_model_name_stored():
    result = EMBEDDER.embed_chunks([_chunk()], show_progress=False)
    assert result[0].model_name == Embedder.MODEL_NAME


def test_embed_chunks_chunk_metadata_preserved():
    c = _chunk(text="Net income: $1.2B", page=7)
    result = EMBEDDER.embed_chunks([c], show_progress=False)
    assert result[0].chunk.page_number == 7
    assert result[0].chunk.doc_id == "test_doc"


def test_embed_chunks_empty_input_returns_empty():
    assert EMBEDDER.embed_chunks([], show_progress=False) == []


def test_embed_chunks_embedding_is_list_of_float():
    result = EMBEDDER.embed_chunks([_chunk()], show_progress=False)
    vec = result[0].embedding
    assert isinstance(vec, list)
    assert all(isinstance(v, float) for v in vec)


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------


def test_embed_query_returns_list():
    result = EMBEDDER.embed_query("What was the net income?")
    assert isinstance(result, list)


def test_embed_query_dim_is_384():
    result = EMBEDDER.embed_query("What was the net income?")
    assert len(result) == Embedder.VECTOR_DIM


def test_embed_query_returns_floats():
    result = EMBEDDER.embed_query("operating margin")
    assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_text_produces_same_vector():
    text = "Operating cash flow was $3.4 billion."
    v1 = EMBEDDER.embed_chunks([_chunk(text)], show_progress=False)[0].embedding
    v2 = EMBEDDER.embed_chunks([_chunk(text)], show_progress=False)[0].embedding
    assert v1 == v2


def test_different_texts_produce_different_vectors():
    v1 = EMBEDDER.embed_chunks([_chunk("Revenue growth")], show_progress=False)[0].embedding
    v2 = EMBEDDER.embed_chunks([_chunk("Credit risk exposure")], show_progress=False)[0].embedding
    assert v1 != v2


# ---------------------------------------------------------------------------
# qdrant_payload integration
# ---------------------------------------------------------------------------


def test_qdrant_payload_has_document_name():
    result = EMBEDDER.embed_chunks([_chunk()], show_progress=False)
    payload = result[0].qdrant_payload
    assert payload["document_name"] == "test.pdf"


def test_qdrant_payload_has_page_number():
    result = EMBEDDER.embed_chunks([_chunk(page=5)], show_progress=False)
    assert result[0].qdrant_payload["page_number"] == 5
