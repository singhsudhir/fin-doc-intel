from datetime import datetime

import pytest

from src.models.schemas import (
    Chunk,
    Citation,
    CompareRequest,
    DocumentMetadata,
    DocumentType,
    EmbeddedChunk,
    IngestRequest,
    IngestResponse,
    IngestStatus,
    QueryRequest,
    QueryResponse,
)


def make_chunk(**kwargs) -> Chunk:
    defaults = dict(
        chunk_id="doc1_p1_c0",
        doc_id="doc1",
        text="Revenue grew 12% year-over-year.",
        page_number=1,
        chunk_index=0,
    )
    return Chunk(**(defaults | kwargs))


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


def test_chunk_strips_whitespace():
    c = make_chunk(text="  hello world  ")
    assert c.text == "hello world"


def test_chunk_page_number_must_be_positive():
    with pytest.raises(Exception):
        make_chunk(page_number=0)


# ---------------------------------------------------------------------------
# EmbeddedChunk
# ---------------------------------------------------------------------------


def test_embedded_chunk_qdrant_payload_keys():
    ec = EmbeddedChunk(chunk=make_chunk(), embedding=[0.1] * 384)
    payload = ec.qdrant_payload
    assert set(payload.keys()) == {
        "chunk_id",
        "doc_id",
        "document_name",
        "text",
        "page_number",
        "chunk_index",
        "token_count",
    }


def test_embedded_chunk_payload_values():
    chunk = make_chunk(page_number=3, chunk_index=2)
    ec = EmbeddedChunk(chunk=chunk, embedding=[0.0] * 384)
    assert ec.qdrant_payload["page_number"] == 3
    assert ec.qdrant_payload["chunk_index"] == 2


# ---------------------------------------------------------------------------
# DocumentMetadata
# ---------------------------------------------------------------------------


def test_document_metadata_defaults():
    meta = DocumentMetadata(
        doc_id="abc123",
        filename="report.pdf",
        total_pages=50,
        file_size_bytes=1024,
        collection_name="financial_docs",
    )
    assert meta.doc_type == DocumentType.OTHER
    assert isinstance(meta.ingested_at, datetime)
    assert meta.extra == {}


# ---------------------------------------------------------------------------
# QueryRequest
# ---------------------------------------------------------------------------


def test_query_request_defaults():
    req = QueryRequest(question="What was the net income?")
    assert req.top_k == 5
    assert req.collection_name == "financial_docs"
    assert req.doc_ids is None


def test_query_request_question_too_short():
    with pytest.raises(Exception):
        QueryRequest(question="hi")


# ---------------------------------------------------------------------------
# IngestRequest
# ---------------------------------------------------------------------------


def test_ingest_request_chunk_overlap_bounds():
    req = IngestRequest(file_paths=["a.pdf"], chunk_size=512, chunk_overlap=64)
    assert req.chunk_overlap == 64


def test_ingest_request_empty_file_paths():
    with pytest.raises(Exception):
        IngestRequest(file_paths=[])


# ---------------------------------------------------------------------------
# CompareRequest
# ---------------------------------------------------------------------------


def test_compare_request_default_topics():
    req = CompareRequest(doc_id_a="a", doc_id_b="b")
    assert "revenue" in req.focus_topics
    assert len(req.focus_topics) > 0
