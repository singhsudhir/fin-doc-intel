"""Integration tests for QdrantStore.

These tests hit the real Qdrant Cloud instance.  They are skipped
automatically when QDRANT_URL is not in the environment.

Uses a dedicated collection "test_financial_docs" that is created and
torn down within the test session.
"""
from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

QDRANT_AVAILABLE = bool(os.getenv("QDRANT_URL"))

pytestmark = pytest.mark.skipif(
    not QDRANT_AVAILABLE,
    reason="QDRANT_URL not set — skipping Qdrant integration tests",
)

TEST_COLLECTION = "test_financial_docs"

from src.embedding.embedder import Embedder
from src.embedding.qdrant_store import QdrantStore
from src.models.schemas import Chunk, EmbeddedChunk

EMBEDDER = Embedder()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def store():
    """QdrantStore pointed at the test collection; cleaned up after all tests."""
    s = QdrantStore(collection_name=TEST_COLLECTION)
    yield s
    # Teardown: drop the test collection entirely
    if s.client.collection_exists(TEST_COLLECTION):
        s.client.delete_collection(TEST_COLLECTION)


def _make_embedded(
    text: str,
    doc_id: str = "doc_test",
    document_name: str = "test.pdf",
    page: int = 1,
    chunk_index: int = 0,
) -> EmbeddedChunk:
    chunk = Chunk(
        chunk_id=f"{doc_id}_p{page}_c{chunk_index}",
        doc_id=doc_id,
        document_name=document_name,
        text=text,
        page_number=page,
        chunk_index=chunk_index,
    )
    return EMBEDDER.embed_chunks([chunk], show_progress=False)[0]


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------


def test_ensure_collection_creates_collection(store):
    store.ensure_collection()
    assert store.client.collection_exists(TEST_COLLECTION)


def test_ensure_collection_is_idempotent(store):
    store.ensure_collection()
    store.ensure_collection()  # second call must not raise
    assert store.client.collection_exists(TEST_COLLECTION)


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


def test_upsert_returns_count(store):
    ec = _make_embedded("ING reported net interest income of €15.2 billion.", chunk_index=0)
    count = store.upsert([ec])
    assert count == 1


def test_upsert_multiple_chunks(store):
    ecs = [
        _make_embedded(f"Sentence number {i} about financial performance.", chunk_index=i)
        for i in range(1, 6)
    ]
    count = store.upsert(ecs)
    assert count == 5


def test_upsert_empty_list_returns_zero(store):
    assert store.upsert([]) == 0


def test_upsert_is_idempotent(store):
    ec = _make_embedded("Capital adequacy ratio was 14.8%.", chunk_index=99)
    store.upsert([ec])
    store.upsert([ec])  # second upsert of same chunk must not raise or duplicate


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_returns_results(store):
    # Seed some known data
    store.upsert([
        _make_embedded("Total equity stood at €52 billion.", doc_id="docA", chunk_index=10),
    ])
    query_vec = EMBEDDER.embed_query("What is the total equity?")
    results = store.search(query_vec, top_k=5)
    assert len(results) > 0


def test_search_result_has_required_fields(store):
    query_vec = EMBEDDER.embed_query("net income")
    results = store.search(query_vec, top_k=1)
    if results:
        r = results[0]
        assert r.chunk_id
        assert r.doc_id
        assert r.text
        assert r.page_number >= 1
        assert 0.0 <= r.score <= 1.0


def test_search_respects_top_k(store):
    # Add 10 chunks of the same doc so there's enough data
    ecs = [
        _make_embedded(
            f"Risk weighted assets increased by {i}% in the reporting period.",
            doc_id="docB",
            chunk_index=i,
        )
        for i in range(10)
    ]
    store.upsert(ecs)
    query_vec = EMBEDDER.embed_query("risk weighted assets")
    results = store.search(query_vec, top_k=3)
    assert len(results) <= 3


def test_search_with_document_filter(store):
    store.upsert([
        _make_embedded(
            "Liquidity coverage ratio was 138%.",
            doc_id="docFilter",
            document_name="filter_doc.pdf",
            chunk_index=0,
        )
    ])
    query_vec = EMBEDDER.embed_query("liquidity coverage ratio")
    results = store.search(query_vec, top_k=10, document_filter="filter_doc.pdf")
    # All returned results must belong to the filtered document
    assert all(r.document_name == "filter_doc.pdf" for r in results)


# ---------------------------------------------------------------------------
# List documents
# ---------------------------------------------------------------------------


def test_list_documents_returns_list(store):
    docs = store.list_documents()
    assert isinstance(docs, list)


def test_list_documents_contains_ingested_doc(store):
    store.upsert([
        _make_embedded(
            "Operating income was €8.1 billion.",
            doc_id="docList",
            document_name="list_test.pdf",
            chunk_index=0,
        )
    ])
    docs = store.list_documents()
    assert "list_test.pdf" in docs


def test_list_documents_sorted(store):
    docs = store.list_documents()
    assert docs == sorted(docs)


def test_list_documents_empty_collection():
    # Fresh store pointing at a non-existent collection
    s = QdrantStore(collection_name="test_nonexistent_xyz")
    assert s.list_documents() == []


# ---------------------------------------------------------------------------
# Delete document
# ---------------------------------------------------------------------------


def test_delete_document_removes_chunks(store):
    store.upsert([
        _make_embedded(
            "Tier 1 capital ratio: 13.4%.",
            doc_id="docDel",
            document_name="to_delete.pdf",
            chunk_index=0,
        )
    ])
    assert "to_delete.pdf" in store.list_documents()
    store.delete_document("to_delete.pdf")
    query_vec = EMBEDDER.embed_query("tier 1 capital")
    results = store.search(query_vec, top_k=10, document_filter="to_delete.pdf")
    assert results == []


def test_delete_nonexistent_document_does_not_raise(store):
    store.delete_document("does_not_exist.pdf")  # must not raise
