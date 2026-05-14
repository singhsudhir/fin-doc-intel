"""Tests for SemanticChunker."""
from __future__ import annotations

import pytest

from src.chunking.semantic_chunker import SemanticChunker, _split_paragraphs, _tail_tokens
from src.models.schemas import Chunk, PageContent

import tiktoken


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


ENC = tiktoken.get_encoding("cl100k_base")

SENTENCE = (
    "The company reported strong earnings growth driven by robust consumer demand. "
    "Net revenue increased by 14.3% year-over-year, reaching $42.6 billion. "
    "Operating margins expanded by 120 basis points to 18.7%. "
    "Management attributed the outperformance to disciplined cost controls. "
)

# ~600-token page: 12 copies of SENTENCE (~50 tokens each)
NORMAL_PAGE_TEXT = "\n\n".join([SENTENCE] * 3)

# Short page: a few sentences
SHORT_PAGE_TEXT = "Total assets: $5.2 billion. Cash and equivalents: $1.1 billion."

# Long paragraph that exceeds 800 tokens on its own
LONG_PARA = " ".join([SENTENCE.strip()] * 20)


def _page(text: str, page_number: int = 1) -> PageContent:
    return PageContent(page_number=page_number, text=text)


def _chunker(**kwargs) -> SemanticChunker:
    return SemanticChunker(min_tokens=500, max_tokens=800, overlap_tokens=100, **kwargs)


# ---------------------------------------------------------------------------
# Basic chunking
# ---------------------------------------------------------------------------


def test_chunk_document_returns_chunks():
    chunker = _chunker()
    pages = [_page(NORMAL_PAGE_TEXT)]
    chunks = chunker.chunk_document(pages, "report.pdf", "doc1")
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_ids_are_unique():
    chunker = _chunker()
    pages = [_page(NORMAL_PAGE_TEXT, 1), _page(NORMAL_PAGE_TEXT, 2)]
    chunks = chunker.chunk_document(pages, "report.pdf", "doc1")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_id_format():
    chunker = _chunker()
    chunks = chunker.chunk_document([_page(NORMAL_PAGE_TEXT)], "report.pdf", "docX")
    for chunk in chunks:
        # Expected: "docX_p1_c0", "docX_p1_c1", …
        assert chunk.chunk_id.startswith("docX_p1_c")


def test_document_name_stored_on_chunk():
    chunker = _chunker()
    chunks = chunker.chunk_document([_page(NORMAL_PAGE_TEXT)], "annual_2023.pdf", "doc1")
    assert all(c.document_name == "annual_2023.pdf" for c in chunks)


def test_page_number_attribution():
    chunker = _chunker()
    p1 = _page("Short page one text.", 1)
    p2 = _page("Short page two text.", 2)
    chunks = chunker.chunk_document([p1, p2], "doc.pdf", "d1")
    page_numbers = {c.page_number for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers


def test_chunk_index_starts_at_zero_per_page():
    chunker = _chunker()
    long_text = "\n\n".join([SENTENCE] * 30)  # force multiple chunks on one page
    chunks = chunker.chunk_document([_page(long_text, 1)], "doc.pdf", "d1")
    page1_chunks = [c for c in chunks if c.page_number == 1]
    assert page1_chunks[0].chunk_index == 0


def test_token_count_stored():
    chunker = _chunker()
    chunks = chunker.chunk_document([_page(NORMAL_PAGE_TEXT)], "r.pdf", "d1")
    assert all(c.token_count is not None and c.token_count > 0 for c in chunks)


# ---------------------------------------------------------------------------
# Token bounds
# ---------------------------------------------------------------------------


def test_no_chunk_exceeds_max_tokens():
    chunker = _chunker()
    # Generate enough text to create many chunks
    big_text = "\n\n".join([SENTENCE] * 60)
    chunks = chunker.chunk_document([_page(big_text)], "big.pdf", "d1")
    for c in chunks:
        assert c.token_count <= chunker.max_tokens + chunker.overlap_tokens, (
            f"Chunk {c.chunk_id} has {c.token_count} tokens, "
            f"max is {chunker.max_tokens + chunker.overlap_tokens}"
        )


def test_oversized_paragraph_is_handled():
    chunker = _chunker()
    chunks = chunker.chunk_document([_page(LONG_PARA)], "big_para.pdf", "d1")
    assert len(chunks) > 0
    for c in chunks:
        assert c.token_count <= chunker.max_tokens + chunker.overlap_tokens


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------


def test_overlap_text_appears_in_second_chunk():
    chunker = SemanticChunker(min_tokens=10, max_tokens=80, overlap_tokens=20)
    # Build text with clearly separated paragraphs
    paras = ["Alpha beta gamma delta epsilon zeta eta theta iota kappa. " * 3] * 6
    text = "\n\n".join(paras)
    chunks = chunker.chunk_document([_page(text)], "ov.pdf", "d1")
    if len(chunks) >= 2:
        # The tail of chunk[0]'s body should appear somewhere in chunk[1]'s text
        first_chunk_tokens = ENC.encode(chunks[0].text)
        overlap_tokens = first_chunk_tokens[-20:]
        overlap_str = ENC.decode(overlap_tokens)
        assert overlap_str.strip() in chunks[1].text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_pages_are_skipped():
    chunker = _chunker()
    pages = [_page("", 1), _page(NORMAL_PAGE_TEXT, 2), _page("   ", 3)]
    chunks = chunker.chunk_document(pages, "r.pdf", "d1")
    page_numbers = {c.page_number for c in chunks}
    assert 1 not in page_numbers
    assert 3 not in page_numbers
    assert 2 in page_numbers


def test_very_short_page_produces_one_chunk():
    chunker = _chunker()
    chunks = chunker.chunk_document([_page(SHORT_PAGE_TEXT)], "short.pdf", "d1")
    assert len(chunks) == 1
    assert "Total assets" in chunks[0].text


def test_multiple_pages_all_produce_chunks():
    chunker = _chunker()
    pages = [_page(NORMAL_PAGE_TEXT, i) for i in range(1, 6)]
    chunks = chunker.chunk_document(pages, "multi.pdf", "d1")
    pages_with_chunks = {c.page_number for c in chunks}
    assert pages_with_chunks == {1, 2, 3, 4, 5}


def test_doc_id_propagated():
    chunker = _chunker()
    chunks = chunker.chunk_document([_page(NORMAL_PAGE_TEXT)], "r.pdf", "my_doc_42")
    assert all(c.doc_id == "my_doc_42" for c in chunks)


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_split_paragraphs_on_blank_line():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird."
    paras = _split_paragraphs(text)
    assert paras == ["First paragraph.", "Second paragraph.", "Third."]


def test_split_paragraphs_skips_empty():
    text = "A\n\n\n\nB"
    paras = _split_paragraphs(text)
    assert paras == ["A", "B"]


def test_tail_tokens_short_text():
    text = "hello world"
    result = _tail_tokens([text], 50, ENC)
    assert result == text


def test_tail_tokens_truncates_long_text():
    long = "word " * 500
    result = _tail_tokens([long], 10, ENC)
    assert len(ENC.encode(result)) <= 10


def test_count_tokens_is_positive():
    chunker = _chunker()
    assert chunker.count_tokens("Revenue grew 10% YoY.") > 0


def test_count_tokens_empty_string():
    chunker = _chunker()
    assert chunker.count_tokens("") == 0
