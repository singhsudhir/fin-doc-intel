"""Tests for PDFParser.

Fixtures build minimal PDFs in-memory so no real files are needed.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import fitz
import pytest

from src.ingestion.pdf_parser import PDFParser, _clean_text
from src.models.schemas import PageContent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pdf(tmp_path: Path, pages: list[str]) -> Path:
    """Create a minimal PDF with one text block per page."""
    out = tmp_path / "test.pdf"
    doc = fitz.open()
    for content in pages:
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text((72, 100), content, fontsize=11)
    doc.save(str(out))
    doc.close()
    return out


def _make_two_column_pdf(tmp_path: Path) -> Path:
    """Create a PDF page that mimics a two-column layout."""
    out = tmp_path / "two_col.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    left_text = "\n".join(
        ["Left column paragraph one.", "Left column paragraph two.",
         "Left column paragraph three.", "Left column paragraph four."]
    )
    right_text = "\n".join(
        ["Right column paragraph one.", "Right column paragraph two.",
         "Right column paragraph three.", "Right column paragraph four."]
    )
    # Left column: x around 72; right column: x around 320
    page.insert_text((72, 100), left_text, fontsize=11)
    page.insert_text((320, 100), right_text, fontsize=11)

    doc.save(str(out))
    doc.close()
    return out


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------


def test_parse_returns_correct_page_count(tmp_path):
    pdf = _make_pdf(tmp_path, ["Page one text.", "Page two text.", "Page three text."])
    pages = PDFParser().parse(pdf)
    assert len(pages) == 3


def test_parse_returns_page_content_objects(tmp_path):
    pdf = _make_pdf(tmp_path, ["Hello financial world."])
    pages = PDFParser().parse(pdf)
    assert all(isinstance(p, PageContent) for p in pages)


def test_page_numbers_are_one_indexed(tmp_path):
    pdf = _make_pdf(tmp_path, ["p1", "p2", "p3"])
    pages = PDFParser().parse(pdf)
    assert [p.page_number for p in pages] == [1, 2, 3]


def test_page_text_is_non_empty(tmp_path):
    pdf = _make_pdf(tmp_path, ["Revenue grew 12% YoY."])
    pages = PDFParser().parse(pdf)
    assert pages[0].text.strip() != ""


def test_page_dimensions_captured(tmp_path):
    pdf = _make_pdf(tmp_path, ["text"])
    pages = PDFParser().parse(pdf)
    assert pages[0].width > 0
    assert pages[0].height > 0


def test_text_contains_original_content(tmp_path):
    pdf = _make_pdf(tmp_path, ["Net income was $1.2 billion."])
    pages = PDFParser().parse(pdf)
    assert "Net income" in pages[0].text or "net income" in pages[0].text.lower()


def test_multipage_each_page_has_text(tmp_path):
    texts = ["Annual Report 2023", "Risk Factors Section", "Financial Statements"]
    pdf = _make_pdf(tmp_path, texts)
    pages = PDFParser().parse(pdf)
    for page in pages:
        assert page.text.strip()


def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        PDFParser().parse("/nonexistent/path/file.pdf")


# ---------------------------------------------------------------------------
# Multi-column detection
# ---------------------------------------------------------------------------


def test_two_column_pdf_parsed_without_error(tmp_path):
    pdf = _make_two_column_pdf(tmp_path)
    pages = PDFParser().parse(pdf)
    assert len(pages) == 1
    assert pages[0].text.strip()


def test_two_column_left_content_present(tmp_path):
    pdf = _make_two_column_pdf(tmp_path)
    pages = PDFParser().parse(pdf)
    assert "Left column" in pages[0].text


def test_two_column_right_content_present(tmp_path):
    pdf = _make_two_column_pdf(tmp_path)
    pages = PDFParser().parse(pdf)
    assert "Right column" in pages[0].text


# ---------------------------------------------------------------------------
# _clean_text unit tests
# ---------------------------------------------------------------------------


def test_clean_text_removes_trailing_space():
    assert _clean_text("hello   world") == "hello world"


def test_clean_text_fixes_hyphenation():
    assert _clean_text("reve-\nnue growth") == "revenue growth"


def test_clean_text_collapses_blank_lines():
    assert "\n\n\n" not in _clean_text("a\n\n\n\nb")


def test_clean_text_preserves_paragraph_break():
    result = _clean_text("paragraph one\n\nparagraph two")
    assert "\n\n" in result


def test_clean_text_strips_outer_whitespace():
    assert _clean_text("  hello  ") == "hello"
