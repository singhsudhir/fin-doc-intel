from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF
import structlog

from src.models.schemas import PageContent

log = structlog.get_logger()

# Blocks are tuples: (x0, y0, x1, y1, text, block_no, block_type)
# block_type == 0 → text, 1 → image
_TEXT_BLOCK = 0


class PDFParser:
    """Extracts per-page text from financial PDFs using PyMuPDF.

    Multi-column handling: detects two-column layouts common in bank annual
    reports by checking whether text blocks cluster into distinct left/right
    halves.  When detected, left column is read top-to-bottom first, then
    right column — giving the correct reading order for a 2-up layout.
    """

    def parse(self, pdf_path: str | Path) -> list[PageContent]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        pages: list[PageContent] = []
        doc = fitz.open(str(path))
        try:
            for idx, page in enumerate(doc):
                text = self._extract_page_text(page)
                pages.append(
                    PageContent(
                        page_number=idx + 1,
                        text=text,
                        width=page.rect.width,
                        height=page.rect.height,
                    )
                )
        finally:
            doc.close()

        log.info("pdf_parsed", path=str(path), total_pages=len(pages))
        return pages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_page_text(self, page: fitz.Page) -> str:
        raw_blocks = page.get_text("blocks")
        text_blocks = [b for b in raw_blocks if b[6] == _TEXT_BLOCK and b[4].strip()]

        if not text_blocks:
            return ""

        if self._is_two_column(text_blocks, page.rect.width):
            ordered = self._sort_two_column(text_blocks, page.rect.width)
        else:
            ordered = sorted(text_blocks, key=lambda b: (round(b[1], 1), b[0]))

        raw = "\n".join(b[4] for b in ordered)
        return _clean_text(raw)

    def _is_two_column(self, blocks: list, page_width: float) -> bool:
        """True when blocks clearly occupy both left and right halves of the page."""
        if len(blocks) < 4:
            return False
        mid = page_width / 2
        margin = page_width * 0.06  # 6% margin keeps headers/footers from triggering this
        left = sum(1 for b in blocks if b[0] < mid - margin)
        right = sum(1 for b in blocks if b[0] > mid + margin)
        return left >= 2 and right >= 2

    def _sort_two_column(self, blocks: list, page_width: float) -> list:
        """Left column (sorted by y) then right column (sorted by y)."""
        mid = page_width / 2
        left = sorted([b for b in blocks if b[0] <= mid], key=lambda b: b[1])
        right = sorted([b for b in blocks if b[0] > mid], key=lambda b: b[1])
        return left + right


# ------------------------------------------------------------------
# Module-level text cleaning (no state needed)
# ------------------------------------------------------------------


def _clean_text(text: str) -> str:
    # Re-join words hyphenated across a line break ("reve-\nnue" → "revenue")
    text = re.sub(r"-\n(\S)", r"\1", text)
    # Collapse runs of spaces/tabs to a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive newlines to a paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
