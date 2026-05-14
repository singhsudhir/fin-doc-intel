from __future__ import annotations

import re

import tiktoken
import structlog

from src.models.schemas import Chunk, PageContent

log = structlog.get_logger()

# Regex that splits on sentence-ending punctuation followed by whitespace.
# Lookbehind keeps the punctuation attached to the sentence it ends.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class SemanticChunker:
    """Splits a list of PageContent objects into token-bounded Chunk objects.

    Strategy
    --------
    1. Process each page independently so every chunk carries an exact page_number.
    2. Accumulate paragraphs (split on blank lines) until we approach max_tokens.
    3. When a single paragraph is over-long, fall back to sentence-level splitting.
    4. Prepend the last `overlap_tokens` of the previous chunk to provide context
       continuity across chunk boundaries on the same page.

    Token counting uses tiktoken cl100k_base (GPT-4 tokenizer), which gives
    consistent counts regardless of the downstream model.
    """

    def __init__(
        self,
        min_tokens: int = 500,
        max_tokens: int = 800,
        overlap_tokens: int = 100,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._enc = tiktoken.get_encoding(encoding_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(
        self,
        pages: list[PageContent],
        document_name: str,
        doc_id: str,
    ) -> list[Chunk]:
        """Return all chunks across all pages of a document."""
        chunks: list[Chunk] = []
        for page in pages:
            if not page.text.strip():
                continue
            chunks.extend(self._chunk_page(page, document_name, doc_id))

        log.info(
            "document_chunked",
            document=document_name,
            input_pages=len(pages),
            output_chunks=len(chunks),
        )
        return chunks

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    # ------------------------------------------------------------------
    # Per-page chunking
    # ------------------------------------------------------------------

    def _chunk_page(
        self,
        page: PageContent,
        document_name: str,
        doc_id: str,
    ) -> list[Chunk]:
        paragraphs = _split_paragraphs(page.text)
        chunks: list[Chunk] = []
        pending: list[str] = []   # paragraphs accumulating in the current chunk
        pending_tokens = 0
        overlap_text = ""

        def flush() -> None:
            nonlocal overlap_text, pending, pending_tokens
            if not pending:
                return
            chunks.append(
                _build_chunk(
                    overlap_text, pending, doc_id, document_name,
                    page.page_number, len(chunks), self._enc,
                )
            )
            overlap_text = _tail_tokens(pending, self.overlap_tokens, self._enc)
            pending = []
            pending_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            if para_tokens > self.max_tokens:
                # Paragraph alone exceeds the limit — flush and sentence-split it
                flush()
                for sentence_group in self._split_oversized(para, overlap_text):
                    chunks.append(
                        _build_chunk(
                            overlap_text, [sentence_group], doc_id, document_name,
                            page.page_number, len(chunks), self._enc,
                        )
                    )
                    overlap_text = _tail_tokens([sentence_group], self.overlap_tokens, self._enc)
                continue

            if pending_tokens + para_tokens > self.max_tokens:
                flush()

            pending.append(para)
            pending_tokens += para_tokens

        flush()
        return chunks

    def _split_oversized(self, text: str, overlap_text: str) -> list[str]:
        """Sentence-level greedy grouping for a paragraph that exceeds max_tokens."""
        sentences = _SENTENCE_SPLIT.split(text)
        groups: list[str] = []
        current: list[str] = []
        # Seed the token count with any overlap that will be prepended
        current_tokens = self.count_tokens(overlap_text) if overlap_text else 0

        for sentence in sentences:
            s_tokens = self.count_tokens(sentence)
            if current_tokens + s_tokens > self.max_tokens and current:
                groups.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(sentence)
            current_tokens += s_tokens

        if current:
            groups.append(" ".join(current))

        return groups


# ------------------------------------------------------------------
# Stateless helpers
# ------------------------------------------------------------------


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _tail_tokens(paras: list[str], n: int, enc: tiktoken.Encoding) -> str:
    """Return the last `n` tokens of the joined paragraphs as a string."""
    joined = "\n\n".join(paras)
    tokens = enc.encode(joined)
    if len(tokens) <= n:
        return joined
    return enc.decode(tokens[-n:])


def _build_chunk(
    overlap: str,
    paras: list[str],
    doc_id: str,
    document_name: str,
    page_number: int,
    chunk_index: int,
    enc: tiktoken.Encoding,
) -> Chunk:
    body = "\n\n".join(paras)
    text = f"{overlap}\n\n{body}".strip() if overlap else body
    return Chunk(
        chunk_id=f"{doc_id}_p{page_number}_c{chunk_index}",
        doc_id=doc_id,
        document_name=document_name,
        text=text,
        page_number=page_number,
        chunk_index=chunk_index,
        token_count=len(enc.encode(text)),
    )
