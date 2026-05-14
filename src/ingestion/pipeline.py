from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import structlog
from tqdm import tqdm

from src.chunking.semantic_chunker import SemanticChunker
from src.embedding.embedder import Embedder
from src.embedding.qdrant_store import QdrantStore
from src.ingestion.pdf_parser import PDFParser

log = structlog.get_logger()


@dataclass
class PipelineResult:
    document_name: str
    doc_id: str
    pages_parsed: int = 0
    chunks_created: int = 0
    vectors_stored: int = 0
    skipped: bool = False
    error: str | None = None


class IngestionPipeline:
    """PDF → parse → chunk → embed → Qdrant, end to end.

    All four components are injectable for testing; production code uses
    the default singletons.
    """

    def __init__(
        self,
        parser: PDFParser | None = None,
        chunker: SemanticChunker | None = None,
        embedder: Embedder | None = None,
        store: QdrantStore | None = None,
    ) -> None:
        self.parser = parser or PDFParser()
        self.chunker = chunker or SemanticChunker()
        self.embedder = embedder or Embedder()
        self.store = store or QdrantStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        pdf_path: str | Path,
        doc_id: str | None = None,
        skip_if_exists: bool = True,
    ) -> PipelineResult:
        """Ingest one PDF file into Qdrant.

        Returns a PipelineResult with .skipped=True if the document was
        already present and skip_if_exists is True.
        """
        path = Path(pdf_path)
        document_name = path.name
        if doc_id is None:
            doc_id = _stable_doc_id(path)

        result = PipelineResult(document_name=document_name, doc_id=doc_id)

        try:
            # ---- 1. Skip check ----------------------------------------
            if skip_if_exists:
                tqdm.write(f"Checking Qdrant for existing document: {document_name}")
                existing = self.store.list_documents()
                if document_name in existing:
                    log.info("skipping_already_ingested", document=document_name)
                    tqdm.write(f"  → Already ingested. Use skip_if_exists=False to re-ingest.")
                    result.skipped = True
                    return result

            # ---- 2. Parse ---------------------------------------------
            tqdm.write(f"\nParsing {document_name} …")
            pages = self.parser.parse(path)
            result.pages_parsed = len(pages)
            tqdm.write(f"  → {len(pages)} pages extracted")

            # ---- 3. Chunk ---------------------------------------------
            tqdm.write("Chunking pages …")
            non_empty = [p for p in pages if p.text.strip()]
            chunks = []
            for page in tqdm(non_empty, desc="  Chunking", unit="page", leave=False):
                chunks.extend(self.chunker._chunk_page(page, document_name, doc_id))
            result.chunks_created = len(chunks)
            tqdm.write(f"  → {len(chunks)} chunks created")

            # ---- 4. Embed ---------------------------------------------
            tqdm.write("Embedding chunks …")
            embedded = self.embedder.embed_chunks(chunks, show_progress=True)

            # ---- 5. Store ---------------------------------------------
            tqdm.write("Storing vectors in Qdrant …")
            result.vectors_stored = self.store.upsert(embedded)
            tqdm.write(f"  → {result.vectors_stored} vectors stored\n")

        except Exception as exc:
            log.exception("pipeline_error", document=document_name, error=str(exc))
            result.error = str(exc)

        return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _stable_doc_id(path: Path) -> str:
    """Deterministic 12-char hex ID from filename + file size."""
    content = f"{path.name}:{path.stat().st_size}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]
