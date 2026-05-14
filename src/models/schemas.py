from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DocumentType(str, Enum):
    ANNUAL_REPORT = "annual_report"
    CREDIT_MEMO = "credit_memo"
    EARNINGS_RELEASE = "earnings_release"
    PROSPECTUS = "prospectus"
    OTHER = "other"


class IngestStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Core document models
# ---------------------------------------------------------------------------


class DocumentMetadata(BaseModel):
    """Metadata extracted from or assigned to a financial PDF."""

    doc_id: str = Field(..., description="Unique document identifier (UUID or hash)")
    filename: str
    doc_type: DocumentType = DocumentType.OTHER
    company: str | None = None
    fiscal_year: int | None = None
    total_pages: int = Field(..., ge=1)
    file_size_bytes: int = Field(..., ge=0)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    collection_name: str = Field(
        ..., description="Qdrant collection this document belongs to"
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary extra metadata"
    )


class Document(BaseModel):
    """Full document representation including all parsed chunks."""

    metadata: DocumentMetadata
    chunks: list[Chunk] = Field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


# ---------------------------------------------------------------------------
# Chunk models
# ---------------------------------------------------------------------------


class PageContent(BaseModel):
    """Raw text extracted from a single PDF page, before chunking."""

    page_number: int = Field(..., ge=1, description="1-indexed page number")
    text: str
    width: float = 0.0
    height: float = 0.0


class Chunk(BaseModel):
    """A single text chunk extracted from a PDF page."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Parent document ID")
    document_name: str | None = Field(None, description="Human-readable source filename")
    text: str = Field(..., min_length=1)
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    chunk_index: int = Field(..., ge=0, description="Sequence index within the page")
    token_count: int | None = None
    bbox: tuple[float, float, float, float] | None = Field(
        None, description="Bounding box (x0, y0, x1, y1) on the page"
    )

    @field_validator("text")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class EmbeddedChunk(BaseModel):
    """Chunk paired with its dense embedding vector, ready for Qdrant upsert."""

    chunk: Chunk
    embedding: list[float] = Field(..., min_length=1)
    model_name: str = Field(
        default="all-MiniLM-L6-v2", description="Embedding model used"
    )

    @property
    def qdrant_payload(self) -> dict[str, Any]:
        """Flat dict stored as Qdrant point payload."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "document_name": self.chunk.document_name,
            "text": self.chunk.text,
            "page_number": self.chunk.page_number,
            "chunk_index": self.chunk.chunk_index,
            "token_count": self.chunk.token_count,
        }


# ---------------------------------------------------------------------------
# Query / retrieval models
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A single page-level citation returned with an answer."""

    doc_id: str
    filename: str
    page_number: int
    chunk_text: str = Field(..., description="The source chunk used")
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class QueryRequest(BaseModel):
    """Incoming user question."""

    question: str = Field(..., min_length=3, max_length=2000)
    collection_name: str = Field(
        default="financial_docs", description="Qdrant collection to search"
    )
    doc_ids: list[str] | None = Field(
        None, description="Restrict search to specific document IDs"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class QueryResponse(BaseModel):
    """Answer with structured citations."""

    question: str
    answer: str
    citations: list[Citation]
    model: str = Field(default="gemini-2.5-pro")
    latency_ms: float | None = None


# ---------------------------------------------------------------------------
# Ingestion models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Request to ingest one or more PDF files by path."""

    file_paths: list[str] = Field(..., min_length=1)
    collection_name: str = Field(default="financial_docs")
    doc_type: DocumentType = DocumentType.OTHER
    company: str | None = None
    fiscal_year: int | None = None
    chunk_size: int = Field(
        default=512, ge=128, le=2048, description="Max tokens per chunk"
    )
    chunk_overlap: int = Field(default=64, ge=0, le=256)


class IngestResponse(BaseModel):
    """Result of an ingestion job."""

    status: IngestStatus
    doc_ids: list[str] = Field(default_factory=list)
    total_chunks: int = 0
    failed_files: list[str] = Field(default_factory=list)
    message: str = ""


# ---------------------------------------------------------------------------
# Document comparison models
# ---------------------------------------------------------------------------


class CompareRequest(BaseModel):
    """Request to compare two financial documents."""

    doc_id_a: str
    doc_id_b: str
    collection_name: str = Field(default="financial_docs")
    focus_topics: list[str] = Field(
        default_factory=lambda: [
            "revenue",
            "net income",
            "debt",
            "risk factors",
            "outlook",
        ],
        description="Topics the comparison should focus on",
    )


class CompareResponse(BaseModel):
    """Side-by-side comparison result."""

    doc_id_a: str
    doc_id_b: str
    comparison: str = Field(..., description="LLM-generated comparative analysis")
    citations_a: list[Citation] = Field(default_factory=list)
    citations_b: list[Citation] = Field(default_factory=list)
    model: str = Field(default="gemini-2.5-pro")
    latency_ms: float | None = None


# ---------------------------------------------------------------------------
# Misc / utility models
# ---------------------------------------------------------------------------


class DocumentSummary(BaseModel):
    """Lightweight document listing entry."""

    doc_id: str
    filename: str
    doc_type: DocumentType
    company: str | None
    fiscal_year: int | None
    total_pages: int
    chunk_count: int
    ingested_at: datetime


class RetrievedChunk(BaseModel):
    """A chunk returned by a Qdrant similarity search."""

    chunk_id: str
    doc_id: str
    document_name: str | None
    text: str
    page_number: int
    chunk_index: int
    token_count: int | None
    score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")

    def to_citation(self) -> "Citation":
        return Citation(
            doc_id=self.doc_id,
            filename=self.document_name or self.doc_id,
            page_number=self.page_number,
            chunk_text=self.text,
            relevance_score=self.score,
        )


class HealthResponse(BaseModel):
    status: str = "ok"
    qdrant_connected: bool = False
    embedding_model_loaded: bool = False
    version: str = "0.1.0"
