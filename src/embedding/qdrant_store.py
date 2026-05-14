from __future__ import annotations

import os
import uuid

import structlog
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

from src.models.schemas import EmbeddedChunk, RetrievedChunk

load_dotenv()
log = structlog.get_logger()

_BATCH_SIZE = 100


class QdrantStore:
    """Manages a single Qdrant collection for financial document chunks.

    The client is created lazily so construction never raises even if env
    vars are missing — the error surfaces only on the first actual call.
    """

    VECTOR_SIZE = 384

    def __init__(self, collection_name: str = "financial_docs") -> None:
        self.collection_name = collection_name
        self._client: QdrantClient | None = None

    # ------------------------------------------------------------------
    # Client (lazy)
    # ------------------------------------------------------------------

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            url = os.getenv("QDRANT_URL")
            api_key = os.getenv("QDRANT_API_KEY")
            if not url:
                raise RuntimeError("QDRANT_URL env var is not set")
            self._client = QdrantClient(url=url, api_key=api_key)
        return self._client

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self) -> None:
        """Create the collection and its payload indexes if they do not exist."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            # Qdrant Cloud requires an explicit keyword index before filtering/
            # deleting by a payload field.
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_name",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="doc_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            log.info("collection_created", name=self.collection_name)
        else:
            log.debug("collection_exists", name=self.collection_name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(
        self,
        embedded_chunks: list[EmbeddedChunk],
        batch_size: int = _BATCH_SIZE,
    ) -> int:
        """Upsert all embedded chunks; returns total points stored."""
        if not embedded_chunks:
            return 0

        self.ensure_collection()

        points = [
            PointStruct(
                # Deterministic UUID so re-ingesting the same chunk is idempotent
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, ec.chunk.chunk_id)),
                vector=ec.embedding,
                payload=ec.qdrant_payload,
            )
            for ec in embedded_chunks
        ]

        batches = range(0, len(points), batch_size)
        for start in tqdm(batches, desc="Upserting to Qdrant", unit="batch"):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + batch_size],
            )

        log.info("upserted", collection=self.collection_name, count=len(points))
        return len(points)

    def delete_document(self, document_name: str) -> None:
        """Delete every chunk whose payload.document_name matches."""
        if not self.client.collection_exists(self.collection_name):
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_name",
                        match=MatchValue(value=document_name),
                    )
                ]
            ),
        )
        log.info("document_deleted", document=document_name)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        document_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k most similar chunks to query_embedding.

        Pass document_filter to restrict results to a single source document.
        """
        query_filter: Filter | None = None
        if document_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_name",
                        match=MatchValue(value=document_filter),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        results: list[RetrievedChunk] = []
        for point in response.points:
            p = point.payload or {}
            results.append(
                RetrievedChunk(
                    chunk_id=p.get("chunk_id", ""),
                    doc_id=p.get("doc_id", ""),
                    document_name=p.get("document_name"),
                    text=p.get("text", ""),
                    page_number=p.get("page_number", 0),
                    chunk_index=p.get("chunk_index", 0),
                    token_count=p.get("token_count"),
                    score=point.score,
                )
            )
        return results

    def list_documents(self) -> list[str]:
        """Return sorted list of unique document names in the collection."""
        if not self.client.collection_exists(self.collection_name):
            return []

        doc_names: set[str] = set()
        offset = None

        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=["document_name"],
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for record in records:
                name = (record.payload or {}).get("document_name")
                if name:
                    doc_names.add(name)

            if next_offset is None:
                break
            offset = next_offset

        return sorted(doc_names)
