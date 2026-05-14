from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_store
from src.embedding.qdrant_store import QdrantStore
from src.models.responses import DocumentInfo

router = APIRouter()


@router.get("/", response_model=list[DocumentInfo], summary="List indexed documents")
async def list_documents(
    store: QdrantStore = Depends(get_store),
) -> list[DocumentInfo]:
    return _aggregate_documents(store)


@router.delete("/{document_name}", summary="Delete a document and all its chunks")
async def delete_document(
    document_name: str,
    store: QdrantStore = Depends(get_store),
) -> dict[str, str]:
    store.delete_document(document_name)
    return {"status": "deleted", "document_name": document_name}


# ------------------------------------------------------------------
# Internal helper — scroll Qdrant and aggregate per-document stats
# ------------------------------------------------------------------


def _aggregate_documents(store: QdrantStore) -> list[DocumentInfo]:
    if not store.client.collection_exists(store.collection_name):
        return []

    docs: dict[str, dict] = {}
    offset = None

    while True:
        records, next_offset = store.client.scroll(
            collection_name=store.collection_name,
            with_payload=["document_name", "doc_id", "page_number"],
            with_vectors=False,
            limit=1000,
            offset=offset,
        )

        for record in records:
            p = record.payload or {}
            name = p.get("document_name")
            if not name:
                continue
            if name not in docs:
                docs[name] = {
                    "document_name": name,
                    "doc_id": p.get("doc_id", ""),
                    "chunk_count": 0,
                    "total_pages": 0,
                }
            docs[name]["chunk_count"] += 1
            docs[name]["total_pages"] = max(
                docs[name]["total_pages"], p.get("page_number", 0)
            )

        if next_offset is None:
            break
        offset = next_offset

    return [
        DocumentInfo(**d)
        for d in sorted(docs.values(), key=lambda x: x["document_name"])
    ]
