from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_embedder, get_store
from src.embedding.embedder import Embedder
from src.embedding.qdrant_store import QdrantStore
from src.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(
    store: QdrantStore = Depends(get_store),
    embedder: Embedder = Depends(get_embedder),
) -> HealthResponse:
    qdrant_ok = False
    try:
        store.client.collection_exists(store.collection_name)
        qdrant_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        qdrant_connected=qdrant_ok,
        embedding_model_loaded=embedder._model is not None,
    )
