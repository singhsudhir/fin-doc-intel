from __future__ import annotations

import structlog
from sentence_transformers import SentenceTransformer

from src.models.schemas import Chunk, EmbeddedChunk

log = structlog.get_logger()


class Embedder:
    """Wraps sentence-transformers to embed Chunk objects.

    The model is loaded lazily on first use and reused for the lifetime of
    the instance — instantiate once and share across the application.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    VECTOR_DIM = 384

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            log.info("loading_embedding_model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = True,
    ) -> list[EmbeddedChunk]:
        """Embed a list of Chunk objects, returning one EmbeddedChunk per input."""
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        log.info("chunks_embedded", count=len(chunks), model=self._model_name)
        return [
            EmbeddedChunk(
                chunk=chunk.model_dump(),
                embedding=vec.tolist(),
                model_name=self._model_name,
            )
            for chunk, vec in zip(chunks, vectors)
        ]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for use in a similarity search."""
        vector = self.model.encode([query], convert_to_numpy=True)[0]
        return vector.tolist()
