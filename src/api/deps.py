"""FastAPI dependency providers — all components live on app.state."""
from __future__ import annotations

from fastapi import Request

from src.embedding.embedder import Embedder
from src.embedding.qdrant_store import QdrantStore
from src.generation.answer_generator import AnswerGenerator
from src.generation.comparator import Comparator
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.retriever import Retriever


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_store(request: Request) -> QdrantStore:
    return request.app.state.store


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_generator(request: Request) -> AnswerGenerator:
    return request.app.state.generator


def get_comparator(request: Request) -> Comparator:
    return request.app.state.comparator


def get_pipeline(request: Request) -> IngestionPipeline:
    return request.app.state.pipeline
