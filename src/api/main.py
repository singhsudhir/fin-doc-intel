from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

load_dotenv()

from src.api.routes import documents, health, ingest, query
from src.embedding.embedder import Embedder
from src.embedding.qdrant_store import QdrantStore
from src.generation.answer_generator import AnswerGenerator
from src.generation.comparator import Comparator
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.retriever import Retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load the embedding model so the first request isn't slow
    embedder = Embedder()
    _ = embedder.model

    store = QdrantStore()
    retriever = Retriever(embedder=embedder, store=store)
    generator = AnswerGenerator()
    comparator = Comparator(embedder=embedder, store=store)
    pipeline = IngestionPipeline(embedder=embedder, store=store)

    app.state.embedder = embedder
    app.state.store = store
    app.state.retriever = retriever
    app.state.generator = generator
    app.state.comparator = comparator
    app.state.pipeline = pipeline

    yield
    # Nothing to clean up — connections are stateless


app = FastAPI(
    title="Financial Document Intelligence API",
    description="RAG system for financial PDFs powered by Gemini 2.5 Pro + Qdrant",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")
