from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.api.deps import get_comparator, get_generator, get_retriever
from src.generation.answer_generator import AnswerGenerator
from src.generation.comparator import Comparator
from src.models.responses import AnswerResponse, CompareResponse
from src.retrieval.retriever import Retriever

router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    document_filter: str | None = Field(
        None, description="Restrict search to this document filename"
    )
    top_k_final: int = Field(default=5, ge=1, le=10)


class CompareRequest(BaseModel):
    document_name_a: str
    document_name_b: str
    focus_topics: list[str] = Field(
        default=["revenue", "net income", "risk factors", "capital ratios", "outlook"]
    )


@router.post("/", response_model=AnswerResponse, summary="Ask a question")
async def query_documents(
    request: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
    generator: AnswerGenerator = Depends(get_generator),
) -> AnswerResponse:
    chunks = await asyncio.to_thread(
        retriever.retrieve,
        request.question,
        top_k_final=request.top_k_final,
        document_filter=request.document_filter,
    )
    return await asyncio.to_thread(generator.generate, request.question, chunks)


@router.post("/compare", response_model=CompareResponse, summary="Compare two documents")
async def compare_documents(
    request: CompareRequest,
    comparator: Comparator = Depends(get_comparator),
) -> CompareResponse:
    return await asyncio.to_thread(
        comparator.compare,
        request.document_name_a,
        request.document_name_b,
        request.focus_topics,
    )
