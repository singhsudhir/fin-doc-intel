from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.api.deps import get_pipeline
from src.ingestion.pipeline import IngestionPipeline
from src.models.schemas import DocumentType, IngestResponse, IngestStatus

router = APIRouter()

_UPLOAD_DIR = Path("data/raw")


@router.post("/", response_model=IngestResponse, summary="Upload and ingest a PDF")
async def ingest_document(
    file: UploadFile = File(..., description="PDF file to ingest"),
    doc_type: DocumentType = Form(DocumentType.OTHER),
    pipeline: IngestionPipeline = Depends(get_pipeline),
) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = _UPLOAD_DIR / file.filename
    dest.write_bytes(await file.read())

    try:
        result = await asyncio.to_thread(pipeline.ingest, dest, skip_if_exists=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    if result.error:
        raise HTTPException(status_code=500, detail=result.error)

    if result.skipped:
        return IngestResponse(
            status=IngestStatus.COMPLETED,
            doc_ids=[result.doc_id],
            total_chunks=0,
            message="Already ingested — delete the document first to re-ingest.",
        )

    return IngestResponse(
        status=IngestStatus.COMPLETED,
        doc_ids=[result.doc_id],
        total_chunks=result.chunks_created,
        message=(
            f"Ingested {result.chunks_created} chunks "
            f"from {result.pages_parsed} pages"
        ),
    )
