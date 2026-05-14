# Financial Document Intelligence Agent

A RAG system for ingesting financial PDFs (annual reports, credit memos), answering questions with page-level citations, and comparing documents side-by-side.

---

## Tech Stack

| Layer | Choice |
|---|---|
| LLM | Google Gemini 2.5 Pro via `google-genai` SDK |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (local, 384-dim) |
| Vector DB | Qdrant Cloud |
| PDF parsing | PyMuPDF (`fitz`) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Config | `python-dotenv` + `.env` |

---

## Project Structure

```
fin-doc-intel/
├── src/
│   ├── models/          # Pydantic schemas (Document, Chunk, QueryRequest, …)
│   ├── ingestion/       # PDF parsing, chunking, embedding, Qdrant upsert
│   ├── retrieval/       # Qdrant similarity search + re-ranking
│   ├── generation/      # Gemini prompt construction and calling
│   ├── api/             # FastAPI app and route handlers
│   │   └── routes/      # ingest.py, query.py
│   └── frontend/        # Streamlit app
├── tests/               # pytest tests
├── data/
│   ├── raw/             # uploaded PDFs (git-ignored)
│   └── processed/       # intermediate JSON (git-ignored)
├── scripts/             # one-off CLI utilities
├── requirements.txt
├── .env                 # never committed
├── .env.example         # committed template
└── CLAUDE.md
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in real values:

```
GEMINI_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
```

---

## Gemini SDK Usage Pattern

```python
from google import genai
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=prompt,
)
answer = response.text
```

**Never use the `anthropic` SDK or OpenAI SDK in this project.** This project exclusively uses `google-genai` for LLM calls.

---

## Embedding Pattern

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(texts, show_progress_bar=True)  # returns np.ndarray
```

Embedding dimension: **384**. Use this as `vector_size` when creating Qdrant collections.

---

## Qdrant Pattern

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

# Create collection (idempotent)
client.recreate_collection(
    collection_name="financial_docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Search
hits = client.search(
    collection_name="financial_docs",
    query_vector=embedding,
    limit=5,
)
```

---

## Ingestion Pipeline (high level)

1. **Parse** — PyMuPDF extracts text per page with bounding boxes
2. **Chunk** — sliding window over tokens (default 512 tokens, 64 overlap)
3. **Embed** — `all-MiniLM-L6-v2` produces 384-dim vectors
4. **Upsert** — `EmbeddedChunk.qdrant_payload` stored in Qdrant Cloud

---

## RAG Query Pipeline (high level)

1. Embed the question with the same model
2. Qdrant cosine search → top-k `EmbeddedChunk` payloads
3. Build prompt: system instruction + retrieved chunks (with page numbers) + question
4. Gemini 2.5 Pro generates answer
5. Return `QueryResponse` with `Citation` list (doc, page, score)

---

## API Endpoints (planned)

| Method | Path | Description |
|---|---|---|
| POST | `/ingest` | Upload and index PDF files |
| POST | `/query` | Ask a question, get answer + citations |
| POST | `/compare` | Compare two documents |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{doc_id}` | Remove a document |
| GET | `/health` | Liveness check |

---

## Data Models (src/models/schemas.py)

Key models:
- `DocumentMetadata` — filename, type, company, fiscal year, Qdrant collection
- `Chunk` — text, page_number, chunk_index, bbox
- `EmbeddedChunk` — Chunk + 384-dim vector; `.qdrant_payload` for upsert
- `Citation` — doc_id, filename, page_number, chunk_text, relevance_score
- `QueryRequest` / `QueryResponse` — question in, answer + citations out
- `CompareRequest` / `CompareResponse` — two doc IDs, comparison text
- `IngestRequest` / `IngestResponse` — file paths, chunk config, status

---

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn src.api.main:app --reload --port 8000

# Run Streamlit frontend
streamlit run src/frontend/app.py

# Run tests
pytest tests/ -v
```

---

## Coding Conventions

- All async I/O in FastAPI routes; Qdrant and Gemini calls use `asyncio.to_thread` if the SDK is sync
- Structured logging via `structlog` (JSON in prod, pretty in dev)
- Pydantic models for all API boundaries — no raw dicts across layers
- Page numbers are always **1-indexed**
- Chunk IDs: `f"{doc_id}_p{page_number}_c{chunk_index}"`
- No mocking in tests that touch Qdrant — use a dedicated test collection prefixed `test_`
