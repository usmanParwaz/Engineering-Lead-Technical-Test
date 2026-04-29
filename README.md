# Document QA API

A clean, production-structured backend that lets you upload documents and ask natural language questions about them using a RAG (Retrieval-Augmented Generation) pipeline.

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/) (used for both Voyage embeddings and Claude)

### 2. Clone & Install

```bash
git clone <your-repo-url>
cd doc-qa-api

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=your_key_here
```

### 4. Run

```bash
uvicorn main:app --reload
```

The API is now available at `http://localhost:8000`.

Interactive API docs: `http://localhost:8000/docs`

Frontend UI: `http://localhost:8000/ui`

---

## Docker

```bash
cp .env.example .env
# Set ANTHROPIC_API_KEY in .env

docker-compose up --build
```

---

## API Reference

### `GET /health`

Returns system status and loaded document stats.

```json
{ "status": "ok", "documents_loaded": 2, "total_chunks": 84 }
```

### `GET /documents`

Lists all ingested documents.

```json
{
  "documents": [
    { "document_id": "abc-123", "filename": "policy.pdf", "chunk_count": 42 }
  ]
}
```

### `POST /ingest`

Upload and index a document.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@your_document.pdf"
```

Response:

```json
{
  "message": "Document ingested successfully.",
  "document_id": "abc-123",
  "chunks_created": 42,
  "filename": "your_document.pdf"
}
```

Supported formats: `.pdf`, `.docx`, `.txt`, `.md`, `.csv`

### `POST /ask`

Ask a question about ingested content.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the document say about refunds?"}'
```

Response:

```json
{
  "answer": "The document states that refunds are allowed within 30 days...",
  "document_id": null,
  "sources": [
    { "text": "...", "score": 0.87, "chunk_index": 12 }
  ]
}
```

**Optional**: set `document_id` in the request body to target a specific document. Omit to search across all ingested documents.

---

## Architecture Overview

```
POST /ingest                           POST /ask
     │                                     │
     ▼                                     ▼
text_extractor.py              embed query (sentence-transformers)
     │                                     │
     ▼                                     ▼
chunker.py                     vector_store.search()
(word-overlap chunks)          (FAISS cosine similarity)
     │                                     │
     ▼                                     ▼
embedder.py                    top-k chunks → llm.py
(sentence-transformers)        (Claude, grounded prompt)
     │                                     │
     ▼                                     ▼
vector_store.py                    AskResponse
(FAISS IndexFlatIP)
```

### Component Responsibilities

| Module | Responsibility |
|--------|---------------|
| `main.py` | FastAPI app setup, CORS, static files |
| `api/routes.py` | HTTP handlers — thin layer, delegates to pipeline |
| `api/dependencies.py` | Singleton pipeline via thread-safe lazy init |
| `services/rag/pipeline.py` | Orchestrates ingest and ask flows |
| `services/rag/embedder.py` | Local embeddings via sentence-transformers |
| `services/rag/vector_store.py` | FAISS index management per document |
| `services/rag/llm.py` | Claude answer generation with grounded prompt |
| `utils/text_extractor.py` | File parsing (PDF, DOCX, TXT) |
| `utils/chunker.py` | Word-overlap text splitting |
| `models/schemas.py` | Pydantic request/response models |
| `models/config.py` | Settings via `pydantic-settings` |

### Key Design Decisions

**One FAISS index per document**

Each ingested document gets its own `IndexFlatIP`. This keeps documents isolated, enables targeted querying by `document_id`, and simplifies deletion (drop the index). A global index would require storing chunk→document mappings separately.

**`IndexFlatIP` + L2 normalisation = cosine similarity**

Exact brute-force search is the right call for document-scale corpora (typically hundreds to a few thousand chunks). Approximate methods (HNSW, IVF) trade recall for speed — unnecessary here and harder to reason about.

**Word-based chunking with overlap**

Chunking by word count approximates token count without needing a tokeniser at runtime. 50-word overlap ensures that sentences spanning a chunk boundary still appear in at least one chunk's full context.

**Voyage embeddings via Anthropic SDK**

Voyage-3 (1024-dim) consistently outperforms OpenAI's ada-002 on retrieval benchmarks and integrates cleanly via the same Anthropic client, keeping the dependency surface small.

**Singleton pipeline with thread-safe lazy init**

FAISS indices live in RAM. `get_pipeline()` uses a `threading.Lock` with double-checked locking to create one shared `RAGPipeline` instance on the first request. All subsequent requests reuse it, so ingested documents persist for the server's lifetime. Indices are also serialised to disk under `data/` so they survive restarts.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | required | Your Anthropic API key |
| `CHUNK_SIZE` | `500` | Target chunk size in words |
| `CHUNK_OVERLAP` | `50` | Overlap in words between chunks |
| `TOP_K_CHUNKS` | `5` | Number of chunks retrieved per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model for embeddings |
| `LLM_MODEL` | `claude-sonnet-4-6` | Claude model for answer generation |

---

## Project Structure

```
doc-qa-api/
├── main.py                  # FastAPI app entrypoint
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
│
├── api/
│   ├── __init__.py
│   ├── routes.py            # HTTP route handlers
│   └── dependencies.py      # Dependency injection
│
├── services/
│   └── rag/
│       ├── __init__.py
│       ├── pipeline.py      # RAG orchestrator
│       ├── embedder.py      # Embedding service
│       ├── vector_store.py  # FAISS wrapper
│       └── llm.py           # LLM answer generation
│
├── models/
│   ├── config.py            # App settings
│   └── schemas.py           # Pydantic schemas
│
├── utils/
│   ├── __init__.py
│   ├── text_extractor.py    # File parsing
│   └── chunker.py           # Text splitting
│
└── frontend/
    └── index.html           # Optional browser UI
```
