# Architecture & Design

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client / Browser UI                       │
└────────────────────┬────────────────────────┬───────────────────┘
                     │  POST /ingest           │  POST /ask
                     ▼                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
│                                                                  │
│  routes.py → validates input → calls RAGPipeline                 │
└────────────────────┬────────────────────────┬───────────────────┘
                     │                         │
          ┌──────────▼──────────┐   ┌──────────▼──────────┐
          │    INGEST FLOW      │   │     QUERY FLOW       │
          │                     │   │                      │
          │  1. extract_text()  │   │  1. embed_query()    │
          │  2. chunk_text()    │   │  2. vector search    │
          │  3. embed(chunks)   │   │  3. build context    │
          │  4. faiss.add()     │   │  4. llm.answer()     │
          └─────────────────────┘   └──────────────────────┘
                     │                         │
          ┌──────────▼──────────┐   ┌──────────▼──────────┐
          │  FAISS IndexFlatIP  │   │  Anthropic Claude    │
          │  (per document)     │   │  (answer generation) │
          └─────────────────────┘   └──────────────────────┘
                     │                         │
          ┌──────────▼──────────────────────────▼──────────┐
          │     sentence-transformers (local, no API key)    │
          │      (embeddings for both ingest + query)       │
          └─────────────────────────────────────────────────┘
```

---

## Current Design Choices

### Storage: In-Process FAISS

FAISS indices are held in RAM as Python objects. One `IndexFlatIP` per document.

**Why**: Simplicity. No external service required, trivial to set up, performs well for typical document sizes (up to ~10k chunks per document before approximate search becomes worthwhile).

**Tradeoff**: Data is lost on server restart. Not horizontally scalable — each process has its own indices.

### Chunking Strategy: Word-Overlap

Documents are split into word windows of configurable size (default: 500 words) with a configurable overlap (default: 50 words).

**Why word count vs token count**: Avoids a tokeniser dependency at ingest time. At `chunk_size=500` words, real token counts stay within 600–700 for typical prose, safely within embedding model limits.

**Overlap rationale**: Sentences at chunk boundaries appear in at least one chunk fully, reducing the chance of a critical sentence being split across two low-scoring chunks.

### Embeddings: sentence-transformers (all-MiniLM-L6-v2)

Embeddings are generated locally using the `sentence-transformers` library. The default model is `all-MiniLM-L6-v2`, which produces 384-dimensional dense vectors. No separate API key or billing is required — the model (~90 MB) is downloaded once on first use and cached locally. The model is configurable via the `EMBEDDING_MODEL` environment variable.

### Similarity: Cosine via L2-Normalised Inner Product

`IndexFlatIP` computes dot product. After L2-normalising both the stored vectors and the query vector, dot product equals cosine similarity. This is exact (no ANN approximation), which is appropriate at this scale.

### LLM: Claude with Grounded Prompt

The system prompt strictly instructs Claude to answer only from provided excerpts and to say so when context is insufficient. This reduces hallucination and makes the system's limitations transparent to users.

Context is placed in the user turn (not the system prompt) so Claude can reference it more directly in its answer.

---

## Production Improvements

### 1. Persistent Vector Store

Replace in-process FAISS with a dedicated vector database:

| Option | Notes |
|--------|-------|
| **Qdrant** | Self-hostable, great filter support, REST + gRPC |
| **Weaviate** | Rich schema, hybrid search (dense + BM25) |
| **Pinecone** | Managed, simple API, no infra |
| **pgvector** | If you already run Postgres, lowest ops overhead |

This unlocks persistence across restarts, horizontal scaling, and shared state across API instances.

### 2. Chunking Improvements

- **Semantic chunking**: Split at paragraph or sentence boundaries rather than fixed word counts. Libraries like `spacy` or `nltk` help here.
- **Structure-aware chunking**: For PDFs, use section headings or page boundaries. For code files, split at function/class boundaries.
- **Hierarchical chunks**: Store both paragraph-level and section-level chunks. Retrieve section summaries first, then drill into paragraphs.

### 3. Retrieval Improvements

- **Hybrid search**: Combine dense (vector) and sparse (BM25/TF-IDF) retrieval, then re-rank with a cross-encoder. This handles both semantic similarity and exact keyword matches better.
- **Re-ranking**: Use a cross-encoder (e.g. Cohere Rerank, Voyage Rerank) as a second-stage ranker on the top-N retrieved chunks.
- **Query expansion**: Generate hypothetical document embeddings (HyDE) — ask the LLM to write what a relevant answer might look like, embed that, and search with it.

### 4. Caching

- Cache embeddings of frequently asked queries (Redis with TTL).
- Cache complete `(question, document_id) → answer` pairs when the document hasn't changed.
- Use a hash of document content as cache key to detect changes.

### 5. Streaming Responses

Use `anthropic.messages.stream()` and FastAPI's `StreamingResponse` with Server-Sent Events to stream token-by-token output to the client, reducing perceived latency on long answers.

### 6. Multi-tenancy & Access Control

Add per-user document namespaces. Each user's documents are isolated. Implement JWT authentication on the API layer.

### 7. Document Management

Add `DELETE /documents/{id}` and `GET /documents/{id}/status` endpoints. Support re-ingestion with version tracking.

### 8. Observability

- Structured logging with correlation IDs per request.
- Metrics: embedding latency, retrieval latency, LLM latency, token usage per query.
- Tracing: OpenTelemetry spans across the ingest and query pipelines.

### 9. Async Ingestion

For large documents, move ingestion to a background task queue (Celery + Redis, or FastAPI `BackgroundTasks` for simple cases). Return a job ID immediately and let the client poll for status.

### 10. Deployment

```
Users → CDN/Load Balancer → API instances (stateless) → Vector DB (shared)
                                         ↓
                                  Background workers (ingestion)
                                         ↓
                                  Object Storage (raw files, S3/GCS)
```

API instances become stateless once the vector store is externalised, enabling horizontal scaling behind a load balancer.
