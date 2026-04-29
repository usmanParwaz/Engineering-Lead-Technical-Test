import logging
from pathlib import Path
from typing import List, Optional, Tuple

from models.config import Settings
from models.schemas import ChunkContext
from utils.chunker import chunk_text
from utils.text_extractor import extract_text

from .embedder import EmbeddingService
from .llm import LLMService
from .vector_store import ChunkRecord, VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrates the full Retrieval-Augmented Generation pipeline.

    Flow:
        Ingest:  raw bytes → text extraction → chunking → embedding → FAISS index
        Query:   question → embedding → vector search → context → LLM → answer

    This class is the single integration point. Routes and API handlers
    call only this class, keeping RAG logic out of the HTTP layer.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = EmbeddingService(
            model=settings.embedding_model,
        )
        self.llm = LLMService(
            api_key=settings.anthropic_api_key,
            model=settings.llm_model,
        )
        self.vector_store = VectorStore(persist_dir=Path(settings.persist_dir))

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, file_bytes: bytes, filename: str) -> Tuple[str, int]:
        """
        Process and index a document.

        Returns:
            (document_id, chunk_count)
        """
        logger.info(f"Ingesting '{filename}' ({len(file_bytes)} bytes)")

        # Clear any previously stored documents before indexing the new one
        self.vector_store.clear_all()

        # 1. Extract text
        text = extract_text(file_bytes, filename)
        if not text.strip():
            raise ValueError("Document appears to be empty or unreadable.")

        # 2. Chunk
        chunks = chunk_text(
            text,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        logger.info(f"Created {len(chunks)} chunks from '{filename}'")

        # 3. Embed
        embeddings = self.embedder.embed(chunks)

        # 4. Store in FAISS
        document_id = self.vector_store.add_document(
            filename=filename,
            chunks=chunks,
            embeddings=embeddings,
        )

        return document_id, len(chunks)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        document_id: Optional[str] = None,
    ) -> Tuple[str, List[ChunkContext]]:
        """
        Answer a question using retrieved document context.

        Returns:
            (answer_text, source_chunks)
        """
        if self.vector_store.document_count == 0:
            raise ValueError("No documents have been ingested. Please POST to /ingest first.")

        # 1. Embed the question
        query_embedding = self.embedder.embed_query(question)

        # 2. Retrieve top-k relevant chunks
        results: List[Tuple[ChunkRecord, float]] = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.settings.top_k_chunks,
            document_id=document_id,
        )

        if not results:
            return "No relevant content found in the document(s) for this question.", []

        # 3. Build context
        context_chunks = [record.text for record, _ in results]
        source_contexts = [
            ChunkContext(text=record.text, score=round(score, 4), chunk_index=record.chunk_index)
            for record, score in results
        ]

        # 4. Generate answer
        answer = self.llm.answer(question=question, context_chunks=context_chunks)

        return answer, source_contexts

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def list_documents(self):
        return self.vector_store.list_documents()

    @property
    def document_count(self) -> int:
        return self.vector_store.document_count

    @property
    def total_chunks(self) -> int:
        return self.vector_store.total_chunks
