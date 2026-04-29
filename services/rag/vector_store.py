import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    chunk_index: int
    text: str
    document_id: str


@dataclass
class DocumentIndex:
    """Holds FAISS index + metadata for a single ingested document."""

    document_id: str
    filename: str
    chunks: List[ChunkRecord]
    index: object  # faiss.IndexFlatIP


class VectorStore:
    """
    In-process FAISS vector store supporting multiple documents.

    When persist_dir is set, each document's FAISS index and chunk metadata
    are written to disk after ingestion and reloaded on startup, so state
    survives process restarts.
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        self._documents: Dict[str, DocumentIndex] = {}
        self._persist_dir = persist_dir
        if persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _doc_dir(self, document_id: str) -> Path:
        return self._persist_dir / document_id

    def _save_to_disk(self, document_id: str) -> None:
        if not self._persist_dir:
            return
        import faiss

        doc = self._documents[document_id]
        doc_dir = self._doc_dir(document_id)
        doc_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(doc.index, str(doc_dir / "index.faiss"))

        metadata = {
            "document_id": doc.document_id,
            "filename": doc.filename,
            "chunks": [
                {"chunk_index": c.chunk_index, "text": c.text, "document_id": c.document_id}
                for c in doc.chunks
            ],
        }
        with open(doc_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)

        logger.info(f"Persisted '{doc.filename}' to {doc_dir}")

    def _load_from_disk(self) -> None:
        if not self._persist_dir or not self._persist_dir.exists():
            return
        import faiss

        for doc_dir in sorted(self._persist_dir.iterdir()):
            if not doc_dir.is_dir():
                continue
            index_path = doc_dir / "index.faiss"
            metadata_path = doc_dir / "metadata.json"
            if not index_path.exists() or not metadata_path.exists():
                continue
            try:
                index = faiss.read_index(str(index_path))
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                chunks = [
                    ChunkRecord(
                        chunk_index=c["chunk_index"],
                        text=c["text"],
                        document_id=c["document_id"],
                    )
                    for c in meta["chunks"]
                ]
                self._documents[meta["document_id"]] = DocumentIndex(
                    document_id=meta["document_id"],
                    filename=meta["filename"],
                    chunks=chunks,
                    index=index,
                )
                logger.info(f"Restored '{meta['filename']}' from disk ({len(chunks)} chunks)")
            except Exception as exc:
                logger.warning(f"Could not load document from {doc_dir}: {exc}")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_document(
        self,
        filename: str,
        chunks: List[str],
        embeddings: List[List[float]],
    ) -> str:
        """
        Index a document's chunks and embeddings.
        Returns the generated document_id.
        """
        try:
            import faiss
        except ImportError:
            raise RuntimeError("faiss-cpu is not installed. Run: pip install faiss-cpu")

        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})")

        document_id = str(uuid.uuid4())
        vectors = np.array(embeddings, dtype="float32")

        # L2-normalise so inner product == cosine similarity
        faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        chunk_records = [
            ChunkRecord(chunk_index=i, text=text, document_id=document_id)
            for i, text in enumerate(chunks)
        ]

        self._documents[document_id] = DocumentIndex(
            document_id=document_id,
            filename=filename,
            chunks=chunk_records,
            index=index,
        )

        self._save_to_disk(document_id)
        logger.info(f"Stored document '{filename}' → {document_id} ({len(chunks)} chunks)")
        return document_id

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Tuple[ChunkRecord, float]]:
        """
        Retrieve the top-k most relevant chunks.

        If document_id is provided, search only that document.
        Otherwise merge results across all documents and re-rank.
        """
        try:
            import faiss
        except ImportError:
            raise RuntimeError("faiss-cpu is not installed.")

        query_vec = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query_vec)

        target_docs = (
            {document_id: self._documents[document_id]}
            if document_id and document_id in self._documents
            else self._documents
        )

        if not target_docs:
            return []

        all_results: List[Tuple[ChunkRecord, float]] = []

        for doc in target_docs.values():
            k = min(top_k, len(doc.chunks))
            scores, indices = doc.index.search(query_vec, k)

            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                all_results.append((doc.chunks[idx], float(score)))

        # Sort by cosine similarity descending and return top_k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_document(self, document_id: str) -> Optional[DocumentIndex]:
        return self._documents.get(document_id)

    def list_documents(self) -> List[Dict]:
        return [
            {
                "document_id": doc.document_id,
                "filename": doc.filename,
                "chunk_count": len(doc.chunks),
            }
            for doc in self._documents.values()
        ]

    @property
    def total_chunks(self) -> int:
        return sum(len(doc.chunks) for doc in self._documents.values())

    @property
    def document_count(self) -> int:
        return len(self._documents)
