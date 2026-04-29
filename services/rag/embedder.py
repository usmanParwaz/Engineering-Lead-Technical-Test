import logging
from typing import List

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generates text embeddings locally using sentence-transformers.

    No API key or billing required. The model is downloaded once on first use
    (~90 MB) and cached by the sentence-transformers library.

    Default model: all-MiniLM-L6-v2
      - 384-dim vectors, fast on CPU, strong retrieval performance
      - Good balance of quality vs. size for document QA
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model_name = model
        logger.info(f"Importing sentence-transformers (PyTorch load — may take 30-60 s)…")
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model '{model}'…")
        self._model = SentenceTransformer(model)
        self._dimension: int = self._model.get_embedding_dimension()
        logger.info(f"Embedding model ready: '{model}' (dim={self._dimension})")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts. Returns list of float vectors."""
        if not texts:
            return []
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        logger.info(f"Embedded {len(texts)} texts using {self.model_name}")
        return vectors.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        vector = self._model.encode(query, convert_to_numpy=True, show_progress_bar=False)
        return vector.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension
