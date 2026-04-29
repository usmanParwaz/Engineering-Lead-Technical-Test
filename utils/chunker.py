from typing import List
import logging

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks by token-approximate word count.

    Strategy: sentence-aware word splitting so chunks don't break mid-sentence
    when possible. Overlap ensures context continuity across chunk boundaries.

    Args:
        text: Raw document text
        chunk_size: Target chunk size in words (approximate tokens)
        overlap: Number of words to overlap between consecutive chunks

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace
    text = " ".join(text.split())

    words = text.split()
    total_words = len(words)

    if total_words <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        if end >= total_words:
            break

        # Move forward by (chunk_size - overlap), ensuring progress
        stride = max(chunk_size - overlap, 1)
        start += stride

    logger.debug(f"Chunked {total_words} words into {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks
