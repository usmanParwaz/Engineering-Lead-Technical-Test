from typing import List, Optional

from pydantic import BaseModel, Field


class ChunkContext(BaseModel):
    text: str
    score: float
    chunk_index: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer from the document(s)")
    document_id: Optional[str] = Field(None, description="Target a specific document; omit to search all")


class AskResponse(BaseModel):
    answer: str
    document_id: Optional[str]
    sources: List[ChunkContext]


class IngestResponse(BaseModel):
    message: str
    document_id: str
    chunks_created: int
    filename: str


class HealthResponse(BaseModel):
    status: str
    documents_loaded: int
    total_chunks: int
