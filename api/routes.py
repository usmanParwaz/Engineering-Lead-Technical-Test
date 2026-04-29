import logging
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from models.schemas import AskRequest, AskResponse, HealthResponse, IngestResponse
from services.rag import RAGPipeline

from .dependencies import get_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health(pipeline: RAGPipeline = Depends(get_pipeline)):
    """Check system health and loaded document stats."""
    return HealthResponse(
        status="ok",
        documents_loaded=pipeline.document_count,
        total_chunks=pipeline.total_chunks,
    )


@router.get("/documents", tags=["Documents"])
async def list_documents(pipeline: RAGPipeline = Depends(get_pipeline)):
    """List all ingested documents with their IDs and chunk counts."""
    return {"documents": pipeline.list_documents()}


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
)
async def ingest_document(
    file: UploadFile = File(..., description="Document to ingest (PDF, DOCX, TXT, MD)"),
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Upload and index a document for later querying.

    Supported formats: PDF, DOCX, TXT, MD, CSV
    The document is chunked, embedded, and stored in a FAISS index.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(file_bytes) > 50 * 1024 * 1024:  # 50 MB guard
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit.")

    try:
        document_id, chunk_count = pipeline.ingest(
            file_bytes=file_bytes,
            filename=file.filename,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception(f"Ingest failed for '{file.filename}'")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")

    return IngestResponse(
        message="Document ingested successfully.",
        document_id=document_id,
        chunks_created=chunk_count,
        filename=file.filename,
    )


@router.post("/ask", response_model=AskResponse, tags=["QA"])
async def ask_question(
    body: AskRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Ask a question about an ingested document.

    Retrieves the most relevant chunks via semantic search and
    generates a grounded answer using Claude.

    Set `document_id` to target a specific document; omit to search all.
    """
    try:
        answer, sources = pipeline.ask(
            question=body.question,
            document_id=body.document_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Ask failed")
        raise HTTPException(status_code=500, detail=f"QA error: {str(e)}")

    return AskResponse(
        answer=answer,
        document_id=body.document_id,
        sources=sources,
    )
