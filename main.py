import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from api.routes import router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Document QA API starting up...")
    yield
    logger.info("Document QA API shutting down.")


app = FastAPI(
    title="Document QA API",
    description=(
        "Upload documents and ask questions about them using a RAG pipeline "
        "(Voyage embeddings + FAISS + Claude)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow all origins for local dev; tighten for production
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
app.include_router(router)

# Serve optional frontend if it exists
frontend_dir = Path(__file__).parent / "frontend"
if (frontend_dir / "index.html").exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    if (frontend_dir / "index.html").exists():
        return RedirectResponse(url="/ui")
    return JSONResponse(
        {"message": "Document QA API", "docs": "/docs", "health": "/health"}
    )
