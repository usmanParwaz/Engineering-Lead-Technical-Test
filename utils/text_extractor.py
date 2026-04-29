import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract raw text from uploaded file bytes.
    Supports: .pdf, .docx, .txt, .md
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _extract_from_pdf(file_bytes)
    elif suffix == ".docx":
        return _extract_from_docx(file_bytes)
    elif suffix in (".txt", ".md", ".csv"):
        return file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: '{suffix}'. Supported types: pdf, docx, txt, md, csv")


def _extract_from_pdf(file_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(f"[Page {i + 1}]\n{text.strip()}")

        if not pages:
            raise ValueError("PDF appears to contain no extractable text (may be image-based).")

        return "\n\n".join(pages)
    except ImportError:
        raise RuntimeError("pypdf is not installed. Run: pip install pypdf")


def _extract_from_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document

        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        if not paragraphs:
            raise ValueError("DOCX appears to be empty.")

        return "\n\n".join(paragraphs)
    except ImportError:
        raise RuntimeError("python-docx is not installed. Run: pip install python-docx")
