import threading
from typing import Optional

from models.config import Settings
from services.rag.pipeline import RAGPipeline

_settings_lock = threading.Lock()
_pipeline_lock = threading.Lock()
_settings: Optional[Settings] = None
_pipeline: Optional[RAGPipeline] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                _settings = Settings()
    return _settings


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = RAGPipeline(settings=get_settings())
    return _pipeline
