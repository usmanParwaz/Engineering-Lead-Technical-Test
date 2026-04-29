from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str

    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_chunks: int = 5

    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "claude-sonnet-4-6"
    persist_dir: str = "data"
