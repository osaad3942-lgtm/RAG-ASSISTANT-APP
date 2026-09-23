"""
app/core/config.py

Application settings, loaded from environment variables / a .env file.
See .env.example for the full list of variables and their defaults.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Vector store ---
    # Path to the persisted Chroma vector store produced by notebooks/rag_pipeline.ipynb
    # (Phase 2.7 - Export). Copy that folder into backend/data/vector_store before running.
    vector_store_path: str = "data/vector_store"

    # --- Ollama ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # --- Retrieval ---
    top_k: int = 3

    # --- CORS ---
    # Comma-separated list of allowed frontend origins, e.g.
    # "http://localhost:8501,http://localhost:7860"
    cors_origins: str = "http://localhost:8501"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def vector_store_dir(self) -> Path:
        return Path(self.vector_store_path)


settings = Settings()
