"""
app/main.py

FastAPI application entrypoint.

- Loads the vector store + embedding model + Ollama connection ONCE at startup
  (via the lifespan context manager), not on every request.
- Adds CORS middleware allowing the frontend's origin(s).
- Registers the /health and /query routes.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import settings
from app.services.generation import GenerationService
from app.services.retrieval import RetrievalService
from app.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: load the vector store, embedding model, and LLM connection once ---
    logger.info("Starting up: loading retrieval and generation services...")

    app.state.retrieval_service = RetrievalService(
        vector_store_dir=settings.vector_store_dir,
        default_top_k=settings.top_k,
    )
    app.state.generation_service = GenerationService(
        model=settings.ollama_model,
        host=settings.ollama_host,
    )

    logger.info("Startup complete.")
    yield
    # --- Shutdown: nothing to clean up explicitly ---
    logger.info("Shutting down.")


app = FastAPI(
    title="RAG-Powered Document Assistant API",
    description="Backend for the ML lecture RAG assistant (Core Track).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
