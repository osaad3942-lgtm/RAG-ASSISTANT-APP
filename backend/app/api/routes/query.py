"""
app/api/routes/query.py

GET /health  - liveness check
POST /query  - retrieve -> build prompt -> call LLM -> return grounded answer
"""
import logging

from fastapi import APIRouter, Depends, Request

from app.schemas.query import QueryRequest, QueryResponse
from app.services.generation import GenerationService, answer_indicates_missing_context
from app.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Dependencies -------------------------------------------------------
# Services are created once at startup (see app/main.py lifespan) and stored on
# app.state. These dependency functions just fetch them per-request, and can be
# swapped out in tests via app.dependency_overrides without needing the real
# vector store / Ollama connection.

def get_retrieval_service(request: Request) -> RetrievalService:
    return request.app.state.retrieval_service


def get_generation_service(request: Request) -> GenerationService:
    return request.app.state.generation_service


# --- Routes ---------------------------------------------------------------

@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    generation_service: GenerationService = Depends(get_generation_service),
) -> QueryResponse:
    retrieved_chunks = retrieval_service.retrieve(payload.question)
    answer = generation_service.generate(payload.question, retrieved_chunks)
    sources = [] if answer_indicates_missing_context(answer) else [
        f"{c['source']} (page {c['page']})" for c in retrieved_chunks
    ]

    logger.info("Answered question with %d retrieved chunk(s).", len(retrieved_chunks))
    return QueryResponse(answer=answer, sources=sources)
