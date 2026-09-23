"""
tests/test_query.py

Two required tests (per the guideline):
1. Happy path - POST /query with a valid body returns a 200 with answer + sources.
2. Invalid input - POST /query with a malformed body returns 422.

The retrieval/generation services are swapped out via FastAPI's dependency_overrides
so these tests don't need a real vector store or a running Ollama instance.
NOTE: the TestClient is used WITHOUT the `with` context manager, so the app's
lifespan (which loads the real services) does not run during these tests.
"""
from fastapi.testclient import TestClient

from app.api.routes.query import get_generation_service, get_retrieval_service
from app.main import app
from app.services.generation import build_prompt


class FakeRetrievalService:
    def retrieve(self, question: str, top_k: int | None = None):
        return [
            {
                "id": "chunk_0",
                "text": "Linear regression models the relationship between a dependent variable and one or more independent variables.",
                "source": "Lecture-2--Linear-Regression-.pdf",
                "page": 3,
                "distance": 0.12,
            }
        ]


class FakeGenerationService:
    def generate(self, question: str, retrieved_chunks) -> str:
        return "Linear regression is used to model and predict a continuous target variable."


class UnavailableGenerationService:
    def generate(self, question: str, retrieved_chunks) -> str:
        return "Unfortunately, the provided context does not contain information about Deep Learning."


app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService()
app.dependency_overrides[get_generation_service] = lambda: FakeGenerationService()

client = TestClient(app)


def test_query_happy_path():
    response = client.post("/query", json={"question": "What is linear regression used for?"})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert data["sources"][0] == "Lecture-2--Linear-Regression-.pdf (page 3)"


def test_query_omits_sources_when_context_does_not_contain_answer():
    original_generation_override = app.dependency_overrides[get_generation_service]
    app.dependency_overrides[get_generation_service] = lambda: UnavailableGenerationService()

    try:
        response = client.post("/query", json={"question": "What is deep learning?"})
    finally:
        app.dependency_overrides[get_generation_service] = original_generation_override

    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_query_invalid_input():
    # Missing the required "question" field entirely
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prompt_does_not_request_duplicate_sources():
    prompt = build_prompt("What is linear regression?", [])

    assert "After the answer, list the sources" not in prompt
