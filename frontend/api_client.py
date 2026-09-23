"""
api_client.py

Wrapper for calling the backend API, so app.py never makes HTTP requests itself.

The backend URL is read from the API_BASE_URL environment variable
(see frontend/.env) and is never hard-coded.
"""
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load frontend/.env no matter which folder the app is started from.
load_dotenv(Path(__file__).resolve().parent / ".env")

# Local LLM answers can take a while, so allow a generous wait.
REQUEST_TIMEOUT_SECONDS = 120


class APIClientError(Exception):
    """Raised with a user-friendly message when a call to the backend fails."""


def get_api_base_url() -> str:
    base_url = os.getenv("API_BASE_URL", "").strip()
    if not base_url:
        raise APIClientError(
            "API_BASE_URL is not set. Add it to frontend/.env (see .env.example)."
        )
    return base_url.rstrip("/")


def ask_question(question: str) -> dict:
    """
    POST /query  ->  {"answer": str, "sources": list[str]}

    Raises APIClientError (with a message that is safe to show to the user)
    if the backend cannot be reached or returns something unusable.
    """
    url = f"{get_api_base_url()}/query"

    try:
        response = requests.post(
            url,
            json={"question": question},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ConnectionError as exc:
        raise APIClientError(
            "Could not connect to the backend. Please try again."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise APIClientError(
            "The backend took too long to respond. Please try again."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise APIClientError(
            "Something went wrong while contacting the backend. Please try again."
        ) from exc

    if response.status_code == 422:
        raise APIClientError(
            "The backend could not accept that question. Please rephrase it and try again."
        )
    if not response.ok:
        raise APIClientError("The backend returned an error. Please try again.")

    try:
        data = response.json()
        return {"answer": data["answer"], "sources": list(data["sources"])}
    except (ValueError, KeyError, TypeError) as exc:
        raise APIClientError(
            "The backend returned an unexpected response. Please try again."
        ) from exc
