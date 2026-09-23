"""
app/schemas/query.py

Request/response models for the /query endpoint.
"""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question.")


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
