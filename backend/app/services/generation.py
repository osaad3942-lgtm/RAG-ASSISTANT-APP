"""
app/services/generation.py

Builds the prompt that combines retrieved context with the user's question
(with citation-style grounding, same template as Phase 2.4 of the notebook),
and calls the local Ollama LLM to produce the answer.
"""
import logging
import re

import ollama

from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

_CONTEXT_NOT_FOUND_PATTERNS = (
    r"information .* not available .* (the )?(provided )?context",
    r"(the )?(provided )?context.*does not contain",
    r"(the )?(provided )?context.*doesn't contain",
    r"not found in (the )?(provided )?(documents|context)",
    r"cannot be found in (the )?(provided )?(documents|context)",
)


def answer_indicates_missing_context(answer: str) -> bool:
    normalized_answer = " ".join(answer.lower().split())
    return any(
        re.search(pattern, normalized_answer)
        for pattern in _CONTEXT_NOT_FOUND_PATTERNS
    )


def build_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[{i}] (source: {chunk['source']}, page {chunk['page']})\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."

    return f"""You are a helpful Machine Learning study assistant. Answer the question using ONLY the context provided below.

Rules:
- Base your answer strictly on the provided context.
- If the answer cannot be found in the context, say clearly that the information is not available in the documents. Do not make anything up.

Context:
{context_text}

Question:
{question}

Answer:"""


class GenerationService:
    def __init__(self, model: str, host: str):
        self.model = model
        self.client = ollama.Client(host=host)
        logger.info("Generation service ready: model='%s', host='%s'", model, host)

    def generate(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        prompt = build_prompt(question, retrieved_chunks)
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
