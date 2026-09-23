"""
app/services/retrieval.py

Loads the persisted Chroma vector store (produced by notebooks/rag_pipeline.ipynb,
Phase 2.3 / 2.7) and the embedding model, and exposes a retrieve() function.

Loading is done once, at app startup (see app/main.py lifespan), not on every request.
"""
import json
import logging
from pathlib import Path
from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RetrievedChunk(TypedDict):
    id: str
    text: str
    source: str
    page: int
    distance: float | None


class RetrievalService:
    """
    Wraps the embedding model + Chroma collection produced by the notebook.

    The embedding model name and Chroma collection name are read from
    <vector_store_path>/config.json (written by the notebook's Phase 2.7 - Export
    step), so the backend always stays in sync with whatever the notebook actually
    built the vector store with, instead of duplicating those values by hand.
    """

    def __init__(self, vector_store_dir: Path, default_top_k: int = 3):
        self.vector_store_dir = vector_store_dir
        self.default_top_k = default_top_k

        config_path = vector_store_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"config.json not found at {config_path}. "
                f"Copy the vector_store folder produced by the notebook "
                f"(Phase 2.7 - Export) into {vector_store_dir} before starting the backend."
            )

        with open(config_path) as f:
            self.config = json.load(f)

        embedding_model_name = self.config["embedding_model_name"]
        collection_name = self.config["collection_name"]

        logger.info("Loading embedding model '%s'...", embedding_model_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)

        logger.info("Connecting to Chroma vector store at %s...", vector_store_dir)
        self.chroma_client = chromadb.PersistentClient(path=str(vector_store_dir))
        self.collection = self.chroma_client.get_collection(name=collection_name)

        logger.info(
            "Retrieval service ready: collection='%s', %d chunk(s) available.",
            collection_name,
            self.collection.count(),
        )

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.default_top_k
        query_embedding = self.embedding_model.encode([question]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(ids)

        retrieved: list[RetrievedChunk] = []
        for cid, text, meta, dist in zip(ids, docs, metas, dists):
            retrieved.append(
                {
                    "id": cid,
                    "text": text,
                    "source": meta.get("source"),
                    "page": meta.get("page"),
                    "distance": dist,
                }
            )
        return retrieved
