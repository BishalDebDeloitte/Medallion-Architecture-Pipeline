"""Persistent document storage for pipeline memory, without semantic search.

Documents (business intents, report summaries) are stored in a local ChromaDB
collection and retrieved by id or in insertion batches. Chroma is given a
constant placeholder embedding for every document, so it never loads or
downloads an embedding model (the default one is ~80 MB fetched from the
network on first use). There is therefore no semantic ranking.
"""

import logging
from typing import Any

import chromadb

from core.paths import CHROMA_DIR

logger = logging.getLogger(__name__)

COLLECTION_NAME = "idamp_memory"
PLACEHOLDER_EMBEDDING = [0.0]


def get_chroma_client() -> Any:
    """Return a persistent ChromaDB client for the configured storage path."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(name: str = COLLECTION_NAME) -> Any:
    """Get or create a collection with automatic embeddings disabled."""
    return get_chroma_client().get_or_create_collection(
        name=name,
        embedding_function=None,
    )


def store_document(
    doc_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Upsert one document; return False (and log) on storage errors so the pipeline can continue."""
    try:
        get_collection().upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {"source": "pipeline"}],
            embeddings=[PLACEHOLDER_EMBEDDING],
        )
        return True
    except Exception as error:
        logger.warning("Could not store memory document %s: %s", doc_id, error)
        return False


def get_document(doc_id: str) -> dict[str, Any] | None:
    """Return one stored document by id, or None if it is missing or storage fails."""
    try:
        found = get_collection().get(ids=[doc_id], include=["documents", "metadatas"])
        if not found["ids"]:
            return None
        return {
            "id": found["ids"][0],
            "document": (found.get("documents") or [None])[0],
            "metadata": (found.get("metadatas") or [None])[0],
        }
    except Exception as error:
        logger.warning("Could not read memory document %s: %s", doc_id, error)
        return None


def query_memory(query_text: str, n_results: int = 5) -> list[dict[str, Any]]:
    """Return stored documents without performing semantic search."""
    del query_text
    try:
        results = get_collection().get(
            limit=n_results,
            include=["documents", "metadatas"],
        )
        ids = results.get("ids", [])
        documents = results.get("documents") or [None] * len(ids)
        metadatas = results.get("metadatas") or [None] * len(ids)
        return [
            {"id": doc_id, "document": document, "metadata": metadata}
            for doc_id, document, metadata in zip(ids, documents, metadatas)
        ]
    except Exception as error:
        logger.warning("Could not query memory: %s", error)
        return []
