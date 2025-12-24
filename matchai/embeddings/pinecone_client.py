"""Pinecone client for cloud-based vector storage.

This module provides a managed vector database that replaces local ChromaDB:
- Serverless, no infrastructure to manage
- Free tier: 1 index, 100K vectors
- Optimized for cosine similarity search
"""

from dataclasses import dataclass
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from matchai.config import PINECONE_API_KEY, PINECONE_DIMENSION, PINECONE_INDEX

_client: Pinecone | None = None
_index: Any | None = None


@dataclass
class VectorRecord:
    """A single vector record bundling ID, embedding, and metadata.

    This ensures data integrity by keeping related fields together,
    eliminating the risk of misaligned lists.
    """

    id: str
    vector: list[float]
    metadata: dict[str, Any] | None = None


def _get_client() -> Pinecone:
    """Lazy load Pinecone client."""
    global _client
    if _client is None:
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable is not set")
        _client = Pinecone(api_key=PINECONE_API_KEY)
    return _client


def _get_index() -> Any:
    """Lazy load Pinecone index, creating it if it doesn't exist."""
    global _index
    if _index is None:
        client = _get_client()

        # Check if index exists
        existing_indexes = [idx.name for idx in client.list_indexes()]

        if PINECONE_INDEX not in existing_indexes:
            # Create index with serverless spec (free tier compatible)
            client.create_index(
                name=PINECONE_INDEX,
                dimension=PINECONE_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        _index = client.Index(PINECONE_INDEX)
    return _index


def upsert_embeddings(records: list[VectorRecord]) -> int:
    """Insert or update embeddings in Pinecone.

    Args:
        records: List of VectorRecord objects, each containing id, vector, and optional metadata.

    Returns:
        Number of vectors upserted.
    """
    if not records:
        return 0

    index = _get_index()

    # Convert VectorRecords to Pinecone format
    pinecone_records = [
        {
            "id": record.id,
            "values": record.vector,
            **({"metadata": record.metadata} if record.metadata else {}),
        }
        for record in records
    ]

    # Upsert in batches of 100 (Pinecone limit)
    batch_size = 100
    total_upserted = 0

    for i in range(0, len(pinecone_records), batch_size):
        batch = pinecone_records[i : i + batch_size]
        index.upsert(vectors=batch)
        total_upserted += len(batch)

    return total_upserted


def query_similar(
    vector: list[float],
    top_k: int = 10,
    filter_dict: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Query for similar vectors.

    Args:
        vector: Query embedding vector.
        top_k: Number of results to return.
        filter_dict: Optional metadata filter.

    Returns:
        List of matches with id, score, and metadata.
    """
    index = _get_index()

    results = index.query(
        vector=vector,
        top_k=top_k,
        filter=filter_dict,
        include_metadata=True,
    )

    return [
        {
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata or {},
        }
        for match in results.matches
    ]


# NOTE: get_existing_ids is commented out because Pinecone doesn't provide a native
# "list all IDs" API. The workaround of querying with a zero vector is not reliable
# for production use. For idempotency checks, we track job UIDs in PostgreSQL instead.
# If this function is needed in the future, consider:
# 1. Using Pinecone's list() API (available in newer versions)
# 2. Maintaining a separate index of IDs in the database
# 3. Using namespaces with known ID patterns
#
# def get_existing_ids(prefix: str | None = None) -> set[str]:
#     """Get all existing vector IDs in the index."""
#     ...


def delete_embeddings(ids: list[str]) -> int:
    """Delete embeddings by IDs.

    Args:
        ids: List of vector IDs to delete.

    Returns:
        Number of vectors deleted.
    """
    if not ids:
        return 0

    index = _get_index()
    index.delete(ids=ids)
    return len(ids)


def fetch_embeddings(ids: list[str]) -> dict[str, list[float]]:
    """Fetch embeddings by IDs.

    Args:
        ids: List of vector IDs to fetch.

    Returns:
        Dictionary mapping ID to embedding vector.
    """
    if not ids:
        return {}

    index = _get_index()
    results = index.fetch(ids=ids)

    return {uid: vec.values for uid, vec in results.vectors.items()}
