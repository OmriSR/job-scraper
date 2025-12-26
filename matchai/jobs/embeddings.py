"""Job embeddings with fastembed (ONNX) and support for local/cloud vector storage.

This module provides a unified interface that:
- Always uses fastembed (ONNX) for embeddings (smaller, faster than PyTorch)
- Uses Pinecone for vector storage in cloud mode
- Uses ChromaDB for vector storage in local development
"""

import numpy as np

from matchai.config import CHROMA_PATH, DATA_DIR, IS_CLOUD
from matchai.embeddings import (
    VectorRecord,
    embed_text_numpy,
    embed_texts_batch_numpy,
    fetch_embeddings,
    upsert_embeddings,
)
from matchai.jobs.preprocessor import extract_details_text
from matchai.schemas.candidate import CandidateProfile
from matchai.schemas.job import Job

# Module-level clients (used only in local mode)
_chroma_client = None
_collection = None

COLLECTION_NAME = "job_embeddings"


def _get_local_chromadb_collection():
    """Lazy load ChromaDB collection for local development."""
    import chromadb  # Import here to avoid loading in cloud mode

    global _chroma_client, _collection
    if _collection is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_text(text: str) -> np.ndarray:
    """Generate embedding for a text string using fastembed (ONNX).

    Args:
        text: Text to embed.

    Returns:
        Embedding vector as numpy array.
    """
    return embed_text_numpy(text)


def embed_and_store_jobs(jobs: list[Job]) -> int:
    """Generate embeddings for jobs and store them in vector database.

    Uses fastembed for embeddings and either Pinecone (cloud) or ChromaDB (local)
    for storage. In cloud mode, marks jobs as embedded in the database after
    successful Pinecone upsert.

    Args:
        jobs: List of Job objects to embed and store.

    Returns:
        Number of jobs embedded and stored.
    """
    import logging

    logger = logging.getLogger(__name__)

    if not jobs:
        return 0

    texts = [extract_details_text(job.details) for job in jobs]
    embeddings = embed_texts_batch_numpy(texts)

    if IS_CLOUD:
        from matchai.jobs.database import mark_jobs_as_embedded

        records = [
            VectorRecord(
                id=job.uid,
                vector=embedding.tolist(),
                metadata={"name": job.name, "company": job.company_name or ""},
            )
            for job, embedding in zip(jobs, embeddings)
        ]

        count = upsert_embeddings(records)

        if count != len(jobs):
            logger.warning(
                f"Pinecone upsert count mismatch: expected {len(jobs)}, got {count}. "
                f"Some jobs will be marked as embedded in the database but may not "
                f"have vectors in Pinecone."
            )

        # Mark jobs as embedded in database after successful Pinecone upsert
        if count > 0:
            uids = [job.uid for job in jobs]
            mark_jobs_as_embedded(uids)

        return count
    else:
        collection = _get_local_chromadb_collection()

        ids = [job.uid for job in jobs]
        metadatas = [{"name": job.name, "company": job.company_name or ""} for job in jobs]

        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=texts,
        )

        return len(jobs)


def get_job_embeddings(job_uids: list[str]) -> dict[str, np.ndarray]:
    """Retrieve embeddings for specific job UIDs.

    Args:
        job_uids: List of job UIDs to retrieve.

    Returns:
        Dictionary mapping job UID to embedding vector.
    """
    if not job_uids:
        return {}

    if IS_CLOUD:
        embeddings_dict = fetch_embeddings(job_uids)
        return {uid: np.array(vec) for uid, vec in embeddings_dict.items()}
    else:
        collection = _get_local_chromadb_collection()
        results = collection.get(ids=job_uids, include=["embeddings"])

        embeddings_dict = {}
        for uid, embedding in zip(results["ids"], results["embeddings"]):
            embeddings_dict[uid] = np.array(embedding)

        return embeddings_dict


def get_existing_embedding_uids() -> set[str]:
    """Get all job UIDs that have embeddings stored.

    In cloud mode, uses the embedded_at column in the database to track
    which jobs have been successfully embedded to Pinecone.
    In local mode, queries ChromaDB directly.
    """
    if IS_CLOUD:
        from matchai.jobs.database import get_embedded_job_uids

        return get_embedded_job_uids()
    else:
        collection = _get_local_chromadb_collection()
        results = collection.get(include=[])
        return set(results["ids"])


def embed_candidate(profile: CandidateProfile) -> np.ndarray:
    """Generate embedding for a candidate profile using fastembed (ONNX).

    Args:
        profile: CandidateProfile to embed.

    Returns:
        Embedding vector as numpy array.
    """
    text_parts = [
        " ".join(profile.skills),
        " ".join(profile.tools_frameworks),
        " ".join(profile.domains),
        " ".join(profile.keywords),
    ]
    text = " ".join(text_parts)
    return embed_text(text)
