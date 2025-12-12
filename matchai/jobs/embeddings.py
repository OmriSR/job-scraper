import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

from matchai.config import CHROMA_PATH, DATA_DIR, EMBEDDING_MODEL
from matchai.jobs.preprocessor import extract_details_text
from matchai.schemas.candidate import CandidateProfile
from matchai.schemas.job import Job

_model = None
_chroma_client = None
_collection = None

COLLECTION_NAME = "job_embeddings"


def _get_model() -> SentenceTransformer:
    """Lazy load sentence-transformers model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection() -> chromadb.Collection:
    """Lazy load ChromaDB collection."""
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
    """Generate embedding for a text string.

    Args:
        text: Text to embed.

    Returns:
        Embedding vector as numpy array.
    """
    model = _get_model()
    return model.encode(text, convert_to_numpy=True)


def embed_and_store_jobs(jobs: list[Job]) -> int:
    """Generate embeddings for jobs and store them in ChromaDB.

    This ensures jobs and their embeddings are always in sync.

    Args:
        jobs: List of Job objects to embed and store.

    Returns:
        Number of jobs embedded and stored.
    """
    if not jobs:
        return 0

    model = _get_model()
    collection = _get_collection()

    texts = [extract_details_text(job.details) for job in jobs]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

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

    collection = _get_collection()
    results = collection.get(ids=job_uids, include=["embeddings"])

    embeddings_dict = {}
    for uid, embedding in zip(results["ids"], results["embeddings"]):
        embeddings_dict[uid] = np.array(embedding)

    return embeddings_dict


def get_existing_embedding_uids() -> set[str]:
    """Get all job UIDs that have embeddings stored."""
    collection = _get_collection()
    results = collection.get(include=[])
    return set(results["ids"])


def embed_candidate(profile: CandidateProfile) -> np.ndarray:
    """Generate embedding for a candidate profile.

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
