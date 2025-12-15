"""Semantic ranking for job matching."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from matchai.config import FILTER_WEIGHT, SIMILARITY_WEIGHT
from matchai.jobs.embeddings import embed_candidate, get_job_embeddings
from matchai.schemas.candidate import CandidateProfile
from matchai.schemas.job import Job
from matchai.schemas.match import MatchResult


def compute_similarities_batch(
    candidate_embedding: np.ndarray,
    job_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between candidate and multiple jobs.

    Args:
        candidate_embedding: Candidate embedding vector (1D).
        job_embeddings: Job embedding matrix (n_jobs, n_features).

    Returns:
        Array of similarity scores (n_jobs,).
    """
    # Sklearn expects 2D arrays (samples, features)
    similarities = cosine_similarity(
        candidate_embedding.reshape(1, -1),
        job_embeddings,
    )[0]
    return np.clip(similarities, 0.0, 1.0)


def compute_final_score(filter_score: float, similarity_score: float) -> float:
    """Compute weighted final score from filter and similarity scores.

    Args:
        filter_score: Deterministic filter score (0-100 from fuzzy matching).
        similarity_score: Semantic similarity score (0-1).

    Returns:
        Combined weighted score (0-1).
    """
    normalized_filter = filter_score / 100.0
    return (FILTER_WEIGHT * normalized_filter) + (SIMILARITY_WEIGHT * similarity_score)


def rank_jobs(
    filtered_jobs: list[tuple[Job, float]],
    candidate: CandidateProfile,
    top_n: int | None = None,
) -> list[MatchResult]:
    """Rank filtered jobs by semantic similarity to candidate.

    Args:
        filtered_jobs: List of (job, filter_score) tuples from apply_filters.
        candidate: Candidate profile to match against.
        top_n: Maximum number of results to return (None for all).

    Returns:
        List of MatchResult objects sorted by final_score descending.
    """
    if not filtered_jobs:
        return []

    candidate_embedding = embed_candidate(candidate)

    job_uids = [job.uid for job, _ in filtered_jobs]
    job_embeddings_dict = get_job_embeddings(job_uids)

    # Build ordered list matching filtered_jobs order
    valid_indices = []
    embeddings_list = []
    for i, uid in enumerate(job_uids):
        embedding = job_embeddings_dict.get(uid)
        if embedding is not None:
            valid_indices.append(i)
            embeddings_list.append(embedding)

    # Batch compute similarities
    similarity_scores = np.zeros(len(filtered_jobs))
    if embeddings_list:
        job_embeddings_matrix = np.vstack(embeddings_list)
        batch_similarities = compute_similarities_batch(
            candidate_embedding, job_embeddings_matrix
        )
        similarity_scores[valid_indices] = batch_similarities

    results = []
    for (job, filter_score), similarity_score in zip(filtered_jobs, similarity_scores):
        final_score = compute_final_score(filter_score, similarity_score)

        result = MatchResult(
            job=job,
            similarity_score=similarity_score,
            filter_score=filter_score / 100.0,
            final_score=final_score,
            explanation=[],
            missing_skills=[],
        )
        results.append(result)

    results.sort(key=lambda r: r.final_score, reverse=True)

    if top_n is not None:
        results = results[:top_n]

    return results
