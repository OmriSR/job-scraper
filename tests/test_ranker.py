"""Tests for semantic job ranking."""

import numpy as np
from unittest.mock import patch

from matchai.matching.ranker import (
    compute_similarities_batch,
    compute_final_score,
    rank_jobs,
)
from tests.test_utils import make_test_candidate, make_test_job


class TestComputeSimilaritiesBatch:
    def test_identical_vectors_return_1(self):
        vec = np.array([1.0, 0.0, 0.0])
        job_vecs = np.array([[1.0, 0.0, 0.0]])

        result = compute_similarities_batch(vec, job_vecs)

        assert len(result) == 1
        assert np.isclose(result[0], 1.0)

    def test_orthogonal_vectors_return_0(self):
        vec = np.array([1.0, 0.0, 0.0])
        job_vecs = np.array([[0.0, 1.0, 0.0]])

        result = compute_similarities_batch(vec, job_vecs)

        assert len(result) == 1
        assert np.isclose(result[0], 0.0)

    def test_multiple_jobs(self):
        candidate_vec = np.array([1.0, 0.0, 0.0])
        job_vecs = np.array([
            [1.0, 0.0, 0.0],  # identical
            [0.0, 1.0, 0.0],  # orthogonal
            [0.5, 0.5, 0.0],  # partial
        ])

        result = compute_similarities_batch(candidate_vec, job_vecs)

        assert len(result) == 3
        assert np.isclose(result[0], 1.0)
        assert np.isclose(result[1], 0.0)
        assert 0 < result[2] < 1

    def test_clipped_to_non_negative(self):
        vec = np.array([1.0, 0.0])
        job_vecs = np.array([[-1.0, 0.0]])

        result = compute_similarities_batch(vec, job_vecs)

        assert result[0] >= 0.0


class TestComputeFinalScore:
    def test_weighted_combination(self):
        filter_score = 80.0  # 0-100 scale
        similarity_score = 0.6  # 0-1 scale

        result = compute_final_score(filter_score, similarity_score)

        # With FILTER_WEIGHT=0.4, SIMILARITY_WEIGHT=0.6:
        # 0.4 * 0.8 + 0.6 * 0.6 = 0.32 + 0.36 = 0.68
        assert 0 <= result <= 1

    def test_max_scores_return_1(self):
        result = compute_final_score(100.0, 1.0)

        assert np.isclose(result, 1.0)

    def test_zero_scores_return_0(self):
        result = compute_final_score(0.0, 0.0)

        assert np.isclose(result, 0.0)


class TestRankJobs:
    def test_empty_jobs_returns_empty(self):
        candidate = make_test_candidate(skills=["python"])

        result = rank_jobs([], candidate)

        assert result == []

    @patch("matchai.matching.ranker.embed_candidate")
    @patch("matchai.matching.ranker.get_job_embeddings")
    def test_returns_match_results(self, mock_get_embeddings, mock_embed_candidate):
        candidate = make_test_candidate(skills=["python"])
        jobs = [
            (make_test_job("1", "Developer", "Python"), 80.0),
            (make_test_job("2", "Engineer", "Java"), 60.0),
        ]

        mock_embed_candidate.return_value = np.array([1.0, 0.0, 0.0])
        mock_get_embeddings.return_value = {
            "1": np.array([0.9, 0.1, 0.0]),
            "2": np.array([0.5, 0.5, 0.0]),
        }

        results = rank_jobs(jobs, candidate)

        assert len(results) == 2
        assert all(hasattr(r, "job") for r in results)
        assert all(hasattr(r, "final_score") for r in results)

    @patch("matchai.matching.ranker.embed_candidate")
    @patch("matchai.matching.ranker.get_job_embeddings")
    def test_sorted_by_final_score_descending(self, mock_get_embeddings, mock_embed_candidate):
        candidate = make_test_candidate(skills=["python"])
        jobs = [
            (make_test_job("1", "Low match"), 50.0),
            (make_test_job("2", "High match"), 90.0),
        ]

        mock_embed_candidate.return_value = np.array([1.0, 0.0, 0.0])
        mock_get_embeddings.return_value = {
            "1": np.array([0.3, 0.7, 0.0]),
            "2": np.array([0.9, 0.1, 0.0]),
        }

        results = rank_jobs(jobs, candidate)

        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    @patch("matchai.matching.ranker.embed_candidate")
    @patch("matchai.matching.ranker.get_job_embeddings")
    def test_top_n_limits_results(self, mock_get_embeddings, mock_embed_candidate):
        candidate = make_test_candidate(skills=["python"])
        jobs = [
            (make_test_job(str(i), f"Job {i}"), 50.0 + i)
            for i in range(10)
        ]

        mock_embed_candidate.return_value = np.array([1.0, 0.0, 0.0])
        mock_get_embeddings.return_value = {
            str(i): np.array([1.0, 0.0, 0.0])
            for i in range(10)
        }

        results = rank_jobs(jobs, candidate, top_n=3)

        assert len(results) == 3

    @patch("matchai.matching.ranker.embed_candidate")
    @patch("matchai.matching.ranker.get_job_embeddings")
    def test_handles_missing_embeddings(self, mock_get_embeddings, mock_embed_candidate):
        candidate = make_test_candidate(skills=["python"])
        jobs = [
            (make_test_job("1", "With embedding"), 80.0),
            (make_test_job("2", "Without embedding"), 80.0),
        ]

        mock_embed_candidate.return_value = np.array([1.0, 0.0, 0.0])
        mock_get_embeddings.return_value = {
            "1": np.array([1.0, 0.0, 0.0]),
            # "2" is missing
        }

        results = rank_jobs(jobs, candidate)

        assert len(results) == 2
        job_with_embedding = next(r for r in results if r.job.uid == "1")
        job_without_embedding = next(r for r in results if r.job.uid == "2")
        assert job_with_embedding.similarity_score > 0
        assert job_without_embedding.similarity_score == 0.0

    @patch("matchai.matching.ranker.embed_candidate")
    @patch("matchai.matching.ranker.get_job_embeddings")
    def test_filter_score_normalized(self, mock_get_embeddings, mock_embed_candidate):
        candidate = make_test_candidate(skills=["python"])
        jobs = [(make_test_job("1", "Dev"), 80.0)]

        mock_embed_candidate.return_value = np.array([1.0, 0.0, 0.0])
        mock_get_embeddings.return_value = {"1": np.array([1.0, 0.0, 0.0])}

        results = rank_jobs(jobs, candidate)

        assert results[0].filter_score == 0.8  # 80/100
