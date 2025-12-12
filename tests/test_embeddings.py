"""Tests for job embeddings and ChromaDB storage."""

from unittest.mock import patch

import numpy as np
import pytest

from matchai.jobs.embeddings import (
    embed_and_store_jobs,
    embed_candidate,
    embed_text,
    get_existing_embedding_uids,
    get_job_embeddings,
)
from matchai.schemas.candidate import CandidateProfile
from matchai.schemas.job import Job, JobDetail


@pytest.fixture
def temp_chroma(tmp_path):
    """Create a temporary ChromaDB for testing."""
    chroma_path = tmp_path / "chroma_db"
    data_dir = tmp_path

    with patch("matchai.jobs.embeddings.CHROMA_PATH", chroma_path), \
         patch("matchai.jobs.embeddings.DATA_DIR", data_dir), \
         patch("matchai.jobs.embeddings._collection", None), \
         patch("matchai.jobs.embeddings._chroma_client", None):
        yield chroma_path


class TestEmbedText:
    def test_returns_numpy_array(self):
        result = embed_text("hello world")
        assert isinstance(result, np.ndarray)

    def test_embedding_dimension(self):
        result = embed_text("hello world")
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        assert result.shape == (384,)


class TestEmbedAndStoreJobs:
    def test_stores_jobs(self, temp_chroma):
        jobs = [
            Job(
                uid="job-1",
                name="Developer",
                details=[JobDetail(name="Description", value="<p>Python developer</p>", order=1)],
            ),
            Job(
                uid="job-2",
                name="Engineer",
                details=[JobDetail(name="Description", value="<p>Software engineer</p>", order=1)],
            ),
        ]
        stored = embed_and_store_jobs(jobs)
        assert stored == 2

    def test_empty_jobs_list(self, temp_chroma):
        stored = embed_and_store_jobs([])
        assert stored == 0


class TestGetJobEmbeddings:
    def test_retrieves_stored_embeddings(self, temp_chroma):
        jobs = [
            Job(
                uid="job-1",
                name="Developer",
                details=[JobDetail(name="Description", value="<p>Python</p>", order=1)],
            ),
        ]
        embed_and_store_jobs(jobs)

        embeddings = get_job_embeddings(["job-1"])
        assert "job-1" in embeddings
        assert isinstance(embeddings["job-1"], np.ndarray)
        assert embeddings["job-1"].shape == (384,)

    def test_empty_uids_list(self, temp_chroma):
        embeddings = get_job_embeddings([])
        assert embeddings == {}


class TestGetExistingEmbeddingUids:
    def test_returns_stored_uids(self, temp_chroma):
        jobs = [
            Job(uid="uid-a", name="Job A", details=[]),
            Job(uid="uid-b", name="Job B", details=[]),
        ]
        embed_and_store_jobs(jobs)

        uids = get_existing_embedding_uids()
        assert uids == {"uid-a", "uid-b"}

    def test_empty_collection(self, temp_chroma):
        uids = get_existing_embedding_uids()
        assert uids == set()


class TestEmbedCandidate:
    def test_embeds_profile(self):
        profile = CandidateProfile(
            skills=["python", "sql"],
            tools_frameworks=["django"],
            seniority="senior",
            domains=["fintech"],
            keywords=["backend"],
            raw_text="",
        )
        result = embed_candidate(profile)
        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)
