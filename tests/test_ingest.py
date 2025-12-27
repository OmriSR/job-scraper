"""Tests for job ingestion pipeline."""

import json
from unittest.mock import patch

import requests

from matchai.jobs.database import (
    get_all_companies,
    get_all_jobs,
    insert_companies,
    insert_jobs_to_db,
)
from matchai.jobs.embeddings import get_existing_embedding_uids
from matchai.jobs.ingest import fetch_positions, ingest_from_api, load_companies_from_file
from matchai.schemas.job import Company, Job

SAMPLE_COMPANIES = [
    {
        "name": "Test Corp",
        "uid": "comp-1",
        "token": "token-1",
        "extracted_from": "https://example.com/1",
    },
    {
        "name": "Another Inc",
        "uid": "comp-2",
        "token": "token-2",
        "extracted_from": "https://example.com/2",
    },
]

SAMPLE_API_RESPONSE = [
    {
        "uid": "job-1",
        "name": "Software Engineer",
        "company_name": "Test Corp",
        "location": "Tel Aviv",
        "details": [
            {"name": "Description", "value": "<p>Python developer</p>", "order": 1},
        ],
    },
    {
        "uid": "job-2",
        "name": "Product Manager",
        "company_name": "Test Corp",
        "location": "Remote",
        "details": [
            {"name": "Description", "value": "<p>Lead product</p>", "order": 1},
        ],
    },
]


class TestLoadCompaniesFromFile:
    def test_loads_companies(self, temp_db, tmp_path):
        file_path = tmp_path / "companies.json"
        file_path.write_text(json.dumps(SAMPLE_COMPANIES))

        inserted = load_companies_from_file(file_path)
        assert inserted == len(SAMPLE_COMPANIES)

        companies = get_all_companies()
        assert len(companies) == len(SAMPLE_COMPANIES)
        assert temp_db.exists()

    def test_idempotent_load(self, temp_db, tmp_path):
        file_path = tmp_path / "companies.json"
        file_path.write_text(json.dumps(SAMPLE_COMPANIES))

        load_companies_from_file(file_path)
        inserted = load_companies_from_file(file_path)
        assert inserted == 0
        assert temp_db.exists()


class TestFetchPositions:
    def test_successful_fetch(self):
        with patch("matchai.jobs.ingest.requests.get") as mock_get:
            mock_get.return_value.json.return_value = SAMPLE_API_RESPONSE
            mock_get.return_value.raise_for_status.return_value = None

            positions, success = fetch_positions("uid-123", "token-abc")

            assert success is True
            assert len(positions) == len(SAMPLE_API_RESPONSE)
            assert positions[0]["uid"] == SAMPLE_API_RESPONSE[0]["uid"]

    def test_handles_request_error(self):
        with patch("matchai.jobs.ingest.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Connection error")

            positions, success = fetch_positions("uid-123", "token-abc")

            assert success is False
            assert positions == []


class TestIngestFromApi:
    def test_no_companies_returns_empty_stats(self, temp_db_and_chroma):
        stats = ingest_from_api()

        assert stats["companies_processed"] == 0
        assert stats["jobs_fetched"] == 0
        assert temp_db_and_chroma["db_path"].exists()

    def test_ingests_jobs_from_api(self, temp_db_and_chroma):
        company = Company(**SAMPLE_COMPANIES[0])
        insert_companies([company])

        with patch("matchai.jobs.ingest.fetch_positions") as mock_fetch:
            mock_fetch.return_value = (SAMPLE_API_RESPONSE, True)

            stats = ingest_from_api()

            assert stats["companies_processed"] == 1
            assert stats["jobs_fetched"] == len(SAMPLE_API_RESPONSE)
            assert stats["jobs_inserted"] == len(SAMPLE_API_RESPONSE)
            assert stats["jobs_embedded"] == len(SAMPLE_API_RESPONSE)

        jobs = get_all_jobs()
        assert len(jobs) == len(SAMPLE_API_RESPONSE)

        embedding_uids = get_existing_embedding_uids()
        expected_uids = {job["uid"] for job in SAMPLE_API_RESPONSE}
        assert embedding_uids == expected_uids
        assert temp_db_and_chroma["chroma_path"].exists()

    def test_skips_existing_jobs(self, temp_db_and_chroma):
        company = Company(**SAMPLE_COMPANIES[0])
        insert_companies([company])

        with patch("matchai.jobs.ingest.fetch_positions") as mock_fetch:
            mock_fetch.return_value = (SAMPLE_API_RESPONSE, True)

            # First ingestion
            ingest_from_api()

            # Second ingestion - should skip existing
            stats = ingest_from_api()

            assert stats["jobs_fetched"] == len(SAMPLE_API_RESPONSE)
            assert stats["jobs_skipped"] == len(SAMPLE_API_RESPONSE)
            assert stats["jobs_inserted"] == 0
            assert temp_db_and_chroma["db_path"].exists()

    def test_multiple_companies(self, temp_db_and_chroma):
        companies = [Company(**c) for c in SAMPLE_COMPANIES]
        insert_companies(companies)

        api_responses = {
            (SAMPLE_COMPANIES[0]["uid"], SAMPLE_COMPANIES[0]["token"]): [
                {"uid": "job-a1", "name": "Dev A"}
            ],
            (SAMPLE_COMPANIES[1]["uid"], SAMPLE_COMPANIES[1]["token"]): [
                {"uid": "job-b1", "name": "Dev B"}
            ],
        }

        def mock_fetch(uid, token):
            return api_responses.get((uid, token), []), True

        with patch("matchai.jobs.ingest.fetch_positions", side_effect=mock_fetch):
            stats = ingest_from_api()

            assert stats["companies_processed"] == len(SAMPLE_COMPANIES)
            assert stats["jobs_fetched"] == len(SAMPLE_COMPANIES)
            assert stats["jobs_inserted"] == len(SAMPLE_COMPANIES)
            assert temp_db_and_chroma["db_path"].exists()

    def test_embeds_existing_jobs_missing_embeddings(self, temp_db_and_chroma):
        """Jobs in DB without embeddings should be embedded on next ingestion."""
        # Insert a company
        company = Company(**SAMPLE_COMPANIES[0])
        insert_companies([company])

        # Insert jobs directly into DB WITHOUT embedding them
        jobs = [
            Job(uid="existing-job-1", name="Existing Dev 1", details=[]),
            Job(uid="existing-job-2", name="Existing Dev 2", details=[]),
        ]
        insert_jobs_to_db(jobs)

        # Verify jobs are in DB but NOT embedded
        all_jobs = get_all_jobs()
        assert len(all_jobs) == 2
        embedding_uids = get_existing_embedding_uids()
        assert len(embedding_uids) == 0

        # Run ingestion (API returns the same jobs - not stale)
        with patch("matchai.jobs.ingest.fetch_positions") as mock_fetch:
            mock_fetch.return_value = (
                [{"uid": "existing-job-1"}, {"uid": "existing-job-2"}],
                True,
            )

            stats = ingest_from_api()

            # Should have embedded the 2 existing jobs that were missing embeddings
            assert stats["jobs_inserted"] == 0
            assert stats["jobs_embedded"] == 2

        # Verify jobs are now embedded
        embedding_uids = get_existing_embedding_uids()
        assert embedding_uids == {"existing-job-1", "existing-job-2"}

    def test_deletes_stale_jobs(self, temp_db_and_chroma):
        """Jobs no longer in API should be deleted from DB and embeddings."""
        company = Company(**SAMPLE_COMPANIES[0])
        insert_companies([company])

        # First ingestion - add jobs
        with patch("matchai.jobs.ingest.fetch_positions") as mock_fetch:
            mock_fetch.return_value = (SAMPLE_API_RESPONSE, True)
            ingest_from_api()

        # Verify jobs exist
        jobs = get_all_jobs()
        assert len(jobs) == 2
        embedding_uids = get_existing_embedding_uids()
        assert len(embedding_uids) == 2

        # Second ingestion - API returns only one job (one was closed)
        with patch("matchai.jobs.ingest.fetch_positions") as mock_fetch:
            mock_fetch.return_value = ([SAMPLE_API_RESPONSE[0]], True)
            stats = ingest_from_api()

        # Verify stale job was deleted
        assert stats["jobs_deleted"] == 1
        jobs = get_all_jobs()
        assert len(jobs) == 1
        assert jobs[0].uid == "job-1"

        # Verify embedding was also deleted
        embedding_uids = get_existing_embedding_uids()
        assert embedding_uids == {"job-1"}

    def test_no_deletion_when_api_fails(self, temp_db_and_chroma):
        """Jobs should NOT be deleted if any API call fails."""
        company = Company(**SAMPLE_COMPANIES[0])
        insert_companies([company])

        # First ingestion - add jobs
        with patch("matchai.jobs.ingest.fetch_positions") as mock_fetch:
            mock_fetch.return_value = (SAMPLE_API_RESPONSE, True)
            ingest_from_api()

        jobs_before = get_all_jobs()
        assert len(jobs_before) == 2

        # Second ingestion - API fails (returns empty with failure flag)
        with patch("matchai.jobs.ingest.fetch_positions") as mock_fetch:
            mock_fetch.return_value = ([], False)
            stats = ingest_from_api()

        # Verify no jobs were deleted due to API failure
        assert stats["jobs_deleted"] == 0
        jobs_after = get_all_jobs()
        assert len(jobs_after) == 2

    def test_no_deletion_when_all_jobs_still_open(self, temp_db_and_chroma):
        """No deletion when all existing jobs are still in API response."""
        company = Company(**SAMPLE_COMPANIES[0])
        insert_companies([company])

        with patch("matchai.jobs.ingest.fetch_positions") as mock_fetch:
            mock_fetch.return_value = (SAMPLE_API_RESPONSE, True)

            # First ingestion
            ingest_from_api()

            # Second ingestion - same jobs
            stats = ingest_from_api()

        assert stats["jobs_deleted"] == 0
        jobs = get_all_jobs()
        assert len(jobs) == 2
