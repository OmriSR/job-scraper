"""Tests for SQLite database operations."""

from matchai.jobs.database import (
    get_all_jobs,
    get_existing_job_uids,
    get_jobs_by_uids,
    insert_companies,
    insert_jobs_to_db,
)
from matchai.schemas.job import Company, Job, JobDetail


class TestInitDatabase:
    def test_creates_database(self, temp_db):
        assert temp_db.exists()


class TestInsertJobs:
    def test_insert_single_job(self, temp_db):
        job = Job(uid="job-1", name="Developer")
        inserted = insert_jobs_to_db([job])
        assert inserted == 1

    def test_insert_job_with_details(self, temp_db):
        job = Job(
            uid="job-2",
            name="Engineer",
            details=[JobDetail(name="Description", value="<p>Test</p>", order=1)],
        )
        inserted = insert_jobs_to_db([job])
        assert inserted == 1

        jobs = get_all_jobs()
        assert len(jobs) == 1
        assert len(jobs[0].details) == 1

    def test_idempotent_insert(self, temp_db):
        job = Job(uid="job-3", name="Manager")
        insert_jobs_to_db([job])
        inserted = insert_jobs_to_db([job])  # Insert again
        assert inserted == 0


class TestInsertCompanies:
    def test_insert_company(self, temp_db):
        company = Company(
            name="Test Corp",
            uid="comp-1",
            token="token123",
            extracted_from="https://example.com",
        )
        inserted = insert_companies([company])
        assert inserted == 1

    def test_idempotent_company_insert(self, temp_db):
        company = Company(name="Corp", uid="comp-2", token="token", extracted_from="url")
        insert_companies([company])
        inserted = insert_companies([company])
        assert inserted == 0


class TestGetJobs:
    def test_get_all_jobs(self, temp_db):
        jobs = [
            Job(uid="j1", name="Dev 1"),
            Job(uid="j2", name="Dev 2"),
        ]
        insert_jobs_to_db(jobs)

        result = get_all_jobs()
        assert len(result) == 2

    def test_get_jobs_by_uids(self, temp_db):
        jobs = [
            Job(uid="j1", name="Dev 1"),
            Job(uid="j2", name="Dev 2"),
            Job(uid="j3", name="Dev 3"),
        ]
        insert_jobs_to_db(jobs)

        result = get_jobs_by_uids(["j1", "j3"])
        assert len(result) == 2
        uids = {j.uid for j in result}
        assert uids == {"j1", "j3"}

    def test_get_jobs_by_uids_empty_list(self, temp_db):
        result = get_jobs_by_uids([])
        assert result == []


class TestGetExistingJobUids:
    def test_returns_existing_uids(self, temp_db):
        jobs = [Job(uid="uid-1", name="J1"), Job(uid="uid-2", name="J2")]
        insert_jobs_to_db(jobs)

        uids = get_existing_job_uids()
        assert uids == {"uid-1", "uid-2"}

    def test_empty_database(self, temp_db):
        uids = get_existing_job_uids()
        assert uids == set()
