"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from matchai.schemas.candidate import CandidateProfile
from matchai.schemas.job import Company, Job, JobDetail
from matchai.schemas.match import MatchResult


class TestCandidateProfile:
    def test_valid_profile(self):
        profile = CandidateProfile(
            skills=["python", "sql"],
            tools_frameworks=["django", "fastapi"],
            seniority="senior",
            years_experience=5,
            domains=["fintech"],
            keywords=["backend", "api"],
            raw_text="Sample CV text",
        )
        assert profile.skills == ["python", "sql"]
        assert profile.seniority == "senior"

    def test_optional_years_experience(self):
        profile = CandidateProfile(
            skills=[],
            tools_frameworks=[],
            seniority="junior",
            domains=[],
            keywords=[],
            raw_text="",
        )
        assert profile.years_experience is None


class TestJobDetail:
    def test_valid_detail(self):
        detail = JobDetail(
            name="Requirements",
            value="<p>Python experience</p>",
            order=1,
        )
        assert detail.name == "Requirements"
        assert detail.order == 1

    def test_optional_value(self):
        detail = JobDetail(name="Why Join Us", order=4)
        assert detail.value is None


class TestJob:
    def test_minimal_job(self):
        job = Job(uid="123", name="Software Engineer")
        assert job.uid == "123"
        assert job.name == "Software Engineer"
        assert job.details == []

    def test_job_with_details(self):
        job = Job(
            uid="456",
            name="Backend Developer",
            company_name="Acme Corp",
            location="Remote",
            details=[
                JobDetail(name="Description", value="<p>Build APIs</p>", order=1),
            ],
        )
        assert len(job.details) == 1
        assert job.details[0].name == "Description"


class TestCompany:
    def test_valid_company(self):
        company = Company(
            uid="comp-123",
            token="secret-token",
            extracted_from="https://careers.example.com",
        )
        assert company.uid == "comp-123"
        assert company.token == "secret-token"


class TestMatchResult:
    def test_valid_match(self):
        job = Job(uid="123", name="Developer")
        match = MatchResult(
            job=job,
            similarity_score=0.85,
            filter_score=0.7,
            final_score=0.8,
            explanation=["Strong Python match", "Experience aligns"],
            missing_skills=["kubernetes"],
        )
        assert match.final_score == 0.8
        assert len(match.explanation) == 2
