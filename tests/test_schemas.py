"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from matchai.schemas.candidate import CandidateProfile, SeniorityLevel
from matchai.schemas.job import Company, Job, JobDetail
from matchai.schemas.match import MatchResult


class TestSeniorityLevel:
    def test_case_insensitive_lookup(self):
        assert SeniorityLevel("senior") == SeniorityLevel.SENIOR
        assert SeniorityLevel("SENIOR") == SeniorityLevel.SENIOR
        assert SeniorityLevel("Senior") == SeniorityLevel.SENIOR

    def test_all_levels(self):
        assert SeniorityLevel("junior") == SeniorityLevel.JUNIOR
        assert SeniorityLevel("mid") == SeniorityLevel.MID
        assert SeniorityLevel("lead") == SeniorityLevel.LEAD
        assert SeniorityLevel("principal") == SeniorityLevel.PRINCIPAL
        assert SeniorityLevel("staff") == SeniorityLevel.STAFF

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            SeniorityLevel("invalid")

    def test_comparison(self):
        assert SeniorityLevel.JUNIOR < SeniorityLevel.MID
        assert SeniorityLevel.SENIOR < SeniorityLevel.LEAD
        assert SeniorityLevel.LEAD < SeniorityLevel.PRINCIPAL


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
        assert profile.seniority == SeniorityLevel.SENIOR

    def test_seniority_case_insensitive(self):
        profile = CandidateProfile(
            skills=[],
            tools_frameworks=[],
            seniority="SENIOR",
            domains=[],
            keywords=[],
            raw_text="",
        )
        assert profile.seniority == SeniorityLevel.SENIOR

    def test_invalid_seniority_raises(self):
        with pytest.raises(ValidationError):
            CandidateProfile(
                skills=[],
                tools_frameworks=[],
                seniority="invalid_level",
                domains=[],
                keywords=[],
                raw_text="",
            )

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
            name="Test Corp",
            uid="comp-123",
            token="secret-token",
            extracted_from="https://careers.example.com",
        )
        assert company.name == "Test Corp"
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
