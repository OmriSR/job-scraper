"""Shared test utility functions."""

from matchai.schemas.candidate import CandidateProfile
from matchai.schemas.job import Job, JobDetail


def make_test_candidate(
    skills: list[str] | None = None,
    tools: list[str] | None = None,
    seniority: str = "mid",
) -> CandidateProfile:
    """Create a dummy candidate for testing."""
    return CandidateProfile(
        skills=skills or [],
        tools_frameworks=tools or [],
        seniority=seniority,
        domains=[],
        keywords=[],
        raw_text="",
    )


def make_test_job(
    uid: str,
    name: str,
    details_text: str = "",
    location: str | None = None,
    workplace_type: str | None = None,
    experience_level: str | None = None,
) -> Job:
    """Create a dummy job for testing."""
    details = []
    if details_text:
        details = [JobDetail(name="Description", value=f"<p>{details_text}</p>", order=1)]
    return Job(
        uid=uid,
        name=name,
        details=details,
        location=location,
        workplace_type=workplace_type,
        experience_level=experience_level,
    )
