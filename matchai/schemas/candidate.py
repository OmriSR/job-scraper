from enum import IntEnum

from pydantic import BaseModel, Field


class SeniorityLevel(IntEnum):
    """Seniority levels in ascending order.

    The integer values represent relative seniority for comparison.
    Use _missing_ for case-insensitive string parsing (e.g., from LLM output).
    """

    JUNIOR = 0
    MID = 1
    SENIOR = 2
    LEAD = 3
    PRINCIPAL = 4
    STAFF = 5

    @classmethod
    def _missing_(cls, value):
        """Allow case-insensitive string lookup for LLM output parsing."""
        if isinstance(value, str):
            value_lower = value.lower()
            for member in cls:
                if member.name.lower() == value_lower:
                    return member
        return None


class CandidateProfile(BaseModel):
    """Structured representation of a candidate's CV extracted by LLM."""

    skills: list[str] = Field(
        description="Technical and soft skills extracted from the CV"
    )
    tools_frameworks: list[str] = Field(
        description="Specific tools, frameworks, and technologies mentioned"
    )
    seniority: SeniorityLevel = Field(
        description="Inferred seniority level: junior, mid, senior, lead, principal, or staff"
    )
    years_experience: int | None = Field(
        default=None,
        description="Total years of professional experience, if determinable"
    )
    domains: list[str] = Field(
        description="Industry domains or areas of expertise (e.g., fintech, healthcare)"
    )
    keywords: list[str] = Field(
        description="Additional relevant keywords for matching"
    )
    raw_text: str = Field(
        description="Original extracted text from the CV"
    )
