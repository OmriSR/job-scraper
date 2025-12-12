from pydantic import BaseModel, Field


class CandidateProfile(BaseModel):
    """Structured representation of a candidate's CV extracted by LLM."""

    skills: list[str] = Field(
        description="Technical and soft skills extracted from the CV"
    )
    tools_frameworks: list[str] = Field(
        description="Specific tools, frameworks, and technologies mentioned"
    )
    seniority: str = Field(
        description="Inferred seniority level: junior, mid, senior, or lead"
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
