from pydantic import BaseModel, Field

from matchai.schemas.job import Job


class MatchResult(BaseModel):
    """Result of matching a candidate to a job position."""

    job: Job = Field(
        description="The matched job position"
    )
    similarity_score: float = Field(
        description="Semantic similarity score between CV and job (0-1)"
    )
    filter_score: float = Field(
        description="Deterministic filter score based on skill/seniority overlap (0-1)"
    )
    final_score: float = Field(
        description="Combined weighted score used for ranking"
    )
    explanation: list[str] = Field(
        description="2-3 bullet points explaining why this job matches"
    )
    missing_skills: list[str] = Field(
        description="Skills required by the job but not found in candidate's CV"
    )
    interview_tips: list[str] = Field(
        default_factory=list,
        description="1-2 actionable tips for interview preparation"
    )
    apply_url: str | None = Field(
        default=None,
        description="Best available URL to apply for the position"
    )
