from pydantic import BaseModel, Field


class JobDetail(BaseModel):
    """A section within a job posting (e.g., Description, Requirements)."""

    name: str = Field(
        description="Section name (e.g., 'Description', 'Requirements', 'Responsibilities')"
    )
    value: str | None = Field(
        default=None,
        description="HTML content of the section"
    )
    order: int = Field(
        description="Display order of the section"
    )


class Job(BaseModel):
    """A job position listing stored in the database for matching."""

    uid: str = Field(description="Unique identifier for the job")
    name: str = Field(description="Job title")
    department: str | None = Field(default=None, description="Department name")
    email: str | None = Field(default=None, description="Contact email")
    email_alias: str | None = Field(default=None, description="Email alias")
    url_comeet_hosted_page: str | None = Field(default=None, description="Comeet hosted page URL")
    url_recruit_hosted_page: str | None = Field(default=None, description="Recruit hosted page URL")
    url_active_page: str | None = Field(default=None, description="Active page URL")
    employment_type: str | None = Field(default=None, description="Employment type (full-time, part-time, etc.)")
    experience_level: str | None = Field(default=None, description="Required experience level")
    location: str | None = Field(default=None, description="Job location")
    internal_use_custom_id: str | None = Field(default=None, description="Internal custom ID")
    is_consent_needed: bool | None = Field(default=None, description="Whether consent is required")
    referrals_reward: str | None = Field(default=None, description="Referral reward amount")
    is_reward: bool | None = Field(default=None, description="Whether there is a reward")
    is_company_reward: bool | None = Field(default=None, description="Whether it's a company reward")
    company_referrals_reward: str | None = Field(default=None, description="Company referral reward")
    url_detected_page: str | None = Field(default=None, description="Detected page URL")
    picture_url: str | None = Field(default=None, description="Company picture URL")
    time_updated: str | None = Field(default=None, description="Last update timestamp")
    company_name: str | None = Field(default=None, description="Company name")
    is_internal: bool | None = Field(default=None, description="Whether it's an internal position")
    linkedin_job_posting_id: str | None = Field(default=None, description="LinkedIn job posting ID")
    workplace_type: str | None = Field(default=None, description="Workplace type (remote, hybrid, on-site)")
    position_url: str | None = Field(default=None, description="Direct URL to the position")
    details: list[JobDetail] = Field(default=[], description="Job description sections")


class Company(BaseModel):
    """Company credentials for fetching job listings from external APIs."""

    uid: str = Field(description="Unique identifier for the company")
    token: str = Field(description="API token used to fetch jobs from the company's career page")
    extracted_from: str = Field(description="URL or source from which to fetch job listings")
