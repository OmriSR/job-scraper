"""Deterministic filters for job matching."""

from rapidfuzz import fuzz

from matchai.config import MAX_JOB_VIEWS, SKILL_MATCH_THRESHOLD
from matchai.db.candidates import get_excluded_job_uids
from matchai.jobs.preprocessor import extract_details_text
from matchai.schemas.candidate import CandidateProfile, SeniorityLevel
from matchai.schemas.job import Job


def _extract_seniority_from_text(text: str) -> SeniorityLevel | None:
    """Extract seniority level from job text."""
    text_lower = text.lower()
    for level in SeniorityLevel:
        if level.name.lower() in text_lower:
            return level
    return None


def filter_by_skills(
    jobs: list[Job],
    candidate: CandidateProfile,
    threshold: int = SKILL_MATCH_THRESHOLD,
) -> list[tuple[Job, float]]:
    """Filter jobs by skill match using fuzzy matching.

    Args:
        jobs: List of jobs to filter.
        candidate: Candidate profile with skills.
        threshold: Minimum fuzzy match score (0-100).

    Returns:
        List of (job, score) tuples for jobs with at least one skill match.
    """
    if not candidate.skills and not candidate.tools_frameworks:
        return [(job, 0.0) for job in jobs]

    candidate_skills = set(
        s.lower() for s in candidate.skills + candidate.tools_frameworks
    )

    results = []
    for job in jobs:
        job_text = extract_details_text(job.details).lower()
        job_text += f" {job.name.lower()}"

        matched_skills = 0
        total_score = 0.0

        for skill in candidate_skills:
            best_score = fuzz.partial_ratio(skill, job_text)
            if best_score >= threshold:
                matched_skills += 1
                total_score += best_score

        if matched_skills > 0:
            avg_score = total_score / len(candidate_skills)
            results.append((job, avg_score))

    return results


def filter_by_seniority(
    jobs: list[Job],
    candidate: CandidateProfile,
) -> list[Job]:
    """Filter jobs by seniority level compatibility.

    Allows jobs at same level or one level above/below candidate.

    Args:
        jobs: List of jobs to filter.
        candidate: Candidate profile with seniority.

    Returns:
        List of jobs matching seniority criteria.
    """
    candidate_level = candidate.seniority

    min_value = max(SeniorityLevel.JUNIOR.value, candidate_level.value - 1)
    max_value = min(SeniorityLevel.STAFF.value, candidate_level.value + 1)
    min_level = SeniorityLevel(min_value)
    max_level = SeniorityLevel(max_value)

    results = []
    for job in jobs:
        job_text = f"{job.name} {job.experience_level or ''}"
        job_level = _extract_seniority_from_text(job_text)

        if job_level is None or min_level <= job_level <= max_level:
            results.append(job)

    return results


def filter_by_view_count(
    jobs: list[Job],
    cv_hash: str | None,
    max_views: int = MAX_JOB_VIEWS,
) -> list[Job]:
    """Filter out jobs that have been shown too many times to this candidate.

    Args:
        jobs: List of jobs to filter.
        cv_hash: Candidate's CV hash (None skips this filter).
        max_views: Maximum times a job can appear in results (0 disables filter).

    Returns:
        List of jobs that haven't exceeded the view limit.
    """
    if cv_hash is None or max_views <= 0:
        return jobs

    excluded_uids = get_excluded_job_uids(cv_hash, max_views)

    if not excluded_uids:
        return jobs

    return [job for job in jobs if job.uid not in excluded_uids]


def filter_by_location(
    jobs: list[Job],
    location: str | None,
) -> list[Job]:
    """Filter jobs by location.

    Args:
        jobs: List of jobs to filter.
        location: Desired location (None means no filter).

    Returns:
        List of jobs matching location criteria.
    """
    if not location:
        return jobs

    location_lower = location.lower()
    results = []

    for job in jobs:
        job_location = (job.location or "").lower()
        workplace = (job.workplace_type or "").lower()

        if "remote" in job_location or "remote" in workplace:
            results.append(job)
        elif location_lower in job_location:
            results.append(job)
        elif "hybrid" in workplace and location_lower in job_location:
            results.append(job)

    return results


def apply_filters(
    jobs: list[Job],
    candidate: CandidateProfile,
    location: str | None = None,
    skill_threshold: int = SKILL_MATCH_THRESHOLD,
    cv_hash: str | None = None,
    max_views: int | None = None,
) -> list[tuple[Job, float]]:
    """Apply all filters and return matching jobs with scores.

    Args:
        jobs: List of jobs to filter.
        candidate: Candidate profile.
        location: Optional location filter.
        skill_threshold: Minimum skill match threshold.
        cv_hash: Candidate's CV hash for view count filtering.
        max_views: Maximum views before exclusion (None uses default, 0 disables).

    Returns:
        List of (job, skill_score) tuples for jobs passing all filters.
    """
    # Apply view count filter first (most efficient - just UID lookup)
    if max_views is None:
        max_views = MAX_JOB_VIEWS
    jobs = filter_by_view_count(jobs, cv_hash, max_views)

    # Then apply other filters
    jobs = filter_by_location(jobs, location)
    jobs = filter_by_seniority(jobs, candidate)
    results = filter_by_skills(jobs, candidate, skill_threshold)

    return results
