"""Match service for running CV-to-job matching pipeline.

This service handles:
- CV parsing with caching (skip re-parsing if already in database)
- Full matching pipeline (filter, rank, explain)
- Running the scheduled match and saving results
"""

import logging
from typing import Any

from matchai.config import DEFAULT_TOP_N
from matchai.cv.parser import parse_cv
from matchai.db.candidates import (
    compute_cv_hash,
    get_candidate,
    get_candidate_by_hash,
    save_candidate,
    save_match_results,
)
from matchai.explainer.generator import (
    find_missing_skills,
    generate_explanation,
    refine_skills_and_tips,
)
from matchai.jobs.database import get_jobs
from matchai.matching.filter import apply_filters
from matchai.matching.ranker import rank_jobs
from matchai.schemas.candidate import CandidateProfile
from matchai.schemas.match import MatchResult

logger = logging.getLogger(__name__)


def get_or_parse_candidate(cv_text: str) -> tuple[CandidateProfile, bool]:
    """Get candidate from cache or parse CV with LLM.

    Checks database cache first using SHA256 hash of CV text.
    If not found, parses CV with LLM and caches the result.

    Args:
        cv_text: Raw text extracted from CV.

    Returns:
        Tuple of (CandidateProfile, was_cached) where was_cached is True
        if the profile was retrieved from cache.
    """
    cv_hash = compute_cv_hash(cv_text)

    # Check cache first
    cached_profile = get_candidate_by_hash(cv_hash)
    if cached_profile is not None:
        logger.info(f"Found cached candidate profile (hash: {cv_hash[:16]}...)")
        return cached_profile, True

    # Parse with LLM
    logger.info("Parsing CV with LLM...")
    profile = parse_cv(cv_text)

    # Save to cache (replaces any existing candidate)
    save_candidate(cv_hash, profile, cv_text)
    logger.info(f"Saved candidate profile to database (hash: {cv_hash[:16]}...)")

    return profile, False


def match_candidate(
    candidate: CandidateProfile,
    location: str | None = None,
    top_n: int = DEFAULT_TOP_N,
    cv_hash: str | None = None,
    max_views: int | None = None,
) -> list[MatchResult]:
    """Run full matching pipeline for a candidate.

    Pipeline:
    1. Load jobs from database (with optional location filter)
    2. Apply view count filter (exclude jobs seen >= max_views times)
    3. Apply skill and seniority filters
    4. Rank by semantic similarity
    5. Generate explanations for top matches
    6. Find missing skills and generate interview tips

    Args:
        candidate: Parsed candidate profile.
        location: Optional location filter.
        top_n: Number of top matches to return.
        cv_hash: Candidate's CV hash for view count filtering.
        max_views: Maximum views before exclusion (None uses default, 0 disables).

    Returns:
        List of top MatchResult objects with explanations.
    """
    # Load jobs with optional location filter
    logger.info("Loading jobs from database...")
    jobs = get_jobs(location=location)

    if not jobs:
        logger.warning("No jobs found matching filters")
        return []

    logger.info(f"Found {len(jobs)} jobs to match against")

    # Apply filters (view count filter runs first if cv_hash provided)
    logger.info("Applying filters...")
    filtered_jobs = apply_filters(
        jobs=jobs,
        candidate=candidate,
        location=None,  # Already filtered at DB level
        cv_hash=cv_hash,
        max_views=max_views,
    )

    if not filtered_jobs:
        logger.warning("No jobs passed filters")
        return []

    logger.info(f"{len(filtered_jobs)} jobs passed filters")

    # Rank jobs
    logger.info("Ranking jobs by similarity...")
    ranked_results = rank_jobs(filtered_jobs=filtered_jobs, candidate=candidate)

    # Get top N
    top_matches = ranked_results[:top_n]
    logger.info(f"Selected top {len(top_matches)} matches")

    # Generate explanations
    logger.info("Generating match explanations...")
    for match in top_matches:
        match.explanation = generate_explanation(
            job=match.job,
            candidate=candidate,
            similarity_score=match.similarity_score,
            filter_score=match.filter_score,
        )
        match.missing_skills = find_missing_skills(
            job=match.job,
            candidate=candidate,
        )

    # Refine skills and generate interview tips
    logger.info("Refining skills and generating interview tips...")
    for match in top_matches:
        if match.missing_skills:
            refined_skills, interview_tips = refine_skills_and_tips(
                candidate=candidate,
                job=match.job,
                raw_missing_skills=match.missing_skills,
            )
            match.missing_skills = refined_skills
            match.interview_tips = interview_tips

    return top_matches


def run_scheduled_matching() -> tuple[dict[str, Any], list[MatchResult]]:
    """Run matching for the stored candidate and save results.

    Called by the scheduled runner (Cloud Run Job). Gets the pre-uploaded CV
    from the database, runs matching against all jobs, and saves results.

    Returns:
        Tuple of (stats dict, list of MatchResult objects).
    """
    stats = {
        "candidate_found": False,
        "matches_generated": 0,
        "results_saved": 0,
    }
    matches: list[MatchResult] = []

    # Get the stored candidate
    result = get_candidate()
    if result is None:
        logger.warning("No candidate found in database. Upload CV first with 'matchai upload-cv'")
        return stats, matches

    cv_hash, candidate = result
    stats["candidate_found"] = True
    logger.info(f"Found stored candidate (hash: {cv_hash[:16]}...)")

    # Run matching (pass cv_hash for view count filtering)
    matches = match_candidate(candidate, cv_hash=cv_hash)
    stats["matches_generated"] = len(matches)

    # Save results
    if matches:
        saved = save_match_results(cv_hash, matches)
        stats["results_saved"] = saved
        logger.info(f"Saved {saved} match results to database")

    logger.info(f"Matching complete: {stats}")
    return stats, matches
