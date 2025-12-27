"""Match service for running CV-to-job matching pipeline.

This service handles:
- CV parsing with caching (skip re-parsing if already in database)
- Full matching pipeline with result caching (filter, rank, explain)
- Running the scheduled match and saving results

Key optimization: Job scores and explanations are computed ONCE and cached forever.
Subsequent runs reuse cached results and only compute for truly new jobs.
"""

import logging
from typing import Any

from matchai.config import DEFAULT_TOP_N, MAX_JOB_VIEWS
from matchai.cv.parser import parse_cv
from matchai.db.candidates import (
    compute_cv_hash,
    get_all_cached_job_uids,
    get_candidate,
    get_candidate_by_hash,
    get_eligible_cached_results,
    save_candidate,
    upsert_match_results,
)
from matchai.explainer.generator import (
    find_missing_skills,
    generate_explanation,
    refine_skills_and_tips,
)
from matchai.jobs.database import get_jobs, get_jobs_by_uids
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
    """Run full matching pipeline for a candidate with result caching.

    Optimized pipeline:
    1. Get ALL cached job UIDs (to identify truly new jobs)
    2. Get eligible cached results (view_count < max_views)
    3. Find TRULY NEW jobs (never computed before)
    4. Compute scores + explanations ONLY for new jobs
    5. Merge cached + new results, sort by final_score
    6. Upsert results (insert new, increment view_count for existing)

    Key optimization: Job scores and LLM explanations are computed ONCE and cached
    forever. Subsequent runs reuse cached results and only compute for truly new jobs.

    Args:
        candidate: Parsed candidate profile.
        location: Optional location filter.
        top_n: Number of top matches to return.
        cv_hash: Candidate's CV hash for caching (required for cache optimization).
        max_views: Maximum views before exclusion (None uses default, 0 disables).

    Returns:
        List of top MatchResult objects with explanations.
    """
    if max_views is None:
        max_views = MAX_JOB_VIEWS

    # Load all jobs with optional location filter
    logger.info("Loading jobs from database...")
    all_jobs = get_jobs(location=location)

    if not all_jobs:
        logger.warning("No jobs found matching filters")
        return []

    logger.info(f"Found {len(all_jobs)} jobs in database")
    all_job_uids = {job.uid for job in all_jobs}

    # If no cv_hash, fall back to computing everything (no caching)
    if cv_hash is None:
        logger.info("No cv_hash provided, computing all matches without caching")
        return _compute_matches_for_jobs(all_jobs, candidate, top_n)

    # Step 1: Get ALL cached job UIDs (to know what's already computed)
    all_cached_uids = get_all_cached_job_uids(cv_hash)
    logger.info(f"Found {len(all_cached_uids)} cached job results")

    # Step 2: Get eligible cached results (view_count < max_views)
    eligible_cached = get_eligible_cached_results(cv_hash, max_views)
    eligible_cached_uids = set(eligible_cached.keys())
    logger.info(
        f"Found {len(eligible_cached_uids)} eligible cached results (view_count < {max_views})"
    )

    # Step 3: Find TRULY NEW jobs (never computed before)
    new_job_uids = all_job_uids - all_cached_uids
    logger.info(f"Found {len(new_job_uids)} truly new jobs to compute")

    # Step 4: Compute scores + explanations ONLY for truly new jobs
    new_results: list[MatchResult] = []
    if new_job_uids:
        new_jobs = [job for job in all_jobs if job.uid in new_job_uids]
        new_results = _compute_matches_for_jobs(new_jobs, candidate, top_n=None)
        logger.info(f"Computed {len(new_results)} new match results")

    # Step 5: Convert eligible cached results to MatchResult objects
    cached_results = _build_match_results_from_cache(eligible_cached)
    logger.info(f"Loaded {len(cached_results)} cached match results")

    # Step 6: Merge, sort by final_score, take top-N
    all_results = cached_results + new_results
    all_results.sort(key=lambda r: r.final_score, reverse=True)
    top_matches = all_results[:top_n]
    logger.info(f"Selected top {len(top_matches)} matches")

    # Step 7: Upsert results (insert new with view_count=1, increment existing)
    if top_matches:
        processed = upsert_match_results(cv_hash, top_matches)
        logger.info(f"Upserted {processed} match results")

    return top_matches


def _compute_matches_for_jobs(
    jobs: list,
    candidate: CandidateProfile,
    top_n: int | None = None,
) -> list[MatchResult]:
    """Compute match scores and explanations for a list of jobs.

    This is the expensive operation that involves:
    - Skill/seniority filtering
    - Embedding similarity ranking
    - LLM explanation generation

    Args:
        jobs: List of jobs to match against.
        candidate: Candidate profile.
        top_n: Optional limit (None means return all matches).

    Returns:
        List of MatchResult objects with scores and explanations.
    """
    if not jobs:
        return []

    # Apply skill and seniority filters
    # (View count filtering is handled at cache level in match_candidate)
    logger.info("Applying filters to new jobs...")
    filtered_jobs = apply_filters(
        jobs=jobs,
        candidate=candidate,
        location=None,  # Already filtered at DB level
    )

    if not filtered_jobs:
        logger.info("No new jobs passed skill/seniority filters")
        return []

    logger.info(f"{len(filtered_jobs)} new jobs passed filters")

    # Rank jobs by semantic similarity
    logger.info("Ranking new jobs by similarity...")
    ranked_results = rank_jobs(filtered_jobs=filtered_jobs, candidate=candidate)

    # Limit if top_n is specified
    if top_n is not None:
        ranked_results = ranked_results[:top_n]

    # Generate explanations for all matches (will be cached)
    logger.info(f"Generating explanations for {len(ranked_results)} matches...")
    for match in ranked_results:
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
    for match in ranked_results:
        if match.missing_skills:
            refined_skills, interview_tips = refine_skills_and_tips(
                candidate=candidate,
                job=match.job,
                raw_missing_skills=match.missing_skills,
            )
            match.missing_skills = refined_skills
            match.interview_tips = interview_tips

    return ranked_results


def _build_match_results_from_cache(
    cached: dict[str, dict],
) -> list[MatchResult]:
    """Build MatchResult objects from cached data.

    Args:
        cached: Dict mapping job_uid to cached result data.

    Returns:
        List of MatchResult objects.
    """
    if not cached:
        return []

    # Fetch all jobs at once for efficiency
    job_uids = list(cached.keys())
    jobs = get_jobs_by_uids(job_uids)
    jobs_by_uid = {job.uid: job for job in jobs}

    results = []
    for job_uid, data in cached.items():
        job = jobs_by_uid.get(job_uid)
        if job is None:
            logger.warning(f"Job {job_uid} not found in database, skipping cached result")
            continue

        results.append(
            MatchResult(
                job=job,
                similarity_score=data["similarity_score"],
                filter_score=data["filter_score"],
                final_score=data["final_score"],
                explanation=data["explanation"],
                missing_skills=data["missing_skills"],
                interview_tips=data["interview_tips"],
            )
        )

    return results


def run_scheduled_matching() -> tuple[dict[str, Any], list[MatchResult]]:
    """Run matching for the stored candidate using cache-first approach.

    Called by the scheduled runner (Cloud Run Job). Gets the pre-uploaded CV
    from the database, runs matching against all jobs with caching, and
    upserts results.

    Key optimization: Only computes scores for truly new jobs. Existing
    matches are retrieved from cache and their view_count is incremented.

    Returns:
        Tuple of (stats dict, list of MatchResult objects).
    """
    stats = {
        "candidate_found": False,
        "matches_generated": 0,
        "cached_results_used": 0,
        "new_jobs_computed": 0,
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

    # Get cached counts before matching (for stats)
    cached_before = len(get_all_cached_job_uids(cv_hash))

    # Run matching with caching (match_candidate now handles upsert internally)
    matches = match_candidate(candidate, cv_hash=cv_hash)
    stats["matches_generated"] = len(matches)

    # Get cached counts after matching (for stats)
    cached_after = len(get_all_cached_job_uids(cv_hash))
    stats["new_jobs_computed"] = cached_after - cached_before
    stats["cached_results_used"] = cached_before

    logger.info(f"Matching complete: {stats}")
    return stats, matches
