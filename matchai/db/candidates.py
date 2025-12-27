"""Candidate and match results database operations."""

import hashlib
import json
from datetime import UTC, datetime

from matchai.db.connection import get_connection
from matchai.jobs.database import get_job_by_uid
from matchai.schemas.candidate import CandidateProfile
from matchai.schemas.match import MatchResult


def compute_cv_hash(cv_text: str) -> str:
    """Compute SHA256 hash of CV text for deduplication.

    Args:
        cv_text: Raw text extracted from CV.

    Returns:
        Hex-encoded SHA256 hash string.
    """
    return hashlib.sha256(cv_text.encode()).hexdigest()


def save_candidate(cv_hash: str, profile: CandidateProfile, raw_text: str) -> None:
    """Save a candidate profile to the database.

    Only one candidate is stored at a time. This replaces any existing candidate.

    Args:
        cv_hash: SHA256 hash of the CV text.
        profile: Parsed candidate profile.
        raw_text: Original CV text.
    """
    with get_connection() as db:
        cursor = db.cursor()
        ph = db.placeholder
        now = datetime.now(UTC).isoformat()

        # Convert profile to JSON, excluding raw_text since we store it separately
        profile_dict = profile.model_dump()
        profile_dict.pop("raw_text", None)
        profile_json = json.dumps(profile_dict)

        # Delete any existing candidate (only one CV at a time)
        cursor.execute("DELETE FROM candidates")

        # Insert the new candidate
        if db.is_postgres:
            cursor.execute(
                f"""
                INSERT INTO candidates (cv_hash, profile_json, raw_text, created_at, last_used_at)
                VALUES ({ph}, {ph}, {ph}, NOW(), NOW())
                """,
                (cv_hash, profile_json, raw_text),
            )
        else:
            cursor.execute(
                f"""
                INSERT INTO candidates (cv_hash, profile_json, raw_text, created_at, last_used_at)
                VALUES ({ph}, {ph}, {ph}, {ph}, {ph})
                """,
                (cv_hash, profile_json, raw_text, now, now),
            )
        db.commit()


def get_candidate_by_hash(cv_hash: str) -> CandidateProfile | None:
    """Retrieve a candidate profile by CV hash.

    Args:
        cv_hash: SHA256 hash of the CV text.

    Returns:
        CandidateProfile if found, None otherwise.
    """
    with get_connection() as db:
        cursor = db.cursor(dictionary=True)
        ph = db.placeholder
        cursor.execute(
            f"SELECT profile_json, raw_text FROM candidates WHERE cv_hash = {ph}",
            (cv_hash,),
        )
        row = cursor.fetchone()

    if row is None:
        return None

    # Parse profile JSON
    profile_data = row["profile_json"]
    if isinstance(profile_data, str):
        profile_data = json.loads(profile_data)

    # Add raw_text back to profile
    profile_data["raw_text"] = row["raw_text"]

    return CandidateProfile(**profile_data)


def get_candidate() -> tuple[str, CandidateProfile] | None:
    """Retrieve the stored candidate from the database.

    Only one candidate is stored at a time.

    Returns:
        Tuple of (cv_hash, CandidateProfile) if exists, None otherwise.
    """
    with get_connection() as db:
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT cv_hash, profile_json, raw_text FROM candidates LIMIT 1")
        row = cursor.fetchone()

    if row is None:
        return None

    profile_data = row["profile_json"]
    if isinstance(profile_data, str):
        profile_data = json.loads(profile_data)
    profile_data["raw_text"] = row["raw_text"]

    return (row["cv_hash"], CandidateProfile(**profile_data))


def save_match_results(cv_hash: str, results: list[MatchResult]) -> int:
    """Save match results to the database.

    Args:
        cv_hash: SHA256 hash of the candidate's CV.
        results: List of match results to save.

    Returns:
        Number of results saved.
    """
    with get_connection() as db:
        cursor = db.cursor()
        ph = db.placeholder

        saved = 0
        for result in results:
            explanation_json = json.dumps(result.explanation)

            if db.is_postgres:
                cursor.execute(
                    f"""
                    INSERT INTO match_results (
                        cv_hash, job_uid, similarity_score, filter_score, final_score,
                        explanation, missing_skills, interview_tips
                    ) VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                    """,
                    (
                        cv_hash,
                        result.job.uid,
                        result.similarity_score,
                        result.filter_score,
                        result.final_score,
                        explanation_json,
                        result.missing_skills,
                        result.interview_tips,
                    ),
                )
            else:
                # SQLite doesn't support arrays, store as JSON
                cursor.execute(
                    f"""
                    INSERT INTO match_results (
                        cv_hash, job_uid, similarity_score, filter_score, final_score,
                        explanation, missing_skills, interview_tips
                    ) VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                    """,
                    (
                        cv_hash,
                        result.job.uid,
                        result.similarity_score,
                        result.filter_score,
                        result.final_score,
                        explanation_json,
                        json.dumps(result.missing_skills),
                        json.dumps(result.interview_tips),
                    ),
                )
            saved += 1

        db.commit()

    return saved


def get_excluded_job_uids(cv_hash: str, max_views: int) -> set[str]:
    """Get job UIDs that have been shown to a candidate too many times.

    Args:
        cv_hash: SHA256 hash of the candidate's CV.
        max_views: Maximum number of times a job can be shown.

    Returns:
        Set of job UIDs to exclude from future matches.
    """
    with get_connection() as db:
        cursor = db.cursor(dictionary=True)
        ph = db.placeholder
        cursor.execute(
            f"""
            SELECT job_uid
            FROM match_results
            WHERE cv_hash = {ph}
            GROUP BY job_uid
            HAVING COUNT(*) >= {ph}
            """,
            (cv_hash, max_views),
        )
        rows = cursor.fetchall()

    return {row["job_uid"] for row in rows}


def get_match_results(cv_hash: str | None = None, limit: int = 10) -> list[MatchResult]:
    """Retrieve match results from the database.

    Args:
        cv_hash: Optional filter by CV hash. If None, returns latest results.
        limit: Maximum number of results to return.

    Returns:
        List of MatchResult objects.
    """
    with get_connection() as db:
        cursor = db.cursor(dictionary=True)
        ph = db.placeholder

        if cv_hash:
            cursor.execute(
                f"""
                SELECT * FROM match_results
                WHERE cv_hash = {ph}
                ORDER BY final_score DESC
                LIMIT {ph}
                """,
                (cv_hash, limit),
            )
        else:
            cursor.execute(
                f"""
                SELECT * FROM match_results
                ORDER BY created_at DESC, final_score DESC
                LIMIT {ph}
                """,
                (limit,),
            )

        rows = cursor.fetchall()

    results = []
    for row in rows:
        job = get_job_by_uid(row["job_uid"])
        if job is None:
            continue

        # Parse JSON fields
        explanation = row["explanation"]
        if isinstance(explanation, str):
            explanation = json.loads(explanation)

        missing_skills = row["missing_skills"]
        if isinstance(missing_skills, str):
            missing_skills = json.loads(missing_skills)
        elif missing_skills is None:
            missing_skills = []

        interview_tips = row["interview_tips"]
        if isinstance(interview_tips, str):
            interview_tips = json.loads(interview_tips)
        elif interview_tips is None:
            interview_tips = []

        results.append(
            MatchResult(
                job=job,
                similarity_score=row["similarity_score"],
                filter_score=row["filter_score"],
                final_score=row["final_score"],
                explanation=explanation,
                missing_skills=missing_skills,
                interview_tips=interview_tips,
            )
        )

    return results
