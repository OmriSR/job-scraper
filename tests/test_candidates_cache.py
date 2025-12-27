"""Tests for candidate caching functionality."""

import json

from matchai.db.candidates import (
    get_all_cached_job_uids,
    get_eligible_cached_results,
    upsert_match_results,
)
from matchai.db.connection import get_connection
from matchai.schemas.match import MatchResult
from tests.test_utils import make_test_job


class TestGetAllCachedJobUids:
    def test_returns_empty_for_no_cache(self, temp_db):
        """When no cached results exist, returns empty set."""
        result = get_all_cached_job_uids("nonexistent_hash")
        assert result == set()

    def test_returns_all_job_uids(self, temp_db):
        """Returns all job UIDs regardless of view_count."""
        cv_hash = "test_hash"

        with get_connection() as db:
            cursor = db.cursor()
            ph = db.placeholder
            # Insert jobs with different view counts
            cursor.execute(
                f"""
                INSERT INTO match_results
                (cv_hash, job_uid, similarity_score, filter_score, final_score,
                 explanation, view_count)
                VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                """,
                (cv_hash, "job1", 0.8, 0.7, 0.75, "[]", 1),
            )
            cursor.execute(
                f"""
                INSERT INTO match_results
                (cv_hash, job_uid, similarity_score, filter_score, final_score,
                 explanation, view_count)
                VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                """,
                (cv_hash, "job2", 0.8, 0.7, 0.75, "[]", 5),
            )
            db.commit()

        result = get_all_cached_job_uids(cv_hash)
        assert result == {"job1", "job2"}


class TestGetEligibleCachedResults:
    def test_returns_empty_for_no_cache(self, temp_db):
        """When no cached results exist, returns empty dict."""
        result = get_eligible_cached_results("nonexistent_hash", max_views=3)
        assert result == {}

    def test_excludes_jobs_over_threshold(self, temp_db):
        """Jobs with view_count >= max_views are excluded."""
        cv_hash = "test_hash"

        with get_connection() as db:
            cursor = db.cursor()
            ph = db.placeholder
            # Job 1: view_count=2 (should be included)
            cursor.execute(
                f"""
                INSERT INTO match_results
                (cv_hash, job_uid, similarity_score, filter_score, final_score,
                 explanation, view_count)
                VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                """,
                (cv_hash, "job1", 0.8, 0.7, 0.75, "[]", 2),
            )
            # Job 2: view_count=3 (should be excluded at max_views=3)
            cursor.execute(
                f"""
                INSERT INTO match_results
                (cv_hash, job_uid, similarity_score, filter_score, final_score,
                 explanation, view_count)
                VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                """,
                (cv_hash, "job2", 0.8, 0.7, 0.75, "[]", 3),
            )
            db.commit()

        result = get_eligible_cached_results(cv_hash, max_views=3)
        assert "job1" in result
        assert "job2" not in result

    def test_returns_all_with_zero_max_views(self, temp_db):
        """When max_views=0, returns all cached results."""
        cv_hash = "test_hash"

        with get_connection() as db:
            cursor = db.cursor()
            ph = db.placeholder
            cursor.execute(
                f"""
                INSERT INTO match_results
                (cv_hash, job_uid, similarity_score, filter_score, final_score,
                 explanation, view_count)
                VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                """,
                (cv_hash, "job1", 0.8, 0.7, 0.75, "[]", 100),
            )
            db.commit()

        result = get_eligible_cached_results(cv_hash, max_views=0)
        assert "job1" in result

    def test_returns_cached_fields(self, temp_db):
        """Cached results include all expected fields."""
        cv_hash = "test_hash"
        expected_explanation = ["Point 1", "Point 2"]
        expected_missing = ["Skill1", "Skill2"]
        expected_tips = ["Tip1"]

        with get_connection() as db:
            cursor = db.cursor()
            ph = db.placeholder
            cursor.execute(
                f"""
                INSERT INTO match_results
                (cv_hash, job_uid, similarity_score, filter_score, final_score,
                 explanation, missing_skills, interview_tips, view_count)
                VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                """,
                (
                    cv_hash, "job1", 0.85, 0.72, 0.78,
                    json.dumps(expected_explanation),
                    json.dumps(expected_missing),
                    json.dumps(expected_tips),
                    1,
                ),
            )
            db.commit()

        result = get_eligible_cached_results(cv_hash, max_views=3)
        assert "job1" in result
        data = result["job1"]
        assert data["similarity_score"] == 0.85
        assert data["filter_score"] == 0.72
        assert data["final_score"] == 0.78
        assert data["explanation"] == expected_explanation
        assert data["missing_skills"] == expected_missing
        assert data["interview_tips"] == expected_tips
        assert data["view_count"] == 1


class TestUpsertMatchResults:
    def test_inserts_new_result(self, temp_db):
        """New jobs are inserted with view_count=1."""
        cv_hash = "test_hash"
        job = make_test_job("job1", "Test Job")
        result = MatchResult(
            job=job,
            similarity_score=0.85,
            filter_score=0.72,
            final_score=0.78,
            explanation=["Point 1"],
            missing_skills=["Skill1"],
            interview_tips=["Tip1"],
        )

        count = upsert_match_results(cv_hash, [result])
        assert count == 1

        # Verify in database
        with get_connection() as db:
            cursor = db.cursor(dictionary=True)
            ph = db.placeholder
            cursor.execute(
                f"SELECT * FROM match_results WHERE cv_hash = {ph} AND job_uid = {ph}",
                (cv_hash, "job1"),
            )
            row = cursor.fetchone()

        assert row is not None
        assert row["view_count"] == 1
        assert row["similarity_score"] == 0.85

    def test_increments_view_count_on_existing(self, temp_db):
        """Existing jobs have their view_count incremented."""
        cv_hash = "test_hash"
        job = make_test_job("job1", "Test Job")

        # Insert initial result
        with get_connection() as db:
            cursor = db.cursor()
            ph = db.placeholder
            cursor.execute(
                f"""
                INSERT INTO match_results
                (cv_hash, job_uid, similarity_score, filter_score, final_score,
                 explanation, view_count)
                VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                """,
                (cv_hash, "job1", 0.8, 0.7, 0.75, "[]", 2),
            )
            db.commit()

        # Upsert the same job
        result = MatchResult(
            job=job,
            similarity_score=0.85,
            filter_score=0.72,
            final_score=0.78,
            explanation=["Point 1"],
            missing_skills=["Skill1"],
            interview_tips=["Tip1"],
        )
        count = upsert_match_results(cv_hash, [result])
        assert count == 1

        # Verify view_count was incremented
        with get_connection() as db:
            cursor = db.cursor(dictionary=True)
            ph = db.placeholder
            cursor.execute(
                f"SELECT view_count FROM match_results WHERE cv_hash = {ph} AND job_uid = {ph}",
                (cv_hash, "job1"),
            )
            row = cursor.fetchone()

        assert row["view_count"] == 3

    def test_handles_multiple_results(self, temp_db):
        """Multiple results can be upserted at once."""
        cv_hash = "test_hash"
        results = [
            MatchResult(
                job=make_test_job("job1", "Job 1"),
                similarity_score=0.8,
                filter_score=0.7,
                final_score=0.75,
                explanation=["P1"],
                missing_skills=[],
                interview_tips=[],
            ),
            MatchResult(
                job=make_test_job("job2", "Job 2"),
                similarity_score=0.9,
                filter_score=0.8,
                final_score=0.85,
                explanation=["P2"],
                missing_skills=[],
                interview_tips=[],
            ),
        ]

        count = upsert_match_results(cv_hash, results)
        assert count == 2

        # Verify both in database
        all_cached = get_all_cached_job_uids(cv_hash)
        assert all_cached == {"job1", "job2"}
