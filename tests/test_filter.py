"""Tests for deterministic job filters."""

from matchai.matching.filter import (
    apply_filters,
    filter_by_location,
    filter_by_seniority,
    filter_by_skills,
    filter_by_view_count,
)
from tests.test_utils import make_test_candidate, make_test_job


class TestFilterBySkills:
    def test_matches_skill_in_job_details(self):
        candidate = make_test_candidate(skills=["python", "sql"])
        jobs = [
            make_test_job("1", "Developer", "We need Python and SQL experience"),
            make_test_job("2", "Designer", "Figma and design skills required"),
        ]

        results = filter_by_skills(jobs, candidate)

        assert len(results) == 1
        assert results[0][0].uid == "1"
        assert results[0][1] > 0

    def test_matches_skill_in_job_title(self):
        candidate = make_test_candidate(skills=["python"])
        jobs = [make_test_job("1", "Python Developer")]

        results = filter_by_skills(jobs, candidate)

        assert len(results) == 1

    def test_fuzzy_matching(self):
        candidate = make_test_candidate(skills=["javascript"])
        jobs = [make_test_job("1", "Developer", "JavaScript/TypeScript experience")]

        results = filter_by_skills(jobs, candidate)

        assert len(results) == 1

    def test_no_skills_returns_all_jobs(self):
        candidate = make_test_candidate(skills=[], tools=[])
        jobs = [make_test_job("1", "Dev"), make_test_job("2", "Designer")]

        results = filter_by_skills(jobs, candidate)

        assert len(results) == len(jobs)

    def test_no_match_returns_empty(self):
        candidate = make_test_candidate(skills=["rust", "go"])
        jobs = [make_test_job("1", "Developer", "Python and Java required")]

        results = filter_by_skills(jobs, candidate)

        assert len(results) == 0

    def test_combines_skills_and_tools(self):
        candidate = make_test_candidate(skills=["python"], tools=["django"])
        jobs = [make_test_job("1", "Developer", "Django framework experience")]

        results = filter_by_skills(jobs, candidate)

        assert len(results) == 1


class TestFilterBySeniority:
    def test_matches_same_level(self):
        candidate = make_test_candidate(seniority="senior")
        jobs = [make_test_job("1", "Senior Developer")]

        results = filter_by_seniority(jobs, candidate)

        assert len(results) == 1

    def test_matches_one_level_above(self):
        candidate = make_test_candidate(seniority="senior")
        jobs = [make_test_job("1", "Lead Developer")]

        results = filter_by_seniority(jobs, candidate)

        assert len(results) == 1

    def test_matches_one_level_below(self):
        candidate = make_test_candidate(seniority="senior")
        jobs = [make_test_job("1", "Mid-level Developer")]

        results = filter_by_seniority(jobs, candidate)

        assert len(results) == 1

    def test_rejects_two_levels_above(self):
        candidate = make_test_candidate(seniority="junior")
        jobs = [make_test_job("1", "Senior Developer")]

        results = filter_by_seniority(jobs, candidate)

        assert len(results) == 0

    def test_includes_jobs_without_level(self):
        candidate = make_test_candidate(seniority="senior")
        jobs = [make_test_job("1", "Software Engineer")]

        results = filter_by_seniority(jobs, candidate)

        assert len(results) == 1

    def test_checks_experience_level_field(self):
        candidate = make_test_candidate(seniority="junior")
        jobs = [make_test_job("1", "Developer", experience_level="Senior")]

        results = filter_by_seniority(jobs, candidate)

        assert len(results) == 0


class TestFilterByLocation:
    def test_no_location_returns_all(self):
        jobs = [
            make_test_job("1", "Dev", location="Tel Aviv"),
            make_test_job("2", "Dev", location="New York"),
        ]

        results = filter_by_location(jobs, None)

        assert len(results) == len(jobs)

    def test_matches_location(self):
        jobs = [
            make_test_job("1", "Dev", location="Tel Aviv"),
            make_test_job("2", "Dev", location="New York"),
        ]

        results = filter_by_location(jobs, "Tel Aviv")

        assert len(results) == 1
        assert results[0].uid == "1"

    def test_includes_remote_jobs(self):
        jobs = [
            make_test_job("1", "Dev", location="Remote"),
            make_test_job("2", "Dev", location="New York"),
        ]

        results = filter_by_location(jobs, "Tel Aviv")

        assert len(results) == 1
        assert results[0].uid == "1"

    def test_includes_remote_workplace_type(self):
        jobs = [make_test_job("1", "Dev", location="USA", workplace_type="Remote")]

        results = filter_by_location(jobs, "Tel Aviv")

        assert len(results) == 1

    def test_case_insensitive(self):
        jobs = [make_test_job("1", "Dev", location="TEL AVIV")]

        results = filter_by_location(jobs, "tel aviv")

        assert len(results) == 1


class TestApplyFilters:
    def test_combines_all_filters(self):
        candidate = make_test_candidate(skills=["python"], seniority="senior")
        jobs = [
            make_test_job("1", "Senior Python Developer", "Python skills", location="Tel Aviv"),
            make_test_job("2", "Junior Java Developer", "Java skills", location="Tel Aviv"),
            make_test_job("3", "Senior Python Developer", "Python skills", location="New York"),
        ]

        results = apply_filters(jobs, candidate, location="Tel Aviv")

        assert len(results) == 1
        assert results[0][0].uid == "1"

    def test_returns_scores(self):
        candidate = make_test_candidate(skills=["python", "django"])
        jobs = [make_test_job("1", "Developer", "Python and Django experience")]

        results = apply_filters(jobs, candidate)

        assert len(results) == 1
        assert results[0][1] > 0


class TestFilterByViewCount:
    def test_no_cv_hash_returns_all_jobs(self):
        """When cv_hash is None, no filtering should occur."""
        jobs = [make_test_job("1", "Dev"), make_test_job("2", "Dev")]

        results = filter_by_view_count(jobs, cv_hash=None)

        assert len(results) == len(jobs)

    def test_zero_max_views_disables_filter(self):
        """When max_views is 0, the filter is disabled."""
        jobs = [make_test_job("1", "Dev")]

        # Even with cv_hash, max_views=0 should return all jobs
        results = filter_by_view_count(jobs, cv_hash="test_hash", max_views=0)

        assert len(results) == 1

    def test_excludes_jobs_over_threshold(self, temp_db):
        """Jobs shown >= max_views times should be excluded."""
        from matchai.db.connection import get_connection

        cv_hash = "test_hash_123"
        jobs = [make_test_job("1", "Dev"), make_test_job("2", "Dev")]

        # Insert 3 match_results for job "1" (should be excluded at max_views=3)
        with get_connection() as db:
            cursor = db.cursor()
            ph = db.placeholder
            for _ in range(3):
                cursor.execute(
                    f"""
                    INSERT INTO match_results
                    (cv_hash, job_uid, similarity_score, filter_score, final_score, explanation)
                    VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                    """,
                    (cv_hash, "1", 0.8, 0.7, 0.75, "[]"),
                )
            # Insert 2 match_results for job "2" (should NOT be excluded at max_views=3)
            for _ in range(2):
                cursor.execute(
                    f"""
                    INSERT INTO match_results
                    (cv_hash, job_uid, similarity_score, filter_score, final_score, explanation)
                    VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                    """,
                    (cv_hash, "2", 0.8, 0.7, 0.75, "[]"),
                )
            db.commit()

        results = filter_by_view_count(jobs, cv_hash=cv_hash, max_views=3)

        assert len(results) == 1
        assert results[0].uid == "2"

    def test_no_excluded_jobs_returns_all(self, temp_db):
        """When no jobs have reached the view threshold, all should be returned."""
        cv_hash = "test_hash_new"
        jobs = [make_test_job("1", "Dev"), make_test_job("2", "Dev")]

        # No match_results in database for this cv_hash
        results = filter_by_view_count(jobs, cv_hash=cv_hash, max_views=3)

        assert len(results) == 2

    def test_different_cv_hash_not_affected(self, temp_db):
        """View counts for one cv_hash shouldn't affect another."""
        from matchai.db.connection import get_connection

        cv_hash_1 = "user_1_hash"
        cv_hash_2 = "user_2_hash"
        jobs = [make_test_job("1", "Dev")]

        # Insert 5 views for user 1
        with get_connection() as db:
            cursor = db.cursor()
            ph = db.placeholder
            for _ in range(5):
                cursor.execute(
                    f"""
                    INSERT INTO match_results
                    (cv_hash, job_uid, similarity_score, filter_score, final_score, explanation)
                    VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                    """,
                    (cv_hash_1, "1", 0.8, 0.7, 0.75, "[]"),
                )
            db.commit()

        # User 1 should have job excluded
        results_1 = filter_by_view_count(jobs, cv_hash=cv_hash_1, max_views=3)
        assert len(results_1) == 0

        # User 2 should still see the job (no views for them)
        results_2 = filter_by_view_count(jobs, cv_hash=cv_hash_2, max_views=3)
        assert len(results_2) == 1
