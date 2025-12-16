"""Tests for job text preprocessing."""

import pytest

from matchai.jobs.preprocessor import (
    extract_details_text,
    extract_job_keywords,
    extract_job_keywords_batch,
    preprocess_job,
    strip_html,
)
from matchai.schemas.job import Job, JobDetail


class TestStripHtml:
    def test_strips_paragraph_tags(self):
        html = "<p>Hello World</p>"
        assert strip_html(html) == "Hello World"

    def test_strips_list_tags(self):
        html = "<ul><li>Item 1</li><li>Item 2</li></ul>"
        result = strip_html(html)
        assert "Item 1" in result
        assert "Item 2" in result

    def test_handles_empty_string(self):
        assert strip_html("") == ""

    def test_handles_none(self):
        assert strip_html(None) == ""

    def test_handles_nested_tags(self):
        html = "<div><p><strong>Bold text</strong></p></div>"
        assert strip_html(html) == "Bold text"


class TestExtractDetailsText:
    def test_concatenates_sections(self):
        details = [
            JobDetail(name="Description", value="<p>Job description</p>", order=1),
            JobDetail(name="Requirements", value="<p>Python skills</p>", order=2),
        ]
        result = extract_details_text(details)
        assert "Description: Job description" in result
        assert "Requirements: Python skills" in result

    def test_sorts_by_order(self):
        details = [
            JobDetail(name="Second", value="<p>B</p>", order=2),
            JobDetail(name="First", value="<p>A</p>", order=1),
        ]
        result = extract_details_text(details)
        assert result.index("First") < result.index("Second")

    def test_skips_none_values(self):
        details = [
            JobDetail(name="Present", value="<p>Content</p>", order=1),
            JobDetail(name="Missing", value=None, order=2),
        ]
        result = extract_details_text(details)
        assert "Present" in result
        assert "Missing" not in result

    def test_empty_details(self):
        result = extract_details_text([])
        assert result == ""


class TestPreprocessJob:
    def test_preprocesses_job_details(self):
        job = Job(
            uid="test-1",
            name="Developer",
            details=[
                JobDetail(name="Requirements", value="<p>Python programming experience</p>", order=1),
            ],
        )
        result = preprocess_job(job)
        # Should contain lemmatized words, no stopwords
        assert "python" in result.lower()
        assert "programming" in result.lower() or "program" in result.lower()

    def test_empty_job_details(self):
        job = Job(uid="test-2", name="Empty")
        result = preprocess_job(job)
        assert result == ""


class TestExtractJobKeywords:
    def test_extracts_nouns_and_adjectives(self):
        job = Job(
            uid="test-3",
            name="Engineer",
            details=[
                JobDetail(
                    name="Requirements",
                    value="<p>Strong Python experience. Cloud infrastructure knowledge.</p>",
                    order=1,
                ),
            ],
        )
        keywords = extract_job_keywords(job)
        assert len(keywords) > 0
        # Should contain relevant keywords
        assert any("python" in kw.lower() for kw in keywords)

    def test_empty_job_returns_empty_list(self):
        job = Job(uid="test-4", name="Empty")
        keywords = extract_job_keywords(job)
        assert keywords == []


class TestExtractJobKeywordsBatch:
    def test_batch_matches_single_processing(self):
        jobs = [
            Job(
                uid="batch-1",
                name="Developer",
                details=[
                    JobDetail(name="Req", value="<p>Python experience</p>", order=1),
                ],
            ),
            Job(
                uid="batch-2",
                name="Engineer",
                details=[
                    JobDetail(name="Req", value="<p>Cloud infrastructure</p>", order=1),
                ],
            ),
        ]
        batch_results = extract_job_keywords_batch(jobs)
        single_results = [extract_job_keywords(job) for job in jobs]

        assert len(batch_results) == len(single_results)
        for batch_kw, single_kw in zip(batch_results, single_results):
            assert set(batch_kw) == set(single_kw)

    def test_handles_empty_jobs_in_batch(self):
        jobs = [
            Job(uid="batch-3", name="Empty"),
            Job(
                uid="batch-4",
                name="Full",
                details=[
                    JobDetail(name="Req", value="<p>Python skills</p>", order=1),
                ],
            ),
            Job(uid="batch-5", name="Empty"),
        ]
        results = extract_job_keywords_batch(jobs)

        assert len(results) == 3
        assert results[0] == []
        assert len(results[1]) > 0
        assert results[2] == []

    def test_all_empty_jobs(self):
        jobs = [
            Job(uid="empty-1", name="Empty1"),
            Job(uid="empty-2", name="Empty2"),
        ]
        results = extract_job_keywords_batch(jobs)

        assert results == [[], []]

    def test_empty_list(self):
        results = extract_job_keywords_batch([])
        assert results == []
