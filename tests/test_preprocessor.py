"""Tests for job text preprocessing."""

import pytest

from matchai.jobs.preprocessor import (
    extract_details_text,
    extract_job_keywords,
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
