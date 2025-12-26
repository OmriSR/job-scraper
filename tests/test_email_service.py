"""Tests for email notification service."""

import smtplib
from unittest.mock import MagicMock, patch

import pytest

from matchai.schemas.job import Job
from matchai.schemas.match import MatchResult
from matchai.services.email_service import (
    _build_html_email,
    _validate_email_config,
    send_match_results_email,
)


@pytest.fixture
def sample_job() -> Job:
    """Create a sample Job for testing."""
    return Job(
        uid="job-123",
        name="Senior Python Developer",
        company_name="TechCorp",
        location="Tel Aviv",
        url_active_page="https://example.com/apply",
    )


@pytest.fixture
def sample_match(sample_job: Job) -> MatchResult:
    """Create a sample MatchResult for testing."""
    return MatchResult(
        job=sample_job,
        similarity_score=0.85,
        filter_score=0.90,
        final_score=0.87,
        explanation=["Strong Python experience", "Backend expertise matches"],
        missing_skills=["Kubernetes", "AWS"],
        interview_tips=["Prepare system design examples"],
        apply_url="https://example.com/apply",
    )


class TestValidateEmailConfig:
    """Tests for _validate_email_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes with all config present."""
        with patch.multiple(
            "matchai.services.email_service",
            EMAIL_SENDER="test@gmail.com",
            EMAIL_RECIPIENT="recipient@example.com",
            EMAIL_APP_PASSWORD="secret123",
        ):
            assert _validate_email_config() is True

    def test_missing_sender(self) -> None:
        """Test validation fails when sender is missing."""
        with patch.multiple(
            "matchai.services.email_service",
            EMAIL_SENDER=None,
            EMAIL_RECIPIENT="recipient@example.com",
            EMAIL_APP_PASSWORD="secret123",
        ):
            assert _validate_email_config() is False

    def test_missing_recipient(self) -> None:
        """Test validation fails when recipient is missing."""
        with patch.multiple(
            "matchai.services.email_service",
            EMAIL_SENDER="test@gmail.com",
            EMAIL_RECIPIENT=None,
            EMAIL_APP_PASSWORD="secret123",
        ):
            assert _validate_email_config() is False

    def test_missing_password(self) -> None:
        """Test validation fails when app password is missing."""
        with patch.multiple(
            "matchai.services.email_service",
            EMAIL_SENDER="test@gmail.com",
            EMAIL_RECIPIENT="recipient@example.com",
            EMAIL_APP_PASSWORD=None,
        ):
            assert _validate_email_config() is False


class TestBuildHtmlEmail:
    """Tests for _build_html_email function."""

    def test_builds_html_with_matches(self, sample_match: MatchResult) -> None:
        """Test HTML is built correctly with match data."""
        html = _build_html_email([sample_match])

        assert "Senior Python Developer" in html
        assert "TechCorp" in html
        assert "Tel Aviv" in html
        assert "87%" in html  # final_score as percentage
        assert "Strong Python experience" in html
        assert "Kubernetes" in html
        assert "Prepare system design examples" in html
        assert "https://example.com/apply" in html
        assert "Apply Now" in html

    def test_builds_html_with_multiple_matches(
        self, sample_job: Job, sample_match: MatchResult
    ) -> None:
        """Test HTML is built correctly with multiple matches."""
        job2 = Job(
            uid="job-456",
            name="Backend Engineer",
            company_name="StartupX",
            location="Remote",
        )
        match2 = MatchResult(
            job=job2,
            similarity_score=0.75,
            filter_score=0.80,
            final_score=0.77,
            explanation=["Good backend skills"],
            missing_skills=[],
            interview_tips=[],
            apply_url=None,
        )

        html = _build_html_email([sample_match, match2])

        assert "Senior Python Developer" in html
        assert "Backend Engineer" in html
        assert "#1" in html
        assert "#2" in html
        assert "Found 2 matching positions" in html

    def test_handles_missing_job_fields(self) -> None:
        """Test HTML handles jobs with missing optional fields."""
        job = Job(uid="job-999", name="Developer")
        match = MatchResult(
            job=job,
            similarity_score=0.5,
            filter_score=0.5,
            final_score=0.5,
            explanation=[],
            missing_skills=[],
            interview_tips=[],
            apply_url=None,
        )

        html = _build_html_email([match])

        assert "Developer" in html
        assert "Unknown Company" in html
        assert "Not specified" in html
        assert "Apply Now" not in html  # No URL

    def test_escapes_html_in_content(self) -> None:
        """Test that user content is properly HTML escaped."""
        job = Job(
            uid="job-xss",
            name="<script>alert('xss')</script>",
            company_name="Evil & Co",
        )
        match = MatchResult(
            job=job,
            similarity_score=0.5,
            filter_score=0.5,
            final_score=0.5,
            explanation=["<b>Bold</b> text"],
            missing_skills=[],
            interview_tips=[],
            apply_url=None,
        )

        html = _build_html_email([match])

        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "Evil &amp; Co" in html
        assert "&lt;b&gt;Bold&lt;/b&gt;" in html


class TestSendMatchResultsEmail:
    """Tests for send_match_results_email function."""

    def test_email_disabled(self, sample_match: MatchResult) -> None:
        """Test returns False when email is disabled."""
        with patch("matchai.services.email_service.EMAIL_ENABLED", False):
            result = send_match_results_email([sample_match])
            assert result is False

    def test_empty_matches(self) -> None:
        """Test returns False when matches list is empty."""
        with patch.multiple(
            "matchai.services.email_service",
            EMAIL_ENABLED=True,
            EMAIL_SENDER="test@gmail.com",
            EMAIL_RECIPIENT="recipient@example.com",
            EMAIL_APP_PASSWORD="secret123",
        ):
            result = send_match_results_email([])
            assert result is False

    def test_sends_email_successfully(self, sample_match: MatchResult) -> None:
        """Test email is sent successfully via SMTP."""
        with (
            patch.multiple(
                "matchai.services.email_service",
                EMAIL_ENABLED=True,
                EMAIL_SENDER="test@gmail.com",
                EMAIL_RECIPIENT="recipient@example.com",
                EMAIL_APP_PASSWORD="secret123",
            ),
            patch("matchai.services.email_service.smtplib.SMTP") as mock_smtp,
        ):
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            result = send_match_results_email([sample_match])

            assert result is True
            mock_smtp.assert_called_once_with("smtp.gmail.com", 587, timeout=30)
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("test@gmail.com", "secret123")
            mock_server.send_message.assert_called_once()

    def test_handles_smtp_auth_error(self, sample_match: MatchResult) -> None:
        """Test handles SMTP authentication errors gracefully."""
        with (
            patch.multiple(
                "matchai.services.email_service",
                EMAIL_ENABLED=True,
                EMAIL_SENDER="test@gmail.com",
                EMAIL_RECIPIENT="recipient@example.com",
                EMAIL_APP_PASSWORD="wrong-password",
            ),
            patch("matchai.services.email_service.smtplib.SMTP") as mock_smtp,
        ):
            mock_server = MagicMock()
            mock_server.login.side_effect = smtplib.SMTPAuthenticationError(
                535, b"Authentication failed"
            )
            mock_smtp.return_value.__enter__.return_value = mock_server

            result = send_match_results_email([sample_match])

            assert result is False

    def test_handles_smtp_exception(self, sample_match: MatchResult) -> None:
        """Test handles general SMTP exceptions gracefully."""
        with (
            patch.multiple(
                "matchai.services.email_service",
                EMAIL_ENABLED=True,
                EMAIL_SENDER="test@gmail.com",
                EMAIL_RECIPIENT="recipient@example.com",
                EMAIL_APP_PASSWORD="secret123",
            ),
            patch("matchai.services.email_service.smtplib.SMTP") as mock_smtp,
        ):
            mock_smtp.return_value.__enter__.side_effect = smtplib.SMTPException(
                "Connection failed"
            )

            result = send_match_results_email([sample_match])

            assert result is False

    def test_handles_config_validation_failure(
        self, sample_match: MatchResult
    ) -> None:
        """Test returns False when config validation fails."""
        with patch.multiple(
            "matchai.services.email_service",
            EMAIL_ENABLED=True,
            EMAIL_SENDER=None,
            EMAIL_RECIPIENT=None,
            EMAIL_APP_PASSWORD=None,
        ):
            result = send_match_results_email([sample_match])
            assert result is False
