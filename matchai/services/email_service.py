"""Email notification service for sending match results via Gmail SMTP."""

import html
import logging
import smtplib
from datetime import UTC, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from matchai.config import (
    EMAIL_APP_PASSWORD,
    EMAIL_ENABLED,
    EMAIL_RECIPIENT,
    EMAIL_SENDER,
    SMTP_HOST,
    SMTP_PORT,
)
from matchai.schemas.match import MatchResult

logger = logging.getLogger(__name__)


def send_match_results_email(matches: list[MatchResult]) -> bool:
    """Send email with match results.

    Args:
        matches: List of MatchResult objects to include in email.

    Returns:
        True if email sent successfully, False otherwise.
    """
    if not EMAIL_ENABLED:
        logger.info("Email notifications disabled")
        return False

    if not _validate_email_config():
        logger.warning("Email configuration incomplete - skipping notification")
        return False

    if not matches:
        logger.info("No matches to send - skipping email")
        return False

    try:
        html_content = _build_html_email(matches)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[MatchAI] {len(matches)} New Job Matches"
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECIPIENT
        msg.attach(MIMEText(html_content, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_APP_PASSWORD)
            server.send_message(msg)

        logger.info(f"Email sent successfully to {EMAIL_RECIPIENT}")
        return True

    except smtplib.SMTPAuthenticationError:
        logger.error(
            "SMTP authentication failed. "
            "Check EMAIL_APP_PASSWORD is a valid Gmail App Password"
        )
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error sending email: {e}")
        return False


def _validate_email_config() -> bool:
    """Validate all required email configuration is present."""
    missing = []
    if not EMAIL_SENDER:
        missing.append("EMAIL_SENDER")
    if not EMAIL_RECIPIENT:
        missing.append("EMAIL_RECIPIENT")
    if not EMAIL_APP_PASSWORD:
        missing.append("EMAIL_APP_PASSWORD")

    if missing:
        logger.warning(f"Missing email configuration: {', '.join(missing)}")
        return False
    return True


def _build_html_email(matches: list[MatchResult]) -> str:
    """Build HTML email body from match results."""
    now = datetime.now(UTC).strftime("%B %d, %Y at %H:%M UTC")

    job_cards = []
    for i, match in enumerate(matches, 1):
        job = match.job
        score_pct = int(match.final_score * 100)

        # Escape user-generated content
        job_title = html.escape(job.name or "Unknown Position")
        company = html.escape(job.company_name or "Unknown Company")
        location = html.escape(job.location or "Not specified")

        # Build explanation list
        explanation_items = "".join(
            f"<li>{html.escape(exp)}</li>" for exp in match.explanation
        )

        # Build missing skills section
        missing_skills_html = ""
        if match.missing_skills:
            skills_items = "".join(
                f"<li>{html.escape(skill)}</li>" for skill in match.missing_skills
            )
            missing_skills_html = f"""
            <div class="missing-skills">
                <div class="section-title">Skills to Develop</div>
                <ul>{skills_items}</ul>
            </div>
            """

        # Build interview tips section
        tips_html = ""
        if match.interview_tips:
            tips_items = "".join(
                f"<li>{html.escape(tip)}</li>" for tip in match.interview_tips
            )
            tips_html = f"""
            <div class="tips">
                <div class="section-title">Interview Tips</div>
                <ul>{tips_items}</ul>
            </div>
            """

        # Apply button
        apply_btn = ""
        if match.apply_url:
            apply_btn = f'<a href="{html.escape(match.apply_url)}" class="apply-btn">Apply Now</a>'

        job_cards.append(f"""
        <div class="job-card">
            <div class="job-number">#{i}</div>
            <h2 class="job-title">{job_title}</h2>
            <p class="company">{company} | {location}</p>
            <span class="score">{score_pct}% Match</span>

            <div class="section-title">Why This Matches</div>
            <ul>{explanation_items}</ul>

            {missing_skills_html}
            {tips_html}
            {apply_btn}
        </div>
        """)

    jobs_html = "\n".join(job_cards)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 600px;
            margin: 0 auto;
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a73e8;
            margin: 0 0 8px 0;
            font-size: 24px;
        }}
        .subtitle {{
            color: #5f6368;
            margin: 0 0 24px 0;
            font-size: 14px;
        }}
        .job-card {{
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            position: relative;
        }}
        .job-number {{
            position: absolute;
            top: 12px;
            right: 12px;
            background: #e8f0fe;
            color: #1a73e8;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        .job-title {{
            font-size: 18px;
            font-weight: bold;
            color: #202124;
            margin: 0 0 4px 0;
            padding-right: 40px;
        }}
        .company {{
            color: #5f6368;
            font-size: 14px;
            margin: 0 0 12px 0;
        }}
        .score {{
            background: #34a853;
            color: white;
            padding: 4px 12px;
            border-radius: 16px;
            font-weight: bold;
            font-size: 14px;
            display: inline-block;
        }}
        .section-title {{
            font-weight: 600;
            color: #202124;
            margin: 16px 0 8px 0;
            font-size: 14px;
        }}
        ul {{
            margin: 0;
            padding-left: 20px;
        }}
        li {{
            margin: 6px 0;
            color: #3c4043;
            font-size: 14px;
            line-height: 1.5;
        }}
        .missing-skills {{
            background: #fff3e0;
            padding: 12px;
            border-radius: 6px;
            margin-top: 12px;
        }}
        .missing-skills .section-title {{
            margin-top: 0;
            color: #e65100;
        }}
        .tips {{
            background: #e8f5e9;
            padding: 12px;
            border-radius: 6px;
            margin-top: 12px;
        }}
        .tips .section-title {{
            margin-top: 0;
            color: #2e7d32;
        }}
        .apply-btn {{
            display: inline-block;
            background: #1a73e8;
            color: white;
            padding: 10px 24px;
            text-decoration: none;
            border-radius: 6px;
            margin-top: 16px;
            font-weight: 600;
            font-size: 14px;
        }}
        .footer {{
            text-align: center;
            color: #9aa0a6;
            font-size: 12px;
            margin-top: 24px;
            padding-top: 16px;
            border-top: 1px solid #e0e0e0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Your Top Job Matches</h1>
        <p class="subtitle">Found {len(matches)} matching positions on {now}</p>

        {jobs_html}

        <div class="footer">
            Sent by MatchAI
        </div>
    </div>
</body>
</html>"""
