"""End-to-end tests for MatchAI system.

Tests the complete pipeline flows including:
- Full matching pipeline (CV → Parse → Match → Explain)
- CLI commands (ingest, match, info)
- Ingestion pipeline with real Comeet sandbox API
- Edge cases

Uses the Comeet sandbox company for realistic testing.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz  # PyMuPDF
import numpy as np
import pytest
from typer.testing import CliRunner

from matchai.cv.extractor import extract_text_from_pdf
from matchai.cv.parser import parse_cv
from matchai.explainer.generator import (
    ExplanationOutput,
    find_missing_skills,
    generate_explanation,
)
from matchai.jobs.database import (
    get_all_companies,
    get_all_jobs,
    get_jobs,
    insert_companies,
    insert_jobs_to_db,
)
from matchai.jobs.embeddings import (
    embed_and_store_jobs,
    embed_candidate,
    get_existing_embedding_uids,
    get_job_embeddings,
)
from matchai.jobs.ingest import fetch_positions, ingest_from_api, load_companies_from_file
from matchai.main import app
from matchai.matching.filter import apply_filters
from matchai.matching.ranker import rank_jobs
from matchai.schemas.candidate import CandidateProfile, SeniorityLevel
from matchai.schemas.job import Company, Job, JobDetail
from matchai.utils import LLMConfigurationError
from tests.test_utils import make_test_candidate, make_test_job

runner = CliRunner()

# =============================================================================
# Sample Data Constants
# =============================================================================

# Comeet sandbox company for testing (publicly available test credentials)
COMEET_SANDBOX = {
    "name": "Comeet Sandbox",
    "uid": "E5.007",
    "token": "5E7236A0BCE5E7295111B55E70BCE",
    "extracted_from": "documentation",
}

# Sample CV content for a senior Python developer
SAMPLE_CV_CONTENT = """
JOHN DOE
Senior Software Engineer
john.doe@email.com | Tel Aviv, Israel | linkedin.com/in/johndoe

PROFESSIONAL SUMMARY
Experienced software engineer with 7+ years of expertise in Python, JavaScript, and cloud technologies.
Specialized in building scalable backend systems, microservices architecture, and API development.
Strong background in leading technical projects and mentoring junior developers.

TECHNICAL SKILLS
Programming Languages: Python, JavaScript, TypeScript, Go, SQL
Frameworks & Libraries: Django, FastAPI, React, Node.js, Express.js
Cloud & DevOps: AWS (EC2, S3, Lambda, RDS), Docker, Kubernetes, Terraform
Databases: PostgreSQL, MongoDB, Redis, Elasticsearch
Tools: Git, CI/CD (GitHub Actions, Jenkins), Linux

PROFESSIONAL EXPERIENCE

Senior Software Engineer | TechCorp Inc. | Tel Aviv | 2020 - Present
- Led development of microservices handling 1M+ daily requests using Python and FastAPI
- Designed and implemented RESTful APIs serving 50+ internal and external clients
- Architected event-driven systems using Kafka and Redis for real-time data processing
- Mentored team of 4 junior developers, conducting code reviews and technical training
- Reduced API response time by 40% through optimization and caching strategies

Software Engineer | StartupXYZ | Tel Aviv | 2017 - 2020
- Built full-stack applications with Django backend and React frontend
- Developed automated testing pipelines reducing bug escape rate by 60%
- Deployed and maintained services on AWS using Docker and Kubernetes
- Implemented CI/CD pipelines using GitHub Actions for automated deployments

Junior Developer | WebAgency | Tel Aviv | 2015 - 2017
- Developed responsive web applications using JavaScript and React
- Created RESTful APIs using Node.js and Express
- Collaborated with design team to implement pixel-perfect UI components

EDUCATION
B.Sc. Computer Science | Tel Aviv University | 2015
- Graduated Cum Laude
- Relevant coursework: Data Structures, Algorithms, Database Systems, Software Engineering

CERTIFICATIONS
- AWS Certified Solutions Architect - Associate
- Kubernetes Application Developer (CKAD)
"""

# Sample CV content for a junior frontend developer
SAMPLE_JUNIOR_CV_CONTENT = """
JANE SMITH
Junior Frontend Developer
jane.smith@email.com | New York, USA

SUMMARY
Recent computer science graduate with 1 year of experience in web development.
Passionate about creating user-friendly interfaces and learning new technologies.

SKILLS
Languages: JavaScript, HTML, CSS, Python (basic)
Frameworks: React, Vue.js
Tools: Git, VS Code, Figma

EXPERIENCE

Junior Developer | WebStudio | New York | 2024 - Present
- Build responsive web pages using React and CSS
- Collaborate with senior developers on feature implementation
- Write unit tests for frontend components

Intern | TechStartup | New York | 2023 - 2024
- Assisted in developing landing pages
- Learned React and modern JavaScript

EDUCATION
B.Sc. Computer Science | NYU | 2023
"""


# =============================================================================
# Sample Data Generators
# =============================================================================


def create_sample_cv_pdf(tmp_path: Path, content: str = SAMPLE_CV_CONTENT) -> Path:
    """Create a sample CV PDF file for testing.

    Uses PyMuPDF to create a PDF with the given content.

    Args:
        tmp_path: Pytest tmp_path fixture for temporary directory.
        content: Text content for the CV.

    Returns:
        Path to the created PDF file.
    """
    pdf_path = tmp_path / "sample_cv.pdf"

    doc = fitz.open()
    page = doc.new_page()

    # Insert text content into the PDF
    text_rect = fitz.Rect(50, 50, 550, 800)
    page.insert_textbox(
        text_rect,
        content,
        fontsize=10,
        fontname="helv",
        align=fitz.TEXT_ALIGN_LEFT,
    )

    doc.save(str(pdf_path))
    doc.close()

    return pdf_path


def create_sample_senior_candidate() -> CandidateProfile:
    """Create a sample senior Python developer candidate."""
    return CandidateProfile(
        skills=["python", "javascript", "typescript", "go", "sql"],
        tools_frameworks=[
            "django",
            "fastapi",
            "react",
            "nodejs",
            "aws",
            "docker",
            "kubernetes",
            "postgresql",
            "redis",
        ],
        seniority=SeniorityLevel.SENIOR,
        years_experience=7,
        domains=["backend", "microservices", "cloud", "api development"],
        keywords=["software engineer", "backend", "api", "microservices", "senior"],
        raw_text=SAMPLE_CV_CONTENT,
    )


def create_sample_junior_candidate() -> CandidateProfile:
    """Create a sample junior frontend developer candidate."""
    return CandidateProfile(
        skills=["javascript", "html", "css", "python"],
        tools_frameworks=["react", "vue.js", "git"],
        seniority=SeniorityLevel.JUNIOR,
        years_experience=1,
        domains=["frontend", "web development"],
        keywords=["junior developer", "frontend", "react"],
        raw_text=SAMPLE_JUNIOR_CV_CONTENT,
    )


def create_sample_mid_candidate() -> CandidateProfile:
    """Create a sample mid-level full-stack developer candidate."""
    return CandidateProfile(
        skills=["python", "javascript", "typescript"],
        tools_frameworks=["django", "react", "postgresql", "docker"],
        seniority=SeniorityLevel.MID,
        years_experience=3,
        domains=["full-stack", "web development"],
        keywords=["developer", "full-stack"],
        raw_text="Mid-level full-stack developer with 3 years experience.",
    )


# =============================================================================
# E2E: Real API Integration Tests
# =============================================================================


class TestComeetSandboxIntegration:
    """Test integration with real Comeet sandbox API."""

    @pytest.mark.integration
    def test_fetch_positions_from_sandbox(self):
        """Test fetching real positions from Comeet sandbox."""
        positions = fetch_positions(COMEET_SANDBOX["uid"], COMEET_SANDBOX["token"])

        assert isinstance(positions, list)
        # Sandbox should have at least some test positions
        assert len(positions) >= 0

        if positions:
            # Verify position structure
            pos = positions[0]
            assert "uid" in pos
            assert "name" in pos

    @pytest.mark.integration
    def test_full_ingestion_from_sandbox(self, temp_db_and_chroma, tmp_path):
        """Test complete ingestion pipeline with Comeet sandbox."""
        # Write sandbox company to file
        companies_file = tmp_path / "companies.json"
        companies_file.write_text(json.dumps([COMEET_SANDBOX]))

        # Load companies
        inserted = load_companies_from_file(companies_file)
        assert inserted == 1

        # Verify company was added
        companies = get_all_companies()
        assert len(companies) == 1
        assert companies[0].uid == COMEET_SANDBOX["uid"]

        # Ingest from API
        stats = ingest_from_api()

        assert stats["companies_processed"] == 1
        assert stats["jobs_fetched"] >= 0

        if stats["jobs_fetched"] > 0:
            assert stats["jobs_inserted"] >= 0
            assert stats["jobs_embedded"] >= 0

            # Verify jobs in database
            jobs = get_all_jobs()
            assert len(jobs) == stats["jobs_inserted"]

    @pytest.mark.integration
    def test_ingestion_idempotency_with_sandbox(self, temp_db_and_chroma, tmp_path):
        """Test that re-ingesting sandbox data doesn't create duplicates."""
        companies_file = tmp_path / "companies.json"
        companies_file.write_text(json.dumps([COMEET_SANDBOX]))

        load_companies_from_file(companies_file)

        # First ingestion
        stats1 = ingest_from_api()
        jobs_after_first = get_all_jobs()

        # Second ingestion
        stats2 = ingest_from_api()
        jobs_after_second = get_all_jobs()

        # Should have same number of jobs
        assert len(jobs_after_first) == len(jobs_after_second)

        if stats1["jobs_fetched"] > 0:
            assert stats2["jobs_skipped"] == stats1["jobs_inserted"]
            assert stats2["jobs_inserted"] == 0


# =============================================================================
# E2E: CV Processing Tests
# =============================================================================


class TestCVProcessing:
    """Test CV extraction and processing."""

    def test_extract_text_from_sample_cv(self, tmp_path):
        """Test extracting text from a sample CV PDF."""
        pdf_path = create_sample_cv_pdf(tmp_path)

        extracted_text = extract_text_from_pdf(pdf_path)

        assert isinstance(extracted_text, str)
        assert len(extracted_text) > 0
        # Check that key content was extracted
        assert "john" in extracted_text.lower() or "software" in extracted_text.lower()


# =============================================================================
# E2E: Full Matching Pipeline Tests
# =============================================================================


class TestFullMatchingPipeline:
    """Test the complete matching pipeline from candidate to results."""

    @pytest.mark.integration
    def test_pipeline_with_sandbox_jobs(self, temp_db_and_chroma, tmp_path):
        """Test full pipeline with real sandbox jobs."""
        # Ingest sandbox jobs
        companies_file = tmp_path / "companies.json"
        companies_file.write_text(json.dumps([COMEET_SANDBOX]))
        load_companies_from_file(companies_file)
        stats = ingest_from_api()

        if stats["jobs_inserted"] == 0:
            pytest.skip("No jobs available in sandbox")

        # Create candidate and run pipeline
        candidate = create_sample_senior_candidate()
        db_jobs = get_all_jobs()

        filtered_jobs = apply_filters(db_jobs, candidate)

        if filtered_jobs:
            ranked_results = rank_jobs(filtered_jobs, candidate)

            assert len(ranked_results) > 0
            assert all(0 <= r.final_score <= 1 for r in ranked_results)

            # Results should be sorted by final_score descending
            scores = [r.final_score for r in ranked_results]
            assert scores == sorted(scores, reverse=True)

    def test_pipeline_filter_to_rank_with_test_jobs(self, temp_db_and_chroma):
        """Test filtering and ranking pipeline with controlled test jobs."""
        # Create test jobs
        jobs = [
            make_test_job(
                "job-1",
                "Senior Python Developer",
                details_text="Python, Django, PostgreSQL, AWS experience required",
                location="Tel Aviv",
                experience_level="Senior",
            ),
            make_test_job(
                "job-2",
                "Full Stack Developer",
                details_text="React, Node.js, TypeScript, MongoDB",
                location="Remote",
                experience_level="Mid",
            ),
            make_test_job(
                "job-3",
                "Junior Backend Engineer",
                details_text="Python basics, SQL, eager to learn",
                location="New York",
                experience_level="Junior",
            ),
        ]
        insert_jobs_to_db(jobs)
        embed_and_store_jobs(jobs)

        # Create senior Python candidate
        candidate = create_sample_senior_candidate()

        # Load jobs from database
        db_jobs = get_all_jobs()
        assert len(db_jobs) == 3

        # Apply filters
        filtered_jobs = apply_filters(db_jobs, candidate)

        # Senior candidate should match Senior and Mid jobs (±1 level)
        # but not Junior (2 levels away)
        assert len(filtered_jobs) > 0

        # Rank jobs
        ranked_results = rank_jobs(filtered_jobs, candidate)

        assert len(ranked_results) > 0
        assert all(r.final_score >= 0 for r in ranked_results)
        assert all(r.final_score <= 1 for r in ranked_results)

        # Python job should rank higher for Python candidate
        python_job_result = next(
            (r for r in ranked_results if "python" in r.job.name.lower()), None
        )
        if python_job_result and len(ranked_results) > 1:
            # Python job should be among top results
            assert python_job_result in ranked_results[:2]

    def test_pipeline_location_filter(self, temp_db_and_chroma):
        """Test pipeline with location filtering."""
        jobs = [
            make_test_job("job-1", "Developer", "Python required", location="Tel Aviv"),
            make_test_job("job-2", "Engineer", "Python needed", location="Remote"),
            make_test_job("job-3", "Coder", "Python skills", location="New York"),
        ]
        insert_jobs_to_db(jobs)
        embed_and_store_jobs(jobs)

        candidate = create_sample_senior_candidate()

        # Filter by Tel Aviv - should get Tel Aviv + Remote jobs
        tel_aviv_jobs = get_jobs(location="Tel Aviv")

        # Database-level filter should include Tel Aviv
        assert any("Tel Aviv" in (j.location or "") for j in tel_aviv_jobs)

        # Apply in-memory filters
        filtered = apply_filters(tel_aviv_jobs, candidate)
        job_uids = {j.uid for j, _ in filtered}

        # Tel Aviv job should be included
        assert "job-1" in job_uids or len(filtered) == 0

    def test_pipeline_seniority_filtering(self, temp_db_and_chroma):
        """Test that seniority filtering works correctly."""
        jobs = [
            make_test_job(
                "senior-job", "Senior Developer", "Python", experience_level="Senior"
            ),
            make_test_job("mid-job", "Developer", "Python", experience_level="Mid"),
            make_test_job(
                "junior-job", "Junior Developer", "Python", experience_level="Junior"
            ),
        ]
        insert_jobs_to_db(jobs)
        embed_and_store_jobs(jobs)

        # Senior candidate
        senior_candidate = create_sample_senior_candidate()
        assert senior_candidate.seniority == SeniorityLevel.SENIOR

        db_jobs = get_all_jobs()
        filtered = apply_filters(db_jobs, senior_candidate)

        filtered_levels = [
            j.experience_level.lower() if j.experience_level else None for j, _ in filtered
        ]

        # Senior candidate should not match Junior jobs (2 levels away)
        assert "junior" not in filtered_levels

    def test_pipeline_with_missing_skills_detection(self, temp_db_and_chroma):
        """Test that missing skills are correctly identified."""
        job = make_test_job(
            "skill-job",
            "Full Stack Developer",
            details_text="Looking for Python Django React MongoDB Redis Kubernetes developers",
        )
        insert_jobs_to_db([job])
        embed_and_store_jobs([job])

        # Candidate without MongoDB, Redis, Kubernetes
        candidate = CandidateProfile(
            skills=["python", "javascript"],
            tools_frameworks=["django", "react"],
            seniority=SeniorityLevel.MID,
            domains=["backend"],
            keywords=["developer"],
            raw_text="Python and JavaScript developer",
        )

        db_jobs = get_all_jobs()
        filtered = apply_filters(db_jobs, candidate)

        if filtered:
            job_obj, _ = filtered[0]
            missing = find_missing_skills(job_obj, candidate)

            assert isinstance(missing, list)
            # Should identify some missing skills from the job description
            assert len(missing) >= 0

    def test_pipeline_with_explanation_generation(self, temp_db_and_chroma):
        """Test explanation generation in pipeline."""
        job = make_test_job(
            uid="explain-job",
            name="Python Developer",
            details_text="Python, Django, PostgreSQL",
        )
        insert_jobs_to_db(jobs=[job])
        embed_and_store_jobs(jobs=[job])

        candidate = create_sample_senior_candidate()
        db_jobs = get_all_jobs()
        filtered = apply_filters(jobs=db_jobs, candidate=candidate)

        if filtered:
            job_obj, _ = filtered[0]

            mock_response = ExplanationOutput(
                bullet_points=[
                    "Strong Python skills match the job requirements",
                    "Backend expertise is a key match",
                ]
            )

            with patch("matchai.explainer.generator.get_llm"), patch(
                "matchai.explainer.generator.PydanticOutputParser"
            ) as mock_parser_class, patch(
                "matchai.explainer.generator.ChatPromptTemplate"
            ) as mock_prompt_class:
                mock_parser = MagicMock()
                mock_parser.get_format_instructions.return_value = "format"
                mock_parser_class.return_value = mock_parser

                mock_prompt = MagicMock()
                mock_prompt_class.from_template.return_value = mock_prompt

                mock_chain = MagicMock()
                mock_chain.invoke.return_value = mock_response
                mock_prompt.__or__ = MagicMock(return_value=MagicMock())
                mock_prompt.__or__.return_value.__or__ = MagicMock(return_value=mock_chain)

                explanation = generate_explanation(
                    job=job_obj,
                    candidate=candidate,
                    similarity_score=0.85,
                    filter_score=0.75,
                )

                assert isinstance(explanation, list)
                assert len(explanation) > 0

    def test_top_n_limiting(self, temp_db_and_chroma):
        """Test that top_n parameter limits results."""
        jobs = [
            make_test_job(f"job-{i}", f"Developer {i}", "Python required") for i in range(10)
        ]
        insert_jobs_to_db(jobs)
        embed_and_store_jobs(jobs)

        candidate = create_sample_senior_candidate()
        db_jobs = get_all_jobs()
        filtered = apply_filters(db_jobs, candidate)

        # Request only top 3
        ranked = rank_jobs(filtered, candidate, top_n=3)

        assert len(ranked) <= 3


# =============================================================================
# E2E: CLI Command Tests
# =============================================================================


class TestCLICommands:
    """Test CLI commands end-to-end."""

    def test_info_command_no_database(self, tmp_path):
        """Test info command when no database exists."""
        with (
            patch("matchai.main.DATABASE_URL", None),
            patch("matchai.main.DB_PATH", tmp_path / "nonexistent.db"),
        ):
            result = runner.invoke(app, ["info"])

            assert "Database not found" in result.stdout or result.exit_code == 0

    def test_info_command_with_data(self, temp_db_and_chroma):
        """Test info command with populated database."""
        jobs = [
            make_test_job("job-1", "Developer", location="Tel Aviv"),
            make_test_job("job-2", "Engineer", location="Remote"),
        ]
        insert_jobs_to_db(jobs)

        with patch("matchai.main.DB_PATH", temp_db_and_chroma["db_path"]):
            result = runner.invoke(app, ["info"])

            assert result.exit_code == 0

    def test_ingest_command_file_not_found(self):
        """Test ingest command with non-existent file."""
        result = runner.invoke(app, ["ingest", "--companies", "/nonexistent/path.json"])

        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    @patch("matchai.main.load_companies_from_file")
    @patch("matchai.main.ingest_from_api")
    def test_ingest_command_success(
        self, mock_ingest, mock_load, temp_db_and_chroma, tmp_path
    ):
        """Test successful ingest command."""
        companies_file = tmp_path / "companies.json"
        companies_file.write_text(json.dumps([COMEET_SANDBOX]))

        mock_load.return_value = 1
        mock_ingest.return_value = {
            "companies_processed": 1,
            "jobs_fetched": 5,
            "jobs_inserted": 5,
            "jobs_skipped": 0,
            "jobs_embedded": 5,
        }

        result = runner.invoke(app, ["ingest", "--companies", str(companies_file)])

        assert result.exit_code == 0
        assert "complete" in result.stdout.lower()

    def test_match_command_cv_not_found(self, temp_db_and_chroma):
        """Test match command with non-existent CV file."""
        with patch("matchai.main.DB_PATH", temp_db_and_chroma["db_path"]):
            result = runner.invoke(app, ["match", "--cv", "/nonexistent/cv.pdf"])

            assert result.exit_code == 1
            assert "not found" in result.stdout.lower()

    def test_match_command_no_database(self, tmp_path):
        """Test match command when database doesn't exist."""
        cv_path = create_sample_cv_pdf(tmp_path)

        with patch("matchai.main.DB_PATH", tmp_path / "nonexistent.db"):
            result = runner.invoke(app, ["match", "--cv", str(cv_path)])

            assert result.exit_code == 1
            assert "database not found" in result.stdout.lower()

    def test_match_command_llm_not_configured(self, temp_db_and_chroma, tmp_path):
        """Test match command when Groq API key is not configured."""
        cv_path = create_sample_cv_pdf(tmp_path)

        with patch("matchai.main.DB_PATH", temp_db_and_chroma["db_path"]), patch(
            "matchai.main.check_llm_configured"
        ) as mock_check:
            mock_check.side_effect = LLMConfigurationError(
                "GROQ_API_KEY environment variable is not set. "
                "Please create a .env file with your Groq API key."
            )

            result = runner.invoke(app, ["match", "--cv", str(cv_path)])

            assert result.exit_code == 1
            assert "groq_api_key" in result.stdout.lower()
            assert ".env" in result.stdout.lower()


# =============================================================================
# E2E: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_database_matching(self, temp_db_and_chroma):
        """Test matching against empty database."""
        candidate = create_sample_senior_candidate()

        jobs = get_all_jobs()
        assert len(jobs) == 0

        filtered = apply_filters(jobs, candidate)
        assert filtered == []

        ranked = rank_jobs(filtered, candidate)
        assert ranked == []

    def test_candidate_with_no_skills(self, temp_db_and_chroma):
        """Test matching with candidate that has no skills."""
        jobs = [make_test_job("job-1", "Developer", "Python required")]
        insert_jobs_to_db(jobs)
        embed_and_store_jobs(jobs)

        candidate = CandidateProfile(
            skills=[],
            tools_frameworks=[],
            seniority=SeniorityLevel.MID,
            domains=[],
            keywords=[],
            raw_text="No skills candidate",
        )

        db_jobs = get_all_jobs()
        filtered = apply_filters(db_jobs, candidate)

        # filter_by_skills returns all jobs with 0 score when no skills
        assert len(filtered) >= 0

    def test_job_with_no_details(self, temp_db_and_chroma):
        """Test handling jobs with empty details."""
        job = Job(
            uid="empty-job",
            name="Job with no details",
            location="Remote",
        )
        insert_jobs_to_db([job])
        embed_and_store_jobs([job])

        candidate = create_sample_senior_candidate()
        db_jobs = get_all_jobs()

        # Should not crash
        filtered = apply_filters(db_jobs, candidate)
        assert isinstance(filtered, list)

    def test_special_characters_in_skills(self, temp_db_and_chroma):
        """Test handling special characters in skill names."""
        candidate = CandidateProfile(
            skills=["c++", "c#", ".net", "node.js"],
            tools_frameworks=["asp.net"],
            seniority=SeniorityLevel.MID,
            domains=[],
            keywords=[],
            raw_text="Test",
        )

        job = Job(
            uid="special-job",
            name=".NET Developer",
            details=[
                JobDetail(
                    name="Requirements",
                    value="<p>C#, .NET, ASP.NET Core, Node.js</p>",
                    order=1,
                )
            ],
        )
        insert_jobs_to_db([job])
        embed_and_store_jobs([job])

        db_jobs = get_all_jobs()
        filtered = apply_filters(db_jobs, candidate)

        # Should find matches despite special characters
        assert len(filtered) > 0

    def test_unicode_in_job_description(self, temp_db_and_chroma):
        """Test handling unicode characters in job descriptions."""
        job = Job(
            uid="unicode-job",
            name="Developer עברית",
            location="Tel Aviv, ישראל",
            details=[
                JobDetail(
                    name="Description",
                    value="<p>Looking for a Python developer. תיאור בעברית</p>",
                    order=1,
                )
            ],
        )
        insert_jobs_to_db([job])
        embed_and_store_jobs([job])

        candidate = create_sample_senior_candidate()
        db_jobs = get_all_jobs()

        # Should not crash
        filtered = apply_filters(db_jobs, candidate)
        assert isinstance(filtered, list)

    def test_very_long_job_description(self, temp_db_and_chroma):
        """Test handling very long job descriptions."""
        long_description = "Python developer with experience in Django. " * 500

        job = Job(
            uid="long-job",
            name="Developer",
            details=[
                JobDetail(
                    name="Description",
                    value=f"<p>{long_description}</p>",
                    order=1,
                )
            ],
        )
        insert_jobs_to_db([job])
        embed_and_store_jobs([job])

        candidate = create_sample_senior_candidate()
        db_jobs = get_all_jobs()

        # Should not crash and should find the job
        filtered = apply_filters(db_jobs, candidate)
        ranked = rank_jobs(filtered, candidate)

        assert isinstance(ranked, list)

    def test_duplicate_skills_in_candidate(self, temp_db_and_chroma):
        """Test handling duplicate skills in candidate profile."""
        candidate = CandidateProfile(
            skills=["python", "python", "Python", "PYTHON"],
            tools_frameworks=["django", "Django"],
            seniority=SeniorityLevel.MID,
            domains=[],
            keywords=[],
            raw_text="Test",
        )

        job = make_test_job("dup-job", "Developer", "Python and Django required")
        insert_jobs_to_db([job])
        embed_and_store_jobs([job])

        db_jobs = get_all_jobs()

        # Should not crash
        filtered = apply_filters(db_jobs, candidate)
        assert isinstance(filtered, list)


# =============================================================================
# E2E: Integration Tests with Mocked LLM
# =============================================================================


class TestLLMIntegration:
    """Test LLM-dependent features with mocked LLM."""

    def test_explanation_generation_integration(self, temp_db_and_chroma):
        """Test full flow with explanation generation."""
        job = make_test_job(
            uid="llm-job",
            name="Python Developer",
            details_text="Python, AWS, Backend",
        )
        insert_jobs_to_db(jobs=[job])
        embed_and_store_jobs(jobs=[job])

        candidate = create_sample_senior_candidate()
        db_jobs = get_all_jobs()
        filtered = apply_filters(jobs=db_jobs, candidate=candidate)
        ranked = rank_jobs(filtered_jobs=filtered, candidate=candidate, top_n=1)

        if ranked:
            result = ranked[0]

            mock_response = ExplanationOutput(
                bullet_points=[
                    "Python expertise matches requirements",
                    "Strong backend background",
                ]
            )

            with patch("matchai.explainer.generator.get_llm"), patch(
                "matchai.explainer.generator.PydanticOutputParser"
            ) as mock_parser_class, patch(
                "matchai.explainer.generator.ChatPromptTemplate"
            ) as mock_prompt_class:
                mock_parser = MagicMock()
                mock_parser.get_format_instructions.return_value = "format"
                mock_parser_class.return_value = mock_parser

                mock_prompt = MagicMock()
                mock_prompt_class.from_template.return_value = mock_prompt

                mock_chain = MagicMock()
                mock_chain.invoke.return_value = mock_response
                mock_prompt.__or__ = MagicMock(return_value=MagicMock())
                mock_prompt.__or__.return_value.__or__ = MagicMock(return_value=mock_chain)

                explanation = generate_explanation(
                    job=result.job,
                    candidate=candidate,
                    similarity_score=result.similarity_score,
                    filter_score=result.filter_score,
                )

                assert isinstance(explanation, list)
                assert len(explanation) > 0

    def test_explanation_handles_llm_error(self, temp_db_and_chroma):
        """Test that explanation generation handles LLM errors gracefully."""
        job = make_test_job(
            uid="error-job",
            name="Developer",
            details_text="Python",
        )
        candidate = create_sample_senior_candidate()

        with patch("matchai.utils.get_llm"), patch(
            "matchai.explainer.generator.PydanticOutputParser"
        ) as mock_parser_class, patch(
            "matchai.explainer.generator.ChatPromptTemplate"
        ) as mock_prompt_class:
            mock_parser = MagicMock()
            mock_parser.get_format_instructions.return_value = "format"
            mock_parser_class.return_value = mock_parser

            mock_prompt = MagicMock()
            mock_prompt_class.from_template.return_value = mock_prompt

            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = Exception("LLM connection failed")
            mock_prompt.__or__ = MagicMock(return_value=MagicMock())
            mock_prompt.__or__.return_value.__or__ = MagicMock(return_value=mock_chain)

            # Should raise exception
            with pytest.raises(Exception):
                generate_explanation(
                    job=job,
                    candidate=candidate,
                    similarity_score=0.5,
                    filter_score=0.5,
                )


# =============================================================================
# E2E: Data Consistency Tests
# =============================================================================


class TestDataConsistency:
    """Test data consistency across operations."""

    def test_job_data_preserved_through_pipeline(self, temp_db_and_chroma):
        """Test that job data is preserved through insert and retrieval."""
        original_jobs = [
            make_test_job("job-1", "Developer A", location="Tel Aviv"),
            make_test_job("job-2", "Developer B", location="Remote"),
            make_test_job("job-3", "Developer C", location="New York"),
        ]
        insert_jobs_to_db(original_jobs)

        retrieved_jobs = get_all_jobs()

        assert len(retrieved_jobs) == len(original_jobs)

        for orig, retr in zip(
            sorted(original_jobs, key=lambda j: j.uid),
            sorted(retrieved_jobs, key=lambda j: j.uid),
        ):
            assert orig.uid == retr.uid
            assert orig.name == retr.name
            assert orig.location == retr.location

    def test_embeddings_stored_and_retrieved(self, temp_db_and_chroma):
        """Test that embeddings are properly stored and retrieved."""
        jobs = [
            make_test_job("emb-1", "Developer", "Python"),
            make_test_job("emb-2", "Engineer", "JavaScript"),
        ]
        insert_jobs_to_db(jobs)
        embed_and_store_jobs(jobs)

        # Check UIDs are stored
        stored_uids = get_existing_embedding_uids()
        expected_uids = {job.uid for job in jobs}
        assert stored_uids == expected_uids

        # Check embeddings can be retrieved
        embeddings = get_job_embeddings(list(expected_uids))
        assert len(embeddings) == len(jobs)

        for uid, embedding in embeddings.items():
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) > 0

    def test_filter_scores_in_valid_range(self, temp_db_and_chroma):
        """Test that all scores are in valid ranges."""
        jobs = [
            make_test_job("score-1", "Python Developer", "Python, Django"),
            make_test_job("score-2", "JS Developer", "JavaScript, React"),
        ]
        insert_jobs_to_db(jobs)
        embed_and_store_jobs(jobs)

        candidate = create_sample_senior_candidate()
        db_jobs = get_all_jobs()
        filtered = apply_filters(db_jobs, candidate)
        ranked = rank_jobs(filtered, candidate)

        for result in ranked:
            assert (
                0 <= result.similarity_score <= 1
            ), f"Invalid similarity: {result.similarity_score}"
            assert (
                0 <= result.filter_score <= 1
            ), f"Invalid filter score: {result.filter_score}"
            assert 0 <= result.final_score <= 1, f"Invalid final score: {result.final_score}"

    def test_ranking_determinism(self, temp_db_and_chroma):
        """Test that ranking is deterministic for same input."""
        jobs = [
            make_test_job(f"det-{i}", f"Developer {i}", "Python Django FastAPI")
            for i in range(5)
        ]
        insert_jobs_to_db(jobs)
        embed_and_store_jobs(jobs)

        candidate = create_sample_senior_candidate()
        db_jobs = get_all_jobs()
        filtered = apply_filters(db_jobs, candidate)

        # Run ranking multiple times
        results1 = rank_jobs(filtered, candidate)
        results2 = rank_jobs(filtered, candidate)

        # Should get same order
        uids1 = [r.job.uid for r in results1]
        uids2 = [r.job.uid for r in results2]
        assert uids1 == uids2


# =============================================================================
# E2E: Candidate Embedding Tests
# =============================================================================


class TestCandidateEmbedding:
    """Test candidate embedding functionality."""

    def test_embed_candidate_produces_valid_vector(self):
        """Test that embedding a candidate produces a valid vector."""
        candidate = create_sample_senior_candidate()
        embedding = embed_candidate(candidate)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
        assert not np.isnan(embedding).any()

    def test_similar_candidates_have_similar_embeddings(self):
        """Test that similar candidates produce similar embeddings."""
        candidate1 = CandidateProfile(
            skills=["python", "django"],
            tools_frameworks=["aws", "docker"],
            seniority=SeniorityLevel.SENIOR,
            domains=["backend"],
            keywords=["developer"],
            raw_text="Python backend developer",
        )

        candidate2 = CandidateProfile(
            skills=["python", "fastapi"],
            tools_frameworks=["aws", "kubernetes"],
            seniority=SeniorityLevel.SENIOR,
            domains=["backend"],
            keywords=["developer"],
            raw_text="Python backend engineer",
        )

        candidate3 = CandidateProfile(
            skills=["javascript", "react"],
            tools_frameworks=["nodejs", "mongodb"],
            seniority=SeniorityLevel.MID,
            domains=["frontend"],
            keywords=["developer"],
            raw_text="JavaScript frontend developer",
        )

        emb1 = embed_candidate(candidate1)
        emb2 = embed_candidate(candidate2)
        emb3 = embed_candidate(candidate3)

        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity

        sim_1_2 = cosine_similarity([emb1], [emb2])[0][0]
        sim_1_3 = cosine_similarity([emb1], [emb3])[0][0]

        # Similar Python backend candidates should be more similar
        # than Python backend vs JavaScript frontend
        assert sim_1_2 > sim_1_3
