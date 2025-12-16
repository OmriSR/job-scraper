"""Tests for CV parsing with LLM."""

from unittest.mock import MagicMock, patch

import pytest

from matchai.cv.parser import LLMCandidateOutput, parse_cv
from matchai.schemas.candidate import SeniorityLevel


class TestParseCv:
    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response with valid candidate data."""
        return LLMCandidateOutput(
            skills=["Python", "Machine Learning", "Communication"],
            tools_frameworks=["TensorFlow", "Docker", "AWS"],
            seniority="senior",
            years_experience=7,
            domains=["FinTech", "Data Science"],
            keywords=["NLP", "Deep Learning"],
        )

    @pytest.fixture
    def sample_cv_text(self):
        return """
        John Doe
        Senior Machine Learning Engineer

        Experience:
        - 7 years of experience in ML and data science
        - Led ML team at FinTech startup
        - Built NLP pipelines using TensorFlow

        Skills:
        - Python, Machine Learning, Communication
        - TensorFlow, Docker, AWS
        """

    def test_parse_cv_returns_candidate_profile(self, mock_llm_response, sample_cv_text):
        with patch("matchai.utils.get_llm"), patch(
            "matchai.cv.parser.PydanticOutputParser"
        ) as mock_parser_class, patch(
            "matchai.cv.parser.ChatPromptTemplate"
        ) as mock_prompt_class:
            mock_parser = MagicMock()
            mock_parser.get_format_instructions.return_value = "format instructions"
            mock_parser_class.return_value = mock_parser

            mock_prompt = MagicMock()
            mock_prompt_class.from_template.return_value = mock_prompt

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_llm_response
            mock_prompt.__or__ = MagicMock(return_value=MagicMock())
            mock_prompt.__or__.return_value.__or__ = MagicMock(return_value=mock_chain)

            result = parse_cv(sample_cv_text)

            assert result.skills == ["python", "machine learning", "communication"]
            assert result.tools_frameworks == ["tensorflow", "docker", "aws"]
            assert result.seniority == SeniorityLevel.SENIOR
            assert result.years_experience == 7
            assert result.domains == ["fintech", "data science"]
            assert result.keywords == ["nlp", "deep learning"]
            assert result.raw_text == sample_cv_text

    def test_parse_cv_normalizes_to_lowercase(self, sample_cv_text):
        mock_response = LLMCandidateOutput(
            skills=["PYTHON", "JavaScript"],
            tools_frameworks=["REACT", "Node.JS"],
            seniority="MID",
            years_experience=3,
            domains=["E-Commerce"],
            keywords=["API"],
        )

        with patch("matchai.utils.get_llm"), patch(
            "matchai.cv.parser.PydanticOutputParser"
        ) as mock_parser_class, patch(
            "matchai.cv.parser.ChatPromptTemplate"
        ) as mock_prompt_class:
            mock_parser = MagicMock()
            mock_parser.get_format_instructions.return_value = "format instructions"
            mock_parser_class.return_value = mock_parser

            mock_prompt = MagicMock()
            mock_prompt_class.from_template.return_value = mock_prompt

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            mock_prompt.__or__ = MagicMock(return_value=MagicMock())
            mock_prompt.__or__.return_value.__or__ = MagicMock(return_value=mock_chain)

            result = parse_cv(sample_cv_text)

            assert result.skills == ["python", "javascript"]
            assert result.tools_frameworks == ["react", "node.js"]
            assert result.seniority == SeniorityLevel.MID
            assert result.domains == ["e-commerce"]
            assert result.keywords == ["api"]

    def test_parse_cv_raises_on_none_response(self, sample_cv_text):
        with patch("matchai.utils.get_llm"), patch(
            "matchai.cv.parser.PydanticOutputParser"
        ) as mock_parser_class, patch(
            "matchai.cv.parser.ChatPromptTemplate"
        ) as mock_prompt_class:
            mock_parser = MagicMock()
            mock_parser.get_format_instructions.return_value = "format instructions"
            mock_parser_class.return_value = mock_parser

            mock_prompt = MagicMock()
            mock_prompt_class.from_template.return_value = mock_prompt

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = None
            mock_prompt.__or__ = MagicMock(return_value=MagicMock())
            mock_prompt.__or__.return_value.__or__ = MagicMock(return_value=mock_chain)

            with pytest.raises(ValueError, match="LLM failed to parse CV"):
                parse_cv(sample_cv_text)


class TestLLMCandidateOutput:
    def test_valid_output_creation(self):
        output = LLMCandidateOutput(
            skills=["python"],
            tools_frameworks=["docker"],
            seniority="senior",
            years_experience=5,
            domains=["fintech"],
            keywords=["backend"],
        )
        assert output.skills == ["python"]
        assert output.seniority == "senior"

    def test_optional_years_experience(self):
        output = LLMCandidateOutput(
            skills=["python"],
            tools_frameworks=["docker"],
            seniority="mid",
            years_experience=None,
            domains=["fintech"],
            keywords=["backend"],
        )
        assert output.years_experience is None
