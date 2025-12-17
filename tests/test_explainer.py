"""Tests for explanation generation."""

from unittest.mock import MagicMock, patch

import pytest

from matchai.explainer.generator import ExplanationOutput, generate_explanation
from tests.test_utils import make_test_candidate, make_test_job


class TestGenerateExplanation:
    def test_generates_bullet_points(self):
        candidate = make_test_candidate(
            skills=["python", "machine learning"],
            tools=["tensorflow", "docker"],
            seniority="senior",
        )
        job = make_test_job(
            uid="job-123",
            name="Senior ML Engineer",
            details_text="We are looking for an ML engineer with Python experience.",
            location="Remote",
        )

        mock_response = ExplanationOutput(
            bullet_points=[
                "Strong match on Python and ML skills",
                "Relevant fintech domain experience",
            ]
        )

        with patch("matchai.explainer.generator.get_llm"), patch(
            "matchai.explainer.generator.PydanticOutputParser"
        ) as mock_parser_class, patch(
            "matchai.explainer.generator.ChatPromptTemplate"
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

            result = generate_explanation(
                job=job,
                candidate=candidate,
                similarity_score=0.85,
                filter_score=0.75,
            )

            assert len(result) == 2
            assert "Python" in result[0]

    def test_raises_on_none_response(self):
        candidate = make_test_candidate(skills=["python"])
        job = make_test_job(uid="job-1", name="Developer")

        with patch("matchai.explainer.generator.get_llm"), patch(
            "matchai.explainer.generator.PydanticOutputParser"
        ) as mock_parser_class, patch(
            "matchai.explainer.generator.ChatPromptTemplate"
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

            with pytest.raises(ValueError, match="LLM failed to generate explanation"):
                generate_explanation(
                    job=job,
                    candidate=candidate,
                    similarity_score=0.85,
                    filter_score=0.75,
                )
