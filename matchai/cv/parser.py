"""CV parsing using LLM to extract structured candidate profile."""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from matchai.schemas.candidate import CandidateProfile, SeniorityLevel
from matchai.utils import get_llm

CV_PARSING_PROMPT = """\
You are a professional CV/resume parser. Analyze the following CV text and extract structured information.

CV TEXT:
{cv_text}

Guidelines:
- Be thorough but avoid duplicating items across categories
- Normalize all values to lowercase
- For seniority, infer from years of experience and role titles:
  - junior: 0-2 years, entry-level roles
  - mid: 2-5 years, independent contributor
  - senior: 5-8 years, mentoring others
  - lead: 8+ years, team leadership
  - principal: 10+ years, technical strategy
  - staff: 12+ years, organization-wide impact

{format_instructions}\
"""


class LLMCandidateOutput(BaseModel):
    """Intermediate schema for LLM CV parsing output.
    This schema defines the structure that the LLM should return when parsinga CV.
    """

    skills: list[str] = Field(
        description="Technical and soft skills extracted from the CV"
    )
    tools_frameworks: list[str] = Field(
        description="Specific tools, frameworks, and technologies mentioned"
    )
    seniority: str = Field(
        description="Inferred seniority level: junior, mid, senior, lead, principal, or staff"
    )
    years_experience: int | None = Field(
        default=None, description="Total years of professional experience, if determinable"
    )
    domains: list[str] = Field(
        description="Industry domains or areas of expertise (e.g., fintech, healthcare)"
    )
    keywords: list[str] = Field(description="Additional relevant keywords for matching")


def parse_cv(cv_text: str) -> CandidateProfile:
    """Parse CV text into a structured CandidateProfile using LLM.

    Args:
        cv_text: Extracted text from the CV.

    Returns:
        CandidateProfile with extracted information.

    Raises:
        ValueError: If LLM fails to parse the CV.
    """
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=LLMCandidateOutput)

    prompt = ChatPromptTemplate.from_template(CV_PARSING_PROMPT)
    chain = prompt | llm | parser

    result = chain.invoke({
        "cv_text": cv_text,
        "format_instructions": parser.get_format_instructions(),
    })

    if result is None:
        raise ValueError("LLM failed to parse CV into structured format")

    seniority = SeniorityLevel(result.seniority)

    return CandidateProfile(
        skills=[s.lower() for s in result.skills],
        tools_frameworks=[t.lower() for t in result.tools_frameworks],
        seniority=seniority,
        years_experience=result.years_experience,
        domains=[d.lower() for d in result.domains],
        keywords=[k.lower() for k in result.keywords],
        raw_text=cv_text,
    )
