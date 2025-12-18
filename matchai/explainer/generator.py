"""LLM-based explanation generation for job matches."""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from matchai.config import SKILL_MATCH_THRESHOLD
from matchai.jobs.preprocessor import extract_details_text, extract_job_keywords
from matchai.schemas.candidate import CandidateProfile
from matchai.schemas.job import Job
from matchai.utils import get_llm

EXPLANATION_PROMPT = """\
You are a career advisor helping a job seeker understand why a position might be a good match.

CANDIDATE PROFILE:
Skills: {candidate_skills}
Tools/Frameworks: {candidate_tools}
Domains: {candidate_domains}
Years of Experience: {years_experience}

JOB POSITION:
Title: {job_title}
Company: {company_name}
Location: {job_location}
Description: {job_description}

MATCH SCORES:
Similarity Score: {similarity_score:.0%}
Skill Match Score: {filter_score:.0%}

Generate 2-3 concise bullet points explaining why this job is a good match for the candidate.
Focus on:
- Matching skills and technologies
- Relevant domain experience
- Career growth opportunities

{format_instructions}\
"""

REFINE_SKILLS_PROMPT = """\
You are a career advisor helping candidates prepare for job interviews.

CANDIDATE PROFILE:
- Skills: {candidate_skills}
- Tools/Frameworks: {candidate_tools}
- Domains: {candidate_domains}
- Seniority: {candidate_seniority}

JOB POSITION:
- Title: {job_title}
- Company: {company_name}

RAW MISSING SKILLS (from keyword extraction):
{raw_missing_skills}

TASKS:
1. FILTER & EDIT the missing skills list:
   - Remove noise (generic words, duplicates, irrelevant terms)
   - Rephrase abbreviations/acronyms to full names (e.g., "k8s" → "Kubernetes")
   - Keep only genuine technical skills or requirements
   - Maximum 5-7 most important skills

2. Provide 1-2 INTERVIEW PREPARATION TIPS:
   - Actionable advice on what to strengthen or prepare
   - Focus on the gap between candidate profile and job requirements
   - Be specific and practical

{format_instructions}\
"""


class ExplanationOutput(BaseModel):
    """Intermediate schema for LLM explanation generation output.

    This schema defines the structure that the LLM should return when generating
    match explanations. The bullet_points field is extracted and returned
    directly by generate_explanation().
    """

    bullet_points: list[str] = Field(
        description="2-3 concise bullet points explaining the match",
        min_length=2,
        max_length=3,
    )


class RefinedSkillsOutput(BaseModel):
    """LLM output for refined missing skills and interview tips."""

    refined_skills: list[str] = Field(
        description="Filtered and rephrased missing skills (max 5-7 items)"
    )
    interview_tips: list[str] = Field(
        description="1-2 actionable tips for interview preparation"
    )


def generate_explanation(
    job: Job,
    candidate: CandidateProfile,
    similarity_score: float,
    filter_score: float,
) -> list[str]:
    """Generate explanation bullet points for a job match using LLM.

    Args:
        job: Matched job position.
        candidate: Candidate profile.
        similarity_score: Semantic similarity score (0-1).
        filter_score: Deterministic filter score (0-1).

    Returns:
        List of 2-3 explanation bullet points.

    Raises:
        ValueError: If LLM fails to generate explanation.
    """
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=ExplanationOutput)

    prompt = ChatPromptTemplate.from_template(EXPLANATION_PROMPT)
    chain = prompt | llm | parser

    job_description = extract_details_text(job.details)
    if len(job_description) > 2000:
        job_description = job_description[:2000] + "..."

    result = chain.invoke({
        "candidate_skills": ", ".join(candidate.skills),
        "candidate_tools": ", ".join(candidate.tools_frameworks),
        "candidate_domains": ", ".join(candidate.domains),
        "years_experience": candidate.years_experience or "Unknown",
        "job_title": job.name,
        "company_name": job.company_name or "Unknown",
        "job_location": job.location or "Not specified",
        "job_description": job_description,
        "similarity_score": similarity_score,
        "filter_score": filter_score,
        "format_instructions": parser.get_format_instructions(),
    })

    if result is None:
        raise ValueError("LLM failed to generate explanation")

    return result.bullet_points


def find_missing_skills(job: Job, candidate: CandidateProfile) -> list[str]:
    """Identify skills required by job that candidate doesn't have.

    Uses deterministic text matching with RapidFuzz to find job skills
    not present in the candidate's profile.

    Args:
        job: Job position to extract keywords from.
        candidate: Candidate profile with skills, tools, and domains.

    Returns:
        List of missing skills/technologies from the job requirements.
    """
    job_keywords = extract_job_keywords(job)
    if not job_keywords:
        return []

    candidate_terms = {
        term.lower()
        for skill_list in [
            candidate.skills,
            candidate.tools_frameworks,
            candidate.domains,
        ]
        for term in skill_list
    }

    missing_skills = [
        keyword
        for keyword in job_keywords
        if not any(
            fuzz.ratio(keyword.lower(), candidate_term) >= SKILL_MATCH_THRESHOLD
            for candidate_term in candidate_terms
        )
    ]

    return missing_skills


def refine_skills_and_tips(
    candidate: CandidateProfile,
    job: Job,
    raw_missing_skills: list[str],
) -> tuple[list[str], list[str]]:
    """Use LLM to filter/edit missing skills and generate interview tips.

    Args:
        candidate: Candidate profile with skills and experience.
        job: Job position to prepare for.
        raw_missing_skills: Raw missing skills from keyword extraction.

    Returns:
        Tuple of (refined_skills, interview_tips).

    Raises:
        ValueError: If LLM fails to generate output.
    """
    if not raw_missing_skills:
        return [], []

    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=RefinedSkillsOutput)

    prompt = ChatPromptTemplate.from_template(REFINE_SKILLS_PROMPT)
    chain = prompt | llm | parser

    result = chain.invoke({
        "candidate_skills": ", ".join(candidate.skills),
        "candidate_tools": ", ".join(candidate.tools_frameworks),
        "candidate_domains": ", ".join(candidate.domains),
        "candidate_seniority": candidate.seniority or "Not specified",
        "job_title": job.name,
        "company_name": job.company_name or "Unknown",
        "raw_missing_skills": ", ".join(raw_missing_skills),
        "format_instructions": parser.get_format_instructions(),
    })

    if result is None:
        raise ValueError("LLM failed to refine skills and generate tips")

    return result.refined_skills, result.interview_tips
