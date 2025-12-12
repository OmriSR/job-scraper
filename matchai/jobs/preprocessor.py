from html.parser import HTMLParser

import spacy

from matchai.config import SPACY_MODEL
from matchai.schemas.job import Job, JobDetail

_nlp = None


def _get_nlp():
    """Lazy load spaCy model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(SPACY_MODEL)
    return _nlp


class HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML content."""

    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data: str) -> None:
        self.text_parts.append(data)

    def get_text(self) -> str:
        return " ".join(self.text_parts)


def strip_html(html: str) -> str:
    """Remove HTML tags and return plain text."""
    if not html:
        return ""
    parser = HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


def extract_details_text(details: list[JobDetail]) -> str:
    """Extract and concatenate text from job details.

    Args:
        details: List of JobDetail objects containing HTML content.

    Returns:
        Cleaned plain text from all detail sections.
    """
    sorted_details = sorted(details, key=lambda d: d.order)
    text_parts = []

    for detail in sorted_details:
        if detail.value:
            section_text = strip_html(detail.value)
            if section_text.strip():
                text_parts.append(f"{detail.name}: {section_text}")

    return "\n\n".join(text_parts)


def preprocess_job(job: Job) -> str:
    """Preprocess job details for embedding.

    Args:
        job: Job object to preprocess.

    Returns:
        Cleaned and lemmatized text from job details.
    """
    raw_text = extract_details_text(job.details)

    if not raw_text:
        return ""

    nlp = _get_nlp()
    doc = nlp(raw_text.lower())

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha
    ]

    return " ".join(tokens)


def extract_job_keywords(job: Job) -> list[str]:
    """Extract keywords from job details.

    Args:
        job: Job object to extract keywords from.

    Returns:
        List of unique keywords (nouns, proper nouns, adjectives).
    """
    raw_text = extract_details_text(job.details)

    if not raw_text:
        return []

    nlp = _get_nlp()
    doc = nlp(raw_text)

    keywords = set()
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN", "ADJ") and not token.is_stop and len(token.text) > 2:
            keywords.add(token.lemma_.lower())

    return list(keywords)
