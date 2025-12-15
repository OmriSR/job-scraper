from html.parser import HTMLParser

import spacy

from matchai.config import SPACY_MODEL
from matchai.schemas.job import Job, JobDetail

_nlp = None


def _get_nlp():
    """Lazy load spaCy model with tokenizer, tagger, and lemmatizer only."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
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
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and token.is_alpha
    ]

    return " ".join(tokens)


def _extract_keywords_from_doc(doc) -> list[str]:
    """Extract keywords from a spaCy Doc."""
    keywords = set()
    for token in doc:
        if (
            token.pos_ in ("NOUN", "PROPN", "ADJ")
            and not token.is_stop
            and len(token.text) > 2
        ):  # TODO: the last term will miss "Go" as a skill
            keywords.add(token.lemma_.lower())
    return list(keywords)


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
    return _extract_keywords_from_doc(doc)


def extract_job_keywords_batch(jobs: list[Job]) -> list[list[str]]:
    """Extract keywords from multiple jobs using batch processing.

    Args:
        jobs: List of Job objects to extract keywords from.

    Returns:
        List of keyword lists, one per job (in same order as input).
    """
    texts = [extract_details_text(job.details) for job in jobs]
    results: list[list[str]] = [[] for _ in jobs]

    non_empty_indices = [i for i, text in enumerate(texts) if text]
    non_empty_texts = [texts[i] for i in non_empty_indices]

    nlp = _get_nlp()
    docs = nlp.pipe(non_empty_texts)

    for i, doc in zip(non_empty_indices, docs):
        results[i] = _extract_keywords_from_doc(doc)

    return results
