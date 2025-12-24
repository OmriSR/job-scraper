"""Ingest service for fetching and storing job listings.

This service wraps the ingestion logic for use in scheduled jobs and CLI.
"""

import logging
from pathlib import Path

from matchai.jobs.ingest import ingest_from_api, load_companies_from_file

logger = logging.getLogger(__name__)


def ingest_jobs() -> dict:
    """Ingest jobs from all configured companies.

    Fetches job listings from Comeet API for all companies stored in the database,
    then stores new jobs in the database and vector store.

    Returns:
        Dict with ingestion statistics:
        - companies_processed: Number of companies queried
        - jobs_fetched: Total jobs returned from API
        - jobs_skipped: Jobs that already existed
        - jobs_inserted: New jobs added to database
        - jobs_embedded: New jobs added to vector store
    """
    logger.info("Starting job ingestion from Comeet API")
    stats = ingest_from_api()
    logger.info(f"Ingestion complete: {stats}")
    return stats


def ingest_companies(file_path: Path) -> int:
    """Ingest companies from a JSON file.

    Args:
        file_path: Path to JSON file containing company credentials.

    Returns:
        Number of companies added.
    """
    logger.info(f"Loading companies from {file_path}")
    count = load_companies_from_file(file_path)
    logger.info(f"Loaded {count} new companies")
    return count
