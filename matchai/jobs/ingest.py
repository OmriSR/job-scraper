import json
import logging
from pathlib import Path

import requests

from matchai.jobs.database import (
    get_all_companies,
    get_existing_job_uids,
    init_database,
    insert_companies,
    insert_jobs_to_db,
)
from matchai.jobs.embeddings import embed_and_store_jobs, get_existing_embedding_uids
from matchai.schemas.job import Company, Job

logger = logging.getLogger(__name__)


def _normalize_position(pos: dict) -> dict:
    """Normalize position data from Comeet API to match Job schema.

    Args:
        pos: Raw position dict from Comeet API.

    Returns:
        Normalized position dict ready for Job model.
    """
    normalized = pos.copy()

    # Handle location - API returns dict, schema expects string
    # Prefer 'city', fall back to 'name'
    location = normalized.get("location")
    if isinstance(location, dict):
        normalized["location"] = location.get("city") or location.get("name")

    return normalized


def load_companies_from_file(file_path: Path) -> int:
    """Load companies from a JSON file and add them to the database.

    Idempotent: skips companies that already exist.

    Args:
        file_path: Path to JSON file containing list of company dicts.

    Returns:
        Number of companies added.
    """
    init_database()

    with open(file_path) as f:
        companies_data = json.load(f)

    companies = [Company(**c) for c in companies_data]
    inserted = insert_companies(companies)

    logger.info(f"Loaded {inserted} new companies from {file_path}")
    return inserted


def fetch_positions(comeet_uid: str, comeet_token: str) -> list[dict]:
    """Fetch open positions from Comeet API.

    Args:
        comeet_uid: Company UID for Comeet API.
        comeet_token: API token for authentication.

    Returns:
        List of position dicts from the API.
    """
    base_url = f"https://www.comeet.co/careers-api/2.0/company/{comeet_uid}/positions"

    headers = {"User-Agent": "MatchAI/1.0"}
    params = {
        "token": comeet_token,
        "status": "open",
        "details": "true",
    }

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()

        positions = response.json()
        logger.info(f"Fetched {len(positions)} positions from Comeet API")
        return positions

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching positions: {e}")
        return []


def ingest_from_api() -> dict:
    """Fetch jobs from all companies in database and ingest them.

    Uses company credentials stored in the database to fetch jobs
    from the Comeet API, then stores them in SQLite and ChromaDB.

    Idempotent: skips jobs that already exist.

    Returns:
        Dict with ingestion statistics.
    """
    init_database()

    stats = {
        "companies_processed": 0,
        "jobs_fetched": 0,
        "jobs_skipped": 0,
        "jobs_inserted": 0,
        "jobs_embedded": 0,
    }

    companies = get_all_companies()
    if not companies:
        logger.warning("No companies found in database")
        return stats

    existing_db_uids = get_existing_job_uids()
    existing_embedding_uids = get_existing_embedding_uids()

    all_new_jobs = []

    for company in companies:
        stats["companies_processed"] += 1

        positions = fetch_positions(company.uid, company.token)
        stats["jobs_fetched"] += len(positions)

        for pos in positions:
            job_uid = pos.get("uid")
            if job_uid in existing_db_uids:
                stats["jobs_skipped"] += 1
            else:
                normalized_pos = _normalize_position(pos)
                job = Job(**normalized_pos)
                all_new_jobs.append(job)

    if all_new_jobs:
        # Insert into SQLite
        inserted = insert_jobs_to_db(all_new_jobs)
        stats["jobs_inserted"] = inserted

        # Embed and store in ChromaDB
        jobs_to_embed = [job for job in all_new_jobs if job.uid not in existing_embedding_uids]
        if jobs_to_embed:
            embedded = embed_and_store_jobs(jobs_to_embed)
            stats["jobs_embedded"] = embedded

    logger.info(
        f"Ingestion complete: {stats['jobs_inserted']} jobs inserted, "
        f"{stats['jobs_embedded']} embedded, {stats['jobs_skipped']} skipped"
    )

    return stats
