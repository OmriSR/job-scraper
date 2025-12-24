"""Job database operations with support for SQLite (local) and PostgreSQL (cloud)."""

import json
import sqlite3
from typing import Any

import psycopg2

from matchai.db.connection import get_connection, init_tables
from matchai.schemas.job import Company, Job, JobDetail


def init_database() -> None:
    """Initialize the database with required tables.

    Delegates to the db module which handles both SQLite and PostgreSQL.
    """
    init_tables()


def insert_jobs_to_db(jobs: list[Job]) -> int:
    """Insert jobs into the database.

    Args:
        jobs: List of Job objects to insert.

    Returns:
        Number of jobs inserted (excludes duplicates).
    """
    inserted = 0

    with get_connection() as db:
        cursor = db.cursor()
        ph = db.placeholder

        for job in jobs:
            details_json = json.dumps([d.model_dump() for d in job.details])

            try:
                cursor.execute(
                    f"""
                    INSERT INTO jobs (
                        uid, name, department, email, email_alias,
                        url_comeet_hosted_page, url_recruit_hosted_page, url_active_page,
                        employment_type, experience_level, location,
                        internal_use_custom_id, is_consent_needed, referrals_reward,
                        is_reward, is_company_reward, company_referrals_reward,
                        url_detected_page, picture_url, time_updated,
                        company_name, is_internal, linkedin_job_posting_id,
                        workplace_type, position_url, details
                    ) VALUES ({", ".join([ph] * 26)})
                    """,
                    (
                        job.uid, job.name, job.department, job.email, job.email_alias,
                        job.url_comeet_hosted_page, job.url_recruit_hosted_page,
                        job.url_active_page, job.employment_type,
                        job.experience_level, job.location,
                        job.internal_use_custom_id, job.is_consent_needed, job.referrals_reward,
                        job.is_reward, job.is_company_reward, job.company_referrals_reward,
                        job.url_detected_page, job.picture_url, job.time_updated,
                        job.company_name, job.is_internal, job.linkedin_job_posting_id,
                        job.workplace_type, job.position_url, details_json,
                    ),
                )
                inserted += 1
            except (sqlite3.IntegrityError, psycopg2.IntegrityError):
                pass  # Job already exists

        db.commit()

    return inserted


def insert_companies(companies: list[Company]) -> int:
    """Insert companies into the database.

    Args:
        companies: List of Company objects to insert.

    Returns:
        Number of companies inserted (excludes duplicates).
    """
    inserted = 0

    with get_connection() as db:
        cursor = db.cursor()
        ph = db.placeholder

        for company in companies:
            try:
                cursor.execute(
                    f"INSERT INTO companies (uid, name, token, extracted_from) "
                    f"VALUES ({ph}, {ph}, {ph}, {ph})",
                    (company.uid, company.name, company.token, company.extracted_from),
                )
                inserted += 1
            except (sqlite3.IntegrityError, psycopg2.IntegrityError):
                pass  # Company already exists

        db.commit()

    return inserted


def get_all_jobs() -> list[Job]:
    """Retrieve all jobs from the database."""
    with get_connection() as db:
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM jobs")
        rows = cursor.fetchall()

    return [_row_to_job(row, db.is_postgres) for row in rows]


def get_jobs_by_uids(uids: list[str]) -> list[Job]:
    """Retrieve jobs by their UIDs.

    Args:
        uids: List of job UIDs to retrieve.

    Returns:
        List of matching Job objects.
    """
    if not uids:
        return []

    with get_connection() as db:
        cursor = db.cursor(dictionary=True)
        ph = db.placeholder
        placeholders = ",".join([ph] * len(uids))
        cursor.execute(f"SELECT * FROM jobs WHERE uid IN ({placeholders})", uids)
        rows = cursor.fetchall()

    return [_row_to_job(row, db.is_postgres) for row in rows]


def get_job_by_uid(uid: str) -> Job | None:
    """Retrieve a single job by its UID.

    Args:
        uid: The job UID to retrieve.

    Returns:
        Job object if found, None otherwise.
    """
    with get_connection() as db:
        cursor = db.cursor(dictionary=True)
        ph = db.placeholder
        cursor.execute(f"SELECT * FROM jobs WHERE uid = {ph}", (uid,))
        row = cursor.fetchone()

    if row is None:
        return None

    return _row_to_job(row, db.is_postgres)


def get_jobs(
    location: str | None = None,
    seniority_level: str | None = None,
) -> list[Job]:
    """Retrieve jobs from database with optional filters.

    This function pushes filtering down to the database level to avoid
    loading all jobs into memory.

    Args:
        location: Optional location filter (case-insensitive partial match).
        seniority_level: Optional seniority level filter (exact match on experience_level).

    Returns:
        List of matching Job objects.
    """
    with get_connection() as db:
        cursor = db.cursor(dictionary=True)
        ph = db.placeholder

        query = "SELECT * FROM jobs"
        params: list[str] = []
        conditions: list[str] = []

        if location:
            conditions.append(f"location LIKE {ph}")
            params.append(f"%{location}%")

        if seniority_level:
            conditions.append(f"experience_level = {ph}")
            params.append(seniority_level)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor.execute(query, params)
        rows = cursor.fetchall()

    return [_row_to_job(row, db.is_postgres) for row in rows]


def get_existing_job_uids() -> set[str]:
    """Get all existing job UIDs for idempotency checks."""
    with get_connection() as db:
        cursor = db.cursor()
        cursor.execute("SELECT uid FROM jobs")
        rows = cursor.fetchall()

    return {row[0] for row in rows}


def get_all_companies() -> list[Company]:
    """Retrieve all companies from the database."""
    with get_connection() as db:
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM companies")
        rows = cursor.fetchall()

    return [
        Company(
            name=row["name"],
            uid=row["uid"],
            token=row["token"],
            extracted_from=row["extracted_from"],
        )
        for row in rows
    ]


def _row_to_job(row: Any, is_postgres: bool = False) -> Job:
    """Convert a database row to a Job object.

    Args:
        row: Database row (dict-like for both SQLite Row and psycopg2 RealDictRow).
        is_postgres: Whether the row comes from PostgreSQL (affects JSON handling).

    Returns:
        Job object populated from the row data.
    """
    # Handle details field - JSONB in Postgres is already parsed, TEXT in SQLite needs parsing
    details_data = row["details"]
    if details_data is None:
        details = []
    elif isinstance(details_data, str):
        details = [JobDetail(**d) for d in json.loads(details_data)]
    else:
        # Already parsed (PostgreSQL JSONB)
        details = [JobDetail(**d) for d in details_data]

    return Job(
        uid=row["uid"],
        name=row["name"],
        department=row["department"],
        email=row["email"],
        email_alias=row["email_alias"],
        url_comeet_hosted_page=row["url_comeet_hosted_page"],
        url_recruit_hosted_page=row["url_recruit_hosted_page"],
        url_active_page=row["url_active_page"],
        employment_type=row["employment_type"],
        experience_level=row["experience_level"],
        location=row["location"],
        internal_use_custom_id=row["internal_use_custom_id"],
        is_consent_needed=(
            bool(row["is_consent_needed"]) if row["is_consent_needed"] is not None else None
        ),
        referrals_reward=row["referrals_reward"],
        is_reward=bool(row["is_reward"]) if row["is_reward"] is not None else None,
        is_company_reward=(
            bool(row["is_company_reward"]) if row["is_company_reward"] is not None else None
        ),
        company_referrals_reward=row["company_referrals_reward"],
        url_detected_page=row["url_detected_page"],
        picture_url=row["picture_url"],
        time_updated=row["time_updated"],
        company_name=row["company_name"],
        is_internal=bool(row["is_internal"]) if row["is_internal"] is not None else None,
        linkedin_job_posting_id=row["linkedin_job_posting_id"],
        workplace_type=row["workplace_type"],
        position_url=row["position_url"],
        details=details,
    )
