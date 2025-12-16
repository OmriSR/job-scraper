import json
import sqlite3

from matchai.config import DATA_DIR, DB_PATH
from matchai.schemas.job import Company, Job, JobDetail


def init_database() -> None:
    """Initialize the SQLite database with required tables."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                uid TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT,
                email TEXT,
                email_alias TEXT,
                url_comeet_hosted_page TEXT,
                url_recruit_hosted_page TEXT,
                url_active_page TEXT,
                employment_type TEXT,
                experience_level TEXT,
                location TEXT,
                internal_use_custom_id TEXT,
                is_consent_needed INTEGER,
                referrals_reward TEXT,
                is_reward INTEGER,
                is_company_reward INTEGER,
                company_referrals_reward TEXT,
                url_detected_page TEXT,
                picture_url TEXT,
                time_updated TEXT,
                company_name TEXT,
                is_internal INTEGER,
                linkedin_job_posting_id TEXT,
                workplace_type TEXT,
                position_url TEXT,
                details TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                uid TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                token TEXT NOT NULL,
                extracted_from TEXT NOT NULL
            )
        """)

        conn.commit()


def insert_jobs_to_db(jobs: list[Job]) -> int:
    """Insert jobs into the database.

    Args:
        jobs: List of Job objects to insert.

    Returns:
        Number of jobs inserted (excludes duplicates).
    """
    inserted = 0

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        for job in jobs:
            details_json = json.dumps([d.model_dump() for d in job.details])

            try:
                cursor.execute(
                    """
                    INSERT INTO jobs (
                        uid, name, department, email, email_alias,
                        url_comeet_hosted_page, url_recruit_hosted_page, url_active_page,
                        employment_type, experience_level, location,
                        internal_use_custom_id, is_consent_needed, referrals_reward,
                        is_reward, is_company_reward, company_referrals_reward,
                        url_detected_page, picture_url, time_updated,
                        company_name, is_internal, linkedin_job_posting_id,
                        workplace_type, position_url, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.uid, job.name, job.department, job.email, job.email_alias,
                        job.url_comeet_hosted_page, job.url_recruit_hosted_page, job.url_active_page,
                        job.employment_type, job.experience_level, job.location,
                        job.internal_use_custom_id, job.is_consent_needed, job.referrals_reward,
                        job.is_reward, job.is_company_reward, job.company_referrals_reward,
                        job.url_detected_page, job.picture_url, job.time_updated,
                        job.company_name, job.is_internal, job.linkedin_job_posting_id,
                        job.workplace_type, job.position_url, details_json,
                    ),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                pass  # Job already exists

        conn.commit()

    return inserted


def insert_companies(companies: list[Company]) -> int:
    """Insert companies into the database.

    Args:
        companies: List of Company objects to insert.

    Returns:
        Number of companies inserted (excludes duplicates).
    """
    inserted = 0

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        for company in companies:
            try:
                cursor.execute(
                    "INSERT INTO companies (uid, name, token, extracted_from) VALUES (?, ?, ?, ?)",
                    (company.uid, company.name, company.token, company.extracted_from),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                pass  # Company already exists

        conn.commit()

    return inserted


def get_all_jobs() -> list[Job]:
    """Retrieve all jobs from the database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs")
        rows = cursor.fetchall()

    return [_row_to_job(row) for row in rows]


def get_jobs_by_uids(uids: list[str]) -> list[Job]:
    """Retrieve jobs by their UIDs.

    Args:
        uids: List of job UIDs to retrieve.

    Returns:
        List of matching Job objects.
    """
    if not uids:
        return []

    placeholders = ",".join("?" * len(uids))

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM jobs WHERE uid IN ({placeholders})", uids)
        rows = cursor.fetchall()

    return [_row_to_job(row) for row in rows]


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
    query = "SELECT * FROM jobs"
    params: list[str] = []
    conditions: list[str] = []

    if location:
        conditions.append("location LIKE ?")
        params.append(f"%{location}%")

    if seniority_level:
        conditions.append("experience_level = ?")
        params.append(seniority_level)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

    return [_row_to_job(row) for row in rows]


def get_existing_job_uids() -> set[str]:
    """Get all existing job UIDs for idempotency checks."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT uid FROM jobs")
        rows = cursor.fetchall()

    return {row[0] for row in rows}


def get_all_companies() -> list[Company]:
    """Retrieve all companies from the database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
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


def _row_to_job(row: sqlite3.Row) -> Job:
    """Convert a database row to a Job object."""
    details_data = json.loads(row["details"]) if row["details"] else []
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
        is_consent_needed=bool(row["is_consent_needed"]) if row["is_consent_needed"] is not None else None,
        referrals_reward=row["referrals_reward"],
        is_reward=bool(row["is_reward"]) if row["is_reward"] is not None else None,
        is_company_reward=bool(row["is_company_reward"]) if row["is_company_reward"] is not None else None,
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
