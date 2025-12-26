"""Database connection factory for SQLite (local) and PostgreSQL (cloud)."""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

from matchai.config import DATA_DIR, DATABASE_URL, DB_PATH


class DatabaseConnection:
    """Wrapper for database connections that provides a consistent interface."""

    def __init__(self, conn: Any, is_postgres: bool = False):
        self.conn = conn
        self.is_postgres = is_postgres
        self._cursor = None

    def cursor(self, dictionary: bool = False) -> Any:
        """Get a cursor for executing database operations.

        A cursor is a database object used to traverse and manipulate query results.
        In PostgreSQL, cursors manage the context of a fetch operation - they maintain
        the position within the result set and handle row-by-row retrieval.

        Args:
            dictionary: If True, rows are returned as dictionaries (column_name: value)
                instead of tuples. In PostgreSQL this uses RealDictCursor which returns
                rows as dict-like objects. In SQLite this uses Row factory. This is
                useful when you need to access columns by name rather than index.

        Returns:
            Database cursor object for executing queries and fetching results.
        """
        if self.is_postgres:
            self._cursor = self.conn.cursor(cursor_factory=RealDictCursor if dictionary else None)
        else:
            self._cursor = self.conn.cursor()
            if dictionary:
                self.conn.row_factory = sqlite3.Row
        return self._cursor

    def execute(self, query: str, params: tuple | list | None = None) -> Any:
        """Execute a query."""
        cursor = self.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor

    def fetchall(self) -> list:
        """Fetch all results from last query."""
        if self._cursor:
            return self._cursor.fetchall()
        return []

    def fetchone(self) -> Any:
        """Fetch one result from last query."""
        if self._cursor:
            return self._cursor.fetchone()
        return None

    def commit(self) -> None:
        """Commit the transaction."""
        self.conn.commit()

    def close(self) -> None:
        """Close the connection."""
        if self._cursor:
            self._cursor.close()
        self.conn.close()

    @property
    def placeholder(self) -> str:
        """Return the parameter placeholder for this database."""
        return "%s" if self.is_postgres else "?"


@contextmanager
def get_connection() -> Generator[DatabaseConnection, None, None]:
    """Get a database connection.

    Uses PostgreSQL if DATABASE_URL is set, otherwise falls back to SQLite.

    Yields:
        DatabaseConnection wrapper with consistent interface.
    """
    if DATABASE_URL:
        conn = psycopg2.connect(DATABASE_URL)
        db = DatabaseConnection(conn, is_postgres=True)
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        db = DatabaseConnection(conn, is_postgres=False)

    try:
        yield db
    finally:
        db.close()


def init_tables() -> None:
    """Initialize database tables.

    Creates all required tables if they don't exist.
    Uses appropriate syntax for PostgreSQL or SQLite.
    """
    with get_connection() as db:
        if db.is_postgres:
            _init_postgres_tables(db)
        else:
            _init_sqlite_tables(db)
        db.commit()


def _init_postgres_tables(db: DatabaseConnection) -> None:
    """Create PostgreSQL tables."""
    cursor = db.cursor()

    # Jobs table
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
            is_consent_needed BOOLEAN,
            referrals_reward TEXT,
            is_reward BOOLEAN,
            is_company_reward BOOLEAN,
            company_referrals_reward TEXT,
            url_detected_page TEXT,
            picture_url TEXT,
            time_updated TEXT,
            company_name TEXT,
            is_internal BOOLEAN,
            linkedin_job_posting_id TEXT,
            workplace_type TEXT,
            position_url TEXT,
            details JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            embedded_at TIMESTAMPTZ
        )
    """)

    # Add embedded_at column if it doesn't exist (migration for existing tables)
    cursor.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'jobs' AND column_name = 'embedded_at'
            ) THEN
                ALTER TABLE jobs ADD COLUMN embedded_at TIMESTAMPTZ;
            END IF;
        END $$;
    """)

    # Companies table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            uid TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            token TEXT NOT NULL,
            extracted_from TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # Candidates table (CV caching)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            cv_hash TEXT PRIMARY KEY,
            profile_json JSONB NOT NULL,
            raw_text TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            last_used_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # Match results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS match_results (
            id SERIAL PRIMARY KEY,
            cv_hash TEXT REFERENCES candidates(cv_hash),
            job_uid TEXT REFERENCES jobs(uid),
            similarity_score FLOAT,
            filter_score FLOAT,
            final_score FLOAT,
            explanation JSONB,
            missing_skills TEXT[],
            interview_tips TEXT[],
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # Index for faster match result lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_match_results_cv_hash
        ON match_results(cv_hash)
    """)


def _init_sqlite_tables(db: DatabaseConnection) -> None:
    """Create SQLite tables (for local development/testing)."""
    cursor = db.cursor()

    # Jobs table
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
            details TEXT,
            embedded_at TEXT
        )
    """)

    # Add embedded_at column if it doesn't exist (migration for existing tables)
    cursor.execute("""
        SELECT COUNT(*) FROM pragma_table_info('jobs') WHERE name='embedded_at'
    """)
    if cursor.fetchone()[0] == 0:
        cursor.execute("ALTER TABLE jobs ADD COLUMN embedded_at TEXT")

    # Companies table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            uid TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            token TEXT NOT NULL,
            extracted_from TEXT NOT NULL
        )
    """)

    # Candidates table (CV caching)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            cv_hash TEXT PRIMARY KEY,
            profile_json TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_used_at TEXT NOT NULL
        )
    """)

    # Match results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS match_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cv_hash TEXT REFERENCES candidates(cv_hash),
            job_uid TEXT REFERENCES jobs(uid),
            similarity_score REAL,
            filter_score REAL,
            final_score REAL,
            explanation TEXT,
            missing_skills TEXT,
            interview_tips TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
