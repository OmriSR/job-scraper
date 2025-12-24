"""Shared pytest fixtures for all tests."""

from unittest.mock import patch

import pytest


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing.

    Patches DB_PATH and DATA_DIR at the connection module level.
    """
    db_path = tmp_path / "test.db"
    data_dir = tmp_path

    # Patch at db.connection where they're used at runtime
    with (
        patch("matchai.db.connection.DB_PATH", db_path),
        patch("matchai.db.connection.DATA_DIR", data_dir),
        patch("matchai.db.connection.DATABASE_URL", None),  # Force SQLite
    ):
        from matchai.db.connection import init_tables

        init_tables()
        yield db_path


@pytest.fixture
def temp_chroma(tmp_path):
    """Create a temporary ChromaDB for testing.

    Patches CHROMA_PATH, DATA_DIR, and resets module-level clients
    to force re-initialization with the new path.
    """
    chroma_path = tmp_path / "chroma_db"
    data_dir = tmp_path

    with (
        patch("matchai.jobs.embeddings.CHROMA_PATH", chroma_path),
        patch("matchai.jobs.embeddings.DATA_DIR", data_dir),
        patch("matchai.jobs.embeddings._collection", None),
        patch("matchai.jobs.embeddings._chroma_client", None),
    ):
        yield chroma_path


@pytest.fixture
def temp_db_and_chroma(temp_db, temp_chroma):
    """Create both temporary database and ChromaDB for integration tests.

    Combines temp_db and temp_chroma fixtures.
    """
    yield {"db_path": temp_db, "chroma_path": temp_chroma}
