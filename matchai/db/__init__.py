"""Database connection module."""

from matchai.db.candidates import (
    compute_cv_hash,
    get_candidate,
    get_candidate_by_hash,
    get_match_results,
    save_candidate,
    save_match_results,
)
from matchai.db.connection import get_connection, init_tables

__all__ = [
    "get_connection",
    "init_tables",
    "compute_cv_hash",
    "save_candidate",
    "get_candidate",
    "get_candidate_by_hash",
    "save_match_results",
    "get_match_results",
]
