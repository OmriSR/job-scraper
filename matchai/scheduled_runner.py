"""Cloud Run Job entrypoint for scheduled execution (runs twice daily).

This module orchestrates the full MatchAI pipeline when triggered by Cloud Scheduler:

1. INGEST: Fetch new job listings from Comeet API → store in Supabase + Pinecone
2. MATCH: Run matching for stored candidate CV(s) → filter, rank, generate explanations
3. SAVE: Store top match results in Supabase `match_results` table

Results are persisted in the database and can be accessed later via:
- CLI: `matchai get-results --limit 10`
- Direct DB query: SELECT * FROM match_results ORDER BY created_at DESC
"""

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main scheduled job execution.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    start_time = time.time()
    logger.info("Starting MatchAI scheduled runner")

    try:
        # Initialize database tables if needed
        from matchai.db.connection import init_tables

        logger.info("Initializing database tables")
        init_tables()

        # Step 1: Ingest new job listings from Comeet API
        from matchai.services.ingest_service import ingest_jobs

        logger.info("Step 1/3: Ingesting jobs from Comeet API")
        ingest_stats = ingest_jobs()
        logger.info(f"Ingestion complete: {ingest_stats}")

        # Step 2: Run matching for stored CV
        from matchai.services.match_service import run_scheduled_matching

        logger.info("Step 2/3: Running matching for stored CV")
        match_stats, matches = run_scheduled_matching()
        logger.info(f"Matching complete: {match_stats}")

        # Step 3: Results are saved to database by run_scheduled_matching()
        logger.info("Step 3/3: Results saved to match_results table")

        # Step 4: Send email notification
        from matchai.services.email_service import send_match_results_email

        if matches:
            logger.info("Sending email notification...")
            email_sent = send_match_results_email(matches)
            if email_sent:
                logger.info(f"Email sent with {len(matches)} matches")
            else:
                logger.warning("Email notification failed or disabled")
        else:
            logger.info("No matches to send via email")

        elapsed = time.time() - start_time
        logger.info(f"Scheduled runner completed successfully in {elapsed:.2f}s")
        return 0

    except Exception as e:
        logger.exception(f"Scheduled runner failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
