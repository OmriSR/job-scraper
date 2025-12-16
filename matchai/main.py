"""MatchAI CLI - Local job matching system."""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from matchai.config import DB_PATH, DEFAULT_TOP_N
from matchai.cv.extractor import extract_text_from_pdf
from matchai.cv.parser import parse_cv
from matchai.explainer.generator import find_missing_skills, generate_explanation
from matchai.jobs.database import get_all_jobs, get_jobs
from matchai.jobs.embeddings import embed_candidate
from matchai.jobs.ingest import ingest_from_api, load_companies_from_file
from matchai.matching.filter import apply_filters
from matchai.matching.ranker import rank_jobs
from matchai.schemas.match import MatchResult
from matchai.utils import OllamaUnavailableError, check_ollama_available

app = typer.Typer(help="MatchAI - Local job matching based on CV analysis")
console = Console()


@app.command()
def ingest(
    companies: Path = typer.Option(
        ..., "--companies", "-c", help="Path to companies JSON file with API credentials"
    ),
) -> None:
    """Ingest jobs from Comeet API using company credentials.

    First loads companies from JSON file, then fetches jobs from Comeet API.
    """
    console.print("[bold cyan]Loading companies and fetching jobs from API...[/bold cyan]")

    if not companies.exists():
        console.print(f"[red]Error: Companies file not found: {companies}[/red]")
        raise typer.Exit(1)

    try:
        # Load companies
        console.print(f"  Loading companies from {companies}...")
        companies_added = load_companies_from_file(companies)
        console.print(f"  Added {companies_added} new companies")

        # Fetch jobs from API
        console.print("  Fetching jobs from Comeet API...")
        stats = ingest_from_api()

        console.print("\n[bold green]Ingestion complete![/bold green]")
        console.print(f"  Companies processed: {stats['companies_processed']}")
        console.print(f"  Jobs fetched: {stats['jobs_fetched']}")
        console.print(f"  Jobs inserted: {stats['jobs_inserted']}")
        console.print(f"  Jobs skipped: {stats['jobs_skipped']}")
        console.print(f"  Jobs embedded: {stats['jobs_embedded']}")

    except Exception as e:
        console.print(f"[red]Error during ingestion: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def match(
    cv: Path = typer.Option(..., "--cv", "-c", help="Path to CV PDF file"),
    location: Optional[str] = typer.Option(
        None, "--location", "-l", help="Filter by job location"
    ),
    top_n: int = typer.Option(
        DEFAULT_TOP_N, "--top-n", "-n", help="Number of top matches to return"
    ),
    output_json: bool = typer.Option(
        False, "--json", help="Output results as JSON instead of pretty format"
    ),
) -> None:
    """Match CV against jobs and display top matches."""
    console.print(f"[bold cyan]Processing CV: {cv}[/bold cyan]")

    if not cv.exists():
        console.print(f"[red]Error: CV file not found: {cv}[/red]")
        raise typer.Exit(1)

    if not DB_PATH.exists():
        console.print(
            "[red]Error: Database not found. Run 'matchai ingest' first.[/red]"
        )
        raise typer.Exit(1)

    # Check Ollama availability before starting expensive operations
    try:
        check_ollama_available()
    except OllamaUnavailableError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    try:
        # Extract and parse CV
        console.print("  Extracting text from PDF...")
        cv_text = extract_text_from_pdf(cv)

        console.print("  Parsing CV with LLM...")
        candidate = parse_cv(cv_text)

        console.print("  Embedding candidate profile...")
        candidate_embedding = embed_candidate(candidate)

        # Load jobs with database-level filtering
        console.print("  Loading jobs from database...")
        jobs_from_db = get_jobs(location=location)

        if not jobs_from_db:
            console.print("[yellow]No jobs found matching your location filter.[/yellow]")
            raise typer.Exit(0)

        # Apply skill and seniority filters (in-memory, more complex logic)
        console.print("  Applying skill and seniority filters...")
        filtered_jobs = apply_filters(
            jobs_from_db, candidate, location=None  # location already filtered at DB
        )

        if not filtered_jobs:
            console.print(
                "[yellow]No jobs match your skills/seniority after filtering.[/yellow]"
            )
            raise typer.Exit(0)

        # Rank jobs
        console.print("  Ranking jobs by similarity...")
        ranked_results = rank_jobs(filtered_jobs, candidate, candidate_embedding)

        # Get top N
        top_matches = ranked_results[:top_n]

        # Generate explanations
        console.print("  Generating match explanations...")
        matches: list[MatchResult] = []
        for job, similarity_score, filter_score, final_score in top_matches:
            explanation = generate_explanation(
                job, candidate, similarity_score, filter_score
            )
            missing_skills = find_missing_skills(job, candidate)

            matches.append(
                MatchResult(
                    job=job,
                    similarity_score=similarity_score,
                    filter_score=filter_score,
                    final_score=final_score,
                    explanation=explanation,
                    missing_skills=missing_skills,
                )
            )

        # Output results
        if output_json:
            _output_json(matches)
        else:
            _output_pretty(matches)

    except Exception as e:
        console.print(f"[red]Error during matching: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """Display system information and database stats."""
    console.print("[bold cyan]MatchAI System Information[/bold cyan]\n")

    # Database status
    if not DB_PATH.exists():
        console.print("[yellow]Database not found. Run 'matchai ingest' first.[/yellow]")
        raise typer.Exit(0)

    try:
        all_jobs = get_all_jobs()
        job_count = len(all_jobs)

        companies = {job.company_name for job in all_jobs if job.company_name}
        company_count = len(companies)

        locations = {job.location for job in all_jobs if job.location}
        location_count = len(locations)

        table = Table(title="Database Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Database Path", str(DB_PATH))
        table.add_row("Total Jobs", str(job_count))
        table.add_row("Unique Companies", str(company_count))
        table.add_row("Unique Locations", str(location_count))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error reading database: {e}[/red]")
        raise typer.Exit(1)


def _output_json(matches: list[MatchResult]) -> None:
    """Output matches as JSON to stdout."""
    output = [match.model_dump(mode="json") for match in matches]
    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")


def _output_pretty(matches: list[MatchResult]) -> None:
    """Output matches in pretty console format."""
    console.print(f"\n[bold green]Found {len(matches)} top matches![/bold green]\n")

    for i, match in enumerate(matches, 1):
        job = match.job

        # Create header
        header = f"[bold]#{i} {job.name}[/bold]"
        if job.company_name:
            header += f" at {job.company_name}"

        # Create content
        content = []

        # Location and scores
        if job.location:
            content.append(f"[cyan]Location:[/cyan] {job.location}")

        content.append(
            f"[cyan]Match Score:[/cyan] {match.final_score:.1%} "
            f"(Similarity: {match.similarity_score:.1%}, "
            f"Skills: {match.filter_score:.1%})"
        )

        # Explanation
        content.append("\n[cyan]Why it's a match:[/cyan]")
        for point in match.explanation:
            content.append(f"  • {point}")

        # Missing skills
        if match.missing_skills:
            content.append(f"\n[cyan]Skills to develop:[/cyan]")
            content.append(f"  {', '.join(match.missing_skills[:5])}")
            if len(match.missing_skills) > 5:
                content.append(f"  ... and {len(match.missing_skills) - 5} more")

        # Job link
        if job.link:
            content.append(f"\n[cyan]Apply:[/cyan] {job.link}")

        panel = Panel(
            "\n".join(content),
            title=header,
            border_style="green" if i == 1 else "blue",
        )
        console.print(panel)
        console.print()


if __name__ == "__main__":
    app()
