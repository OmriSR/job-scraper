"""MatchAI CLI - Local job matching system."""

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from matchai.config import DATABASE_URL, DB_PATH, DEFAULT_TOP_N
from matchai.cv.extractor import extract_text_from_pdf
from matchai.cv.parser import parse_cv
from matchai.db.candidates import (
    compute_cv_hash,
    get_candidate,
    get_match_results,
    save_candidate,
)
from matchai.db.connection import init_tables
from matchai.explainer.generator import (
    find_missing_skills,
    generate_explanation,
    refine_skills_and_tips,
)
from matchai.jobs.database import (
    get_all_companies,
    get_all_jobs,
    get_jobs,
    insert_companies,
)
from matchai.jobs.ingest import ingest_from_api, load_companies_from_file
from matchai.matching.filter import apply_filters
from matchai.matching.ranker import rank_jobs
from matchai.schemas.job import Company
from matchai.schemas.match import MatchResult
from matchai.utils import LLMConfigurationError, check_llm_configured

app = typer.Typer(help="MatchAI - Local job matching based on CV analysis")
console = Console()


@app.command()
def ingest(
    companies: Path = typer.Option(
        ...,
        "--companies",
        "-c",
        help="Path to companies JSON file with API credentials",
    ),
) -> None:
    """Ingest jobs from Comeet API using company credentials.

    First loads companies from JSON file, then fetches jobs from Comeet API.
    """
    console.print(
        "[bold cyan]Loading companies and fetching jobs from API...[/bold cyan]"
    )

    if not companies.exists():
        console.print(f"[red]Error: Companies file not found: {companies}[/red]")
        raise typer.Exit(1)

    try:
        # Load companies
        console.print(f"  Loading companies from {companies}...")
        companies_added = load_companies_from_file(file_path=companies)
        console.print(f"  Added {companies_added} new companies")

        # Fetch jobs from API
        console.print("  Fetching jobs...")
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
    location: str | None = typer.Option(
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

    # Check LLM configuration before starting expensive operations
    try:
        check_llm_configured()
    except LLMConfigurationError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    try:
        # Extract and parse CV
        console.print("  Extracting text from PDF...")
        cv_text = extract_text_from_pdf(file_path=cv)

        console.print("  Parsing CV with LLM...")
        candidate = parse_cv(cv_text=cv_text)

        # Load jobs with database-level filtering
        console.print("  Loading jobs from database...")
        jobs_from_db = get_jobs(location=location)

        if not jobs_from_db:
            console.print(
                "[yellow]No jobs found matching your location filter.[/yellow]"
            )
            raise typer.Exit(0)

        # Apply skill and seniority filters (in-memory, more complex logic)
        console.print("  Applying skill and seniority filters...")
        filtered_jobs = apply_filters(
            jobs=jobs_from_db,
            candidate=candidate,
            location=None,  # location already filtered at DB
        )

        if not filtered_jobs:
            console.print(
                "[yellow]No jobs match your skills/seniority after filtering.[/yellow]"
            )
            raise typer.Exit(0)

        # Rank jobs
        console.print("  Ranking jobs by similarity...")
        ranked_results = rank_jobs(filtered_jobs=filtered_jobs, candidate=candidate)

        # Get top N
        top_matches = ranked_results[:top_n]

        # Generate explanations
        console.print("  Generating match explanations...")
        for match in top_matches:
            match.explanation = generate_explanation(
                job=match.job,
                candidate=candidate,
                similarity_score=match.similarity_score,
                filter_score=match.filter_score,
            )
            match.missing_skills = find_missing_skills(
                job=match.job, candidate=candidate
            )

        # Refine missing skills and generate interview tips using LLM
        console.print("  Refining skills and generating interview tips...")
        for match in top_matches:
            if match.missing_skills:
                refined_skills, interview_tips = refine_skills_and_tips(
                    candidate=candidate,
                    job=match.job,
                    raw_missing_skills=match.missing_skills,
                )
                match.missing_skills = refined_skills
                match.interview_tips = interview_tips

        # Output results
        if output_json:
            _output_json(matches=top_matches)
        else:
            _output_pretty(matches=top_matches)

    except Exception as e:
        console.print(f"[red]Error during matching: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """Display system information and database stats."""
    console.print("[bold cyan]MatchAI System Information[/bold cyan]\n")

    # Check database availability
    is_cloud = DATABASE_URL is not None
    if not is_cloud and not DB_PATH.exists():
        console.print(
            "[yellow]Database not found. Run 'matchai ingest' first.[/yellow]"
        )
        raise typer.Exit(0)

    try:
        all_jobs = get_all_jobs()
        job_count = len(all_jobs)

        companies = {job.company_name for job in all_jobs if job.company_name}
        company_count = len(companies)

        locations = {job.location for job in all_jobs if job.location}
        location_count = len(locations)

        # Check for stored candidate
        candidate_info = get_candidate()
        has_candidate = candidate_info is not None

        table = Table(title="Database Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        if is_cloud:
            table.add_row("Database", "PostgreSQL (cloud)")
        else:
            table.add_row("Database", str(DB_PATH))
        table.add_row("Total Jobs", str(job_count))
        table.add_row("Unique Companies", str(company_count))
        table.add_row("Unique Locations", str(location_count))
        table.add_row("CV Uploaded", "Yes" if has_candidate else "No")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error reading database: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="upload-cv")
def upload_cv(
    cv: Path = typer.Option(..., "--cv", "-c", help="Path to CV PDF file"),
) -> None:
    """Parse CV locally and upload parsed profile to database.

    Run this locally (not in cloud) to set up your CV for scheduled matching.
    The PDF is parsed on your machine, and only the parsed profile is uploaded.
    Only one CV is stored at a time - uploading a new one replaces the old.
    """
    console.print(f"[bold cyan]Uploading CV: {cv}[/bold cyan]")

    if not cv.exists():
        console.print(f"[red]Error: CV file not found: {cv}[/red]")
        raise typer.Exit(1)

    try:
        check_llm_configured()
    except LLMConfigurationError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    try:
        # Initialize database tables
        console.print("  Initializing database...")
        init_tables()

        # Extract and parse CV
        console.print("  Extracting text from PDF...")
        cv_text = extract_text_from_pdf(file_path=cv)

        console.print("  Parsing CV with LLM...")
        candidate = parse_cv(cv_text=cv_text)

        # Compute hash and save
        cv_hash = compute_cv_hash(cv_text)
        console.print("  Saving to database...")
        save_candidate(cv_hash=cv_hash, profile=candidate, raw_text=cv_text)

        console.print("\n[bold green]CV uploaded successfully![/bold green]")
        console.print(f"  Hash: {cv_hash[:16]}...")
        console.print(f"  Name: {candidate.name or 'Not detected'}")
        console.print(f"  Skills: {len(candidate.skills)} detected")

    except Exception as e:
        console.print(f"[red]Error uploading CV: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command(name="add-company")
def add_company(
    name: str = typer.Option(..., "--name", "-n", help="Company name"),
    uid: str = typer.Option(..., "--uid", "-u", help="Comeet company UID"),
    token: str = typer.Option(..., "--token", "-t", help="Comeet API token"),
) -> None:
    """Add a company to the database for job fetching.

    Run this locally to register companies for scheduled job fetching.
    The scheduled cloud job will use these credentials to fetch jobs.
    """
    try:
        # Initialize database tables
        init_tables()

        company = Company(name=name, uid=uid, token=token, extracted_from="cli")
        count = insert_companies([company])

        if count > 0:
            console.print(f"[bold green]Company '{name}' added![/bold green]")
        else:
            console.print(f"[yellow]Company '{name}' already exists.[/yellow]")

    except Exception as e:
        console.print(f"[red]Error adding company: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="import-companies")
def import_companies(
    companies_file: Path = typer.Option(
        ..., "--file", "-f", help="Path to companies JSON file"
    ),
) -> None:
    """Import companies from a JSON file.

    The JSON file should contain an array of company objects:
    [{"name": "...", "uid": "...", "token": "...", "extracted_from": "..."}]

    This is the bulk alternative to 'add-company' for importing multiple companies.
    """
    if not companies_file.exists():
        console.print(f"[red]Error: File not found: {companies_file}[/red]")
        raise typer.Exit(1)

    try:
        # Initialize database tables
        init_tables()

        # Load companies from file
        count = load_companies_from_file(file_path=companies_file)

        if count > 0:
            console.print(
                f"[bold green]Imported {count} companies from {companies_file}[/bold green]"
            )
        else:
            console.print("[yellow]No new companies imported (all already exist).[/yellow]")

    except Exception as e:
        console.print(f"[red]Error importing companies: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command(name="list-companies")
def list_companies() -> None:
    """List all companies in the database."""
    try:
        companies = get_all_companies()

        if not companies:
            console.print("[yellow]No companies found.[/yellow]")
            raise typer.Exit(0)

        table = Table(title="Registered Companies")
        table.add_column("Name", style="cyan")
        table.add_column("UID", style="dim")
        table.add_column("Source", style="green")

        for company in companies:
            table.add_row(company.name, company.uid, company.extracted_from)

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error listing companies: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="get-results")
def get_results(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results to show"),
    output_json: bool = typer.Option(
        False, "--json", help="Output results as JSON"
    ),
) -> None:
    """Fetch latest match results from the database.

    Shows results from the most recent scheduled matching run.
    Run this locally after the cloud job has executed.
    """
    try:
        # Check if CV is uploaded
        candidate_info = get_candidate()
        if candidate_info is None:
            console.print(
                "[yellow]No CV uploaded. Run 'matchai upload-cv' first.[/yellow]"
            )
            raise typer.Exit(0)

        cv_hash, _ = candidate_info
        results = get_match_results(cv_hash=cv_hash, limit=limit)

        if not results:
            console.print(
                "[yellow]No match results found. "
                "Wait for the scheduled job to run or run matching manually.[/yellow]"
            )
            raise typer.Exit(0)

        if output_json:
            _output_json(matches=results)
        else:
            _output_pretty(matches=results)

    except Exception as e:
        console.print(f"[red]Error fetching results: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


def _output_json(matches: list[MatchResult]) -> None:
    """Output matches as JSON to stdout."""
    output = [match.model_dump(mode="json") for match in matches]
    json.dump(obj=output, fp=sys.stdout, indent=2)
    sys.stdout.write("\n")


def _output_pretty(matches: list[MatchResult]) -> None:
    """Output matches in pretty console format."""
    console.print(f"\n[bold green]Found {len(matches)} top matches![/bold green]\n")

    for i, match in enumerate(iterable=matches, start=1):
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
            content.append("\n[cyan]Skills to develop:[/cyan]")
            content.append(f"  {', '.join(match.missing_skills[:7])}")
            if len(match.missing_skills) > 7:
                content.append(f"  ... and {len(match.missing_skills) - 7} more")

        # Interview tips
        if match.interview_tips:
            content.append("\n[yellow]Prepare for interview:[/yellow]")
            for tip in match.interview_tips:
                content.append(f"  • {tip}")

        # Job link
        job_url = (
            job.url_active_page
            or job.position_url
            or job.url_comeet_hosted_page
            or job.url_recruit_hosted_page
        )
        if job_url:
            content.append(f"\n[cyan]Apply:[/cyan] {job_url}")

        panel = Panel(
            renderable="\n".join(content),
            title=header,
            border_style="green" if i == 1 else "blue",
        )
        console.print(panel)
        console.print()


if __name__ == "__main__":
    app()
