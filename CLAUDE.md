# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
MatchAI - A local, zero-cost job matching system that ranks job positions based on CV analysis.

**Tech Stack:**
- Python 3.11+
- LLM: Ollama (llama3.2) via LangChain
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Vector Store: ChromaDB
- Database: SQLite
- NLP: spaCy (en_core_web_sm)
- CLI: typer

## Development Setup

1. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Download spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. Install and start Ollama, then pull the model:
   ```bash
   ollama pull llama3.2
   ```

## Common Commands

```bash
# Install project
pip install -e ".[dev]"

# Run CLI
matchai --help
matchai ingest --jobs <path> --companies <path>
matchai match --cv <path> [--location <loc>] [--top-n 5]
matchai info

# Run tests
pytest

# Lint
ruff check .
```

## Architecture Notes

Modular monolith structure:
```
matchai/
├── main.py           # CLI entrypoint (typer)
├── config.py         # Configuration settings
├── schemas/          # Pydantic models
├── cv/               # CV processing (PDF extraction, LLM parsing)
├── jobs/             # Job data layer (SQLite, preprocessing, embeddings)
├── matching/         # Matching pipeline (filters, ranking)
└── explainer/        # LLM explanation generation
```

**Pipeline Flow:**
1. CV → PDF extraction → LLM parsing → CandidateProfile
2. Jobs → SQLite storage → preprocessing → ChromaDB embeddings
3. Matching: deterministic filters → semantic ranking → LLM explanations
4. Output: Top-N matches with scores and explanations

## Testing

```bash
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest --cov=matchai      # With coverage
```

## Rules

1. Be minimalistic; take small steps
2. Use context managers (`with`) when possible
3. Use Pydantic `Field` with descriptions for all schemas
4. Mark completed tasks in `tasks.md`
5. Prefer deterministic methods over LLM-based ones
6. LLM only for: CV parsing and explanation generation
7. Write tests for each feature/module as it's implemented
