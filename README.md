# MatchAI

**A job matching system that ranks positions based on CV analysis, with support for local development and cloud deployment.**

MatchAI uses Groq's fast LLM API, semantic search, and deterministic filters to match candidates with job positions. It can run locally for development or as a scheduled Cloud Run Job on GCP.

## Features

- **Fast Inference**: Uses Groq's optimized LLM API for quick CV parsing and explanations
- **Lightweight Embeddings**: Uses fastembed (ONNX Runtime) - 75% smaller than PyTorch
- **Smart Matching**: Combines semantic similarity with deterministic filters (skills, seniority, location)
- **Explainable**: Generates human-readable explanations for each match
- **Interview Tips**: LLM-generated preparation suggestions based on skill gaps
- **Cloud-Native**: Runs as scheduled Cloud Run Job with Supabase and Pinecone
- **Local Development**: SQLite + ChromaDB fallback for local testing

## Architecture

### Local Mode
```
CV (PDF) → Text Extraction → LLM Parsing → CandidateProfile
                                                ↓
Job API → SQLite Storage → Preprocessing → ChromaDB Embeddings
                                                ↓
                          Filtering (Location, Seniority, Skills)
                                                ↓
                          Semantic Ranking (Cosine Similarity)
                                                ↓
                          LLM Explanations → Top-N Matches
```

### Cloud Mode
```
┌─────────────────────────────────────────────────────────────────┐
│                    Google Cloud Platform                         │
│  ┌────────────────┐       ┌─────────────────────────────────┐   │
│  │ Cloud Scheduler│──────▶│        Cloud Run Job            │   │
│  │ (8 AM, 8 PM)   │       │  • Fetch jobs from Comeet API   │   │
│  └────────────────┘       │  • Match against stored CV      │   │
│                           │  • Store results in Supabase    │   │
│  ┌────────────────┐       └────────────────┬────────────────┘   │
│  │ Secret Manager │──────▶ API Keys        │                    │
│  └────────────────┘                        │                    │
└────────────────────────────────────────────┼────────────────────┘
                    ┌────────────────────────┴────────────────────┐
                    ▼                                              ▼
           ┌─────────────────┐                        ┌─────────────────┐
           │    Supabase     │                        │    Pinecone     │
           │   (Postgres)    │                        │   (Vectors)     │
           │ • jobs          │                        │ • job_embeddings│
           │ • candidates    │                        └─────────────────┘
           │ • match_results │
           └─────────────────┘
```

## Tech Stack

| Component | Local | Cloud |
|-----------|-------|-------|
| Database | SQLite | Supabase (PostgreSQL) |
| Vector Store | ChromaDB | Pinecone |
| Embeddings | fastembed (ONNX) | fastembed (ONNX) |
| LLM | Groq API | Groq API |
| NLP | spaCy | spaCy |
| Execution | CLI | Cloud Run Job |

## Installation

### Prerequisites

1. **Groq API Key** (required): https://console.groq.com
2. **Supabase Project** (for cloud): https://supabase.com
3. **Pinecone Account** (for cloud): https://pinecone.io

### Local Setup

```bash
# Clone and install
cd job-scraper
pip install -e ".[dev]"

# Download spaCy model
python -m spacy download en_core_web_sm

# Set up environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Verify Installation

```bash
matchai --help
```

## Usage

### Local Mode (Development)

#### Step 1: Ingest Jobs

```bash
# Create companies.json with Comeet credentials
matchai ingest --companies companies.json
```

#### Step 2: Match Your CV

```bash
matchai match --cv /path/to/cv.pdf --location "Tel Aviv" --top-n 10
```

#### Step 3: View Info

```bash
matchai info
```

### Cloud Mode (Production)

#### Step 1: Set Up Cloud Services

```bash
# Set environment variables for cloud
export DATABASE_URL="postgresql://user:pass@db.supabase.co:5432/postgres"
export PINECONE_API_KEY="your-pinecone-key"
export GROQ_API_KEY="your-groq-key"
```

#### Step 2: Upload CV (Run Locally)

```bash
# Parse CV locally and upload profile to Supabase
matchai upload-cv --cv /path/to/cv.pdf
```

#### Step 3: Import Companies (Run Locally)

```bash
# Import companies from JSON file (same format as local mode)
matchai import-companies --file companies.json

# Or add one at a time
matchai add-company --name "Acme Corp" --uid "company-uid" --token "api-token"

# List registered companies
matchai list-companies
```

#### Step 4: Deploy to GCP

```bash
# Full deployment (APIs, Docker, Cloud Run, Scheduler)
./scripts/deploy-gcp.sh deploy

# Or just build and push image
./scripts/deploy-gcp.sh build
```

#### Step 5: Get Results

```bash
# After scheduled job runs (8 AM and 8 PM daily)
matchai get-results --limit 10
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `matchai ingest --companies FILE` | Ingest jobs from Comeet API (local mode) |
| `matchai match --cv FILE` | Match CV against jobs (local mode) |
| `matchai info` | Show database statistics |
| `matchai upload-cv --cv FILE` | Upload parsed CV to cloud database |
| `matchai import-companies --file FILE` | Import companies from JSON file |
| `matchai add-company` | Add single company credentials |
| `matchai list-companies` | List registered companies |
| `matchai get-results` | Fetch match results from cloud |

## Configuration

### Environment Variables

```bash
# Required
GROQ_API_KEY=your-groq-key

# Cloud mode (optional for local)
DATABASE_URL=postgresql://...
PINECONE_API_KEY=your-pinecone-key
```

### Application Settings

See [matchai/config.py](matchai/config.py):

```python
# Matching weights
SIMILARITY_WEIGHT = 0.6        # Semantic similarity weight
FILTER_WEIGHT = 0.4            # Skill match weight
DEFAULT_TOP_N = 5              # Default results count

# Embedding model (ONNX-based)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Pinecone settings (cloud)
PINECONE_INDEX = "job-embeddings"
PINECONE_DIMENSION = 384
```

## Project Structure

```
matchai/
├── main.py              # CLI entrypoint (typer)
├── config.py            # Configuration settings
├── scheduled_runner.py  # Cloud Run Job entrypoint
├── schemas/             # Pydantic models
├── cv/                  # CV processing (PDF extraction, LLM parsing)
├── jobs/                # Job data layer (database, embeddings, ingest)
├── db/                  # Database abstraction (SQLite + PostgreSQL)
├── embeddings/          # Embedding clients (fastembed, Pinecone)
├── matching/            # Matching pipeline (filters, ranking)
├── explainer/           # LLM explanation generation
├── services/            # Service layer (ingest, match)
└── utils.py             # Utility functions
```

## How Matching Works

### 1. CV Parsing
- Extracts text from PDF using PyMuPDF
- LLM parses into structured `CandidateProfile`:
  - Skills, tools/frameworks, domains
  - Seniority level (junior → staff)
  - Years of experience

### 2. Job Processing
- Jobs fetched from Comeet API
- HTML details parsed and cleaned
- Text preprocessed with spaCy (lemmatization)
- Keywords extracted for skill matching
- Embeddings generated via fastembed (ONNX)

### 3. Filtering
- **Database level**: Location, seniority (efficient SQL)
- **In-memory**: Skills (fuzzy matching with RapidFuzz)

### 4. Ranking
- Cosine similarity between candidate and job embeddings
- Skill match score (% of candidate skills in job)
- Combined: `final_score = 0.6 * similarity + 0.4 * skill_match`

### 5. Explanation & Tips
- LLM generates match explanations
- Identifies missing skills
- Generates interview preparation tips

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
pytest -v
pytest --cov=matchai

# Lint
ruff check .
ruff check . --fix
```

## Troubleshooting

### "GROQ_API_KEY environment variable is not set"
```bash
cp .env.example .env
# Add your key from https://console.groq.com
```

### "No jobs found in database"
```bash
matchai ingest --companies companies.json
```

### "spaCy model not found"
```bash
python -m spacy download en_core_web_sm
```

### Cloud: "Connection refused to database"
Check your `DATABASE_URL` environment variable points to Supabase.

## Limitations

1. **Comeet API Only**: Currently only supports Comeet for job fetching
2. **PDF CVs Only**: Only supports PDF format
3. **English Only**: spaCy model and LLM prompts are English-only
4. **Single CV**: Cloud mode stores one CV at a time

## License

This project is for educational and personal use.

## Credits

Built with:
- [Groq](https://groq.com) - Fast LLM inference API
- [LangChain](https://langchain.com) - LLM orchestration
- [fastembed](https://github.com/qdrant/fastembed) - ONNX-based embeddings
- [Pinecone](https://pinecone.io) - Cloud vector database
- [ChromaDB](https://www.trychroma.com) - Local vector database
- [Supabase](https://supabase.com) - Cloud PostgreSQL
- [spaCy](https://spacy.io) - NLP preprocessing
- [Typer](https://typer.tiangolo.com) - CLI framework
- [Rich](https://rich.readthedocs.io) - Beautiful terminal output
