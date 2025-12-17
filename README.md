# MatchAI

**A local, zero-cost job matching system that ranks job positions based on CV analysis.**

MatchAI uses Groq's fast LLM API, semantic search, and deterministic filters to match candidates with job positions. CV parsing and explanations use Groq's free tier API, while embeddings are generated locally.

## Features

- **Fast Inference**: Uses Groq's optimized LLM API for quick CV parsing and explanations
- **Zero Cost**: Uses Groq's free tier API and local sentence-transformers for embeddings
- **Smart Matching**: Combines semantic similarity with deterministic filters (skills, seniority, location)
- **Explainable**: Generates human-readable explanations for each match
- **Efficient**: Database-level filtering to handle large job datasets
- **Idempotent**: Re-running ingestion won't create duplicates

## Architecture

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

## Tech Stack

- **Python**: 3.11+
- **LLM**: Groq API (llama-3.3-70b-versatile) via LangChain
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB
- **Database**: SQLite
- **NLP**: spaCy (en_core_web_sm)
- **CLI**: typer + rich

## Installation

### 1. Prerequisites

Get a free Groq API key:
1. Go to https://console.groq.com
2. Sign up and create an API key
3. Copy your API key for the next step

### 2. Install MatchAI

```bash
# Clone the repository
cd job-scraper

# Install in development mode
pip install -e ".[dev]"

# Download spaCy model
python -m spacy download en_core_web_sm

# Set up environment variables
cp .env.example .env
# Edit .env and add your Groq API key
```

### 3. Verify Installation

```bash
matchai --help
```

## Usage

### Step 1: Prepare Company Data

Create a `companies.json` file with Comeet API credentials:

```json
[
  {
    "name": "Example Company",
    "uid": "company-uid-from-comeet",
    "token": "your-comeet-api-token",
    "extracted_from": "comeet"
  }
]
```

### Step 2: Ingest Jobs

```bash
matchai ingest --companies companies.json
```

This will:
1. Load companies into the database
2. Fetch jobs from Comeet API
3. Store jobs in SQLite
4. Generate embeddings and store in ChromaDB

### Step 3: Match Your CV

```bash
matchai match --cv /path/to/your/cv.pdf
```

Options:
- `--location "New York"` - Filter by location
- `--top-n 10` - Return top 10 matches (default: 5)
- `--json` - Output as JSON instead of pretty format

Example with filters:
```bash
matchai match --cv ~/Documents/my_cv.pdf --location "Tel Aviv" --top-n 10
```

### Step 4: View Database Info

```bash
matchai info
```

Shows statistics about jobs, companies, and locations in your database.

## Configuration

### Environment Variables

Create a `.env` file (copy from `.env.example`):

```bash
GROQ_API_KEY=your-groq-api-key-here
```

### Application Settings

All other configuration is in [matchai/config.py](matchai/config.py):

```python
# Paths
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "matchai.db"
CHROMA_PATH = DATA_DIR / "chroma_db"

# LLM settings (Groq)
GROQ_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.0

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Matching settings
SIMILARITY_WEIGHT = 0.6        # Weight for semantic similarity
FILTER_WEIGHT = 0.4            # Weight for skill/seniority match
DEFAULT_TOP_N = 5              # Default number of results
SKILL_MATCH_THRESHOLD = 80     # RapidFuzz threshold for skill matching
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
- Embeddings generated and stored in ChromaDB

### 3. Filtering (Database + In-Memory)
- **Database level**: Location, seniority (for efficiency)
- **In-memory**: Skills (fuzzy matching with RapidFuzz)

### 4. Ranking
- Computes cosine similarity between candidate and job embeddings
- Computes skill match score (% of candidate skills in job)
- Combines scores: `final_score = 0.6 * similarity + 0.4 * skill_match`

### 5. Explanation
- LLM generates 2-3 bullet points explaining the match
- Deterministically identifies missing skills

## Performance Optimizations

1. **Database-level filtering**: Location and seniority filters run in SQL, not Python
2. **Idempotent ingestion**: Checks existing UIDs before inserting
3. **Batch embeddings**: Uses sentence-transformers' batch processing
4. **Efficient similarity**: NumPy/sklearn for fast cosine similarity computation

## Common Commands

```bash
# Install project
pip install -e ".[dev]"

# Run CLI
matchai --help
matchai ingest --companies companies.json
matchai match --cv cv.pdf --location "New York" --top-n 5
matchai info

# Run tests
pytest
pytest -v
pytest --cov=matchai

# Lint
ruff check .
```

## Troubleshooting

### "GROQ_API_KEY environment variable is not set"
Create a `.env` file with your API key:
```bash
cp .env.example .env
# Edit .env and add your Groq API key from https://console.groq.com
```

### "No jobs found in database"
Run ingestion first:
```bash
matchai ingest --companies companies.json
```

### "Database not found"
The database is created automatically on first ingestion. Make sure the `data/` directory is writable.

### "spaCy model not found"
Download the model:
```bash
python -m spacy download en_core_web_sm
```

## Limitations

1. **Comeet API Only**: Currently only supports Comeet API for job fetching
2. **PDF CVs Only**: Only supports PDF format (not DOCX, TXT, etc.)
3. **English Only**: spaCy model and LLM prompts are English-only

## Future Enhancements

- [ ] Support for local job JSON files (no API required)
- [ ] Multi-format CV support (DOCX, TXT)
- [ ] Multiple job board integrations (LinkedIn, Indeed, etc.)
- [ ] Web UI for easier interaction
- [ ] Batch CV processing
- [ ] Historical match tracking
- [ ] Interview preparation suggestions

## License

This project is for educational and personal use.

## Credits

Built with:
- [Groq](https://groq.com) - Fast LLM inference API
- [LangChain](https://langchain.com) - LLM orchestration
- [Sentence Transformers](https://www.sbert.net) - Text embeddings
- [ChromaDB](https://www.trychroma.com) - Vector database
- [spaCy](https://spacy.io) - NLP preprocessing
- [Typer](https://typer.tiangolo.com) - CLI framework
- [Rich](https://rich.readthedocs.io) - Beautiful terminal output
