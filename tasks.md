# MatchAI Build Tasks

## Phase 1: Project Setup
- [x] 1.1 Create `pyproject.toml` with dependencies
- [x] 1.2 Create `.gitignore`
- [x] 1.3 Create directory structure (all `__init__.py` files)
- [x] 1.4 Create `matchai/config.py` with settings

## Phase 2: Pydantic Schemas
- [x] 2.1 Create `schemas/candidate.py` - CandidateProfile model
- [x] 2.2 Create `schemas/job.py` - JobDetail, Job, Company models
- [x] 2.3 Create `schemas/match.py` - MatchResult model

## Phase 3.1: PDF Text Extraction
- [x] 3.1.1 Create `cv/extractor.py`
- [x] 3.1.2 Implement `extract_text_from_pdf()` using PyMuPDF
- [x] 3.1.3 Handle multi-page PDFs and clean whitespace

## Phase 3.2: LLM CV Parsing
- [x] 3.2.1 Create `cv/parser.py`
- [x] 3.2.2 Set up Ollama via LangChain
- [x] 3.2.3 Create CV parsing prompt
- [x] 3.2.4 Implement `parse_cv()` → CandidateProfile

## Phase 4.1: SQLite Database
- [x] 4.1.1 Create `jobs/database.py`
- [x] 4.1.2 Implement `init_database()` - create tables
- [x] 4.1.3 Implement `insert_jobs()`
- [x] 4.1.4 Implement `insert_companies()`
- [x] 4.1.5 Implement `get_all_jobs()`
- [x] 4.1.6 Implement `get_jobs_by_uids()`
- [x] 4.1.7 Implement `get_existing_job_uids()` for idempotency

## Phase 4.2: Text Preprocessing
- [x] 4.2.1 Create `jobs/preprocessor.py`
- [x] 4.2.2 Implement `extract_details_text()` - parse HTML from details
- [x] 4.2.3 Implement `preprocess_job()` - clean and lemmatize
- [x] 4.2.4 Implement `extract_job_keywords()`

## Phase 4.3: Embeddings + ChromaDB
- [x] 4.3.1 Create `jobs/embeddings.py`
- [x] 4.3.2 Initialize sentence-transformers model
- [x] 4.3.3 Initialize ChromaDB client
- [x] 4.3.4 Implement `embed_text()`
- [x] 4.3.5 Implement `embed_and_store_jobs()` - combined embed + store
- [x] 4.3.6 Implement `get_job_embeddings()`
- [x] 4.3.7 Implement `get_existing_embedding_uids()`
- [x] 4.3.8 Implement `embed_candidate()`

## Phase 5: Ingestion Pipeline
- [x] 5.1 Create ingestion function in `jobs/ingest.py`
- [x] 5.2 Load and validate JSON with Pydantic
- [x] 5.3 Check existing uids (idempotency)
- [x] 5.4 Insert new jobs only
- [x] 5.5 Preprocess and embed new jobs
- [x] 5.6 Log ingestion stats
- [x] 5.7 Write tests for ingestion pipeline

## Phase 6.1: Deterministic Filters
- [x] 6.1.1 Create `matching/filter.py`
- [x] 6.1.2 Implement `filter_by_skills()` with RapidFuzz
- [x] 6.1.3 Implement `filter_by_seniority()` with SeniorityLevel enum
- [x] 6.1.4 Implement `filter_by_location()`
- [x] 6.1.5 Implement `apply_filters()` - combine all
- [x] 6.1.6 Write tests for filters

## Phase 6.2: Semantic Ranking
- [x] 6.2.1 Create `matching/ranker.py`
- [x] 6.2.2 Implement `compute_similarities_batch()` - cosine (sklearn)
- [x] 6.2.3 Implement `rank_jobs()`
- [x] 6.2.4 Implement `compute_final_score()`
- [x] 6.2.5 Write tests for ranker

## Phase 7: LLM Explanation
- [x] 7.1 Create `explainer/generator.py`
- [x] 7.2 Create explanation prompt template
- [x] 7.3 Implement `generate_explanation()`
- [x] 7.4 Implement `find_missing_skills()` (deterministic)

## Phase 8: CLI
- [x] 8.1 Create `main.py` with typer
- [x] 8.2 Implement `ingest` command
- [x] 8.3 Implement `match` command
- [x] 8.4 Implement `info` command
- [x] 8.5 Implement pretty console output
- [x] 8.6 Implement JSON output option
- [x] 8.7 Add database-level filtering to avoid loading all jobs

## Phase 9: Testing
- [x] 9.1 Create sample test data (CV PDF, jobs JSON)
- [x] 9.2 Test PDF extraction
- [x] 9.3 Test Pydantic validation
- [x] 9.4 Test SQLite operations
- [x] 9.5 Test preprocessing
- [x] 9.6 Test embeddings
- [x] 9.7 Test filters
- [x] 9.8 Test ranking
- [x] 9.9 Integration test: full pipeline

---

# Cloud-Native Transformation (GCP)

## Phase 10: Database Migration (SQLite → Supabase)

### 10.1 Supabase Setup
- [ ] Create Supabase account (free tier)
- [ ] Create new project
- [ ] Note connection string and API keys

### 10.2 PostgreSQL Schema Creation
- [ ] Create `jobs` table with JSONB for details field
- [ ] Create `companies` table
- [ ] Create `candidates` table for CV caching
- [ ] Create `match_results` table for storing results
- [ ] Add foreign key constraints
- [ ] Add created_at timestamps with defaults

### 10.3 Install PostgreSQL Dependencies
- [ ] Add `psycopg2-binary>=2.9.0` to pyproject.toml
- [ ] Add `supabase>=2.0.0` to pyproject.toml (optional, for REST API)

### 10.4 Update Config Module
- [ ] Add `DATABASE_URL` from environment variable
- [ ] Add `SUPABASE_URL` and `SUPABASE_KEY` constants
- [ ] Add environment detection (local vs cloud)

### 10.5 Create Database Abstraction Layer
- [ ] Create `matchai/db/connection.py` with connection factory
- [ ] Implement `get_connection()` for PostgreSQL
- [ ] Handle connection pooling for cloud usage

### 10.6 Migrate Database Functions
- [ ] Update `init_database()` for PostgreSQL syntax
- [ ] Update `insert_jobs_to_db()` - use JSONB for details
- [ ] Update `insert_companies()` for PostgreSQL
- [ ] Update `get_all_jobs()` for PostgreSQL
- [ ] Update `get_jobs_by_uids()` for PostgreSQL
- [ ] Update `get_jobs()` with filters for PostgreSQL
- [ ] Update `get_existing_job_uids()` for PostgreSQL
- [ ] Update `get_all_companies()` for PostgreSQL

### 10.7 Add New Database Functions
- [ ] Implement `save_candidate(cv_hash, profile, raw_text)`
- [ ] Implement `get_candidate_by_hash(cv_hash)`
- [ ] Implement `save_match_results(cv_hash, results)`
- [ ] Implement `get_match_results(cv_hash, limit)`

### 10.8 Test Database Migration
- [ ] Write tests for PostgreSQL connection
- [ ] Test CRUD operations with Supabase
- [ ] Verify JSON/JSONB serialization works correctly

---

## Phase 11: Embeddings Migration (PyTorch → ONNX)

### 11.1 Update Dependencies
- [ ] Remove `sentence-transformers>=3.0.0` from pyproject.toml
- [ ] Add `fastembed>=0.3.0` to pyproject.toml

### 11.2 Create Pinecone Account
- [ ] Create Pinecone account (free starter plan)
- [ ] Create index `job-embeddings` (dimension: 384, metric: cosine)
- [ ] Note API key and environment

### 11.3 Update Dependencies for Pinecone
- [ ] Remove `chromadb>=0.5.0` from pyproject.toml
- [ ] Add `pinecone-client>=3.0.0` to pyproject.toml

### 11.4 Update Config for Embeddings
- [ ] Add `PINECONE_API_KEY` from environment
- [ ] Add `PINECONE_INDEX = "job-embeddings"`
- [ ] Add `EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"`

### 11.5 Create New Embeddings Module
- [ ] Create `matchai/embeddings/fastembed_client.py`
- [ ] Initialize TextEmbedding model from fastembed
- [ ] Implement `embed_text(text: str) -> list[float]`
- [ ] Implement `embed_texts_batch(texts: list[str]) -> list[list[float]]`

### 11.6 Create Pinecone Client Module
- [ ] Create `matchai/embeddings/pinecone_client.py`
- [ ] Initialize Pinecone client with API key
- [ ] Implement `upsert_embeddings(ids, vectors, metadata)`
- [ ] Implement `query_similar(vector, top_k)`
- [ ] Implement `get_existing_ids() -> set[str]`
- [ ] Implement `delete_embeddings(ids)`

### 11.7 Update Embeddings Integration
- [ ] Update `embed_and_store_jobs()` to use fastembed + Pinecone
- [ ] Update `get_job_embeddings()` for Pinecone
- [ ] Update `embed_candidate()` to use fastembed

### 11.8 Test Embeddings Migration
- [ ] Test fastembed produces same dimension (384)
- [ ] Test Pinecone upsert and query
- [ ] Verify cosine similarity results are consistent

---

## Phase 12: Containerization

### 12.1 Create Dockerfile
- [ ] Create `Dockerfile` in project root
- [ ] Use `python:3.11-slim` as base image
- [ ] Install minimal system dependencies
- [ ] Copy and install Python dependencies
- [ ] Pre-download spaCy model at build time
- [ ] Pre-download fastembed model at build time
- [ ] Set entrypoint to `python -m matchai.job`

### 12.2 Create .dockerignore
- [ ] Create `.dockerignore` file
- [ ] Exclude `data/`, `*.pyc`, `__pycache__/`
- [ ] Exclude `.git/`, `.env`, `*.db`
- [ ] Exclude `tests/`, `*.md` (except README)

### 12.3 Create Job Entrypoint
- [ ] Create `matchai/job.py`
- [ ] Implement `main()` function for scheduled execution
- [ ] Add logging with INFO level
- [ ] Call `ingest_jobs()` from service layer
- [ ] Call `run_batch_matching()` from service layer
- [ ] Handle exceptions and log errors

### 12.4 Test Docker Build Locally
- [ ] Build Docker image: `docker build -t matchai-job .`
- [ ] Verify image size is ~600MB or less
- [ ] Run container locally with env vars
- [ ] Verify job executes successfully

---

## Phase 13: Service Layer Refactoring

### 13.1 Create Services Directory
- [ ] Create `matchai/services/` directory
- [ ] Create `matchai/services/__init__.py`

### 13.2 Create Ingest Service
- [ ] Create `matchai/services/ingest_service.py`
- [ ] Implement `get_companies_from_db() -> list[Company]`
- [ ] Implement `ingest_jobs() -> dict` (stats)
- [ ] Handle Comeet API fetching
- [ ] Store jobs in Supabase + embeddings in Pinecone
- [ ] Return ingestion statistics

### 13.3 Create Match Service
- [ ] Create `matchai/services/match_service.py`
- [ ] Implement `get_candidate_from_db() -> CandidateProfile`
- [ ] Implement `match_candidate(candidate) -> list[MatchResult]`
- [ ] Implement `run_batch_matching() -> dict` (stats)
- [ ] Implement `save_results_to_db(cv_hash, results)`
- [ ] Handle full pipeline: filter → rank → explain

### 13.4 Create Supabase Client Helper
- [ ] Create `matchai/services/supabase_client.py`
- [ ] Implement connection helper with retry logic
- [ ] Implement common query patterns
- [ ] Handle connection errors gracefully

### 13.5 Update Imports
- [ ] Update `matchai/services/__init__.py` with exports
- [ ] Ensure all services are importable

---

## Phase 14: GCP Deployment

### 14.1 GCP Project Setup
- [ ] Create GCP project: `matchai-prod`
- [ ] Enable Cloud Run API
- [ ] Enable Artifact Registry API
- [ ] Enable Cloud Scheduler API
- [ ] Enable Secret Manager API

### 14.2 Set Up Secrets
- [ ] Create secret: `groq-api-key`
- [ ] Create secret: `database-url` (Supabase connection string)
- [ ] Create secret: `pinecone-api-key`
- [ ] Create secret: `comeet-credentials` (if needed)

### 14.3 Set Up Artifact Registry
- [ ] Create Docker repository in Artifact Registry
- [ ] Configure Docker authentication for GCP

### 14.4 Build and Push Image
- [ ] Tag image for Artifact Registry
- [ ] Push image to Artifact Registry
- [ ] Verify image is accessible

### 14.5 Create Cloud Run Job
- [ ] Create Cloud Run Job with image
- [ ] Configure memory (2Gi) and CPU (2)
- [ ] Set max-retries to 1
- [ ] Set task-timeout to 30m
- [ ] Attach secrets as environment variables

### 14.6 Create Cloud Scheduler
- [ ] Create scheduler job with cron: `0 8,20 * * *`
- [ ] Configure HTTP target to Cloud Run Job
- [ ] Set up service account for invocation
- [ ] Test manual trigger

### 14.7 Test End-to-End
- [ ] Trigger job manually from Cloud Console
- [ ] Verify logs in Cloud Logging
- [ ] Check results in Supabase match_results table
- [ ] Verify Pinecone has job embeddings

---

## Phase 15: CLI Setup Commands

### 15.1 Add Upload CV Command
- [ ] Implement `matchai upload-cv --cv <path>` command
- [ ] Extract text from PDF locally
- [ ] Parse CV with LLM (Groq API)
- [ ] Compute SHA256 hash of CV text
- [ ] Upload to Supabase candidates table
- [ ] Display success message with hash

### 15.2 Add Company Management Commands
- [ ] Implement `matchai add-company --name --uid --token`
- [ ] Upload company to Supabase companies table
- [ ] Implement `matchai list-companies`
- [ ] Fetch and display companies from Supabase

### 15.3 Add Results Command
- [ ] Implement `matchai get-results --limit 10`
- [ ] Fetch latest match results from Supabase
- [ ] Display formatted results with Rich
- [ ] Add `--json` flag for JSON output

### 15.4 Update Info Command
- [ ] Update `matchai info` to show cloud stats
- [ ] Display Supabase connection status
- [ ] Display Pinecone index stats
- [ ] Display candidate count

### 15.5 Test CLI Commands
- [ ] Test `matchai upload-cv` with sample CV
- [ ] Test `matchai add-company` with sample company
- [ ] Test `matchai list-companies`
- [ ] Test `matchai get-results`
- [ ] Test `matchai info`

---

## Phase 16: Infrastructure as Code (Optional)

### 16.1 Create Terraform Directory
- [ ] Create `terraform/` directory
- [ ] Create `terraform/main.tf`
- [ ] Create `terraform/variables.tf`
- [ ] Create `terraform/outputs.tf`

### 16.2 Define GCP Provider
- [ ] Configure Google provider
- [ ] Set project and region variables

### 16.3 Define Cloud Run Job Resource
- [ ] Create `google_cloud_run_v2_job` resource
- [ ] Configure container, memory, CPU
- [ ] Attach secrets from Secret Manager

### 16.4 Define Cloud Scheduler Resource
- [ ] Create `google_cloud_scheduler_job` resource
- [ ] Configure HTTP target
- [ ] Set up OAuth token for authentication

### 16.5 Define IAM Resources
- [ ] Create service account for invoker
- [ ] Grant Cloud Run invoker role
- [ ] Grant Secret Manager accessor role

### 16.6 Test Terraform
- [ ] Run `terraform init`
- [ ] Run `terraform plan`
- [ ] Run `terraform apply` (optional)

---

## Phase 17: Final Polish

### 17.1 Update pyproject.toml
- [ ] Update version to `2.0.0`
- [ ] Verify all dependencies are correct
- [ ] Remove unused dependencies

### 17.2 Update Documentation
- [ ] Update README.md with cloud architecture
- [ ] Document CLI commands for setup
- [ ] Document environment variables

### 17.3 Clean Up Old Code
- [ ] Remove ChromaDB-specific code (after migration verified)
- [ ] Remove SQLite-specific code (after migration verified)
- [ ] Remove sentence-transformers imports

### 17.4 Final Testing
- [ ] Run all unit tests
- [ ] Run linter: `ruff check .`
- [ ] Test CLI locally with cloud backends
- [ ] Trigger Cloud Run Job and verify results

---

## Summary

| Phase | Description | New Files |
|-------|-------------|-----------|
| 10 | Database Migration | `db/connection.py` |
| 11 | Embeddings Migration | `embeddings/fastembed_client.py`, `embeddings/pinecone_client.py` |
| 12 | Containerization | `Dockerfile`, `.dockerignore`, `job.py` |
| 13 | Service Layer | `services/ingest_service.py`, `services/match_service.py` |
| 14 | GCP Deployment | - |
| 15 | CLI Commands | Updated `main.py` |
| 16 | Terraform (Optional) | `terraform/*.tf` |
| 17 | Final Polish | - |
