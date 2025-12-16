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
- [ ] 6.1.1 Create `matching/filter.py`
- [ ] 6.1.2 Implement `filter_by_skills()` with RapidFuzz
- [ ] 6.1.3 Implement `filter_by_seniority()`
- [ ] 6.1.4 Implement `filter_by_location()`
- [ ] 6.1.5 Implement `apply_filters()` - combine all

## Phase 6.2: Semantic Ranking
- [ ] 6.2.1 Create `matching/ranker.py`
- [ ] 6.2.2 Implement `compute_similarity()` - cosine
- [ ] 6.2.3 Implement `rank_jobs()`
- [ ] 6.2.4 Implement `compute_final_score()`

## Phase 7: LLM Explanation
- [x] 7.1 Create `explainer/generator.py`
- [x] 7.2 Create explanation prompt template
- [x] 7.3 Implement `generate_explanation()`
- [ ] 7.4 Implement `find_missing_skills()` (deterministic)

## Phase 8: CLI
- [ ] 8.1 Create `main.py` with typer
- [ ] 8.2 Implement `ingest` command
- [ ] 8.3 Implement `match` command
- [ ] 8.4 Implement `info` command
- [ ] 8.5 Implement pretty console output
- [ ] 8.6 Implement JSON output option

## Phase 9: Testing
- [ ] 9.1 Create sample test data (CV PDF, jobs JSON)
- [ ] 9.2 Test PDF extraction
- [ ] 9.3 Test Pydantic validation
- [ ] 9.4 Test SQLite operations
- [ ] 9.5 Test preprocessing
- [ ] 9.6 Test embeddings
- [ ] 9.7 Test filters
- [ ] 9.8 Test ranking
- [ ] 9.9 Integration test: full pipeline
