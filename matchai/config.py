import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Environment detection
IS_CLOUD = os.getenv("CLOUD_RUN_JOB", "") != "" or os.getenv("DATABASE_URL", "") != ""

# Paths (local development only)
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "matchai.db"
CHROMA_PATH = DATA_DIR / "chroma_db"

# Database (Supabase PostgreSQL)
DATABASE_URL = os.getenv("DATABASE_URL")

# Pinecone (Vector Store)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = "job-embeddings"
PINECONE_DIMENSION = 384

# LLM settings (Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.0

# Embedding model (fastembed uses ONNX Runtime)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# spaCy model
SPACY_MODEL = "en_core_web_sm"

# Matching settings
SIMILARITY_WEIGHT = 0.6
FILTER_WEIGHT = 1 - SIMILARITY_WEIGHT
DEFAULT_TOP_N = 5
SKILL_MATCH_THRESHOLD = 80
MAX_JOB_VIEWS = 3  # Exclude jobs shown this many times to the same candidate

# Email settings (Gmail SMTP)
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "true").lower() == "true"
