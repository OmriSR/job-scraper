import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "matchai.db"
CHROMA_PATH = DATA_DIR / "chroma_db"

# LLM settings (Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.0

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# spaCy model
SPACY_MODEL = "en_core_web_sm"

# Matching settings
SIMILARITY_WEIGHT = 0.6
FILTER_WEIGHT = 1 - SIMILARITY_WEIGHT
DEFAULT_TOP_N = 5
SKILL_MATCH_THRESHOLD = 80
