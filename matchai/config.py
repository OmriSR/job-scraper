from pathlib import Path

# Paths
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "matchai.db"
CHROMA_PATH = DATA_DIR / "chroma_db"

# LLM settings (Ollama)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
OLLAMA_TEMPERATURE = 0.0

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# spaCy model
SPACY_MODEL = "en_core_web_sm"

# Matching settings
SIMILARITY_WEIGHT = 0.6
FILTER_WEIGHT = 1 - SIMILARITY_WEIGHT
DEFAULT_TOP_N = 5
SKILL_MATCH_THRESHOLD = 80
