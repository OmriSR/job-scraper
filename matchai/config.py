from pathlib import Path

# Paths
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "matchai.db"
CHROMA_PATH = DATA_DIR / "chroma_db"

# LLM settings (Ollama)
OLLAMA_MODEL = "llama3.2"
OLLAMA_TEMPERATURE = 0.0

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# spaCy model
SPACY_MODEL = "en_core_web_sm"

# Matching settings
SIMILARITY_WEIGHT = 0.7
FILTER_WEIGHT = 0.3
DEFAULT_TOP_N = 5
