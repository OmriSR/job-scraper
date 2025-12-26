# MatchAI Cloud Run Job Container
# Optimized for small image size (~600MB) using ONNX Runtime instead of PyTorch

FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy dependency file first for better layer caching
COPY pyproject.toml .

# Install Python dependencies (excluding local-only deps like chromadb)
# We use pip install with specific packages to avoid installing chromadb in the cloud image
RUN pip install --no-cache-dir \
    pymupdf>=1.24.0 \
    langchain>=0.3.0 \
    langchain-groq>=0.2.0 \
    python-dotenv>=1.0.0 \
    pydantic>=2.0.0 \
    fastembed>=0.3.0 \
    pinecone>=5.0.0 \
    psycopg2-binary>=2.9.0 \
    spacy>=3.7.0 \
    rapidfuzz>=3.0.0 \
    typer>=0.12.0 \
    rich>=13.0.0 \
    requests>=2.31.0 \
    scikit-learn>=1.5.0 \
    && pip cache purge

# Download spaCy model at build time
RUN python -m spacy download en_core_web_sm

# Pre-download fastembed model at build time (avoids runtime download)
# This ensures consistent model version and faster cold starts
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
COPY matchai/ matchai/

# Set environment variable to indicate cloud mode
ENV CLOUD_RUN_JOB=true

# Set entrypoint for Cloud Run Job
ENTRYPOINT ["python", "-m", "matchai.scheduled_runner"]
