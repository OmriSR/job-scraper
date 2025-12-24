"""Embeddings module with ONNX-based fastembed and Pinecone vector store."""

from matchai.embeddings.fastembed_client import (
    embed_text,
    embed_text_numpy,
    embed_texts_batch,
    embed_texts_batch_numpy,
)
from matchai.embeddings.pinecone_client import (
    VectorRecord,
    delete_embeddings,
    fetch_embeddings,
    query_similar,
    upsert_embeddings,
)

__all__ = [
    "embed_text",
    "embed_text_numpy",
    "embed_texts_batch",
    "embed_texts_batch_numpy",
    "VectorRecord",
    "upsert_embeddings",
    "query_similar",
    "fetch_embeddings",
    "delete_embeddings",
]
