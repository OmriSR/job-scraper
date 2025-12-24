"""FastEmbed client for ONNX-based text embeddings.

This module uses fastembed which runs on ONNX Runtime, providing:
- ~15x smaller footprint than PyTorch (~100MB vs 1.5GB)
- 2-3x faster inference
- Same embedding quality as sentence-transformers
"""

from collections.abc import Iterator

import numpy as np
from fastembed import TextEmbedding

from matchai.config import EMBEDDING_MODEL

_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    """Lazy load fastembed model.

    The model is loaded once and cached for subsequent calls.
    fastembed automatically downloads and caches the ONNX model.
    """
    global _model
    if _model is None:
        _model = TextEmbedding(EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> list[float]:
    """Generate embedding for a single text string.

    Args:
        text: Text to embed.

    Returns:
        Embedding vector as list of floats (dimension 384 for MiniLM).
    """
    model = _get_model()
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist()


def embed_texts_batch(texts: list[str], show_progress: bool = False) -> list[list[float]]:
    """Generate embeddings for multiple texts efficiently.

    FastEmbed processes texts in batches internally for optimal performance.

    Args:
        texts: List of texts to embed.
        show_progress: Whether to show progress bar (default False).

    Returns:
        List of embedding vectors, each as a list of floats.
    """
    if not texts:
        return []

    model = _get_model()
    embeddings: Iterator[np.ndarray] = model.embed(texts)
    return [emb.tolist() for emb in embeddings]


def embed_text_numpy(text: str) -> np.ndarray:
    """Generate embedding as numpy array (for compatibility with existing code).

    Args:
        text: Text to embed.

    Returns:
        Embedding vector as numpy array.
    """
    model = _get_model()
    embeddings = list(model.embed([text]))
    return embeddings[0]


def embed_texts_batch_numpy(texts: list[str]) -> np.ndarray:
    """Generate embeddings as numpy array (for compatibility with existing code).

    Args:
        texts: List of texts to embed.

    Returns:
        2D numpy array of shape (n_texts, embedding_dim).
    """
    if not texts:
        return np.array([])

    model = _get_model()
    embeddings = list(model.embed(texts))
    return np.array(embeddings)
