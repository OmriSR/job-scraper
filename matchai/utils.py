"""Shared utilities for matchai."""

import httpx
from langchain_ollama import ChatOllama

from matchai.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE


class OllamaUnavailableError(Exception):
    """Raised when Ollama server is not reachable."""

    pass


def check_ollama_available() -> None:
    """Check if Ollama server is running and reachable.

    Raises:
        OllamaUnavailableError: If Ollama is not running or unreachable.
    """
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        response.raise_for_status()
    except httpx.ConnectError:
        raise OllamaUnavailableError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Please ensure Ollama is running with: ollama serve"
        )
    except httpx.TimeoutException:
        raise OllamaUnavailableError(
            f"Connection to Ollama at {OLLAMA_BASE_URL} timed out. "
            "Please check if Ollama is responding."
        )
    except httpx.HTTPStatusError as e:
        raise OllamaUnavailableError(
            f"Ollama returned an error: {e.response.status_code}. "
            "Please check your Ollama installation."
        )


def get_llm() -> ChatOllama:
    """Initialize Ollama LLM with configured settings."""
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMPERATURE,
        base_url=OLLAMA_BASE_URL,
    )
