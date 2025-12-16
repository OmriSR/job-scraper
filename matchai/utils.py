"""Shared utilities for matchai."""

from langchain_ollama import ChatOllama

from matchai.config import OLLAMA_MODEL, OLLAMA_TEMPERATURE


def get_llm() -> ChatOllama:
    """Initialize Ollama LLM with configured settings."""
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMPERATURE,
    )
