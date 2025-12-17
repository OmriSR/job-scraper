"""Shared utilities for matchai."""

from langchain_groq import ChatGroq

from matchai.config import GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE

_llm_instance: ChatGroq | None = None


class LLMConfigurationError(Exception):
    """Raised when LLM is not properly configured."""

    pass


def check_llm_configured() -> None:
    """Check if Groq API key is configured.

    Raises:
        LLMConfigurationError: If GROQ_API_KEY is not set.
    """
    if not GROQ_API_KEY:
        raise LLMConfigurationError(
            "GROQ_API_KEY environment variable is not set. "
            "Please create a .env file with your Groq API key. "
            "Get your free API key at https://console.groq.com"
        )


def get_llm() -> ChatGroq:
    """Get singleton Groq LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatGroq(
            model=GROQ_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=GROQ_API_KEY,
        )
    return _llm_instance
