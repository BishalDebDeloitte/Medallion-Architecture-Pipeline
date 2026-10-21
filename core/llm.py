"""Construct chat models for the configured provider on demand."""

from typing import Any

from core import config


_PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY", "github": "GITHUB_TOKEN", "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY", "gemini": "GOOGLE_API_KEY",
}


def configuration_problem() -> str | None:
    """Describe what is missing for the configured provider, or None if it can be used."""
    provider = config.LLM_PROVIDER
    key_name = _PROVIDER_KEYS.get(provider)
    if key_name is None:
        return f"Unsupported LLM provider: {provider}"
    if not getattr(config, key_name, None):
        return f"{key_name} is not set (provider: {provider}). Add it to the .env file and restart the app."
    return None


def create_chat_model() -> Any:
    """Create a LangChain chat model or fail clearly when its key is missing."""
    provider = config.LLM_PROVIDER
    if provider == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required for the anthropic provider")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=0,
        )
    if provider == "github":
        if not config.GITHUB_TOKEN:
            raise ValueError("GITHUB_TOKEN is required for the github provider")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.GITHUB_MODEL,
            api_key=config.GITHUB_TOKEN,
            base_url=config.GITHUB_BASE_URL,
            temperature=0,
        )
    if provider == "openai":
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for the openai provider")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0,
        )
    if provider == "groq":
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required for the groq provider")
        from langchain_groq import ChatGroq

        options = {
            "model": config.GROQ_MODEL,
            "api_key": config.GROQ_API_KEY,
            "temperature": 0,
        }
        if config.GROQ_MODEL.startswith("openai/gpt-oss-"):
            options["reasoning_format"] = "hidden"
        return ChatGroq(**options)
    if provider == "gemini":
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required for the gemini provider")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")
