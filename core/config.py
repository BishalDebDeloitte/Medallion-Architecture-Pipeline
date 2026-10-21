"""Central project paths and environment-based provider configuration."""

import os

from dotenv import load_dotenv
from core.paths import (
    BASE_DIR, DATA_DIR, LANDING_DIR, PROFILES_DIR, STTM_DIR, BRONZE_DIR,
    SILVER_DIR, GOLD_DIR, TRACES_DIR, REPORTS_DIR, RUNS_DIR, LAYER_REPORTS_DIR, AUDIT_DIR, CHROMA_DIR,
)


load_dotenv(BASE_DIR / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_MODEL = os.getenv("GITHUB_MODEL", "openai/gpt-4.1-mini")
GITHUB_BASE_URL = "https://models.inference.ai.azure.com"

_SUPPORTED_PROVIDERS = {"anthropic", "github", "openai", "groq", "gemini"}
_CONFIGURED_PROVIDER = os.getenv("LLM_PROVIDER")
if _CONFIGURED_PROVIDER:
    LLM_PROVIDER = _CONFIGURED_PROVIDER.strip().lower()
    if LLM_PROVIDER not in _SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(_SUPPORTED_PROVIDERS))
        raise ValueError(f"Unsupported LLM_PROVIDER {LLM_PROVIDER!r}; choose from: {supported}")
else:
    _PROVIDER_KEYS = (
        ("anthropic", ANTHROPIC_API_KEY),
        ("github", GITHUB_TOKEN),
        ("openai", OPENAI_API_KEY),
        ("groq", GROQ_API_KEY),
        ("gemini", GOOGLE_API_KEY),
    )
    LLM_PROVIDER = next(
        (provider for provider, api_key in _PROVIDER_KEYS if api_key),
        "anthropic",
    )

for _directory in (
    DATA_DIR,
    LANDING_DIR,
    PROFILES_DIR,
    STTM_DIR,
    BRONZE_DIR,
    SILVER_DIR,
    GOLD_DIR,
    TRACES_DIR,
    REPORTS_DIR,
    RUNS_DIR,
    LAYER_REPORTS_DIR,
    AUDIT_DIR,
    CHROMA_DIR,
):
    _directory.mkdir(parents=True, exist_ok=True)
