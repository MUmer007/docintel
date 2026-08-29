"""
Centralized, typed configuration.

Design notes for reviewers:
- All config is env-driven (12-factor), never hardcoded, never committed.
- Split into logical sub-settings so each module only depends on what it needs
  (e.g. the retrieval layer doesn't need to know about Groq API keys).
- `Settings` is a singleton accessed via `get_settings()`, cached, so we don't
  re-parse env vars on every call but tests can still override it cleanly.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    groq_api_key: str = Field(default="", description="Primary generation provider")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    anthropic_api_key: str = Field(default="", description="Used for independent LLM-as-judge")
    judge_model: str = Field(default="claude-sonnet-4-6")
    openai_api_key: str = Field(default="", description="Optional fallback / embeddings")

    request_timeout_s: float = 30.0
    max_retries: int = 3


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", extra="ignore")

    chroma_persist_dir: Path = PROJECT_ROOT / "data" / "chroma"
    collection_name: str = "docintel_chunks"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_final: int = 5
    use_hybrid_search: bool = True
    use_reranker: bool = True


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OTEL_", extra="ignore")

    service_name: str = "docintel"
    otlp_endpoint: str = ""  # empty = log spans locally instead of exporting
    log_level: str = "INFO"
    trace_sample_rate: float = 1.0


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    environment: str = Field(default="development")
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()