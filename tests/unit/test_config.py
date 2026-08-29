import pytest

from docintel.core.config import AppSettings, get_settings


@pytest.mark.unit
def test_settings_load_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should load even with no .env present, using safe defaults."""
    monkeypatch.delenv("LLM_GROQ_API_KEY", raising=False)
    settings = AppSettings()
    assert settings.environment == "development"
    assert settings.retrieval.top_k_final == 5
    assert settings.retrieval.use_hybrid_search is True


@pytest.mark.unit
def test_get_settings_is_cached() -> None:
    """get_settings() should return the same object on repeated calls (lru_cache)."""
    assert get_settings() is get_settings()


@pytest.mark.unit
def test_llm_settings_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM_ prefixed env vars should populate the nested LLMSettings correctly."""
    monkeypatch.setenv("LLM_GROQ_MODEL", "llama-3.1-8b-instant")
    settings = AppSettings()
    assert settings.llm.groq_model == "llama-3.1-8b-instant"