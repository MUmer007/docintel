"""
Thin wrapper around the Groq client with retry logic.

Why wrap rather than call the SDK directly: every call site gets consistent
retry/backoff behavior and a single place to swap providers later (the
generation layer depends on this module's interface, not on Groq
specifically -- if we wanted to A/B test against OpenAI, only this file
would need a new implementation).
"""

from __future__ import annotations

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from docintel.core.config import get_settings

_client: Groq | None = None


def get_groq_client() -> Groq:
    global _client
    if _client is None:
        settings = get_settings()
        _client = Groq(api_key=settings.llm.groq_api_key, timeout=settings.llm.request_timeout_s)
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def complete(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """
    Single-turn completion. Retries on transient failures (rate limits,
    timeouts) with exponential backoff, up to 3 attempts.

    temperature=0.0 by default: for a RAG Q&A system over financial filings,
    we want deterministic, low-creativity answers, not varied phrasing.
    """
    settings = get_settings()
    client = get_groq_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model or settings.llm.groq_model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    return content or ""