"""
Wraps hybrid_search as an agent-callable tool, so the agent can decide
whether/when to search filing text -- as opposed to Phase 4's rag.py, which
always retrieves first. This gives the agent the ability to skip retrieval
entirely for pure-math follow-ups, or call it multiple times for multi-part
questions.
"""

from __future__ import annotations

from docintel.retrieval.fusion import hybrid_search

RETRIEVAL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_filings",
        "description": "Search filing text for qualitative info (risks, strategy, description).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A natural-language search query."}
            },
            "required": ["query"],
        },
    },
}


def search_filings(query: str, top_k: int = 3) -> list[dict[str, str | float]]:
    """Tool-callable wrapper around hybrid_search, returning plain dicts for JSON serialization."""
    results = hybrid_search(query, top_k=top_k)
    return [
        {
            "chunk_id": r.chunk_id,
            "text": r.text,
            "ticker": str(r.metadata.get("ticker", "")),
            "filing_date": str(r.metadata.get("filing_date", "")),
        }
        for r in results
    ]