"""
RAG generation: takes a question + retrieved chunks, produces a structured,
cited answer.

Design choices:
- JSON-structured output (not free text) so citations are machine-parseable
  and we can validate/display them, not just trust the model said something
  citation-shaped in prose.
- Explicit "insufficient_context" flag: the model is instructed to say it
  doesn't know rather than hallucinate when retrieved chunks don't actually
  answer the question. We measure how well it follows this in Phase 6 evals.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from docintel.generation.llm_client import complete
from docintel.retrieval.fusion import RetrievedChunk, hybrid_search

SYSTEM_PROMPT = """You are a financial research assistant answering questions \
about SEC filings (10-K annual reports) using ONLY the provided context chunks.

Rules:
1. Answer using ONLY information present in the provided context. Do not use \
outside knowledge about these companies.
2. Every claim in your answer must be traceable to a specific chunk. Cite \
chunk IDs in the "citations" field.
3. If the context does not contain enough information to answer the question, \
set "insufficient_context" to true and explain what's missing in "answer" -- \
do NOT guess or fill gaps with outside knowledge.
4. Respond with ONLY a JSON object matching this exact schema, no other text:
{"answer": "...", "citations": ["chunk_id_1", "chunk_id_2"], "insufficient_context": false}
"""


class RAGResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    answer: str
    citations: list[str] = Field(default_factory=list)
    insufficient_context: bool = False
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list, exclude=True)


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for chunk in chunks:
        ticker = chunk.metadata.get("ticker", "unknown")
        filing_date = chunk.metadata.get("filing_date", "unknown")
        parts.append(
            f"[chunk_id: {chunk.chunk_id}] (Source: {ticker} 10-K, {filing_date})\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, top_k: int = 5) -> RAGResponse:
    """Retrieve relevant chunks and generate a cited answer via Groq."""
    chunks = hybrid_search(question, top_k=top_k)

    if not chunks:
        return RAGResponse(
            answer="No relevant content was found in the indexed filings for this question.",
            citations=[],
            insufficient_context=True,
            retrieved_chunks=[],
        )

    context = _build_context_block(chunks)
    prompt = f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"

    raw_response = complete(prompt=prompt, system=SYSTEM_PROMPT, temperature=0.0, max_tokens=1024)

    # Groq's JSON-mode-less models can sometimes wrap output in markdown fences; strip defensively.
    cleaned = (
        raw_response.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Model didn't follow the JSON format -- surface this as insufficient rather than crash.
        return RAGResponse(
            answer=f"Model returned non-JSON output: {raw_response[:200]}",
            citations=[],
            insufficient_context=True,
            retrieved_chunks=chunks,
        )

    return RAGResponse(
        answer=parsed.get("answer", ""),
        citations=parsed.get("citations", []),
        insufficient_context=parsed.get("insufficient_context", False),
        retrieved_chunks=chunks,
    )