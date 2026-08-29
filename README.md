# DocIntel

Production-style RAG + agent platform for enterprise document Q&A, built with a first-class LLM eval harness.

**Status: Phase 0 complete** — project scaffold, typed config, CLI, CI.

## Quickstart

\`\`\`bash
uv sync
cp .env.example .env   # fill in LLM_GROQ_API_KEY, LLM_ANTHROPIC_API_KEY
uv run docintel info
uv run pytest tests/unit -v
\`\`\`