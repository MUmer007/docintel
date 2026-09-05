"""
LLM-as-judge: scores generated RAG answers against explicit rubrics using a
model independent of the generator where possible.

Independence tradeoff (read before trusting these scores blindly):
- IDEAL: judge with a different provider/lab entirely (e.g. Claude judging
  Groq-generated answers). This is the strongest mitigation against
  self-preference bias -- a model rating its own family's outputs more
  favorably than an independent judge would.
- FALLBACK (what this module defaults to): judge with a DIFFERENT MODEL on
  the SAME PROVIDER (Groq) as the generator, e.g. Qwen judging gpt-oss
  outputs. This is a partial mitigation only -- different training data and
  architecture reduce but do not eliminate shared-infrastructure bias
  (e.g. both models may have been RLHF-tuned with similar preference data
  or share safety-training artifacts). This fallback was adopted here
  because Anthropic billing was not configured; Claude-as-judge remains
  available and preferred if LLM_ANTHROPIC_API_KEY + billing are set up
  (see get_settings().llm.anthropic_api_key).
- We mitigate residual bias with an explicit, narrow rubric per dimension
  rather than an open-ended "rate 1-10" prompt, but this does not fully
  substitute for true cross-provider independence or human evaluation.

KNOWN LIMITATION -- judge context handling: observed empirically on a real
query ("Where is Apple headquartered?"), the judge scored faithfulness=1 and
citation_accuracy=1 despite the correct answer chunk being genuinely present
in the 5-chunk context passed to it (confirmed by inspecting retrieved_chunks
directly). The context is joined from 5 chunks and truncated at 4000 chars
(see context[:4000] below) before reaching the judge; either the truncation
cut off the relevant sentence, or the smaller judge model (Qwen, chosen as a
free-tier fallback -- see independence tradeoff above) failed to locate the
answer within a longer multi-chunk context. Not investigated further here;
flagged as a concrete example of why LLM-as-judge scores should be treated
as a noisy signal to spot-check against, not ground truth -- a lesson this
project surfaces empirically rather than asserting abstractly.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from docintel.core.config import get_settings
from docintel.generation.llm_client import get_groq_client


class JudgeScore(BaseModel):
    faithfulness: int
    relevance: int
    citation_accuracy: int
    reasoning: str


JUDGE_PROMPT_TEMPLATE = """Evaluate this RAG system's answer.

Question: {question}

Context provided:
{context}

Answer: {answer}

Cited chunk IDs: {citations}

Score 1-5 (5=best) on:
1. faithfulness: every claim supported by context, no hallucination
2. relevance: answer addresses the question
3. citation_accuracy: cited IDs plausibly support the claims

Respond with ONLY JSON, no other text:
{{"faithfulness": <1-5>, "relevance": <1-5>, "citation_accuracy": <1-5>, \
"reasoning": "<one sentence>"}}
"""


def judge_answer(question: str, context: str, answer: str, citations: list[str]) -> JudgeScore:
    """Score a generated RAG answer using the configured judge model (see module
    docstring for the independence tradeoff of the default Groq-based judge)."""
    settings = get_settings()
    client = get_groq_client()

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        context=context[:4000],
        answer=answer,
        citations=", ".join(citations) if citations else "(none)",
    )

    response = client.chat.completions.create(
        model=settings.llm.judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    )
    raw_text = response.choices[0].message.content or ""
    cleaned = (
        raw_text.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )

    try:
        parsed = json.loads(cleaned)
        return JudgeScore(**parsed)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Judge returned unparseable response: {raw_text[:300]}") from e
