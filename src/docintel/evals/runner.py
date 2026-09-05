"""
Ties together retrieval metrics, generation quality (LLM-as-judge), and the
gold dataset into one runnable eval suite. This is the harness that turns
"we built a RAG system" into "we can measure whether changes make it better
or worse" -- the actual point of Phase 6.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from docintel.evals.llm_judge import judge_answer
from docintel.evals.retrieval_metrics import compute_retrieval_metrics
from docintel.generation.rag import answer_question

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GOLD_DATASET_PATH = PROJECT_ROOT / "data" / "eval_datasets" / "gold_qa.json"


class GoldQuestion(BaseModel):
    """One labeled question in the gold eval dataset (data/eval_datasets/gold_qa.json)."""

    id: str
    question: str
    question_type: str
    expected_answer_contains: list[str] = []
    relevant_chunk_ids: list[str] = []
    ticker: str | None = None
    should_refuse: bool = False
    known_correct_table: str | None = None


class EvalResult(BaseModel):
    question_id: str
    question: str
    question_type: str
    answer: str
    citations: list[str]
    insufficient_context: bool
    recall_at_k: float | None = None
    reciprocal_rank: float | None = None
    faithfulness: int | None = None
    relevance: int | None = None
    citation_accuracy: int | None = None
    correctly_refused: bool | None = None
    contains_expected_terms: bool | None = None


def _load_gold_dataset(path: Path = GOLD_DATASET_PATH) -> list[GoldQuestion]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return [GoldQuestion(**item) for item in raw]


def evaluate_single_question(item: GoldQuestion) -> EvalResult:
    """Run the full RAG pipeline on one gold question and score it against
    every metric we have ground truth or a judge available for."""
    response = answer_question(item.question)
    retrieved_ids = [c.chunk_id for c in response.retrieved_chunks]

    result = EvalResult(
        question_id=item.id,
        question=item.question,
        question_type=item.question_type,
        answer=response.answer,
        citations=response.citations,
        insufficient_context=response.insufficient_context,
    )

    # Retrieval metrics -- only computable when we have gold chunk labels
    if item.relevant_chunk_ids:
        metrics = compute_retrieval_metrics(retrieved_ids, item.relevant_chunk_ids)
        result.recall_at_k = metrics.recall_at_k
        result.reciprocal_rank = metrics.reciprocal_rank

    # Refusal correctness -- for out-of-scope questions, insufficient_context should be True
    if item.should_refuse:
        result.correctly_refused = response.insufficient_context

    # Expected-term coverage -- crude but useful "does the answer mention the right facts" check
    if item.expected_answer_contains:
        answer_lower = response.answer.lower()
        result.contains_expected_terms = any(
            term.lower() in answer_lower for term in item.expected_answer_contains
        )

    # LLM-as-judge scoring -- skip for out-of-scope questions where we WANT a refusal,
    # not a grounded answer (faithfulness scoring doesn't apply to "I don't know" responses)
    if not item.should_refuse and not response.insufficient_context:
        context = "\n\n".join(c.text for c in response.retrieved_chunks)
        try:
            judge_score = judge_answer(
                item.question, context, response.answer, response.citations
            )
            result.faithfulness = judge_score.faithfulness
            result.relevance = judge_score.relevance
            result.citation_accuracy = judge_score.citation_accuracy
        except ValueError:
            # judge failed to return parseable output -- leave scores as None rather than crash
            pass

    return result


def run_eval_suite(dataset_path: Path = GOLD_DATASET_PATH) -> list[EvalResult]:
    gold = _load_gold_dataset(dataset_path)
    return [evaluate_single_question(item) for item in gold]


def summarize_results(results: list[EvalResult]) -> dict[str, float | int]:
    """Aggregate metrics across the whole eval run for a top-line summary."""
    recalls = [r.recall_at_k for r in results if r.recall_at_k is not None]
    faithfulness_scores = [r.faithfulness for r in results if r.faithfulness is not None]
    relevance_scores = [r.relevance for r in results if r.relevance is not None]
    refusal_checks = [r.correctly_refused for r in results if r.correctly_refused is not None]
    term_checks = [
        r.contains_expected_terms for r in results if r.contains_expected_terms is not None
    ]

    summary: dict[str, float | int] = {"total_questions": len(results)}
    if recalls:
        summary["mean_recall_at_k"] = sum(recalls) / len(recalls)
    if faithfulness_scores:
        summary["mean_faithfulness"] = sum(faithfulness_scores) / len(faithfulness_scores)
    if relevance_scores:
        summary["mean_relevance"] = sum(relevance_scores) / len(relevance_scores)
    if refusal_checks:
        summary["refusal_accuracy"] = sum(refusal_checks) / len(refusal_checks)
    if term_checks:
        summary["expected_term_coverage"] = sum(term_checks) / len(term_checks)

    return summary

def save_results(
    results: list[EvalResult], summary: dict[str, float | int], output_path: Path
) -> None:
    """Persist eval results + summary to JSON for tracking over time (e.g. across
    prompt/model changes, or as CI artifacts for regression comparison)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "results": [r.model_dump() for r in results],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
