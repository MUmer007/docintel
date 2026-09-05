"""
Regression fixtures for known LLM-as-judge behavior, documented as findings
rather than pass/fail assertions on judge quality itself (judge output is
non-deterministic and model-dependent, so we don't assert exact scores).

These tests are marked `eval` (not `unit`) since they make real API calls
and are excluded from the fast unit-test loop; run explicitly via:
    uv run pytest tests/evals -v -m eval
"""

import pytest

from docintel.evals.llm_judge import judge_answer


@pytest.mark.eval
def test_known_case_correct_answer_with_relevant_context_present() -> None:
    """
    KNOWN FINDING (see llm_judge.py module docstring): on this real question,
    the judge scored faithfulness=1 despite the correct chunk being present
    in context. This test doesn't assert a "correct" score (there isn't a
    stable one to assert against a live model) -- it exists so this specific
    case is easy to re-run and compare if the judge prompt or model changes.
    """
    question = "Where is Apple headquartered?"
    context = (
        "Item 2. Properties\nThe Company's headquarters is located in "
        "Cupertino, California. As of September 27, 2025, the Company owned "
        "or leased facilities..."
    )
    answer = "Apple's headquarters is located in Cupertino, California."
    citations = ["AAPL_2025-10-31_merged_39"]

    score = judge_answer(question, context, answer, citations)

    # Documenting behavior, not asserting a specific "correct" outcome --
    # print for visibility when run manually/in CI logs.
    print(f"\nFaithfulness: {score.faithfulness}, Relevance: {score.relevance}, "
          f"Citation: {score.citation_accuracy}")
    print(f"Reasoning: {score.reasoning}")

    # The only hard assertion: the judge should at least recognize relevance,
    # since the answer directly addresses the question regardless of the
    # faithfulness-scoring quirk documented above.
    assert score.relevance >= 3