import pytest

from docintel.evals.runner import EvalResult, summarize_results


def _make_result(**overrides: object) -> EvalResult:
    defaults: dict[str, object] = {
        "question_id": "q_test",
        "question": "test?",
        "question_type": "qualitative",
        "answer": "test answer",
        "citations": [],
        "insufficient_context": False,
    }
    defaults.update(overrides)
    return EvalResult(**defaults)  # type: ignore[arg-type]


@pytest.mark.unit
def test_summarize_computes_mean_recall() -> None:
    results = [_make_result(recall_at_k=1.0), _make_result(recall_at_k=0.5)]
    summary = summarize_results(results)
    assert summary["mean_recall_at_k"] == pytest.approx(0.75)


@pytest.mark.unit
def test_summarize_skips_missing_metrics_rather_than_treating_as_zero() -> None:
    """A question with no gold chunk labels has recall_at_k=None, not 0.0 --
    it must not silently drag down the mean as if retrieval failed."""
    results = [_make_result(recall_at_k=1.0), _make_result(recall_at_k=None)]
    summary = summarize_results(results)
    assert summary["mean_recall_at_k"] == 1.0  # only the one labeled question counts


@pytest.mark.unit
def test_summarize_omits_metric_entirely_when_no_data_available() -> None:
    results = [_make_result()]  # no recall_at_k on any result
    summary = summarize_results(results)
    assert "mean_recall_at_k" not in summary


@pytest.mark.unit
def test_summarize_refusal_accuracy() -> None:
    results = [
        _make_result(correctly_refused=True),
        _make_result(correctly_refused=True),
        _make_result(correctly_refused=False),
    ]
    summary = summarize_results(results)
    assert summary["refusal_accuracy"] == pytest.approx(2 / 3)