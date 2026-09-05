import pytest

from docintel.evals.retrieval_metrics import compute_retrieval_metrics


@pytest.mark.unit
def test_perfect_retrieval() -> None:
    metrics = compute_retrieval_metrics(
        retrieved_chunk_ids=["a", "b", "c"], relevant_chunk_ids=["a"]
    )
    assert metrics.recall_at_k == 1.0
    assert metrics.reciprocal_rank == 1.0  # found at rank 1


@pytest.mark.unit
def test_relevant_chunk_found_at_rank_3() -> None:
    metrics = compute_retrieval_metrics(
        retrieved_chunk_ids=["x", "y", "a"], relevant_chunk_ids=["a"]
    )
    assert metrics.recall_at_k == 1.0
    assert metrics.reciprocal_rank == pytest.approx(1 / 3)


@pytest.mark.unit
def test_relevant_chunk_not_found() -> None:
    metrics = compute_retrieval_metrics(
        retrieved_chunk_ids=["x", "y", "z"], relevant_chunk_ids=["a"]
    )
    assert metrics.recall_at_k == 0.0
    assert metrics.reciprocal_rank == 0.0


@pytest.mark.unit
def test_precision_counts_only_relevant_hits() -> None:
    metrics = compute_retrieval_metrics(
        retrieved_chunk_ids=["a", "x", "y"], relevant_chunk_ids=["a", "b"]
    )
    assert metrics.precision_at_k == pytest.approx(1 / 3)  # 1 of 3 retrieved was relevant
    assert metrics.recall_at_k == pytest.approx(1 / 2)  # 1 of 2 relevant was retrieved


@pytest.mark.unit
def test_empty_relevant_ids_raises() -> None:
    with pytest.raises(ValueError, match="undefined"):
        compute_retrieval_metrics(retrieved_chunk_ids=["a"], relevant_chunk_ids=[])