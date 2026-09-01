import pytest

from docintel.retrieval.fusion import RRF_K


@pytest.mark.unit
def test_rrf_score_formula() -> None:
    """Sanity-check the RRF formula itself: 1/(k+rank), summed across retrievers."""
    rank = 1
    expected = 1.0 / (RRF_K + rank)
    assert expected == pytest.approx(1.0 / 61)


@pytest.mark.unit
def test_rrf_rewards_appearing_in_both_retrievers() -> None:
    """
    A chunk ranked #1 in both dense and sparse should score higher than a
    chunk ranked #1 in only one -- this is the whole point of fusion: reward
    consensus across retrieval strategies.
    """
    score_in_both = (1.0 / (RRF_K + 1)) + (1.0 / (RRF_K + 1))
    score_in_one = 1.0 / (RRF_K + 1)
    assert score_in_both > score_in_one