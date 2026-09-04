import pytest

from docintel.agents.tools.calculator import calculate


@pytest.mark.unit
def test_calculate_basic_arithmetic() -> None:
    result = calculate("2 + 2")
    assert result.result == 4.0
    assert result.error is None


@pytest.mark.unit
def test_calculate_percentage_growth() -> None:
    result = calculate("(391035 - 383285) / 383285 * 100")
    assert result.result is not None
    assert result.result == pytest.approx(2.02, abs=0.01)


@pytest.mark.unit
def test_calculate_rejects_code_injection_attempt() -> None:
    """
    Security property: numexpr must never execute arbitrary Python, since
    expressions here originate from LLM tool-call output (untrusted input).
    """
    result = calculate('__import__("os").system("echo pwned")')
    assert result.result is None
    assert result.error is not None


@pytest.mark.unit
def test_calculate_rejects_malformed_expression() -> None:
    result = calculate("not a valid expression!!!")
    assert result.result is None
    assert result.error is not None