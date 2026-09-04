import pytest

from docintel.agents.orchestrator import MAX_TOOL_RESULT_CHARS, _truncate


@pytest.mark.unit
def test_truncate_leaves_short_text_unchanged() -> None:
    short_text = "hello world"
    assert _truncate(short_text) == short_text


@pytest.mark.unit
def test_truncate_cuts_long_text_and_notes_original_length() -> None:
    long_text = "x" * (MAX_TOOL_RESULT_CHARS + 500)
    result = _truncate(long_text)
    assert len(result) < len(long_text)
    assert "truncated" in result
    assert str(len(long_text)) in result