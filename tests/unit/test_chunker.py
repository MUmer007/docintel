import pytest

from docintel.ingestion.chunker import merge_into_chunks
from docintel.ingestion.models import DocumentMetadata, SourceType, TextChunk

METADATA = DocumentMetadata(
    ticker="TEST",
    filing_date="2025-01-01",
    source_type=SourceType.SEC_10K,
    source_path="test.htm",
)


def _make_chunk(text: str, position: int) -> TextChunk:
    return TextChunk(
        chunk_id=f"raw_{position}",
        text=text,
        element_type="Text",
        position=position,
        metadata=METADATA,
    )


@pytest.mark.unit
def test_merges_small_adjacent_elements_into_one_chunk() -> None:
    raw = [
        _make_chunk("Item 1. Business", 0),
        _make_chunk("We make software.", 1),
        _make_chunk("We have many customers.", 2),
    ]
    merged = merge_into_chunks(raw, METADATA, target_tokens=512)
    assert len(merged) == 1
    assert "Item 1. Business" in merged[0].text
    assert "We make software." in merged[0].text


@pytest.mark.unit
def test_never_merges_across_section_boundary() -> None:
    raw = [
        _make_chunk("Item 1. Business", 0),
        _make_chunk("We make software.", 1),
        _make_chunk("Item 1A. Risk Factors", 2),
        _make_chunk("Our business could suffer.", 3),
    ]
    merged = merge_into_chunks(raw, METADATA, target_tokens=512)
    assert len(merged) == 2
    assert "Item 1. Business" in merged[0].text
    assert "Item 1A. Risk Factors" not in merged[0].text
    assert "Item 1A. Risk Factors" in merged[1].text


@pytest.mark.unit
def test_oversized_single_paragraph_kept_intact_not_split() -> None:
    """
    Design decision: target_tokens is a SOFT cap. A single source paragraph
    longer than target_tokens is preserved whole rather than split mid-
    sentence, because a coherent oversized chunk retrieves better than a
    fragment truncated mid-thought.
    """
    long_paragraph = "This is a very important sentence. " * 200  # well over 512 tokens
    raw = [_make_chunk(long_paragraph, 0)]
    merged = merge_into_chunks(raw, METADATA, target_tokens=512)
    assert len(merged) == 1
    assert merged[0].text.strip() == long_paragraph.strip()