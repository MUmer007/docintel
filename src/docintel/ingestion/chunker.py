"""
    Merge small `unstructured` elements into larger chunks, respecting
    section boundaries (never merging across an "Item N." header) and
    roughly targeting `target_tokens` per chunk with `overlap_tokens` of
    trailing context carried into the next chunk for retrieval continuity.

    `target_tokens` is a SOFT target, not a hard cap: if a single source
    element (e.g. one long Risk Factor paragraph) already exceeds
    target_tokens, we keep it whole rather than splitting mid-sentence.
    A slightly oversized-but-coherent chunk retrieves and reads better than
    a chunk truncated mid-thought. In practice this means a small number of
    chunks per document will exceed target_tokens -- that's expected.
    """

from __future__ import annotations

import re

import tiktoken

from docintel.ingestion.models import DocumentMetadata, TextChunk

# Matches SEC 10-K/10-Q section headers, e.g. "Item 1A. Risk Factors"
SECTION_HEADER_PATTERN = re.compile(r"^Item\s+\d+[A-Z]?\.?\s", re.IGNORECASE)

_ENCODER = tiktoken.get_encoding("cl100k_base")  # tokenizer used just for length estimates


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _is_section_boundary(text: str) -> bool:
    return bool(SECTION_HEADER_PATTERN.match(text.strip()))


def merge_into_chunks(
    raw_chunks: list[TextChunk],
    metadata: DocumentMetadata,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[TextChunk]:
    """
    Merge small `unstructured` elements into larger chunks, respecting
    section boundaries (never merging across an "Item N." header) and
    roughly targeting `target_tokens` per chunk with `overlap_tokens` of
    trailing context carried into the next chunk for retrieval continuity.
    """
    merged: list[TextChunk] = []
    current_texts: list[str] = []
    current_tokens = 0
    current_start_pos = 0
    merged_index = 0

    def flush() -> None:
        nonlocal merged_index
        if not current_texts:
            return
        merged.append(
            TextChunk(
                chunk_id=f"{metadata.ticker}_{metadata.filing_date}_merged_{merged_index}",
                text="\n\n".join(current_texts),
                element_type="MergedSection",
                position=current_start_pos,
                metadata=metadata,
            )
        )
        merged_index += 1

    for raw in raw_chunks:
        text = raw.text.strip()
        is_boundary = _is_section_boundary(text)
        text_tokens = _count_tokens(text)

        would_exceed = current_tokens + text_tokens > target_tokens
        must_break = is_boundary and current_texts  # new section starts -> close prior chunk

        if must_break or (would_exceed and current_texts):
            flush()
            # carry a small overlap of the previous chunk's tail forward for context continuity,
            # unless we just crossed a section boundary (overlap across sections isn't meaningful)
            if not must_break and overlap_tokens > 0 and current_texts:
                tail_text = current_texts[-1]
                tail_tokens = _count_tokens(tail_text)
                if tail_tokens <= overlap_tokens:
                    current_texts = [tail_text]
                    current_tokens = tail_tokens
                else:
                    current_texts = []
                    current_tokens = 0
            else:
                current_texts = []
                current_tokens = 0
            current_start_pos = raw.position

        if not current_texts:
            current_start_pos = raw.position
        current_texts.append(text)
        current_tokens += text_tokens

    flush()  # don't lose the final in-progress chunk
    return merged