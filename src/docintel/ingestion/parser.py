"""
Parses raw SEC filing HTML into structured TextChunks and ExtractedTables.

We use `unstructured`'s HTML partitioner (validated interactively in
scripts/explore_parsing.py before writing this) because it correctly
separates prose from tables in iXBRL filings, which a naive HTML-tag-strip
approach does not do reliably.
"""

from __future__ import annotations

import re
from pathlib import Path

from unstructured.documents.elements import ListItem, NarrativeText, Table, Text
from unstructured.partition.html import partition_html

from docintel.ingestion.models import (
    DocumentMetadata,
    ExtractedTable,
    ParsedDocument,
    SourceType,
    TextChunk,
)

# Matches filenames like "AAPL_10K_2025-10-31.htm"
FILENAME_PATTERN = re.compile(r"^([A-Z]+)_10([KQ])_(\d{4}-\d{2}-\d{2})")


def _metadata_from_filename(path: Path) -> DocumentMetadata:
    """Derive ticker/date/form-type from our own download naming convention."""
    match = FILENAME_PATTERN.match(path.stem)
    if not match:
        raise ValueError(
            f"Filename '{path.name}' doesn't match expected pattern "
            f"'<TICKER>_10<K|Q>_<YYYY-MM-DD>...'"
        )
    ticker, form_letter, filing_date = match.groups()
    source_type = SourceType.SEC_10K if form_letter == "K" else SourceType.SEC_10Q
    return DocumentMetadata(
        ticker=ticker,
        filing_date=filing_date,
        source_type=source_type,
        source_path=str(path),
    )


def parse_filing(path: Path) -> ParsedDocument:
    """Parse one raw SEC filing HTML file into chunks + tables with metadata."""
    metadata = _metadata_from_filename(path)
    elements = partition_html(filename=str(path))

    chunks: list[TextChunk] = []
    tables: list[ExtractedTable] = []

    for position, element in enumerate(elements):
        text = str(element).strip()
        if not text:
            continue  # skip empty elements (unstructured produces a few)

        if isinstance(element, Table):
            tables.append(
                ExtractedTable(
                    table_id=f"{metadata.ticker}_{metadata.filing_date}_table_{position}",
                    html=getattr(element.metadata, "text_as_html", "") or "",
                    text=text,
                    position=position,
                    metadata=metadata,
                )
            )
        elif isinstance(element, (NarrativeText, Text, ListItem)):
            chunks.append(
                TextChunk(
                    chunk_id=f"{metadata.ticker}_{metadata.filing_date}_chunk_{position}",
                    text=text,
                    element_type=type(element).__name__,
                    position=position,
                    metadata=metadata,
                )
            )
        # Other element types (e.g. Image) are intentionally skipped for now --
        # a real production system might OCR images or extract alt-text.

    return ParsedDocument(metadata=metadata, chunks=chunks, tables=tables)