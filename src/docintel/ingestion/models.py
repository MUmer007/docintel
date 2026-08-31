"""
Data model for ingested document content.

Design rationale: we keep prose chunks and tables as distinct types rather
than forcing everything into one "Chunk" shape. A table flattened into prose
text loses its structure and becomes useless for both retrieval (embeddings
don't represent tabular relationships well) and for the agent's SQL tool
(Phase 5), which needs the table's actual rows/columns intact.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    SEC_10K = "sec_10k"
    SEC_10Q = "sec_10q"


class DocumentMetadata(BaseModel):
    """Metadata common to everything extracted from one source filing."""

    ticker: str
    filing_date: str  # ISO date string, e.g. "2025-10-31"
    source_type: SourceType
    source_path: str  # original file path, for traceability/debugging


class TextChunk(BaseModel):
    """A prose chunk destined for the vector store."""

    chunk_id: str
    text: str
    element_type: str  # "NarrativeText" | "Text" | "ListItem" (from unstructured)
    position: int  # order within the document, for citation/debugging
    metadata: DocumentMetadata


class ExtractedTable(BaseModel):
    """
    A table extracted intact (not chunked). Kept as HTML/text representation
    for now; Phase 2/5 will add structured row/column extraction into DuckDB.
    """

    table_id: str
    html: str  # raw HTML representation from `unstructured`, preserves structure
    text: str  # plain-text fallback representation
    position: int
    metadata: DocumentMetadata


class ParsedDocument(BaseModel):
    """Result of parsing one source filing."""

    metadata: DocumentMetadata
    chunks: list[TextChunk] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)