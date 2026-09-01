"""
Ties parser -> chunker together and persists results to data/processed/ as
JSON, so downstream phases (embedding, Chroma indexing) don't need to
re-parse raw HTML on every run -- parsing a 13MB filing is not free.
"""

from __future__ import annotations

from pathlib import Path

from docintel.ingestion.chunker import merge_into_chunks
from docintel.ingestion.models import ParsedDocument
from docintel.ingestion.parser import parse_filing

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def process_filing(path: Path) -> ParsedDocument:
    """Parse one filing and replace its raw chunks with merged, section-aware ones."""
    doc = parse_filing(path)
    merged_chunks = merge_into_chunks(doc.chunks, doc.metadata)
    return ParsedDocument(metadata=doc.metadata, chunks=merged_chunks, tables=doc.tables)


def save_processed(doc: ParsedDocument, output_dir: Path = PROCESSED_DIR) -> Path:
    """Write a ParsedDocument to disk as JSON, named after ticker + filing date."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{doc.metadata.ticker}_{doc.metadata.filing_date}.json"
    out_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def run_pipeline(raw_dir: Path = RAW_DIR, output_dir: Path = PROCESSED_DIR) -> list[Path]:
    """Process every filing in raw_dir and persist results. Returns output paths."""
    filing_paths = sorted(raw_dir.glob("*.htm")) + sorted(raw_dir.glob("*.pdf"))
    if not filing_paths:
        raise FileNotFoundError(
            f"No .htm or .pdf filings found in {raw_dir}. "
            f"Run scripts/download_filings.py first."
        )

    output_paths = []
    for path in filing_paths:
        print(f"[processing] {path.name}")
        doc = process_filing(path)
        out_path = save_processed(doc, output_dir)
        print(
            f"[saved] {out_path.name} "
            f"({len(doc.chunks)} chunks, {len(doc.tables)} tables)"
        )
        output_paths.append(out_path)

    return output_paths


if __name__ == "__main__":
    run_pipeline()