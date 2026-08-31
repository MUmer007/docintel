from pathlib import Path

import pytest

from docintel.ingestion.parser import parse_filing

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


@pytest.mark.integration
def test_parse_real_aapl_filing_produces_chunks_and_tables() -> None:
    candidates = list(RAW_DATA_DIR.glob("AAPL_10K_*.htm"))
    if not candidates:
        pytest.skip("No AAPL 10-K found in data/raw/ -- run scripts/download_filings.py first")

    doc = parse_filing(candidates[0])

    assert doc.metadata.ticker == "AAPL"
    assert len(doc.chunks) > 100
    assert len(doc.tables) > 5
    assert all(chunk.text.strip() for chunk in doc.chunks)
    assert all(table.text.strip() for table in doc.tables)