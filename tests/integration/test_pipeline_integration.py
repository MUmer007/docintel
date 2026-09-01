from pathlib import Path

import pytest

from docintel.ingestion.models import ParsedDocument
from docintel.ingestion.pipeline import process_filing, save_processed

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


@pytest.mark.integration
def test_pipeline_round_trips_unicode_content(tmp_path: Path) -> None:
    """
    Regression test: SEC filings contain non-ASCII characters (smart quotes,
    em-dashes, trademark symbols). Reading/writing processed JSON must use
    explicit UTF-8 -- Windows defaults to cp1252, which crashes on these
    characters. This test would have caught that bug before it shipped.
    """
    candidates = list(RAW_DATA_DIR.glob("AAPL_10K_*.htm"))
    if not candidates:
        pytest.skip("No AAPL 10-K found in data/raw/ -- run scripts/download_filings.py first")

    doc = process_filing(candidates[0])
    out_path = save_processed(doc, output_dir=tmp_path)

    # Reload using explicit UTF-8, exactly as downstream phases will.
    reloaded = ParsedDocument.model_validate_json(out_path.read_text(encoding="utf-8"))

    assert reloaded.metadata.ticker == doc.metadata.ticker
    assert len(reloaded.chunks) == len(doc.chunks)
    assert reloaded.chunks[0].text == doc.chunks[0].text