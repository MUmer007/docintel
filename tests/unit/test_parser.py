from pathlib import Path

import pytest

from docintel.ingestion.models import SourceType
from docintel.ingestion.parser import _metadata_from_filename


@pytest.mark.unit
def test_metadata_from_filename_parses_ticker_date_type() -> None:
    path = Path("AAPL_10K_2025-10-31.htm")
    metadata = _metadata_from_filename(path)
    assert metadata.ticker == "AAPL"
    assert metadata.filing_date == "2025-10-31"
    assert metadata.source_type == SourceType.SEC_10K


@pytest.mark.unit
def test_metadata_from_filename_rejects_bad_pattern() -> None:
    with pytest.raises(ValueError, match="doesn't match expected pattern"):
        _metadata_from_filename(Path("not_a_valid_filename.htm"))