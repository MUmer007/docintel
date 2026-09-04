"""
Loads messy HTML tables (extracted by `unstructured` from SEC filings) into
clean pandas DataFrames, then persists them into DuckDB for the agent's SQL
tool.

SEC filing tables are notoriously messy: many empty spacer cells (used for
visual alignment of currency symbols in the original document, not real
data), currency symbols split into their own cells adjacent to their value,
and header content that isn't in row 0 (often preceded by blank spacer
rows). This module does best-effort structural cleanup:

  1. Find the first row with any real (non-null) content -- SEC tables often
     have a fully blank row 0 used for visual spacing in the original doc.
  2. Use that row as headers; forward-fill blank header cells (a header like
     "September 2025" is often followed by blank cells for its paired
     currency-symbol column).
  3. Drop columns that are entirely currency symbols ("$") -- these exist
     only to visually align the symbol next to a number in the original
     table and carry no data of their own once merged conceptually with
     their neighboring value column.
  4. Drop fully-empty rows/columns after the above.

Tables that don't clean up into something with enough real rows/columns are
DROPPED rather than loaded as garbage -- an honest missing table is better
than a silently wrong SQL query result. We report the drop rate rather than
hiding it, since it's a genuine, measurable data-quality signal about how
much of this messy real-world corpus is machine-queryable as-is.

KNOWN LIMITATION: header-row detection assumes the first non-blank row is
the header row. For tables where currency-symbol columns get dropped BEFORE
a mostly-blank leading row is fully removed, the "first non-blank row" can
shift to a data row instead of the true header row, producing a cleaned
table with data values as column names. This was observed empirically (see
tests/unit/test_table_loader.py::test_known_limitation_header_misalignment)
on multi-year stock-performance comparison tables with irregular spacer-row
layouts. A more robust fix would score candidate header rows by "looks like
a label" (text-heavy, low numeric density) rather than "first non-blank" --
left as a documented follow-up rather than pursued further here, since (a)
most tables in this corpus are simpler single-header-row layouts that clean
up correctly, and (b) chasing every edge case in a fundamentally ambiguous
format (SEC HTML tables have no reliable semantic header markup) has
diminishing returns relative to project scope.
"""

from __future__ import annotations

import re
from io import StringIO

import pandas as pd

from docintel.ingestion.models import ExtractedTable

MIN_USEFUL_COLUMNS = 2
MIN_USEFUL_ROWS = 1

_CURRENCY_ONLY_PATTERN = re.compile(r"^\s*[$€£¥]\s*$")


def _is_currency_symbol_or_empty(value: object) -> bool:
    text = str(value).strip()
    return bool(_CURRENCY_ONLY_PATTERN.match(text)) or text in ("", "nan", "None")


def _find_header_row_index(df: pd.DataFrame) -> int | None:
    """First row index with at least one non-null cell, or None if the whole table is empty."""
    for i in range(len(df)):
        if df.iloc[i].notna().any():
            return i
    return None


def _promote_header_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use the first non-blank row as column headers (forward-filled to cover
    paired currency-symbol columns), then drop that row from the data body.
    """
    header_idx = _find_header_row_index(df)
    if header_idx is None:
        return df

    header_row = df.iloc[header_idx].ffill()
    new_columns = [
        str(val) if pd.notna(val) else f"col_{i}" for i, val in enumerate(header_row)
    ]
    df = df.iloc[header_idx + 1 :].copy()
    df.columns = new_columns
    return df


def _drop_currency_symbol_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [col for col in df.columns if df[col].apply(_is_currency_symbol_or_empty).all()]
    return df.drop(columns=cols_to_drop) if cols_to_drop else df


def _drop_empty_rows_and_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    return df


def _drop_currency_symbol_columns_using_body(df: pd.DataFrame, header_idx: int) -> pd.DataFrame:
    """
    Detect all-currency-symbol columns using only the DATA rows (after the
    header row), not the header row itself -- the header row's own cells
    (e.g. "September 2020") would never match the currency pattern, so this
    must run before header promotion overwrites column identity.
    """
    body = df.iloc[header_idx + 1 :]
    cols_to_drop = [
        col for col in df.columns if body[col].apply(_is_currency_symbol_or_empty).all()
    ]
    return df.drop(columns=cols_to_drop) if cols_to_drop else df


def clean_table(table: ExtractedTable) -> pd.DataFrame | None:
    """
    Attempt to parse and clean one extracted table's HTML into a usable
    DataFrame with real column headers. Returns None if the table doesn't
    clean up into something SQL-worthy.
    """
    if not table.html.strip():
        return None

    try:
        dfs = pd.read_html(StringIO(table.html))
    except ValueError:
        return None

    if not dfs:
        return None

    df = dfs[0]
    header_idx = _find_header_row_index(df)
    if header_idx is None:
        return None

    df = _drop_currency_symbol_columns_using_body(df, header_idx)
    df = _promote_header_row(df)
    df = _drop_empty_rows_and_columns(df)

    if df.shape[1] < MIN_USEFUL_COLUMNS or df.shape[0] < MIN_USEFUL_ROWS:
        return None

    return df