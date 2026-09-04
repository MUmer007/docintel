"""
Loads cleaned tables into a DuckDB database file, providing the agent's SQL
tool with a queryable structured-data backend.

Each cleaned table becomes one DuckDB table, named after its source
(ticker + table position) so the agent can discover what's available via
a schema-listing query before writing a SELECT.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

from docintel.agents.tools.table_loader import clean_table
from docintel.ingestion.models import ParsedDocument
from docintel.retrieval.indexer import PROCESSED_DIR, load_processed_documents

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DUCKDB_PATH = PROJECT_ROOT / "data" / "chroma" / "structured_data.duckdb"

_INVALID_TABLE_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_]")


def _safe_table_name(table_id: str) -> str:
    """DuckDB table names must be valid identifiers; sanitize our table_id."""
    name = _INVALID_TABLE_NAME_CHARS.sub("_", table_id)
    return f"t_{name}" if name[0].isdigit() else name


def build_sql_store(
    processed_dir: Path = PROCESSED_DIR, db_path: Path = DUCKDB_PATH
) -> dict[str, int]:
    """
    Clean and load every extracted table from every processed document into
    DuckDB. Returns a summary dict: {"loaded": N, "skipped": N} -- an honest
    count of how much of the raw table corpus became queryable, since not
    every messy source table survives cleanup (see table_loader.py).
    """
    docs: list[ParsedDocument] = load_processed_documents(processed_dir)
    if not docs:
        raise FileNotFoundError(
            f"No processed documents found in {processed_dir}. Run `docintel ingest run` first."
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()  # rebuild fresh each time -- avoids stale/duplicate tables across runs

    conn = duckdb.connect(str(db_path))
    loaded = 0
    skipped = 0
    table_registry: list[dict[str, str]] = []

    for doc in docs:
        for table in doc.tables:
            df = clean_table(table)
            if df is None:
                skipped += 1
                continue

            table_name = _safe_table_name(table.table_id)
            conn.register("temp_df", df)
            conn.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM temp_df')
            conn.unregister("temp_df")

            table_registry.append(
                {
                    "table_name": table_name,
                    "ticker": table.metadata.ticker,
                    "filing_date": table.metadata.filing_date,
                    "position": str(table.position),
                }
            )
            loaded += 1

    registry_df = pd.DataFrame(table_registry)
    conn.register("temp_registry", registry_df)
    conn.execute('CREATE TABLE "_table_registry" AS SELECT * FROM temp_registry')
    conn.unregister("temp_registry")

    conn.close()
    return {"loaded": loaded, "skipped": skipped}


def get_connection(db_path: Path = DUCKDB_PATH) -> duckdb.DuckDBPyConnection:
    if not db_path.exists():
        raise FileNotFoundError(
            f"No DuckDB store found at {db_path}. Run `docintel ingest index` first."
        )
    return duckdb.connect(str(db_path), read_only=True)


def list_available_tables(db_path: Path = DUCKDB_PATH) -> pd.DataFrame:
    """Returns the registry of loaded tables (name, ticker, filing_date, position)."""
    conn = get_connection(db_path)
    result = conn.execute("SELECT * FROM _table_registry").fetchdf()
    conn.close()
    return result


def run_sql_query(query: str, db_path: Path = DUCKDB_PATH) -> pd.DataFrame:
    """Execute a read-only SQL query against the structured data store."""
    conn = get_connection(db_path)
    try:
        return conn.execute(query).fetchdf()
    finally:
        conn.close()
# Tool schema for Groq function calling
SQL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_sql_query",
        "description": (
            "Run a read-only SQL query for precise figures; call list_available_tables first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A DuckDB SQL SELECT query.",
                }
            },
            "required": ["query"],
        },
    },
}

LIST_TABLES_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_available_tables",
        "description": "List available SQL tables (name, ticker, filing_date) before querying.",
        "parameters": {"type": "object", "properties": {}},
    },
}        