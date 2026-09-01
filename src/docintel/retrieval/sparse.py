"""
BM25 sparse retrieval -- catches exact-match queries dense embeddings are
weak at (specific figures like "$391 billion", tickers, exact phrases).
Dense embeddings encode semantic similarity, not lexical/numeric precision;
combining both (see fusion.py) covers both failure modes.

Chroma has no first-class BM25 support, so we build and persist our own
lightweight index using rank_bm25, keyed by the same chunk_ids used in the
dense (Chroma) index -- this lets fusion.py merge results by ID.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from docintel.retrieval.indexer import PROCESSED_DIR, load_processed_documents

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BM25_INDEX_PATH = PROJECT_ROOT / "data" / "chroma" / "bm25_index.pkl"

_TOKEN_PATTERN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer. Good enough for BM25; no need for a
    heavyweight tokenizer here since BM25 just needs consistent term matching."""
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunk_ids: list[str]
    texts: list[str]
    metadatas: list[dict[str, str | int]]


def build_bm25_index(processed_dir: Path = PROCESSED_DIR) -> BM25Index:
    docs = load_processed_documents(processed_dir)
    if not docs:
        raise FileNotFoundError(
            f"No processed documents found in {processed_dir}. Run `docintel ingest run` first."
        )

    chunk_ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, str | int]] = []

    for doc in docs:
        for chunk in doc.chunks:
            chunk_ids.append(chunk.chunk_id)
            texts.append(chunk.text)
            metadatas.append(
                {
                    "ticker": chunk.metadata.ticker,
                    "filing_date": chunk.metadata.filing_date,
                    "source_type": chunk.metadata.source_type.value,
                    "element_type": chunk.element_type,
                    "position": chunk.position,
                }
            )

    tokenized_corpus = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    return BM25Index(bm25=bm25, chunk_ids=chunk_ids, texts=texts, metadatas=metadatas)


def save_bm25_index(index: BM25Index, path: Path = BM25_INDEX_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(index, f)


def load_bm25_index(path: Path = BM25_INDEX_PATH) -> BM25Index:
    if not path.exists():
        raise FileNotFoundError(
            f"No BM25 index found at {path}. Run `docintel ingest index` first."
        )

    with path.open("rb") as f:
        result: BM25Index = pickle.load(f)
        return result


def query_bm25(
    index: BM25Index, query: str, top_k: int = 10
) -> list[tuple[str, float, str, dict[str, str | int]]]:
    """Returns list of (chunk_id, score, text, metadata) sorted by descending BM25 score."""
    tokenized_query = _tokenize(query)
    scores = index.bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(index.chunk_ids, scores, index.texts, index.metadatas, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_k]