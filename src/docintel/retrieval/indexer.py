"""
Embeds processed chunks and upserts them into a persistent Chroma collection.

We use a local sentence-transformers model (all-MiniLM-L6-v2 by default) --
small, fast, no API cost, good enough for portfolio-scale corpora. A real
enterprise deployment might swap this for a larger/hosted embedding model;
that swap is isolated to this one module because we depend only on the
model name from config, not on any hardcoded embedding logic elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from docintel.core.config import get_settings
from docintel.ingestion.models import ParsedDocument

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def get_chroma_client() -> chromadb.api.ClientAPI:
    settings = get_settings()
    settings.retrieval.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.retrieval.chroma_persist_dir))


def get_or_create_collection(client: chromadb.api.ClientAPI) -> chromadb.Collection:
    settings = get_settings()
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.retrieval.embedding_model
    )
    collection: chromadb.Collection = client.get_or_create_collection(
        name=settings.retrieval.collection_name,
        embedding_function=embedding_fn,  # type: ignore[arg-type]  # chromadb's stub signature for
        # EmbeddingFunction is broader (accepts ndarray input) than SentenceTransformerEmbeddingFunction's
        # actual signature; this is a stub mismatch in chromadb, not a real type error -- confirmed
        # working at runtime.
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def load_processed_documents(processed_dir: Path = PROCESSED_DIR) -> list[ParsedDocument]:
    docs = []
    for path in sorted(processed_dir.glob("*.json")):
        docs.append(ParsedDocument.model_validate_json(path.read_text(encoding="utf-8")))
    return docs


def index_documents(docs: list[ParsedDocument]) -> int:
    """Embed and upsert every chunk from every document into Chroma. Returns count indexed."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    for doc in docs:
        for chunk in doc.chunks:
            ids.append(chunk.chunk_id)
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

    if not ids:
        return 0

    # Chroma upsert is idempotent on ID, so re-running ingestion is safe.
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)  # type: ignore[arg-type]
    # chromadb's Metadata type union is stricter than necessary for our plain str/int dict;
    # confirmed working at runtime (see test_indexer.py).
    return len(ids)


def run_indexing(processed_dir: Path = PROCESSED_DIR) -> int:
    docs = load_processed_documents(processed_dir)
    if not docs:
        raise FileNotFoundError(
            f"No processed documents found in {processed_dir}. "
            f"Run `docintel ingest run` first."
        )
    return index_documents(docs)


if __name__ == "__main__":
    count = run_indexing()
    print(f"Indexed {count} chunks.")