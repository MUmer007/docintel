"""
Hybrid search: combines dense (Chroma) and sparse (BM25) rankings using
Reciprocal Rank Fusion (RRF).

Why RRF over a weighted score blend: dense (cosine distance) and sparse
(BM25) scores live on completely different scales and distributions, so
directly weighting/summing them requires careful, corpus-specific tuning.
RRF sidesteps this by fusing on RANK POSITION instead of raw score --
score(doc) = sum(1 / (k + rank_i)) across each retriever it appears in. It's
simple, has no distributional assumptions, and is a well-established
baseline in IR/RAG literature (used in Elasticsearch's hybrid search, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

from docintel.core.config import get_settings
from docintel.retrieval.indexer import get_chroma_client, get_or_create_collection
from docintel.retrieval.sparse import BM25Index, load_bm25_index, query_bm25

RRF_K = 60  # standard constant from the original RRF paper; dampens the impact of rank 1 vs rank 2


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | int]
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None


def _dense_search(query: str, top_k: int) -> list[tuple[str, str, dict[str, str | int]]]:
    """Returns list of (chunk_id, text, metadata) ranked by Chroma dense similarity."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    results = collection.query(query_texts=[query], n_results=top_k)

    ids = results["ids"][0]
    docs = results["documents"][0] if results["documents"] else [""] * len(ids)
    metas_raw = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
    metas: list[dict[str, str | int]] = [dict(m) for m in metas_raw]  # type: ignore[arg-type]
    return list(zip(ids, docs, metas, strict=True))


def _sparse_search(
    query: str, top_k: int, bm25_index: BM25Index
) -> list[tuple[str, str, dict[str, str | int]]]:
    """Returns list of (chunk_id, text, metadata) ranked by BM25 score."""
    results = query_bm25(bm25_index, query, top_k=top_k)
    return [(chunk_id, text, meta) for chunk_id, _score, text, meta in results]


def hybrid_search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """
    Run dense + sparse search, fuse rankings via RRF, return top_k results
    sorted by fused score (descending).
    """
    settings = get_settings()
    top_k = top_k or settings.retrieval.top_k_final
    fetch_k = max(settings.retrieval.top_k_dense, settings.retrieval.top_k_sparse)

    dense_results = _dense_search(query, fetch_k)
    bm25_index = load_bm25_index()
    sparse_results = _sparse_search(query, fetch_k, bm25_index)

    # rank position (1-indexed) per chunk_id in each retriever's result list
    dense_ranks = {chunk_id: i + 1 for i, (chunk_id, _, _) in enumerate(dense_results)}
    sparse_ranks = {chunk_id: i + 1 for i, (chunk_id, _, _) in enumerate(sparse_results)}

    # union of all chunk_ids seen by either retriever, keeping text/metadata from whichever found it
    lookup: dict[str, tuple[str, dict[str, str | int]]] = {}
    for chunk_id, text, meta in dense_results:
        lookup[chunk_id] = (text, meta)
    for chunk_id, text, meta in sparse_results:
        lookup.setdefault(chunk_id, (text, meta))

    fused: list[RetrievedChunk] = []
    for chunk_id, (text, meta) in lookup.items():
        d_rank = dense_ranks.get(chunk_id)
        s_rank = sparse_ranks.get(chunk_id)
        score = 0.0
        if d_rank is not None:
            score += 1.0 / (RRF_K + d_rank)
        if s_rank is not None:
            score += 1.0 / (RRF_K + s_rank)
        fused.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                metadata=meta,
                rrf_score=score,
                dense_rank=d_rank,
                sparse_rank=s_rank,
            )
        )

    fused.sort(key=lambda r: r.rrf_score, reverse=True)
    return fused[:top_k]