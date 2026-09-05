"""
Retrieval quality metrics: recall@k, precision@k, MRR (Mean Reciprocal Rank).

Measured against a gold set of (question -> relevant_chunk_ids), independent
of generation quality. This isolates retrieval as a variable: if generation
quality is bad, these metrics tell us whether the problem is "didn't find
the right chunks" (a retrieval problem) or "found them but answered badly"
(a generation problem) -- conflating the two makes debugging RAG systems
much harder.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float  # 0 if no relevant chunk found in results


def compute_retrieval_metrics(
    retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str]
) -> RetrievalMetrics:
    """
    recall@k: fraction of relevant chunks that were retrieved.
    precision@k: fraction of retrieved chunks that were relevant.
    reciprocal_rank: 1/rank of the first relevant chunk found (0 if none).

    If relevant_chunk_ids is empty (no gold labels for this question), all
    metrics are undefined -- callers should skip such questions rather than
    treat 0/0 as a real score.
    """
    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids is empty; metrics are undefined for this question")

    relevant_set = set(relevant_chunk_ids)
    retrieved_set = set(retrieved_chunk_ids)

    hits = relevant_set & retrieved_set
    recall = len(hits) / len(relevant_set)
    precision = len(hits) / len(retrieved_chunk_ids) if retrieved_chunk_ids else 0.0

    reciprocal_rank = 0.0
    for i, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_set:
            reciprocal_rank = 1.0 / i
            break

    return RetrievalMetrics(
        recall_at_k=recall, precision_at_k=precision, reciprocal_rank=reciprocal_rank
    )