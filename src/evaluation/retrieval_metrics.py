"""Measurable top-k retrieval metrics against verified identifiers."""

from __future__ import annotations

from collections.abc import Sequence
from math import log2


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 3) -> float:
    if k < 1:
        raise ValueError("k phải >= 1")
    return len(set(retrieved[:k]) & relevant) / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 3) -> float:
    if k < 1:
        raise ValueError("k phải >= 1")
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 3) -> float:
    return float(bool(set(retrieved[:k]) & relevant))


def mean_reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    return next(
        (1.0 / rank for rank, chunk_id in enumerate(retrieved, 1) if chunk_id in relevant),
        0.0,
    )


def ndcg_at_k(
    retrieved: Sequence[str],
    relevance: dict[str, float] | set[str],
    k: int = 3,
) -> float:
    if k < 1:
        raise ValueError("k phải >= 1")
    scores = (
        {identifier: 1.0 for identifier in relevance}
        if isinstance(relevance, set)
        else relevance
    )
    if any(score < 0 for score in scores.values()):
        raise ValueError("Relevance score phải không âm")
    actual = sum(
        (2**scores.get(identifier, 0.0) - 1) / log2(rank + 1)
        for rank, identifier in enumerate(retrieved[:k], start=1)
    )
    ideal_scores = sorted(scores.values(), reverse=True)[:k]
    ideal = sum(
        (2**score - 1) / log2(rank + 1)
        for rank, score in enumerate(ideal_scores, start=1)
    )
    return actual / ideal if ideal else 0.0
