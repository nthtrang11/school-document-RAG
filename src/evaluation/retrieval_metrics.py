"""Measurable top-k retrieval metrics against verified chunk identifiers."""

from __future__ import annotations

from collections.abc import Sequence


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 3) -> float:
    if k < 1:
        raise ValueError("k phải >= 1")
    return len(set(retrieved[:k]) & relevant) / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 3) -> float:
    if k < 1:
        raise ValueError("k phải >= 1")
    if not relevant:
        raise ValueError("Cần ít nhất một chunk liên quan đã xác minh")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 3) -> float:
    return float(bool(set(retrieved[:k]) & relevant))


def mean_reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    return next(
        (1.0 / rank for rank, chunk_id in enumerate(retrieved, 1) if chunk_id in relevant),
        0.0,
    )
