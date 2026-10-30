"""Reciprocal rank fusion without mixing incomparable source scores."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[str, float]]], k: int = 60
) -> list[tuple[str, float]]:
    if k < 1:
        raise ValueError("RRF k phải >= 1")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, (chunk_id, _) in enumerate(ranking, start=1):
            if chunk_id not in seen:
                scores[chunk_id] += 1.0 / (k + rank)
                seen.add(chunk_id)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
