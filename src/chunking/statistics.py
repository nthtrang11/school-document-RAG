"""Chunk corpus quality summaries for each experiment strategy."""

from __future__ import annotations

from collections import Counter

from src.models import ChunkRecord


def summarize_chunks(chunks: list[ChunkRecord]) -> dict[str, object]:
    sizes = [len(chunk.text) for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "average_characters": sum(sizes) / len(sizes) if sizes else 0,
        "minimum_characters": min(sizes, default=0),
        "maximum_characters": max(sizes, default=0),
        "empty_chunks": sum(not chunk.text.strip() for chunk in chunks),
        "chunks_missing_metadata": sum(not chunk.metadata for chunk in chunks),
        "chunks_per_document": dict(Counter(chunk.document_id for chunk in chunks)),
    }
