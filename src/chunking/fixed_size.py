"""Fixed-character chunks with configurable overlap and page provenance."""

from __future__ import annotations

from src.chunking.common import (
    make_chunk,
    make_table_chunks,
    section_for,
    word_aligned_end,
    word_aligned_start,
)
from src.config import settings
from src.models import ChunkRecord, DocumentMetadata, PageRecord


def chunk_pages(
    pages: list[PageRecord],
    metadata_by_document: dict[str, DocumentMetadata] | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[ChunkRecord]:
    size = chunk_size or settings.chunk_size
    step_overlap = settings.chunk_overlap if overlap is None else overlap
    if size < 1 or step_overlap < 0 or step_overlap >= size:
        raise ValueError("Cần chunk_size > overlap >= 0")

    chunks: list[ChunkRecord] = []
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        section: str | None = None
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                end = word_aligned_end(text, end)
            chunk_text = text[start:end].strip()
            if not chunk_text:
                break
            section = section_for(text[:end], section)
            chunks.append(
                make_chunk(
                    page,
                    chunk_text,
                    "fixed_size",
                    start,
                    section,
                    (metadata_by_document or {}).get(page.document_id),
                )
            )
            if end == len(text):
                break
            next_start = word_aligned_start(text, start, end - step_overlap)
            start = next_start if next_start > start else end
        chunks.extend(
            make_table_chunks(
                page,
                "fixed_size",
                (metadata_by_document or {}).get(page.document_id),
            )
        )
    return chunks
