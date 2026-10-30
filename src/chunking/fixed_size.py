"""Fixed-character chunks with configurable overlap and page provenance."""

from __future__ import annotations

from src.chunking.common import make_chunk, section_for
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
        for start in range(0, len(text), size - step_overlap):
            end = min(start + size, len(text))
            if not text[start:end].strip():
                continue
            section = section_for(text[:end], section)
            chunks.append(
                make_chunk(
                    page,
                    text[start:end],
                    "fixed_size",
                    start,
                    section,
                    (metadata_by_document or {}).get(page.document_id),
                )
            )
            if end == len(text):
                break
    return chunks
