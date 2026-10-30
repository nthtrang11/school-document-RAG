"""Recursive chunks prioritizing headings, paragraphs, and sentence boundaries."""

from __future__ import annotations

import re

from src.chunking.common import make_chunk, section_for
from src.config import settings
from src.models import ChunkRecord, DocumentMetadata, PageRecord

_SPLIT_PATTERN = re.compile(
    r"(?m)(?=^(?:CHƯƠNG\s+[IVXLCDM\d]+|Điều\s+\d+|Mục\s+\d+|"
    r"\d+(?:\.\d+){0,3}\.?\s+[A-ZÀ-ỸĐ]))|(?<=\n\n)|(?<=[.!?;])\s+"
)


def _split_text(text: str, size: int, overlap: int) -> list[tuple[int, str]]:
    spans = [(match.start(), match.end()) for match in _SPLIT_PATTERN.finditer(text)]
    boundaries = [0] + [end for _, end in spans if end > 0] + [len(text)]
    boundaries = sorted(set(boundaries))
    segments: list[tuple[int, str]] = []
    start = 0
    for boundary in boundaries[1:]:
        if boundary - start <= size:
            continue
        candidate = text[start:boundary].strip()
        if candidate:
            segments.append((start, candidate))
        start = boundary
    tail = text[start:].strip()
    if tail:
        segments.append((start, tail))

    output: list[tuple[int, str]] = []
    for offset, segment in segments:
        if len(segment) <= size:
            output.append((offset, segment))
            continue
        cursor = 0
        while cursor < len(segment):
            end = min(cursor + size, len(segment))
            if end < len(segment):
                split = segment.rfind(" ", cursor, end)
                if split > cursor + size // 2:
                    end = split
            output.append((offset + cursor, segment[cursor:end].strip()))
            if end == len(segment):
                break
            cursor = max(cursor + 1, end - overlap)
    return output


def chunk_pages(
    pages: list[PageRecord],
    metadata_by_document: dict[str, DocumentMetadata] | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[ChunkRecord]:
    size = chunk_size or settings.chunk_size
    actual_overlap = settings.chunk_overlap if overlap is None else overlap
    if size < 1 or actual_overlap < 0 or actual_overlap >= size:
        raise ValueError("Cần chunk_size > overlap >= 0")
    chunks: list[ChunkRecord] = []
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        section: str | None = None
        for offset, part in _split_text(text, size, actual_overlap):
            section = section_for(text[: offset + len(part)], section)
            chunks.append(
                make_chunk(
                    page,
                    part,
                    "recursive",
                    offset,
                    section,
                    (metadata_by_document or {}).get(page.document_id),
                )
            )
    return chunks
