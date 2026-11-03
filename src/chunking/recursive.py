"""Recursive chunks prioritizing headings, paragraphs, and sentence boundaries."""

from __future__ import annotations

import re

from src.chunking.common import (
    make_chunk,
    make_table_chunks,
    section_for,
    word_aligned_end,
    word_aligned_start,
)
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
        raw_candidate = text[start:boundary]
        leading = len(raw_candidate) - len(raw_candidate.lstrip())
        candidate = raw_candidate.strip()
        if candidate:
            segments.append((start + leading, candidate))
        start = boundary
    raw_tail = text[start:]
    leading = len(raw_tail) - len(raw_tail.lstrip())
    tail = raw_tail.strip()
    if tail:
        segments.append((start + leading, tail))

    output: list[tuple[int, str]] = []
    for offset, segment in segments:
        if len(segment) <= size:
            output.append((offset, segment))
            continue
        cursor = 0
        while cursor < len(segment):
            end = min(cursor + size, len(segment))
            if end < len(segment):
                end = word_aligned_end(segment, end)
            output.append((offset + cursor, segment[cursor:end].strip()))
            if end == len(segment):
                break
            next_cursor = word_aligned_start(segment, cursor, end - overlap)
            cursor = next_cursor if next_cursor > cursor else end
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
        if not text and not page.table_rows:
            continue
        section: str | None = None
        for offset, part in _split_text(text, size, actual_overlap) if text else []:
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
        chunks.extend(
            make_table_chunks(
                page,
                "recursive",
                (metadata_by_document or {}).get(page.document_id),
            )
        )
    return chunks
