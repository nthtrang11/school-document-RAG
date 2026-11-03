"""Shared section detection and chunk construction."""

from __future__ import annotations

import hashlib
import re

from src.models import ChunkRecord, DocumentMetadata, PageRecord

_SECTION = re.compile(
    r"^\s*((?:CHƯƠNG|Chương)\s+[IVXLCDM\d]+[^\n]*|"
    r"(?:Điều|Mục)\s+\d+[^\n]*|"
    r"[IVXLCDM]+\s*/\s+[^\n]*|"
    r"\d+(?:\.\d+){0,3}\.?\s+[A-ZÀ-ỸĐ][^\n]*)",
    re.MULTILINE,
)


def word_aligned_end(text: str, end: int) -> int:
    while end < len(text) and not text[end].isspace():
        end += 1
    return end


def word_aligned_start(text: str, start: int, target: int) -> int:
    target = max(start, target)
    if target < len(text) and text[target].isspace():
        while target < len(text) and text[target].isspace():
            target += 1
        return target
    while target > start and not text[target - 1].isspace():
        target -= 1
    while target < len(text) and text[target].isspace():
        target += 1
    return target


def section_for(text: str, current: str | None = None) -> str | None:
    matches = list(_SECTION.finditer(text))
    roman_sections = [
        match
        for match in matches
        if re.match(r"[IVXLCDM]+\s*/", match.group(1), re.IGNORECASE)
    ]
    if roman_sections:
        return roman_sections[-1].group(1).strip()
    return matches[-1].group(1).strip() if matches else current


def make_chunk(
    page: PageRecord,
    text: str,
    strategy: str,
    start: int,
    section: str | None,
    metadata: DocumentMetadata | None,
) -> ChunkRecord:
    section = section_for(text, section)
    stable = f"{page.document_id}|{strategy}|{page.page_no}|{start}|{text}"
    chunk_id = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:20]
    meta = metadata.model_dump(mode="json") if metadata else {}
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=page.document_id,
        text=text.strip(),
        page=page.page_no,
        end_page=page.page_no,
        section=section,
        strategy=strategy,  # type: ignore[arg-type]
        metadata=meta,
    )


def make_table_chunks(
    page: PageRecord,
    strategy: str,
    metadata: DocumentMetadata | None,
) -> list[ChunkRecord]:
    chunks = []
    for row in page.table_rows:
        text = "\n".join(row.cells).strip()
        if not text:
            continue
        search_text = "\n".join(
            f"{header}: {cell}"
            for header, cell in zip(row.headers, row.cells)
            if cell
        )
        document_title = metadata.title_subject if metadata else None
        if document_title:
            search_text = f"{document_title}\n{search_text}"
        chunk = make_chunk(
            page,
            text,
            strategy,
            row.row_index,
            None,
            metadata,
        )
        chunks.append(
            chunk.model_copy(
                update={
                    "search_text": search_text,
                    "metadata": {
                        **chunk.metadata,
                        "table_id": row.table_id,
                        "table_row_index": row.row_index,
                        "table_headers": row.headers,
                        "table_cells": row.cells,
                        "table_bboxes": row.bboxes,
                    },
                }
            )
        )
    return chunks
