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
        header_rows = row.header_rows or [row.headers]
        headers = []
        for column, fallback in enumerate(row.headers):
            levels = list(
                dict.fromkeys(
                    header_row[column].strip()
                    for header_row in header_rows
                    if column < len(header_row) and header_row[column].strip()
                )
            )
            if not levels and column < len(row.header_cell_spans):
                span = row.header_cell_spans[column]
                if span and span.column_end - span.column_start > 1:
                    parent_header = next(
                        (
                            header_row[span.column_start].strip()
                            for header_row in header_rows
                            if span.column_start < len(header_row)
                            and header_row[span.column_start].strip()
                        ),
                        "",
                    )
                    if parent_header:
                        levels = [
                            parent_header,
                            "[nhãn cột chưa xác định]",
                        ]
            headers.append(" > ".join(levels) if levels else fallback)
        text = "\n".join(row.cells).strip()
        if not text:
            continue
        search_lines = []
        for index, (header, cell) in enumerate(zip(headers, row.cells)):
            state = row.cell_states[index] if row.cell_states else None
            if not cell:
                if state == "empty_or_unknown":
                    search_lines.append(f"{header}: [trống hoặc chưa xác định]")
                elif state == "text_not_extracted":
                    search_lines.append(
                        f"{header}: [có thể có nội dung nhưng chưa trích xuất được]"
                    )
                elif state == "covered_by_merged_cell":
                    search_lines.append(f"{header}: [phạm vi của ô gộp]")
                continue
            suffix = (
                " [giá trị kế thừa từ ô gộp]"
                if state == "inherited_from_merged_cell"
                else ""
            )
            search_lines.append(f"{header}: {cell}{suffix}")
        search_text = "\n".join(search_lines)
        document_title = metadata.title_subject if metadata else None
        if row.table_title:
            search_text = f"{row.table_title}\n{search_text}"
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
                        "table_group_id": row.table_group_id or row.table_id,
                        "table_row_index": row.row_index,
                        "table_headers": headers,
                        "table_cells": row.cells,
                        "table_bboxes": row.bboxes,
                        "table_title": row.table_title,
                        "table_header_rows": row.header_rows,
                        "table_header_spans": [
                            span.model_dump(mode="json") if span else None
                            for span in row.header_cell_spans
                        ],
                        "table_cell_states": row.cell_states,
                        "table_cell_origins": row.cell_origins,
                        "table_cell_spans": [
                            span.model_dump(mode="json") if span else None
                            for span in row.cell_spans
                        ],
                        "table_bbox": row.table_bbox,
                    },
                }
            )
        )
    return chunks
