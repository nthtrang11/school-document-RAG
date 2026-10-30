"""Shared section detection and chunk construction."""

from __future__ import annotations

import hashlib
import re

from src.models import ChunkRecord, DocumentMetadata, PageRecord

_SECTION = re.compile(
    r"^\s*((?:CHƯƠNG|Chương)\s+[IVXLCDM\d]+[^\n]*|"
    r"(?:Điều|Mục)\s+\d+[^\n]*|"
    r"\d+(?:\.\d+){0,3}\.?\s+[A-ZÀ-ỸĐ][^\n]*)",
    re.MULTILINE,
)


def section_for(text: str, current: str | None = None) -> str | None:
    matches = list(_SECTION.finditer(text))
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
