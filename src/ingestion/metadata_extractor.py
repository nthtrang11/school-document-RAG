"""Conservative metadata extraction; unknown values remain null."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from src.models import DocumentMetadata, PageRecord

_NUMBER = re.compile(
    r"(?im)^\s*Số\s*[:.]?\s*(\d{1,5}/[A-ZĐ0-9][A-ZĐ0-9./-]{2,})\b"
)
_DATE = re.compile(
    r"\bHà\s+Nội,\s*ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})\b",
    re.I,
)
_DOCUMENT_TYPE = {
    "thông báo": "Thông báo",
    "kế hoạch": "Kế hoạch",
    "quyết định": "Quyết định",
    "công văn": "Công văn",
}


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1).strip(" .,:;") if match else None


def extract_metadata(
    path: Path,
    data_root: Path,
    document_id: str,
    file_hash: str,
    pages: list[PageRecord],
    source_url: str | None = None,
) -> DocumentMetadata:
    text = "\n".join(page.text for page in pages[:2])
    number = _first_match(_NUMBER, text)
    header = "\n".join(text.splitlines()[:12])
    date_match = _DATE.search(header)
    issue_date = None
    if date_match:
        day, month, year = (int(value) for value in date_match.groups())
        try:
            issue_date = datetime(year, month, day).date().isoformat()
        except ValueError:
            issue_date = None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    doc_type = None
    for line in lines[:12]:
        for key, value in _DOCUMENT_TYPE.items():
            if re.match(rf"^\s*\d*\s*{re.escape(key)}\b", line, re.I):
                doc_type = value
                break
        if doc_type:
            break
    title = None
    for index, line in enumerate(lines[:20]):
        if line.casefold().startswith("về việc"):
            title = line[:500]
            break
        if line.casefold() in _DOCUMENT_TYPE:
            doc_type = _DOCUMENT_TYPE[line.casefold()]
            if index + 1 < len(lines):
                title = lines[index + 1][:500]
                break
        if line.casefold().startswith(tuple(f"{key} " for key in _DOCUMENT_TYPE)):
            doc_type = _DOCUMENT_TYPE[next(
                key for key in _DOCUMENT_TYPE if line.casefold().startswith(f"{key} ")
            )]
            title = line[:500]
            break
    authority = None
    lowered_text = text.casefold()
    if "trường đại học gtvt" in lowered_text:
        authority = "Trường Đại học GTVT"
    elif "đại học giao thông vận tải" in lowered_text:
        authority = "Trường Đại học Giao thông vận tải"
    return DocumentMetadata(
        document_id=document_id,
        document_type=doc_type,
        number_symbol=number,
        title_subject=title,
        issue_date=issue_date,
        issuing_authority=authority,
        source_path=path.resolve().relative_to(data_root.resolve()).as_posix(),
        source_url=source_url,
        accessed_at=datetime.now(timezone.utc),
        file_hash=file_hash,
        page_count=len(pages),
    )
