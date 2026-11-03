"""Conservative metadata extraction; unknown values remain null."""

from __future__ import annotations

import re
import unicodedata
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.pdf_loader import OCR_LOW_CONFIDENCE_PREFIX, read_pdf_header_ocr
from src.models import DocumentMetadata, PageRecord

LOGGER = logging.getLogger(__name__)
_NUMBER_CANDIDATE = re.compile(
    r"(?P<raw>(?P<serial>\d{1,5})(?:\s*/\s*\d{4})?\s*/\s*"
    r"(?P<kind>[A-ZĐ]{1,8})\s*-\s*"
    r"(?P<authority>[A-Z0-9Đ]+(?:\s*-\s*[A-Z0-9Đ]+)*))"
    r"(?=$|[\s,;.)])",
    re.I,
)
_PARTIAL_NUMBER = re.compile(
    r"(?P<raw>/\s*(?P<kind>[A-ZĐ]{1,8})\s*-\s*"
    r"(?P<authority>[A-Z0-9Đ]+(?:\s*-\s*[A-Z0-9Đ]+)*))"
    r"(?=$|[\s,;.)])",
    re.I,
)
_WRITTEN_DATE = re.compile(
    r"(?P<day>\d{1,2})\s*tháng\s*(?P<month>\d{1,2})\s*"
    r"năm\s*(?P<year>\d{4})",
    re.I,
)
_NUMERIC_DATE = re.compile(
    r"(?<!\d)(?P<day>\d{1,2})\s*[/\-]\s*"
    r"(?P<month>\d{1,2})\s*[/\-]\s*(?P<year>\d{4})(?!\d)"
)
_PARTIAL_WRITTEN_DATE = re.compile(
    r"(?P<day>\d{1,2})?\s*tháng\s*(?P<month>\d{1,2})?\s*"
    r"năm\s*(?P<year>\d{4})",
    re.I,
)
_ISSUE_CONTEXT = re.compile(
    r"\b(?:ha\s+n(?:oi|i)|ngay\s+ky|ky\s+ngay|ban\s+hanh\s+ngay)\b", re.I
)
_NON_ISSUE_CONTEXT = re.compile(
    r"\b(?:hieu\s+luc|thuc\s+hien|het\s+han|dang\s+tai|dang\s+ngay|"
    r"ap\s+dung|bat\s+dau|cham\s+nhat|can\s+cu|luat\s+so|"
    r"nghi\s+dinh\s+so|thong\s+tu\s+so|cong\s+van\s+so|"
    r"thong\s+bao\s+so|quyet\s+dinh\s+so|van\s+ban\s+so|"
    r"thoi\s+gian|thoi\s+han|tu\s+ngay|den\s+ngay)\b",
    re.I,
)
_SCHEDULE_CONTEXT = re.compile(
    r"\b(?:hoc\s+tu\s+ngay|thi\s+tu\s+ngay|ngay\s+thi|lich\s+thi)\b", re.I
)
_NUMBER_LABEL = re.compile(r"^\s*s(?:o)?\s*[:.]\s*(.*)$", re.I)
_DOCUMENT_TYPE = {
    "thông báo": "Thông báo",
    "kế hoạch": "Kế hoạch",
    "quyết định": "Quyết định",
    "công văn": "Công văn",
}


def _fold_vietnamese(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(
        character
        for character in decomposed.replace("đ", "d")
        if unicodedata.category(character) != "Mn"
    )


def _ordered_pages(
    pages: list[PageRecord], ocr_pages: list[PageRecord]
) -> list[tuple[PageRecord, str]]:
    native_first = [page for page in pages if page.page_no == 1]
    ocr_first = [page for page in ocr_pages if page.page_no == 1]
    native_rest = [page for page in pages if page.page_no != 1]
    ocr_rest = [page for page in ocr_pages if page.page_no != 1]
    return [
        *((page, "text_layer") for page in native_first),
        *((page, "ocr") for page in ocr_first),
        *((page, "text_layer") for page in native_rest),
        *((page, "ocr") for page in ocr_rest),
    ]


def _normalize_number(raw: str, page_text: str) -> str:
    normalized = re.sub(r"\s+", "", raw)
    if "dai hoc gtvt" in _fold_vietnamese(page_text):
        normalized = re.sub(r"(?i)(?<=-)dhgtvt", "ĐHGTVT", normalized)
    return normalized


def _number_symbol(
    pages: list[PageRecord], ocr_pages: list[PageRecord]
) -> tuple[str | None, str | None, str | None, int | None]:
    partial: tuple[str, str, int] | None = None
    native_text_by_page = {page.page_no: page.text for page in pages}
    for page, source in _ordered_pages(pages, ocr_pages):
        page_lines = page.text.splitlines()[:30]
        low_confidence = [
            line.startswith(OCR_LOW_CONFIDENCE_PREFIX) for line in page_lines
        ]
        lines = [
            line.removeprefix(OCR_LOW_CONFIDENCE_PREFIX).strip()
            for line in page_lines
        ]
        for index, line in enumerate(lines):
            label = _NUMBER_LABEL.match(_fold_vietnamese(line))
            if not label:
                continue
            evidence = "\n".join(lines[index:index + 3])
            value = evidence[evidence.find(":") + 1:] if ":" in evidence else label.group(1)
            match = _NUMBER_CANDIDATE.search(value)
            if match:
                raw = match.group("raw").strip()
                continuation_is_untrusted = "\n" in raw and any(
                    low_confidence[index + 1:index + 3]
                )
                if low_confidence[index] or continuation_is_untrusted:
                    if partial is None:
                        partial = (raw, source, page.page_no)
                    continue
                page_context = page.text + native_text_by_page.get(page.page_no, "")
                return (
                    _normalize_number(raw, page_context),
                    raw,
                    source,
                    page.page_no,
                )
            match = _PARTIAL_NUMBER.search(value)
            if match and partial is None:
                partial = (match.group("raw").strip(), source, page.page_no)
    if partial:
        raw, source, page_no = partial
        return None, raw, source, page_no
    return None, None, None, None


def _issue_date(
    pages: list[PageRecord], ocr_pages: list[PageRecord]
) -> tuple[str | None, str | None, int | None, str | None, int | None]:
    partial: tuple[str, int, str, int] | None = None
    for page, source in _ordered_pages(pages, ocr_pages):
        page_lines = page.text.splitlines()[:30]
        low_confidence = [
            line.startswith(OCR_LOW_CONFIDENCE_PREFIX) for line in page_lines
        ]
        lines = [
            line.removeprefix(OCR_LOW_CONFIDENCE_PREFIX).strip()
            for line in page_lines
        ]
        for index, line in enumerate(lines):
            prefix = " ".join(lines[max(0, index - 2):index + 1])
            date_window = " ".join(lines[index:index + 3])
            folded_prefix = _fold_vietnamese(prefix)
            folded_line = _fold_vietnamese(line)
            explicit = bool(_ISSUE_CONTEXT.search(folded_prefix))
            window_start = max(0, index - 2)
            window_lines = lines[window_start:index + 3]
            cue_index = next(
                (
                    offset
                    for offset, candidate_line in enumerate(window_lines)
                    if _ISSUE_CONTEXT.search(_fold_vietnamese(candidate_line))
                ),
                0,
            )
            raw_end = next(
                (
                    offset
                    for offset, candidate_line in enumerate(window_lines[cue_index:], cue_index)
                    if re.search(r"\b\d{4}\b", candidate_line)
                ),
                min(index - window_start, len(window_lines) - 1),
            )
            evidence_raw = " ".join(window_lines[cue_index:raw_end + 1]).strip()
            blocked_line = _NON_ISSUE_CONTEXT.search(folded_line) or _SCHEDULE_CONTEXT.search(
                folded_line
            )
            blocked_prefix = _NON_ISSUE_CONTEXT.search(folded_prefix) or _SCHEDULE_CONTEXT.search(
                folded_prefix
            )
            if blocked_line or (blocked_prefix and not explicit):
                continue
            standalone_date = bool(
                re.fullmatch(
                    r"\s*(?:ngay\s+)?\d{1,2}\s*(?:[/\-]\s*\d{1,2}"
                    r"\s*[/\-]\s*\d{4}|thang\s*\d{1,2}\s*nam\s*\d{4})\s*",
                    folded_line,
                )
            )
            top_header_date = index < 10 and (
                explicit
                or standalone_date
                or bool(re.match(r"^\s*ngay\b", folded_line))
            )
            annex_header_date = page.page_no > 1 and index < 10 and explicit
            if not (top_header_date or annex_header_date):
                continue

            for pattern in (_WRITTEN_DATE, _NUMERIC_DATE):
                match = pattern.search(date_window)
                if not match:
                    continue
                day, month, year = (
                    int(match.group(key)) for key in ("day", "month", "year")
                )
                raw = evidence_raw if explicit else line.strip()
                try:
                    normalized = datetime(year, month, day).date().isoformat()
                except ValueError:
                    if partial is None:
                        partial = (raw, year, source, page.page_no)
                    continue
                if any(low_confidence[index:index + 3]):
                    if partial is None:
                        partial = (raw, year, source, page.page_no)
                    continue
                return normalized, raw, year, source, page.page_no

            for match in _PARTIAL_WRITTEN_DATE.finditer(date_window):
                if match.group("day") and match.group("month"):
                    continue
                year = int(match.group("year"))
                if partial is None:
                    partial = (
                        evidence_raw if explicit else date_window.strip(),
                        year,
                        source,
                        page.page_no,
                    )

    if partial:
        raw, year, source, page_no = partial
        return None, raw, year, source, page_no
    return None, None, None, None, None


def needs_header_ocr(pages: list[PageRecord]) -> bool:
    if not pages:
        return False
    number, _, _, _ = _number_symbol(pages, [])
    issue_date, _, _, _, _ = _issue_date(pages, [])
    return number is None or issue_date is None


def extract_pdf_metadata(
    path: Path,
    data_root: Path,
    document_id: str,
    file_hash: str,
    pages: list[PageRecord],
) -> DocumentMetadata:
    metadata = extract_metadata(path, data_root, document_id, file_hash, pages)
    if not needs_header_ocr(pages):
        return metadata
    try:
        ocr_page = read_pdf_header_ocr(path, document_id)
    except (ImportError, OSError, RuntimeError, ValueError):
        LOGGER.exception("OCR vùng header thất bại cho %s", path)
        return metadata.model_copy(
            update={"metadata_verification_status": "needs_manual_review"}
        )
    return extract_metadata(
        path,
        data_root,
        document_id,
        file_hash,
        pages,
        ocr_pages=[ocr_page],
    )


def extract_metadata(
    path: Path,
    data_root: Path,
    document_id: str,
    file_hash: str,
    pages: list[PageRecord],
    source_url: str | None = None,
    ocr_pages: list[PageRecord] | None = None,
) -> DocumentMetadata:
    text = "\n".join(page.text for page in pages[:2])
    ocr_pages = ocr_pages or []
    number, number_raw, number_source, number_page = _number_symbol(pages, ocr_pages)
    issue_date, issue_date_raw, issue_year, date_source, date_page = _issue_date(
        pages, ocr_pages
    )

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
        number_symbol_raw=number_raw,
        number_symbol_source=number_source,
        number_symbol_page=number_page,
        title_subject=title,
        issue_date=issue_date,
        issue_date_raw=issue_date_raw,
        issue_year=issue_year,
        issue_date_source=date_source,
        issue_date_page=date_page,
        metadata_verification_status=(
            "needs_manual_review" if ocr_pages else "unverified"
        ),
        issuing_authority=authority,
        source_path=path.resolve().relative_to(data_root.resolve()).as_posix(),
        source_url=source_url,
        accessed_at=datetime.now(timezone.utc),
        file_hash=file_hash,
        page_count=len(pages),
    )
