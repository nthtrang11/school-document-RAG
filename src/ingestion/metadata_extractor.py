"""Conservative metadata extraction; unknown values remain null."""

from __future__ import annotations

import re
import unicodedata
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.pdf_loader import OCR_LOW_CONFIDENCE_PREFIX, read_pdf_header_ocr
from src.models import (
    DeadlineMetadata,
    DocumentMetadata,
    DocumentRelation,
    MetadataEvidence,
    OCRTextRegion,
    PageRecord,
)

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
_EFFECTIVE_CUE = re.compile(
    r"\b(?:co hieu luc|hieu luc thi hanh|thuc hien ke tu|ap dung tu|"
    r"bat dau co hieu luc)\b",
    re.I,
)
_DEADLINE_CUE = re.compile(
    r"\b(?:den het(?: ngay)?|truoc(?: ngay)?|cham nhat(?: vao)?|"
    r"hoan thanh truoc|gia han den|nop truoc|thuc hien tu ngay|"
    r"thoi gian thuc hien tu|tu ngay)\b",
    re.I,
)
_ISSUE_HEADER_CUE = re.compile(
    r"\b(?:ha\s+n(?:oi|i)|ngay\s+ky|ky\s+ngay|ban\s+hanh\s+ngay)\b",
    re.I,
)
_CURRENT_DOCUMENT_CUE = re.compile(
    r"\b(?:van\s+ban|thong\s+bao|quyet\s+dinh|ke\s+hoach|cong\s+van)\s+nay\b",
    re.I,
)
_REFERENCE_CONTEXT = re.compile(
    r"\b(?:can\s+cu|luat(?:\s+so)?|nghi\s+dinh(?:\s+so)?|"
    r"thong\s+tu(?:\s+so)?|quyet\s+dinh(?:\s+so)?|"
    r"cong\s+van(?:\s+so)?|thong\s+bao(?:\s+so)?)\b",
    re.I,
)
_RELATION_CUES = (
    (
        "extends",
        re.compile(
            r"\b(?:gia\s+han|thay\s+cho[\s\S]{0,100}th(?:oi|di)\s+han)\b",
            re.I,
        ),
    ),
    ("replaces", re.compile(r"\bthay\s+the\b", re.I)),
    (
        "amends",
        re.compile(
            r"\b(?:sua\s+doi|dieu\s+chinh)\b|"
            r"\bbo\s+sung\b(?=.{0,40}\b(?:noi\s+dung|quy\s+dinh)\b)",
            re.I,
        ),
    ),
    ("revokes", re.compile(r"\b(?:bai\s+bo|thu\s+hoi)\b", re.I)),
)
_REFERENCED_DOCUMENT = re.compile(
    r"\b(?:can cu|luat so|nghi dinh so|thong tu so|quyet dinh so|"
    r"cong van so|thong bao so|van ban so)\b",
    re.I,
)


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
    def source_for(page: PageRecord) -> str:
        return "ocr" if page.text_source == "hybrid" else page.text_source

    page_variants = [
        variant
        for page in pages
        for variant in _page_variants(page)
    ]
    native_first = [item for item in page_variants if item[0].page_no == 1]
    ocr_first = [
        (page, source_for(page)) for page in ocr_pages if page.page_no == 1
    ]
    native_rest = [item for item in page_variants if item[0].page_no != 1]
    ocr_rest = [
        (page, source_for(page)) for page in ocr_pages if page.page_no != 1
    ]
    return [
        *native_first,
        *ocr_first,
        *native_rest,
        *ocr_rest,
    ]


def _page_variants(page: PageRecord) -> list[tuple[PageRecord, str]]:
    source = "ocr" if page.text_source in {"ocr", "hybrid"} else "text_layer"
    variants = [(page, source)]
    if (
        page.text_source == "hybrid"
        and page.text_layer_text
        and page.text_layer_text != page.text
    ):
        variants.append(
            (
                page.model_copy(update={"text": page.text_layer_text}),
                "text_layer",
            )
        )
    return variants


def _normalize_number(raw: str, page_text: str) -> str:
    normalized = re.sub(r"\s+", "", raw)
    if "dai hoc gtvt" in _fold_vietnamese(page_text):
        normalized = re.sub(r"(?i)(?<=-)dhgtvt", "ĐHGTVT", normalized)
    return normalized


def _number_symbol(
    pages: list[PageRecord], ocr_pages: list[PageRecord]
) -> tuple[str | None, str | None, str | None, int | None]:
    partial: tuple[str, str, int] | None = None
    native_text_by_page = {
        page.page_no: page.text_layer_text or page.text for page in pages
    }
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
        page_lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        if not page_lines:
            continue
        header_limit = max(8, (len(page_lines) + 3) // 4)
        for index, line in enumerate(page_lines):
            folded = _fold_vietnamese(line)
            if any(
                re.match(rf"^\s*{re.escape(_fold_vietnamese(kind))}\b", folded)
                for kind in _DOCUMENT_TYPE
            ):
                header_limit = max(header_limit, index + 2)
                break
        lines = page_lines[:header_limit]
        low_confidence = [
            line.startswith(OCR_LOW_CONFIDENCE_PREFIX) for line in lines
        ]
        lines = [
            line.removeprefix(OCR_LOW_CONFIDENCE_PREFIX).strip()
            for line in lines
        ]
        for index, line in enumerate(lines):
            prefix = " ".join(lines[max(0, index - 2):index + 1])
            date_window = " ".join(lines[max(0, index - 1):index + 3])
            folded_prefix = _fold_vietnamese(prefix)
            folded_line = _fold_vietnamese(line)
            explicit = bool(_ISSUE_HEADER_CUE.search(folded_prefix))
            window_start = max(0, index - 1)
            window_lines = lines[window_start:index + 3]
            cue_index = next(
                (
                    offset
                    for offset, candidate_line in enumerate(window_lines)
                    if _ISSUE_HEADER_CUE.search(_fold_vietnamese(candidate_line))
                ),
                max(0, index - window_start),
            )
            raw_end = next(
                (
                    offset
                    for offset, candidate_line in enumerate(window_lines[cue_index:], cue_index)
                    if re.search(r"\b\d{4}\b", _fold_vietnamese(candidate_line))
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
            top_header_date = (
                explicit
                or standalone_date
                or bool(re.search(r"\bngay\b", folded_line))
            )
            annex_header_date = page.page_no > 1 and explicit
            if not (top_header_date or annex_header_date):
                continue

            folded_window = _fold_vietnamese(date_window)
            issue_cue = _ISSUE_HEADER_CUE.search(folded_window)
            issue_cue_position = issue_cue.start() if issue_cue else -1
            date_candidates = [
                match
                for match in _date_matches(date_window)
                if match.start() >= issue_cue_position
            ]
            for match in date_candidates:
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


def _date_matches(text: str) -> list[re.Match[str]]:
    return sorted(
        [*_WRITTEN_DATE.finditer(text), *_NUMERIC_DATE.finditer(text)],
        key=lambda match: match.start(),
    )


def _parse_date(raw_value: str) -> str | None:
    for pattern in (_WRITTEN_DATE, _NUMERIC_DATE):
        match = pattern.search(raw_value)
        if not match:
            continue
        try:
            return datetime(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            ).date().isoformat()
        except ValueError:
            return None
    return None


def _effective_date(
    pages: list[PageRecord],
) -> tuple[str | None, str | None, str | None, int | None, str | None]:
    for page, source in _ordered_pages(pages, []):
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", page.text)
            if paragraph.strip()
        ]
        if not paragraphs:
            paragraphs = [page.text.strip()]
        for evidence in paragraphs:
            folded_evidence = _fold_vietnamese(evidence)
            for cue in _EFFECTIVE_CUE.finditer(folded_evidence):
                subject_context = folded_evidence[:cue.start()]
                explicit_current_subject = bool(
                    _CURRENT_DOCUMENT_CUE.search(subject_context)
                )
                referenced_subject = bool(
                    _REFERENCE_CONTEXT.search(subject_context)
                    or _REFERENCED_DOCUMENT.search(subject_context)
                )
                if referenced_subject and not explicit_current_subject:
                    continue
                following = [
                    match
                    for match in _date_matches(evidence)
                    if match.start() >= cue.end()
                ]
                if not following:
                    continue
                selected = following[0]
                raw_value = selected.group(0)
                return (
                    _parse_date(raw_value),
                    raw_value,
                    evidence,
                    page.page_no,
                    source,
                )
    return None, None, None, None, None


def _deadline_records(pages: list[PageRecord]) -> list[DeadlineMetadata]:
    records: list[DeadlineMetadata] = []
    record_positions: dict[tuple[int, str], list[int]] = {}

    def type_tokens(value: str | None) -> set[str]:
        ignored = {
            "bao", "den", "duoc", "dupe", "gia", "han", "het", "hien",
            "ngay", "nop", "thdi", "thoi", "thuc", "trudc", "tu",
        }
        return {
            token
            for token in re.findall(r"[^\W_]+", _fold_vietnamese(value or ""))
            if len(token) > 2 and token not in ignored
        }

    def type_strength(value: str | None) -> int:
        return len(type_tokens(value))

    for page, source in _ordered_pages(pages, []):
        sources = [(page.text, None)]
        for row in page.table_rows:
            row_text = "\n".join(
                f"{header}: {cell}"
                for header, cell in zip(row.headers, row.cells)
                if cell.strip()
            )
            if row_text:
                boxes = [box for box in row.bboxes if box is not None]
                row_bbox = (
                    (
                        min(box[0] for box in boxes),
                        min(box[1] for box in boxes),
                        max(box[2] for box in boxes),
                        max(box[3] for box in boxes),
                    )
                    if boxes
                    else row.table_bbox
                )
                sources.append((row_text, row_bbox))

        for source_text, table_bbox in sources:
            text = source_text.strip()
            folded = _fold_vietnamese(text)
            cues = list(_DEADLINE_CUE.finditer(folded))
            dates = _date_matches(text)
            for cue_index, cue in enumerate(cues):
                cue_line_start = folded.rfind("\n", 0, cue.start()) + 1
                next_cue = cues[cue_index + 1].start() if cue_index + 1 < len(cues) else len(text)
                previous_line = text.rfind("\n", 0, cue_line_start)
                previous_previous_line = (
                    text.rfind("\n", 0, previous_line)
                    if previous_line >= 0
                    else -1
                )
                local_start = max(0, previous_previous_line + 1)
                following_dates = [
                    match for match in dates
                    if cue.end() <= match.start() < max(next_cue, cue.end())
                ]
                prefix_context = _fold_vietnamese(text[local_start:cue.start()])
                if (
                    _REFERENCE_CONTEXT.search(prefix_context)
                    and not re.search(
                        r"\b(?:nop|thuc hien|dang ky|hoan thanh|gia han|thu phi|dong phi)\b",
                        prefix_context,
                    )
                ):
                    continue
                if not following_dates:
                    continue
                cue_text = _fold_vietnamese(text[cue.start():cue.end()])
                is_start_cue = bool(re.search(r"\btu\s+ngay\b", cue_text))
                is_range_end = bool(
                    not is_start_cue
                    and re.search(r"\btu\s+ngay\b", prefix_context)
                    and re.search(r"\bden\b", cue_text)
                )
                if is_start_cue:
                    continue
                end_match = following_dates[0]
                preceding_dates = [
                    match for match in dates
                    if local_start <= match.start() < end_match.start()
                ]
                start_match = preceding_dates[-1] if is_range_end and preceding_dates else None
                normalized = _parse_date(end_match.group(0))
                if normalized is None:
                    continue
                evidence = text[local_start:end_match.end()].strip()
                folded_evidence = _fold_vietnamese(evidence)
                if _REFERENCE_CONTEXT.search(folded_evidence) and not re.search(
                    r"\b(?:nop|thuc hien|dang ky|hoan thanh|gia han|thu phi|dong phi)\b",
                    folded_evidence,
                ):
                    continue
                prefix_start = cue_line_start
                prefix = text[prefix_start:cue.start()].strip(" \n\t:;-")
                if type_strength(prefix) < 2 and prefix_start > 0:
                    previous_line_start = text.rfind("\n", 0, prefix_start - 1) + 1
                    prefix = f"{text[previous_line_start:prefix_start]} {prefix}".strip()
                last_clause = re.split(r"[:;.!?]", prefix)[-1].strip()
                if type_strength(last_clause) >= 2:
                    prefix = last_clause
                prefix = re.sub(r"^\s*(?:[-•]\s*)?(?:\d+(?:\.\d+)*[.)]?\s*)", "", prefix)
                deadline_type = " ".join(prefix.split()) or None
                if type_strength(deadline_type) < 2:
                    deadline_type = None
                region = _matching_ocr_region(page, evidence) if source == "ocr" else None
                candidate = DeadlineMetadata(
                    deadline_type=deadline_type,
                    value=normalized,
                    raw_value=end_match.group(0),
                    start_value=_parse_date(start_match.group(0)) if start_match else None,
                    start_raw_value=start_match.group(0) if start_match else None,
                    source_page=page.page_no,
                    evidence=evidence,
                    source_type=source,
                    extraction_method="ocr" if source == "ocr" else "rule_based",
                    confidence=region.confidence if region else None,
                    bbox=region.bbox if region else table_bbox,
                )
                key = (page.page_no, normalized)
                previous_positions = record_positions.get(key, [])
                previous_position = next(
                    (
                        position
                        for position in previous_positions
                        if records[position].deadline_type is None
                        or candidate.deadline_type is None
                        or _fold_vietnamese(records[position].deadline_type or "")
                        == _fold_vietnamese(candidate.deadline_type or "")
                        or (
                            min(
                                type_strength(records[position].deadline_type),
                                type_strength(candidate.deadline_type),
                            )
                            <= 2
                            and len(
                                type_tokens(records[position].deadline_type)
                                & type_tokens(candidate.deadline_type)
                            )
                            >= 2
                        )
                    ),
                    None,
                )
                if previous_position is None:
                    record_positions.setdefault(key, []).append(len(records))
                    records.append(candidate)
                else:
                    previous = records[previous_position]
                    merged_evidence = "\n".join(
                        dict.fromkeys([previous.evidence, candidate.evidence])
                    )
                    best = max(
                        (previous, candidate),
                        key=lambda item: type_strength(item.deadline_type),
                    )
                    records[previous_position] = best.model_copy(
                        update={"evidence": merged_evidence}
                    )
    return records


def _document_relations(pages: list[PageRecord]) -> list[DocumentRelation]:
    records: list[DocumentRelation] = []
    seen: dict[tuple[str, int], int] = {}
    document_context = "\n".join(
        f"{page.text}\n{page.text_layer_text or ''}" for page in pages
    )
    for page, source in _ordered_pages(pages, []):
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        for index in range(len(lines)):
            evidence = "\n".join(lines[max(0, index - 1):index + 2])
            clauses = re.split(r"(?<=[;.!?])\s*|\n{2,}", evidence)
            for clause in clauses:
                matches = list(_NUMBER_CANDIDATE.finditer(clause))
                if not matches:
                    continue
                folded = _fold_vietnamese(clause)
                for match in matches:
                    prefix = folded[:match.start()]
                    active_cues = [
                        (kind, cue.search(folded))
                        for kind, cue in _RELATION_CUES
                    ]
                    active_cues = [
                        (kind, cue_match)
                        for kind, cue_match in active_cues
                        if cue_match is not None and cue_match.start() < match.start()
                    ]
                    latest_active = max(
                        active_cues,
                        key=lambda item: item[1].start(),
                        default=None,
                    )
                    reference_matches = [
                        cue_match
                        for cue_match in (
                            re.search(r"\b(?:can\s+cu|theo)\b", folded),
                        )
                        if cue_match is not None and cue_match.start() < match.start()
                    ]
                    latest_reference = max(
                        reference_matches,
                        key=lambda item: item.start(),
                        default=None,
                    )
                    if (
                        latest_reference is not None
                        and (
                            latest_active is None
                            or latest_reference.start() > latest_active[1].start()
                        )
                    ):
                        relation_type = "references"
                    elif latest_active is not None:
                        relation_type = latest_active[0]
                    elif _REFERENCED_DOCUMENT.search(prefix):
                        relation_type = "references"
                    else:
                        continue
                    raw_value = match.group("raw").strip()
                    referenced_number = _normalize_number(
                        raw_value, f"{clause}\n{document_context}"
                    )
                    key = (referenced_number, page.page_no)
                    region = (
                        _matching_ocr_region(page, clause)
                        if source == "ocr"
                        else None
                    )
                    candidate = DocumentRelation(
                        relation_type=relation_type,
                        referenced_number=referenced_number,
                        raw_value=raw_value,
                        source_page=page.page_no,
                        evidence=clause.strip(),
                        source_type=source,
                        extraction_method="ocr" if source == "ocr" else "rule_based",
                        confidence=region.confidence if region else None,
                        bbox=region.bbox if region else None,
                    )
                    previous_index = seen.get(key)
                    if previous_index is None:
                        seen[key] = len(records)
                        records.append(candidate)
                    elif (
                        records[previous_index].relation_type == "references"
                        and relation_type != "references"
                    ):
                        records[previous_index] = candidate
                    elif (
                        records[previous_index].relation_type != "references"
                        and relation_type == "references"
                    ):
                        continue
                    elif len(candidate.evidence) > len(records[previous_index].evidence):
                        records[previous_index] = candidate
    return records


def resolve_document_relations(
    metadata_records: list[DocumentMetadata],
) -> list[DocumentMetadata]:
    def key(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", _fold_vietnamese(value))

    documents_by_number: dict[str, list[str]] = {}
    for metadata in metadata_records:
        if metadata.number_symbol:
            documents_by_number.setdefault(key(metadata.number_symbol), []).append(
                metadata.document_id
            )

    resolved = []
    for metadata in metadata_records:
        relations = []
        ambiguous = False
        for relation in metadata.document_relations:
            candidates = sorted(
                set(documents_by_number.get(key(relation.referenced_number), []))
                - {metadata.document_id}
            )
            target_id = candidates[0] if len(candidates) == 1 else None
            match_status = (
                "matched"
                if target_id
                else "ambiguous"
                if candidates
                else "unresolved"
            )
            ambiguous = ambiguous or match_status == "ambiguous"
            relations.append(
                relation.model_copy(
                    update={
                        "target_document_id": target_id,
                        "target_match_status": match_status,
                        "target_document_candidates": candidates,
                    }
                )
            )
        warnings = [
            warning
            for warning in metadata.metadata_extraction_warnings
            if warning != "ambiguous_document_relation"
        ]
        if ambiguous:
            warnings.append("ambiguous_document_relation")
        resolved.append(
            metadata.model_copy(
                update={
                    "document_relations": relations,
                    "metadata_extraction_warnings": warnings,
                    "metadata_verification_status": (
                        "needs_manual_review"
                        if ambiguous
                        else metadata.metadata_verification_status
                    ),
                }
            )
        )
    return resolved


def _matching_ocr_region(
    page: PageRecord, evidence: str
) -> OCRTextRegion | None:
    normalized_evidence = " ".join(evidence.casefold().split())
    return next(
        (
            region
            for region in page.ocr_regions
            if " ".join(region.text.casefold().split()) in normalized_evidence
            or normalized_evidence in " ".join(region.text.casefold().split())
        ),
        None,
    )


def _field_evidence(
    value: str | None,
    raw_value: str | None,
    evidence: str | None,
    source_page: int | None,
    source: str | None = None,
    page: PageRecord | None = None,
) -> list[MetadataEvidence]:
    if not evidence or source_page is None:
        return []
    matching_region = (
        _matching_ocr_region(page, evidence)
        if source == "ocr" and page is not None
        else None
    )
    return [
        MetadataEvidence(
            value=value,
            raw_value=raw_value,
            evidence=evidence,
            source_page=source_page,
            source_type=source,
            extraction_method=(
                "ocr" if source == "ocr" else "rule_based"
            ),
            confidence=matching_region.confidence if matching_region else None,
            bbox=matching_region.bbox if matching_region else None,
        )
    ]


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
    ocr_pages = ocr_pages or []
    ordered_pages = _ordered_pages(pages, ocr_pages)
    header_lines = [
        (page, source, line.strip())
        for page, source in ordered_pages
        if page.page_no <= 2
        for line in page.text.splitlines()[:30]
        if line.strip()
    ]
    number, number_raw, number_source, number_page = _number_symbol(pages, ocr_pages)
    issue_date, issue_date_raw, issue_year, date_source, date_page = _issue_date(
        pages, ocr_pages
    )
    effective_date, effective_raw, effective_evidence, effective_page, effective_source = (
        _effective_date(pages)
    )
    deadlines = _deadline_records(pages)
    relations = _document_relations(pages)

    doc_type = None
    doc_type_evidence = None
    doc_type_page = None
    doc_type_source = None
    type_index = None
    for index, (page, source, line) in enumerate(header_lines):
        for key, value in _DOCUMENT_TYPE.items():
            folded_line = _fold_vietnamese(line)
            if re.match(
                rf"^\s*\d*\s*{re.escape(_fold_vietnamese(key))}\b",
                folded_line,
                re.I,
            ):
                doc_type = value
                doc_type_evidence = line
                doc_type_page = page.page_no
                doc_type_source = source
                type_index = index
                break
        if doc_type:
            break
    title = None
    title_evidence = None
    title_page = None
    title_source = None
    for index, (page, source, line) in enumerate(header_lines):
        folded_line = _fold_vietnamese(line)
        if folded_line.startswith("ve viec"):
            title = line[:500]
            title_evidence = line
            title_page = page.page_no
            title_source = source
            break
        if type_index is not None and index == type_index + 1:
            title = line[:500]
            title_evidence = f"{doc_type_evidence}\n{line}"
            title_page = page.page_no
            title_source = source
            break
    authority = None
    authority_evidence = None
    authority_page = None
    authority_source = None
    for page, source, line in header_lines:
        folded_line = _fold_vietnamese(line)
        if "truong" in folded_line and "dai" in folded_line and "gtvt" in folded_line:
            authority = "Trường Đại học GTVT"
        elif "dai hoc giao thong van tai" in folded_line:
            authority = "Trường Đại học Giao thông vận tải"
        else:
            continue
        authority_evidence = line
        authority_page = page.page_no
        authority_source = source
        break

    def matching_page(page_no: int | None, source: str | None) -> PageRecord | None:
        return next(
            (
                page for page, candidate_source in ordered_pages
                if page.page_no == page_no and candidate_source == source
            ),
            None,
        )

    field_provenance = {
        "document_type": _field_evidence(
            doc_type,
            doc_type,
            doc_type_evidence,
            doc_type_page,
            doc_type_source,
            matching_page(doc_type_page, doc_type_source),
        ),
        "title_subject": _field_evidence(
            title,
            title,
            title_evidence,
            title_page,
            title_source,
            matching_page(title_page, title_source),
        ),
        "number_symbol": _field_evidence(
            number,
            number_raw,
            number_raw,
            number_page,
            number_source,
            matching_page(number_page, number_source),
        ),
        "issue_date": _field_evidence(
            issue_date,
            issue_date_raw,
            issue_date_raw,
            date_page,
            date_source,
            matching_page(date_page, date_source),
        ),
        "issuing_authority": _field_evidence(
            authority,
            authority_evidence,
            authority_evidence,
            authority_page,
            authority_source,
            matching_page(authority_page, authority_source),
        ),
    }
    if effective_evidence and effective_page:
        field_provenance["effective_date"] = [
            MetadataEvidence(
                value=effective_date,
                raw_value=effective_raw,
                evidence=effective_evidence,
                source_page=effective_page,
                source_type=effective_source,
                extraction_method="ocr" if effective_source == "ocr" else "rule_based",
            )
        ]
    if deadlines:
        primary_deadline = deadlines[0]
        field_provenance["deadline"] = [
            MetadataEvidence(
                value=primary_deadline.value,
                raw_value=primary_deadline.raw_value,
                evidence=primary_deadline.evidence,
                source_page=primary_deadline.source_page,
                source_type=primary_deadline.source_type,
                extraction_method=primary_deadline.extraction_method,
                confidence=primary_deadline.confidence,
                bbox=primary_deadline.bbox,
            )
        ]
    for deadline in deadlines:
        field_provenance.setdefault("deadlines", []).append(
            MetadataEvidence(
                value=deadline.value,
                raw_value=deadline.raw_value,
                evidence=deadline.evidence,
                source_page=deadline.source_page,
                source_type=deadline.source_type,
                extraction_method=deadline.extraction_method,
                confidence=deadline.confidence,
                bbox=deadline.bbox,
            )
        )
    for relation in relations:
        field_provenance.setdefault(relation.relation_type, []).append(
            MetadataEvidence(
                value=relation.referenced_number,
                raw_value=relation.raw_value,
                evidence=relation.evidence,
                source_page=relation.source_page,
                source_type=relation.source_type,
                extraction_method=relation.extraction_method,
                confidence=relation.confidence,
                bbox=relation.bbox,
            )
        )

    warnings = [
        f"missing:{field}"
        for field, value in (
            ("document_type", doc_type),
            ("number_symbol", number),
            ("issue_date", issue_date),
            ("title_subject", title),
            ("issuing_authority", authority),
        )
        if value is None
    ]
    return DocumentMetadata(
        document_id=document_id,
        document_type=doc_type,
        number_symbol=number,
        number_symbol_raw=number_raw,
        number_symbol_source=number_source,
        number_symbol_page=number_page,
        title_subject=title,
        title_subject_raw=title,
        title_subject_corrected=None,
        issue_date=issue_date,
        issue_date_raw=issue_date_raw,
        issue_year=issue_year,
        issue_date_source=date_source,
        issue_date_page=date_page,
        metadata_verification_status=(
            "needs_manual_review"
            if ocr_pages
            or any(
                page.ocr_quality is not None
                and page.ocr_quality.status not in {"not_needed", "completed"}
                for page in pages
            )
            or any(
                page.ocr_quality is not None
                and page.ocr_quality.status == "completed"
                for page in pages
            )
            else "unverified"
        ),
        issuing_authority=authority,
        effective_date=effective_date,
        effective_date_raw=effective_raw,
        deadline=deadlines[0].value if deadlines else None,
        replaces=next(
            (
                relation.referenced_number for relation in relations
                if relation.relation_type == "replaces"
            ),
            None,
        ),
        amends=next(
            (
                relation.referenced_number for relation in relations
                if relation.relation_type == "amends"
            ),
            None,
        ),
        extends=next(
            (
                relation.referenced_number for relation in relations
                if relation.relation_type == "extends"
            ),
            None,
        ),
        deadlines=deadlines,
        document_relations=relations,
        field_provenance=field_provenance,
        metadata_extraction_warnings=warnings,
        source_path=path.resolve().relative_to(data_root.resolve()).as_posix(),
        source_url=source_url,
        accessed_at=datetime.now(timezone.utc),
        file_hash=file_hash,
        page_count=len(pages),
    )
