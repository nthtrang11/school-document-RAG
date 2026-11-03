"""Resolve PDF evidence and create a page-targeted, highlighted viewer copy."""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Literal

import pymupdf

from src.config import settings


def _normalize_token(token: str) -> str:
    return unicodedata.normalize("NFKC", unicodedata.normalize("NFC", token))


def _normalize_text(text: str) -> str:
    return " ".join(_normalize_token(token) for token in text.split())


def resolve_pdf_path(source_path: str) -> Path:
    root = settings.pdf_dir.resolve()
    candidate = (root / source_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Đường dẫn tài liệu nằm ngoài thư mục nguồn đã cấu hình.") from exc
    if candidate.suffix.casefold() != ".pdf":
        raise ValueError("Citation này không trỏ đến PDF.")
    if not candidate.is_file():
        raise FileNotFoundError(f"Không tìm thấy PDF nguồn: {candidate.name}")
    return candidate


def _exact_word_quads(
    page: pymupdf.Page, evidence_text: str
) -> list[pymupdf.Quad]:
    expected = "".join(_normalize_token(token) for token in evidence_text.split())
    if not expected:
        return []

    words = page.get_text("words", sort=True)
    actual_parts: list[str] = []
    word_ranges: list[tuple[int, int]] = []
    position = 0
    for word in words:
        token = _normalize_token(word[4])
        actual_parts.append(token)
        word_ranges.append((position, position + len(token)))
        position += len(token)
    actual = "".join(actual_parts)
    matches = [
        start
        for start in range(len(actual) - len(expected) + 1)
        if actual[start : start + len(expected)] == expected
    ]

    complete_matches: list[tuple[int, int]] = []
    for start in matches:
        end = start + len(expected)
        first_word = next(
            (
                index
                for index, (word_start, word_end) in enumerate(word_ranges)
                if word_start <= start < word_end
            ),
            None,
        )
        last_word = next(
            (
                index
                for index, (word_start, word_end) in enumerate(word_ranges)
                if word_start < end <= word_end
            ),
            None,
        )
        if (
            first_word is not None
            and last_word is not None
            and word_ranges[first_word][0] == start
            and word_ranges[last_word][1] == end
        ):
            complete_matches.append((first_word, last_word))

    if len(complete_matches) != 1:
        return []

    first_word, last_word = complete_matches[0]
    quads = []
    for word in words[first_word : last_word + 1]:
        rect = pymupdf.Rect(word[:4])
        quads.append(pymupdf.Quad(rect.tl, rect.tr, rect.bl, rect.br))
    return quads


def _evidence_quads(page: pymupdf.Page, evidence_text: str) -> list[pymupdf.Quad]:
    return _exact_word_quads(page, evidence_text)


def _evidence_segment_quads(
    page: pymupdf.Page, evidence_segments: list[str]
) -> tuple[list[pymupdf.Quad], Literal["matched", "partial_match", "not_found"]]:
    matches = [_exact_word_quads(page, segment) for segment in evidence_segments]
    matched = [quad for quads in matches for quad in quads]
    if matches and all(matches):
        return matched, "matched"
    if matched:
        return [], "partial_match"
    return [], "not_found"


def _table_cell_quads(
    page: pymupdf.Page, locations: list[dict[str, object]]
) -> tuple[list[pymupdf.Quad], Literal["matched", "partial_match", "not_found"]]:
    quads: list[pymupdf.Quad] = []
    for location in locations:
        expected = location.get("text")
        bbox = location.get("bbox")
        if not isinstance(expected, str) or not expected.strip():
            continue
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            rect = pymupdf.Rect(*(float(value) for value in bbox))
        except (TypeError, ValueError):
            continue
        if (
            rect.is_empty
            or rect.is_infinite
            or not page.rect.contains(rect)
        ):
            continue
        actual = page.get_text("text", clip=rect, sort=True)
        if _normalize_text(actual) != _normalize_text(expected):
            continue
        quads.append(pymupdf.Quad(rect.tl, rect.tr, rect.bl, rect.br))

    if len(quads) == len(locations) and quads:
        return quads, "matched"
    if quads:
        return [], "partial_match"
    return [], "not_found"


def build_pdf_view(
    pdf_path: Path,
    page_number: int,
    evidence_text: str,
    table_locations: list[dict[str, object]] | None = None,
    evidence_segments: list[str] | None = None,
) -> tuple[
    bytes,
    list[float] | None,
    Literal["matched", "partial_match", "not_found"],
]:
    """Return the full PDF with a cited-page target and evidence-only highlight."""
    try:
        document = pymupdf.open(pdf_path)
    except (pymupdf.FileDataError, OSError) as exc:
        raise ValueError(
            "Không thể đọc PDF nguồn; tệp có thể bị hỏng hoặc không hợp lệ."
        ) from exc

    with document:
        if document.is_encrypted:
            raise ValueError("PDF nguồn được mã hóa.")
        if page_number < 1 or page_number > document.page_count:
            raise ValueError(
                f"Trang citation không hợp lệ: {page_number}; PDF có {document.page_count} trang."
            )

        page = document[page_number - 1]
        if table_locations is not None:
            quads, highlight_status = _table_cell_quads(page, table_locations)
        elif evidence_segments is not None:
            quads, highlight_status = _evidence_segment_quads(
                page, evidence_segments
            )
        else:
            quads = _evidence_quads(page, evidence_text)
            highlight_status = "matched" if quads else "not_found"
        highlight_rect: list[float] | None = None
        if quads:
            annotation = page.add_highlight_annot(quads)
            annotation.set_colors(stroke=(1.0, 0.82, 0.1))
            annotation.update()
            union = quads[0].rect
            for quad in quads[1:]:
                union |= quad.rect
            highlight_rect = [round(value, 2) for value in union]

        catalog = document.pdf_catalog()
        document.xref_set_key(catalog, "OpenAction", f"[{page.xref} 0 R /Fit]")
        pdf_bytes = document.tobytes(garbage=4, deflate=True)
        return pdf_bytes, highlight_rect, highlight_status
