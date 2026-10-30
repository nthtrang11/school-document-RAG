"""Resolve PDF evidence and create a page-targeted, highlighted viewer copy."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pymupdf

from src.config import settings

_WHITESPACE = re.compile(r"\s+")


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


def _candidate_phrases(evidence_text: str) -> list[str]:
    phrases: set[str] = set()
    for line in evidence_text.splitlines():
        normalized = _WHITESPACE.sub(" ", line).strip()
        if len(normalized) >= 24:
            phrases.add(normalized)
        words = normalized.split()
        window_size = 14
        if len(words) > window_size:
            for start in range(0, len(words) - window_size + 1, 7):
                phrases.add(" ".join(words[start : start + window_size]))
    return sorted(phrases, key=len, reverse=True)


def _evidence_quads(
    page: pymupdf.Page, evidence_text: str
) -> list[pymupdf.Quad]:
    for phrase in _candidate_phrases(evidence_text):
        quads = page.search_for(phrase, quads=True)
        if quads:
            return quads
    return []


def _validated_bbox(page: pymupdf.Page, bbox: Any) -> pymupdf.Rect | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in bbox
    ):
        return None
    coordinates = tuple(float(value) for value in bbox)
    if not all(math.isfinite(value) for value in coordinates):
        return None
    rectangle = pymupdf.Rect(coordinates)
    if rectangle.is_empty or not page.rect.contains(rectangle):
        return None
    return rectangle


def build_pdf_view(
    pdf_path: Path, page_number: int, evidence_text: str, bbox: Any = None
) -> tuple[bytes, list[float] | None]:
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
        quads = _evidence_quads(page, evidence_text)
        highlight_rect: list[float] | None = None
        if quads:
            annotation = page.add_highlight_annot(quads)
            annotation.set_colors(stroke=(1.0, 0.82, 0.1))
            annotation.update()
            union = quads[0].rect
            for quad in quads[1:]:
                union |= quad.rect
            highlight_rect = [round(value, 2) for value in union]
        else:
            rectangle = _validated_bbox(page, bbox)
            if rectangle is not None:
                annotation = page.add_highlight_annot(rectangle)
                annotation.set_colors(stroke=(1.0, 0.82, 0.1))
                annotation.update()
                highlight_rect = [round(value, 2) for value in rectangle]

        catalog = document.pdf_catalog()
        document.xref_set_key(catalog, "OpenAction", f"[{page.xref} 0 R /Fit]")
        pdf_bytes = document.tobytes(garbage=4, deflate=True)
        return pdf_bytes, highlight_rect
