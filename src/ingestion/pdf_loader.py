"""Read PDF files page by page with stable source identifiers."""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pymupdf
from rapidocr import RapidOCR

from src.config import settings
from src.models import (
    OCRQuality,
    OCRTextRegion,
    PageRecord,
    PageTextQuality,
    TableCellSpan,
    TableRowRecord,
)

LOGGER = logging.getLogger(__name__)
OCR_LOW_CONFIDENCE_PREFIX = "[[OCR_LOW_CONFIDENCE]] "
_CELL_SPACES = re.compile(r"[^\S\r\n]+")
_VALID_PUNCTUATION = frozenset(".,;:!?()[]{}<>/\\-–—_+%#@&*'\"°§$€£")


@lru_cache(maxsize=1)
def _ocr_engine() -> RapidOCR:
    return RapidOCR()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_id_for(path: Path, data_root: Path) -> str:
    relative = path.resolve().relative_to(data_root.resolve()).as_posix().casefold()
    return "doc_" + hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]


def _cell_text(page: pymupdf.Page, bbox: tuple[float, ...] | None) -> str:
    if bbox is None:
        return ""
    text = page.get_text("text", clip=pymupdf.Rect(bbox), sort=True)
    lines = [_CELL_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _extract_table_rows(
    page: pymupdf.Page, document_id: str, page_no: int
) -> list[TableRowRecord]:
    extracted: list[TableRowRecord] = []
    page_blocks = page.get_text("blocks")
    for table_number, table in enumerate(page.find_tables().tables, start=1):
        if not table.rows:
            continue
        source_headers = [_cell_text(page, bbox) for bbox in table.rows[0].cells]
        if not any(source_headers):
            continue
        headers = [
            header or f"Cột {index + 1}"
            for index, header in enumerate(source_headers)
        ]
        rows = table.rows
        row_starts: list[float | None] = []
        all_boxes: list[tuple[int, int, tuple[float, float, float, float]]] = []
        x_boundaries: set[float] = set()
        for row_index, row in enumerate(rows):
            boxes = [
                tuple(float(value) for value in bbox)
                for bbox in row.cells
                if bbox is not None
            ]
            row_starts.append(min((box[1] for box in boxes), default=None))
            for column, bbox in enumerate(row.cells):
                if bbox is None:
                    continue
                normalized = tuple(float(value) for value in bbox)
                all_boxes.append((row_index, column, normalized))
                x_boundaries.update((normalized[0], normalized[2]))

        column_edges = sorted(x_boundaries)
        column_centers = [
            (column_edges[index] + column_edges[index + 1]) / 2
            for index in range(min(len(column_edges) - 1, len(headers)))
        ]
        table_bbox = tuple(float(value) for value in table.bbox)
        image_bboxes = [
            tuple(float(value) for value in image["bbox"])
            for image in page.get_image_info()
            if image.get("bbox")
        ]
        nearby_blocks = [
            block
            for block in page_blocks
            if len(block) > 4
            and float(block[3]) <= table_bbox[1]
            and table_bbox[1] - float(block[3]) <= 48
            and float(block[0]) < table_bbox[2]
            and float(block[2]) > table_bbox[0]
        ]
        title_lines = [
            line.strip()
            for block in nearby_blocks
            for line in str(block[4]).splitlines()
            if re.match(r"(?i)^\s*(?:bảng|table)\s*(?:số|no\.?)?\s*\d*\b", line)
        ]
        if not title_lines:
            title_lines = [
                str(block[4]).strip()
                for block in nearby_blocks
                if sum(char.isalpha() for char in str(block[4])) >= 10
                and sum(char.isupper() for char in str(block[4]) if char.isalpha())
                / sum(char.isalpha() for char in str(block[4]))
                >= 0.9
            ]
        table_title = " ".join(title_lines) if title_lines else None

        row_bands: list[tuple[float, float] | None] = []
        for row_index, start in enumerate(row_starts):
            if start is None:
                row_bands.append(None)
                continue
            later_starts = [
                value
                for value in row_starts[row_index + 1 :]
                if value is not None and value > start + 0.1
            ]
            row_end = min(later_starts, default=table_bbox[3])
            row_bands.append((start, row_end) if start < row_end else None)

        def find_owner(
            row_index: int, column: int
        ) -> tuple[int, int, tuple[float, float, float, float]] | None:
            band = row_bands[row_index]
            if band is None or column >= len(column_centers):
                return None
            center_x = column_centers[column]
            center_y = sum(band) / 2
            candidates = [
                candidate
                for candidate in all_boxes
                if candidate[2][0] - 0.1 <= center_x <= candidate[2][2] + 0.1
                and candidate[2][1] - 0.1 <= center_y <= candidate[2][3] + 0.1
            ]
            return min(
                candidates,
                key=lambda candidate: (
                    (candidate[2][2] - candidate[2][0])
                    * (candidate[2][3] - candidate[2][1]),
                    candidate[0],
                    candidate[1],
                ),
                default=None,
            )

        def span_for(
            owner: tuple[int, int, tuple[float, float, float, float]]
        ) -> TableCellSpan:
            _, _, bbox = owner
            covered_rows = [
                index
                for index, band in enumerate(row_bands)
                if band is not None
                and bbox[1] <= sum(band) / 2 <= bbox[3]
            ]
            covered_columns = [
                index
                for index, center in enumerate(column_centers)
                if bbox[0] <= center <= bbox[2]
            ]
            return TableCellSpan(
                row_start=min(covered_rows, default=owner[0]),
                row_end=max(covered_rows, default=owner[0]) + 1,
                column_start=min(covered_columns, default=owner[1]),
                column_end=max(covered_columns, default=owner[1]) + 1,
            )

        header_cell_spans = []
        for column in range(len(headers)):
            owner = find_owner(0, column)
            header_cell_spans.append(span_for(owner) if owner else None)

        for row_index in range(1, len(rows)):
            band = row_bands[row_index]
            if band is None:
                continue
            cells: list[str] = []
            cell_bboxes: list[tuple[float, float, float, float] | None] = []
            cell_states: list[str] = []
            cell_origins: list[tuple[int, int] | None] = []
            cell_spans: list[TableCellSpan | None] = []
            for column in range(len(headers)):
                owner = find_owner(row_index, column)
                band = row_bands[row_index]
                center_x = (
                    column_centers[column]
                    if column < len(column_centers)
                    else None
                )
                center_y = sum(band) / 2 if band is not None else None
                covered_by_image = bool(
                    center_x is not None
                    and center_y is not None
                    and any(
                        bbox[0] <= center_x <= bbox[2]
                        and bbox[1] <= center_y <= bbox[3]
                        for bbox in image_bboxes
                    )
                )
                if owner is None:
                    selected_bbox = None
                    value = ""
                    state = (
                        "text_not_extracted"
                        if covered_by_image
                        else "empty_or_unknown"
                    )
                    origin = None
                    span = None
                else:
                    owner_row, owner_column, selected_bbox = owner
                    owner_value = _cell_text(page, selected_bbox)
                    span = span_for(owner)
                    origin = (owner_row, owner_column)
                    if not owner_value:
                        value = ""
                        state = (
                            "text_not_extracted"
                            if covered_by_image
                            else "empty_or_unknown"
                        )
                    elif owner_column != column:
                        value = ""
                        state = "covered_by_merged_cell"
                    elif owner_row != row_index:
                        value = owner_value
                        state = "inherited_from_merged_cell"
                    else:
                        value = owner_value
                        state = "explicit_value"
                cells.append(value)
                cell_bboxes.append(selected_bbox)
                cell_states.append(state)
                cell_origins.append(origin)
                cell_spans.append(span)

            if any(cells) or any(
                state
                in {
                    "inherited_from_merged_cell",
                    "covered_by_merged_cell",
                    "text_not_extracted",
                }
                for state in cell_states
            ):
                extracted.append(
                    TableRowRecord(
                        table_id=f"{document_id}:p{page_no}:t{table_number}",
                        table_group_id=f"{document_id}:p{page_no}:t{table_number}",
                        row_index=row_index,
                        headers=headers,
                        cells=cells,
                        bboxes=cell_bboxes,
                        table_title=table_title,
                        header_rows=[source_headers],
                        header_cell_spans=header_cell_spans,
                        cell_states=cell_states,
                        cell_origins=cell_origins,
                        cell_spans=cell_spans,
                        table_bbox=table_bbox,
                    )
                )
    return extracted


def _link_continued_tables(pages: list[PageRecord]) -> list[PageRecord]:
    linked = list(pages)
    for page_index in range(len(linked) - 1):
        previous = linked[page_index]
        following = linked[page_index + 1]
        previous_ids = list(dict.fromkeys(row.table_id for row in previous.table_rows))
        following_ids = list(dict.fromkeys(row.table_id for row in following.table_rows))
        if not previous_ids or not following_ids:
            continue
        previous_id = previous_ids[-1]
        following_id = following_ids[0]
        previous_rows = [
            row for row in previous.table_rows if row.table_id == previous_id
        ]
        following_rows = [
            row for row in following.table_rows if row.table_id == following_id
        ]
        previous_headers = previous_rows[0].header_rows or [previous_rows[0].headers]
        following_headers = following_rows[0].header_rows or [following_rows[0].headers]
        normalized_previous = tuple(
            tuple(" ".join(cell.casefold().split()) for cell in header_row)
            for header_row in previous_headers
        )
        normalized_following = tuple(
            tuple(" ".join(cell.casefold().split()) for cell in header_row)
            for header_row in following_headers
        )
        previous_title = previous_rows[0].table_title
        following_title = following_rows[0].table_title
        if (
            normalized_previous != normalized_following
            or (
                previous_title
                and following_title
                and previous_title.casefold() != following_title.casefold()
            )
        ):
            continue
        group_id = previous_rows[0].table_group_id or previous_id
        linked[page_index] = previous.model_copy(
            update={
                "table_rows": [
                    row.model_copy(update={"table_group_id": group_id})
                    if row.table_id == previous_id
                    else row
                    for row in previous.table_rows
                ]
            }
        )
        linked[page_index + 1] = following.model_copy(
            update={
                "table_rows": [
                    row.model_copy(update={"table_group_id": group_id})
                    if row.table_id == following_id
                    else row
                    for row in following.table_rows
                ]
            }
        )
    return linked


def _valid_character_ratio(text: str) -> float:
    if not text:
        return 0.0
    valid = sum(
        character != "\ufffd"
        and (
            unicodedata.category(character)[0] != "C"
            or character in "\n\r\t"
        )
        for character in text
    )
    return valid / len(text)


def evaluate_text_quality(
    text: str,
    page: pymupdf.Page,
    *,
    ocr_regions: list[OCRTextRegion] | None = None,
) -> PageTextQuality:
    if not text.strip():
        return PageTextQuality(status="empty", reasons=["Trang không có văn bản."])

    text_length = len(text)
    denominator = max(1, text_length)
    replacement_ratio = sum(
        character == "\ufffd"
        or (
            unicodedata.category(character).startswith("C")
            and character not in "\n\r\t"
        )
        for character in text
    ) / denominator
    garbage_ratio = sum(
        not (
            character.isalnum()
            or character.isspace()
            or character in _VALID_PUNCTUATION
        )
        for character in text
    ) / denominator
    words = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    visible_characters = sum(not character.isspace() for character in text)
    word_ratio = sum(len(word) for word in words) / max(1, visible_characters)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    layout_quality = sum(len(line) >= 3 for line in lines) / max(1, len(lines))

    if ocr_regions is not None:
        boxes = [region.bbox for region in ocr_regions if region.bbox]
    else:
        boxes = [
            tuple(float(value) for value in block["bbox"])
            for block in page.get_text("dict")["blocks"]
            if block.get("type") == 0 and block.get("lines")
        ]
    page_area = max(1.0, page.rect.width * page.rect.height)
    text_area = sum(
        max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        for box in boxes
    )
    text_coverage = min(1.0, text_area / page_area)
    image_coverage = _image_area_ratio(page)
    vn_ratio = _valid_character_ratio(text)
    coverage_score = min(1.0, text_coverage / 0.08)
    score = (
        0.30 * vn_ratio
        + 0.15 * word_ratio
        + 0.20 * (1.0 - replacement_ratio)
        + 0.15 * (1.0 - garbage_ratio)
        + 0.10 * coverage_score
        + 0.10 * layout_quality
    )
    reasons = []
    if replacement_ratio > 0:
        reasons.append("Có ký tự thay thế hoặc ký tự điều khiển.")
    if text_length < settings.ocr_min_text_characters:
        reasons.append("Văn bản ngắn hơn ngưỡng ký tự cấu hình.")
    if score < settings.ocr_min_text_quality_score:
        reasons.append("Điểm chất lượng tổng hợp thấp hơn ngưỡng cấu hình.")
    status = (
        "poor"
        if score < settings.ocr_min_text_quality_score
        or text_length < settings.ocr_min_text_characters
        else "degraded"
        if score < 0.8 or reasons
        else "good"
    )
    return PageTextQuality(
        text_length=text_length,
        vn_ratio=vn_ratio,
        replacement_ratio=replacement_ratio,
        garbage_ratio=garbage_ratio,
        word_ratio=word_ratio,
        text_coverage=text_coverage,
        image_coverage=image_coverage,
        layout_quality=layout_quality,
        score=max(0.0, min(1.0, score)),
        status=status,
        reasons=reasons,
    )


def _image_area_ratio(page: pymupdf.Page) -> float:
    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return 0.0
    image_area = sum(
        max(0.0, float(image["bbox"][2]) - float(image["bbox"][0]))
        * max(0.0, float(image["bbox"][3]) - float(image["bbox"][1]))
        for image in page.get_image_info()
        if image.get("bbox")
    )
    return min(1.0, image_area / page_area)


def _ocr_bbox(
    polygon: Any, page: pymupdf.Page, image_width: int, image_height: int
) -> tuple[float, float, float, float] | None:
    try:
        points = np.asarray(polygon, dtype=float).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if len(points) < 2 or not np.isfinite(points).all():
        return None
    scale_x = page.rect.width / image_width
    scale_y = page.rect.height / image_height
    x0 = max(0.0, float(points[:, 0].min()) * scale_x)
    y0 = max(0.0, float(points[:, 1].min()) * scale_y)
    x1 = min(page.rect.width, float(points[:, 0].max()) * scale_x)
    y1 = min(page.rect.height, float(points[:, 1].max()) * scale_y)
    if x0 >= x1 or y0 >= y1:
        return None
    return (x0, y0, x1, y1)


def _ocr_regions(
    result: Any, page: pymupdf.Page, image_width: int, image_height: int
) -> list[OCRTextRegion]:
    texts = getattr(result, "txts", None)
    if texts is None:
        return []
    scores = getattr(result, "scores", None)
    boxes = getattr(result, "boxes", None)
    regions = []
    for index, value in enumerate(texts):
        text = str(value).strip()
        if not text:
            continue
        confidence = None
        if scores is not None and index < len(scores):
            try:
                score = float(scores[index])
                if np.isfinite(score) and 0.0 <= score <= 1.0:
                    confidence = score
            except (TypeError, ValueError):
                confidence = None
        bbox = (
            _ocr_bbox(boxes[index], page, image_width, image_height)
            if boxes is not None and index < len(boxes)
            else None
        )
        regions.append(OCRTextRegion(text=text, confidence=confidence, bbox=bbox))
    return regions


def _extract_ocr_page(
    page: pymupdf.Page,
    document_id: str,
    page_no: int,
    native_text: str,
    *,
    render: bool,
) -> PageRecord:
    native_quality = evaluate_text_quality(native_text, page)
    valid_ratio = _valid_character_ratio(native_text)
    image_area_ratio = _image_area_ratio(page)
    has_native_text = bool(native_text.strip())
    if (
        has_native_text
        and valid_ratio >= settings.ocr_min_valid_character_ratio
        and native_quality.score >= settings.ocr_min_text_quality_score
        and image_area_ratio < settings.ocr_hybrid_image_area_ratio
    ):
        return PageRecord(
            document_id=document_id,
            page_no=page_no,
            text=native_text,
            raw_text=native_text,
            page_classification="native_text",
            text_source="text_layer",
            ocr_quality=OCRQuality(
                status="not_needed",
                recognized_character_count=len(native_text),
                valid_character_ratio=valid_ratio,
                indexable=True,
            ),
            page_quality=native_quality,
            page_dimensions=(page.rect.width, page.rect.height),
            page_rotation=page.rotation,
            processing_version="2",
            processed_at=datetime.now(timezone.utc),
        )

    classification = "hybrid" if has_native_text else "ocr_required"
    if not render:
        return PageRecord(
            document_id=document_id,
            page_no=page_no,
            text=native_text,
            raw_text=native_text,
            page_classification=classification,
            text_source="text_layer",
            ocr_quality=OCRQuality(
                status="not_needed",
                recognized_character_count=len(native_text),
                valid_character_ratio=valid_ratio,
                indexable=has_native_text,
                reason="Chưa chạy OCR trong bước kiểm tra cấu trúc.",
            ),
            page_quality=native_quality,
            page_dimensions=(page.rect.width, page.rect.height),
            page_rotation=page.rotation,
            processing_version="2",
            processed_at=datetime.now(timezone.utc),
        )

    ocr_started = perf_counter()
    try:
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(settings.ocr_render_scale, settings.ocr_render_scale),
            alpha=False,
        )
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        regions = _ocr_regions(
            _ocr_engine()(image), page, pixmap.width, pixmap.height
        )
        ocr_duration = perf_counter() - ocr_started
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        LOGGER.exception("OCR toàn trang thất bại ở trang %d", page_no)
        return PageRecord(
            document_id=document_id,
            page_no=page_no,
            text=(
                native_text
                if has_native_text
                and valid_ratio >= settings.ocr_min_valid_character_ratio
                else ""
            ),
            raw_text=native_text or None,
            page_classification=(
                "hybrid"
                if has_native_text
                and valid_ratio >= settings.ocr_min_valid_character_ratio
                else "unreadable"
            ),
            text_source=(
                "text_layer"
                if has_native_text
                and valid_ratio >= settings.ocr_min_valid_character_ratio
                else "ocr"
            ),
            ocr_quality=OCRQuality(
                status="failed",
                reason=str(exc),
                indexable=(
                    has_native_text
                    and valid_ratio >= settings.ocr_min_valid_character_ratio
                ),
            ),
            page_quality=native_quality,
            page_dimensions=(page.rect.width, page.rect.height),
            page_rotation=page.rotation,
            processing_version="2",
            processed_at=datetime.now(timezone.utc),
            ocr_engine="RapidOCR",
            ocr_dpi=round(settings.ocr_render_scale * 72),
            ocr_duration_seconds=perf_counter() - ocr_started,
        )

    raw_ocr_text = "\n".join(region.text for region in regions)
    confidence_values = [
        region.confidence for region in regions if region.confidence is not None
    ]
    confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values and len(confidence_values) == len(regions)
        else None
    )
    valid_ocr_ratio = _valid_character_ratio(raw_ocr_text)
    ocr_text_quality = evaluate_text_quality(
        raw_ocr_text, page, ocr_regions=regions
    )
    reasons = []
    if len(raw_ocr_text) < settings.ocr_min_text_characters:
        reasons.append("Không đủ ký tự OCR để đạt ngưỡng cấu hình.")
    if valid_ocr_ratio < settings.ocr_min_valid_character_ratio:
        reasons.append("Tỷ lệ ký tự hợp lệ thấp hơn ngưỡng cấu hình.")
    if confidence is None:
        reasons.append("OCR engine không cung cấp confidence cho mọi vùng.")
    elif confidence < settings.ocr_min_confidence:
        reasons.append("Confidence OCR trung bình thấp hơn ngưỡng cấu hình.")
    accepted = (
        bool(regions)
        and not reasons
        and ocr_text_quality.score >= settings.ocr_min_text_quality_score
    )
    if (
        bool(regions)
        and not reasons
        and ocr_text_quality.score < settings.ocr_min_text_quality_score
    ):
        reasons.append("Điểm chất lượng OCR tổng hợp thấp hơn ngưỡng cấu hình.")
    if not regions:
        reasons.append("Không nhận diện được vùng văn bản.")
    use_native_text = (
        has_native_text
        and valid_ratio >= settings.ocr_min_valid_character_ratio
        and native_quality.score >= settings.ocr_min_text_quality_score
        and (
            not accepted
            or native_quality.score >= ocr_text_quality.score
        )
    )
    if not accepted or use_native_text:
        return PageRecord(
            document_id=document_id,
            page_no=page_no,
            text=(
                native_text
                if use_native_text
                else ""
            ),
            raw_text=native_text if use_native_text else raw_ocr_text or None,
            page_classification=(
                "hybrid"
                if has_native_text
                else "unreadable"
                if not accepted
                else "ocr_required"
            ),
            text_source=(
                "text_layer"
                if use_native_text
                else "ocr"
            ),
            ocr_quality=OCRQuality(
                status="needs_review" if regions else "unreadable",
                recognized_character_count=len(raw_ocr_text),
                valid_character_ratio=valid_ocr_ratio,
                confidence_score=confidence,
                region_count=len(regions),
                reason=" ".join(reasons) or "Text layer được giữ do chất lượng cao hơn OCR.",
                indexable=use_native_text,
            ),
            page_quality=(
                native_quality if use_native_text else ocr_text_quality
            ),
            text_layer_text=native_text if has_native_text else None,
            ocr_regions=regions,
            page_dimensions=(page.rect.width, page.rect.height),
            page_rotation=page.rotation,
            processing_version="2",
            processed_at=datetime.now(timezone.utc),
            ocr_engine="RapidOCR",
            ocr_dpi=round(settings.ocr_render_scale * 72),
            ocr_duration_seconds=ocr_duration,
        )

    return PageRecord(
        document_id=document_id,
        page_no=page_no,
        text=raw_ocr_text,
        raw_text=raw_ocr_text,
        page_classification=classification,
        text_source="hybrid" if has_native_text else "ocr",
        ocr_quality=OCRQuality(
            status="completed",
            recognized_character_count=len(raw_ocr_text),
            valid_character_ratio=valid_ocr_ratio,
            confidence_score=confidence,
            region_count=len(regions),
            indexable=True,
        ),
        page_quality=ocr_text_quality,
        text_layer_text=native_text if has_native_text else None,
        ocr_regions=regions,
        page_dimensions=(page.rect.width, page.rect.height),
        page_rotation=page.rotation,
        processing_version="2",
        processed_at=datetime.now(timezone.utc),
        ocr_engine="RapidOCR",
        ocr_dpi=round(settings.ocr_render_scale * 72),
        ocr_duration_seconds=ocr_duration,
    )


def read_pdf(
    path: Path, data_root: Path, *, perform_ocr: bool = True
) -> tuple[str, str, list[PageRecord]]:
    file_hash = sha256_file(path)
    document_id = document_id_for(path, data_root)
    try:
        document = pymupdf.open(path)
    except (pymupdf.FileDataError, OSError) as exc:
        raise ValueError(f"Không thể mở PDF: {exc}") from exc

    with document:
        if document.is_encrypted:
            raise ValueError("PDF được mã hóa và chưa cung cấp mật khẩu")
        pages = []
        for number, page in enumerate(document, start=1):
            native_text = page.get_text("text", sort=True)
            extracted = _extract_ocr_page(
                page,
                document_id,
                number,
                native_text,
                render=perform_ocr,
            )
            pages.append(
                extracted.model_copy(
                    update={
                        "table_rows": _extract_table_rows(page, document_id, number)
                    }
                )
            )
    pages = _link_continued_tables(pages)
    return document_id, file_hash, pages


def read_pdf_header_ocr(
    path: Path,
    document_id: str,
    page_no: int = 1,
) -> PageRecord:
    with pymupdf.open(path) as document:
        if page_no < 1 or page_no > len(document):
            raise ValueError(f"Số trang OCR ngoài phạm vi PDF: {page_no}")
        page = document[page_no - 1]
        page_dimensions = (page.rect.width, page.rect.height)
        page_rotation = page.rotation
        native_text = page.get_text("text").strip()
        ocr_started = perf_counter()
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(settings.ocr_render_scale, settings.ocr_render_scale),
            alpha=False,
        )
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        result = _ocr_engine()(image)
        regions = _ocr_regions(result, page, pixmap.width, pixmap.height)
        text = "\n".join(
            (
                region.text
                if region.confidence is not None
                and region.confidence >= settings.ocr_min_confidence
                else f"{OCR_LOW_CONFIDENCE_PREFIX}{region.text}"
            )
            for region in regions
        )
        page_quality = evaluate_text_quality(
            text, page, ocr_regions=regions
        )
        ocr_duration = perf_counter() - ocr_started
    return PageRecord(
        document_id=document_id,
        page_no=page_no,
        text=text,
        raw_text=text,
        page_classification="hybrid" if text else "unreadable",
        text_source="ocr",
        text_layer_text=native_text or None,
        page_quality=page_quality,
        ocr_regions=regions,
        page_dimensions=page_dimensions,
        page_rotation=page_rotation,
        processing_version="2",
        processed_at=datetime.now(timezone.utc),
        ocr_engine="RapidOCR",
        ocr_dpi=round(settings.ocr_render_scale * 72),
        ocr_duration_seconds=ocr_duration,
    )
