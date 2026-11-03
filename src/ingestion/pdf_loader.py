"""Read PDF files page by page with stable source identifiers."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pymupdf
from rapidocr import RapidOCR

from src.models import PageRecord, TableCellSpan, TableRowRecord

_OCR_CONFIDENCE_THRESHOLD = 0.9
_HEADER_HEIGHT_RATIO = 0.3
OCR_LOW_CONFIDENCE_PREFIX = "[[OCR_LOW_CONFIDENCE]] "
_CELL_SPACES = re.compile(r"[^\S\r\n]+")


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


def read_pdf(path: Path, data_root: Path) -> tuple[str, str, list[PageRecord]]:
    file_hash = sha256_file(path)
    document_id = document_id_for(path, data_root)
    try:
        document = pymupdf.open(path)
    except (pymupdf.FileDataError, OSError) as exc:
        raise ValueError(f"Không thể mở PDF: {exc}") from exc

    with document:
        if document.is_encrypted:
            raise ValueError("PDF được mã hóa và chưa cung cấp mật khẩu")
        pages = [
            PageRecord(
                document_id=document_id,
                page_no=number,
                text=page.get_text("text", sort=True),
                table_rows=_extract_table_rows(page, document_id, number),
            )
            for number, page in enumerate(document, start=1)
        ]
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
        clip = pymupdf.Rect(
            0, 0, page.rect.width, page.rect.height * _HEADER_HEIGHT_RATIO
        )
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(3, 3), clip=clip, alpha=False)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        result = _ocr_engine()(image)
        if result is None:
            text = ""
        else:
            text = "\n".join(
                (
                    line.strip()
                    if float(score) >= _OCR_CONFIDENCE_THRESHOLD
                    else f"{OCR_LOW_CONFIDENCE_PREFIX}{line.strip()}"
                )
                for line, score in zip(result.txts, result.scores)
                if line.strip()
            )
    return PageRecord(document_id=document_id, page_no=page_no, text=text)
