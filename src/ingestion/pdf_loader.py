"""Read PDF files page by page with stable source identifiers."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pymupdf
from rapidocr import RapidOCR

from src.models import PageRecord, TableRowRecord

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
    for table_number, table in enumerate(page.find_tables().tables, start=1):
        if not table.rows:
            continue
        headers = [_cell_text(page, bbox) for bbox in table.rows[0].cells]
        if not any(headers):
            continue
        headers = [
            header or f"Cột {index + 1}"
            for index, header in enumerate(headers)
        ]
        data_rows = table.rows[1:]
        row_bands: list[tuple[float, float] | None] = []
        for row in data_rows:
            boxes = [pymupdf.Rect(cell) for cell in row.cells if cell is not None]
            if not boxes:
                row_bands.append(None)
                continue
            top = max(box.y0 for box in boxes)
            bottom = min(box.y1 for box in boxes)
            row_bands.append((top, bottom) if top < bottom else None)

        logical_row = 0
        for row_index, (row, band) in enumerate(zip(data_rows, row_bands)):
            if band is None:
                continue
            top, bottom = band
            cells: list[str] = []
            cell_bboxes: list[tuple[float, float, float, float] | None] = []
            for column, bbox in enumerate(row.cells):
                if bbox is not None:
                    selected_bbox = tuple(float(value) for value in bbox)
                else:
                    covering_cells = []
                    for source_row_index, source_row in enumerate(data_rows):
                        source_bbox = source_row.cells[column]
                        source_band = row_bands[source_row_index]
                        if source_bbox is None or source_band is None:
                            continue
                        source_rect = pymupdf.Rect(source_bbox)
                        if (
                            source_rect.y0 <= top + 0.1
                            and source_rect.y1 >= bottom - 0.1
                            and _cell_text(page, source_bbox)
                        ):
                            covering_cells.append(
                                (source_rect.height, source_row_index, source_bbox)
                            )
                    selected_bbox = (
                        tuple(float(value) for value in min(covering_cells)[2])
                        if covering_cells
                        else None
                    )

                cells.append(_cell_text(page, selected_bbox))
                cell_bboxes.append(selected_bbox)

            if any(cells):
                logical_row += 1
                extracted.append(
                    TableRowRecord(
                        table_id=f"{document_id}:p{page_no}:t{table_number}",
                        row_index=logical_row,
                        headers=headers,
                        cells=cells,
                        bboxes=cell_bboxes,
                    )
                )
    return extracted


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
