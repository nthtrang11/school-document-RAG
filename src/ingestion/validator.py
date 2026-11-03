"""Validation and exclusion handling for local source documents."""

from __future__ import annotations

from pathlib import Path
import re

from src.config import settings
from src.ingestion.pdf_loader import document_id_for, sha256_file, read_pdf


def personal_data_warning(text: str) -> str | None:
    if re.search(r"(?<!\d)(?:\d{9}|\d{12})(?!\d)", text):
        return "Phát hiện chuỗi có thể là số định danh"
    return None


def validate_pdf(path: Path, data_root: Path, seen_hashes: set[str]) -> tuple[bool, str | None]:
    if not path.is_file():
        return False, "Tệp không tồn tại"
    if path.suffix.casefold() != ".pdf":
        return False, "Định dạng không phải PDF"
    try:
        with path.open("rb") as file:
            if file.read(5) != b"%PDF-":
                return False, "Thiếu chữ ký PDF hợp lệ"
        file_hash = sha256_file(path)
        if file_hash in seen_hashes:
            return False, "Tệp trùng SHA-256 với tài liệu đã xử lý"
        if any(
            excluded.casefold() in {
                path.name.casefold(),
                path.resolve().as_posix().casefold(),
                document_id_for(path, data_root).casefold(),
            }
            for excluded in settings.excluded_documents
        ):
            return False, "Nằm trong danh sách loại trừ cấu hình"
        _, _, pages = read_pdf(path, data_root)
        if not pages:
            return False, "PDF không có trang"
        if not any(page.text.strip() or page.table_rows for page in pages):
            return False, "Không trích xuất được văn bản; cần OCR trước khi index"
        combined = "\n".join(
            "\n".join(
                [page.text]
                + [
                    "\n".join(row.cells)
                    for row in page.table_rows
                ]
            )
            for page in pages
        )
        personal_data = personal_data_warning(combined)
        if personal_data:
            return False, f"{personal_data}; cần rà soát trước khi index"
        seen_hashes.add(file_hash)
        return True, None
    except (OSError, ValueError) as exc:
        return False, str(exc)
