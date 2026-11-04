"""Validation and exclusion handling for local source documents."""

from __future__ import annotations

from pathlib import Path
import re

from src.config import settings
from src.ingestion.pdf_loader import document_id_for, sha256_file, read_pdf
from src.models import PageRecord


def personal_data_warning(text: str) -> str | None:
    if re.search(r"(?<!\d)(?:\d{9}|\d{12})(?!\d)", text):
        return "Phát hiện chuỗi có thể là số định danh"
    return None


def validate_pdf(
    path: Path,
    data_root: Path,
    seen_hashes: set[str],
    *,
    pages: list[PageRecord] | None = None,
    inspect_content: bool = True,
) -> tuple[bool, str | None]:
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
        if pages is None and not inspect_content:
            import pymupdf

            try:
                with pymupdf.open(path) as document:
                    if document.is_encrypted:
                        return False, "PDF được mã hóa và chưa cung cấp mật khẩu"
                    if not len(document):
                        return False, "PDF không có trang"
                    native_text = "\n".join(
                        page.get_text("text", sort=True) for page in document
                    )
            except pymupdf.FileDataError as exc:
                return False, f"Không thể mở PDF: {exc}"
            personal_data = personal_data_warning(native_text)
            if personal_data:
                return False, f"{personal_data}; cần rà soát trước khi index"
            return True, None

        if pages is None:
            _, _, pages = read_pdf(path, data_root)
        if not pages:
            return False, "PDF không có trang"
        if not any(page.text.strip() or page.table_rows for page in pages):
            return False, (
                "Không có trang văn bản hoặc OCR đạt chất lượng để lập chỉ mục"
            )
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
