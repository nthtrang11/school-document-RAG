"""Read PDF files page by page with stable source identifiers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf

from src.models import PageRecord


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_id_for(path: Path, data_root: Path) -> str:
    relative = path.resolve().relative_to(data_root.resolve()).as_posix().casefold()
    return "doc_" + hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]


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
            PageRecord(document_id=document_id, page_no=number, text=page.get_text("text"))
            for number, page in enumerate(document, start=1)
        ]
    return document_id, file_hash, pages
