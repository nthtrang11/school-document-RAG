"""Run validation, page extraction, normalization, and manifest generation."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings
from src.ingestion.cleaner import clean_pages
from src.ingestion.metadata_extractor import extract_metadata
from src.ingestion.pdf_loader import read_pdf
from src.ingestion.validator import validate_pdf
from src.ingestion.web_loader import load_web_file
from src.ingestion.validator import personal_data_warning
from src.models import PageRecord

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def run_ingestion(pdf_dir: Path | None = None) -> dict[str, int]:
    source_dir = pdf_dir or settings.pdf_dir
    output_dir = settings.processed_dir
    files = sorted(source_dir.rglob("*.pdf")) if source_dir.exists() else []
    seen_hashes: set[str] = set()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    raw_pages: list[dict[str, Any]] = []
    clean_page_records: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    data_root = source_dir

    for path in files:
        valid, reason = validate_pdf(path, data_root, seen_hashes)
        if not valid:
            rejected.append({"file": str(path), "reason": reason or "Không hợp lệ"})
            continue
        try:
            document_id, file_hash, pages = read_pdf(path, data_root)
            before, after = clean_pages(pages)
            metadata = extract_metadata(path, data_root, document_id, file_hash, pages)
            accepted.append(
                {"document_id": document_id, "path": str(path), "sha256": file_hash}
            )
            registry.append(metadata.model_dump(mode="json"))
            raw_pages.extend(page.model_dump(mode="json") for page in pages)
            clean_page_records.extend(page.model_dump(mode="json") for page in after)
        except (OSError, ValueError) as exc:
            rejected.append({"file": str(path), "reason": str(exc)})
            LOGGER.exception("Không thể xử lý %s", path)

    web_root = settings.web_dir
    if web_root.exists():
        for path in sorted(
            item for item in web_root.rglob("*")
            if item.is_file() and item.suffix.casefold() in {".html", ".htm", ".txt"}
        ):
            try:
                web_data_root = web_root.parent
                document_id, file_hash, page, source_url = load_web_file(path, web_data_root)
                if file_hash in seen_hashes:
                    rejected.append({"file": str(path), "reason": "Tệp trùng SHA-256 với tài liệu đã xử lý"})
                    continue
                if not page.text.strip():
                    rejected.append({"file": str(path), "reason": "Không trích xuất được văn bản"})
                    continue
                personal_data = personal_data_warning(page.text)
                if personal_data:
                    rejected.append({
                        "file": str(path),
                        "reason": f"{personal_data}; cần rà soát trước khi index",
                    })
                    continue
                seen_hashes.add(file_hash)
                before, after = clean_pages([page])
                metadata = extract_metadata(
                    path, web_data_root, document_id, file_hash, before, source_url=source_url
                )
                accepted.append(
                    {"document_id": document_id, "path": str(path), "sha256": file_hash}
                )
                registry.append(metadata.model_dump(mode="json"))
                raw_pages.append(page.model_dump(mode="json"))
                clean_page_records.extend(item.model_dump(mode="json") for item in after)
            except (OSError, ValueError) as exc:
                rejected.append({"file": str(path), "reason": str(exc)})
                LOGGER.exception("Không thể xử lý %s", path)

    _write_jsonl(output_dir / "pages_raw.jsonl", raw_pages)
    _write_jsonl(output_dir / "pages_clean.jsonl", clean_page_records)
    _write_jsonl(output_dir / "documents.jsonl", registry)
    _write_jsonl(output_dir / "ingestion_valid.jsonl", accepted)
    _write_jsonl(output_dir / "ingestion_rejected.jsonl", rejected)
    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "valid_documents": len(accepted),
        "rejected_documents": len(rejected),
        "pages": len(clean_page_records),
        "text_pages": sum(bool(item["text"].strip()) for item in clean_page_records),
    }
    (output_dir / "ingestion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Ingestion hoàn tất: %s", summary)
    return {key: value for key, value in summary.items() if key != "run_at"}


def load_page_records(path: Path | None = None) -> list[PageRecord]:
    source = path or (settings.processed_dir / "pages_clean.jsonl")
    if not source.exists():
        raise FileNotFoundError("Chưa có pages_clean.jsonl; hãy chạy ingestion trước")
    return [
        PageRecord.model_validate_json(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    run_ingestion()
