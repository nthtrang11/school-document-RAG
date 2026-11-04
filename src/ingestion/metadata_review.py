"""Persist explicit human review decisions for extracted document metadata."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings
from src.indexing.index_manager import build_strategy, load_strategy
from src.ingestion.upload_service import _publish_directories
from src.models import DocumentMetadata


def load_metadata_records(path: Path) -> list[DocumentMetadata]:
    if not path.is_file():
        return []
    return [
        DocumentMetadata.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_reviewed_evidence(metadata: DocumentMetadata) -> None:
    populated_fields = {
        "document_type": metadata.document_type,
        "title_subject": metadata.title_subject,
        "number_symbol": metadata.number_symbol,
        "issue_date": metadata.issue_date,
        "effective_date": metadata.effective_date,
        "issuing_authority": metadata.issuing_authority,
    }
    for field, value in populated_fields.items():
        if value is None:
            continue
        matching = [
            evidence
            for evidence in metadata.field_provenance.get(field, [])
            if evidence.evidence.strip()
            and evidence.source_page <= metadata.page_count
            and evidence.value == str(value)
        ]
        if not matching:
            raise ValueError(
                f"Trường {field} cần provenance có giá trị khớp và trang hợp lệ."
            )
    if any(not item.evidence.strip() for item in metadata.deadlines):
        raise ValueError("Mỗi deadline cần có bằng chứng trước khi xác minh.")
    if any(not item.evidence.strip() for item in metadata.document_relations):
        raise ValueError("Mỗi quan hệ văn bản cần có bằng chứng trước khi xác minh.")


def save_metadata_review(
    path: Path,
    reviewed: DocumentMetadata,
    *,
    reviewer: str,
    status: str,
    notes: str | None = None,
) -> DocumentMetadata:
    if status not in {"verified", "rejected"}:
        raise ValueError("Trạng thái review phải là verified hoặc rejected.")
    if not reviewer.strip():
        raise ValueError("Cần ghi tên người xác minh.")
    records = load_metadata_records(path)
    index = next(
        (
            position for position, item in enumerate(records)
            if item.document_id == reviewed.document_id
        ),
        None,
    )
    if index is None:
        raise ValueError("Không tìm thấy tài liệu cần cập nhật trong metadata registry.")
    current = records[index]
    if current.file_hash != reviewed.file_hash:
        raise ValueError("File nguồn đã thay đổi; cần chạy lại ingestion trước khi review.")
    if status == "verified":
        _validate_reviewed_evidence(reviewed)

    provenance = {
        field: [
            item.model_copy(update={"verification_status": status})
            for item in evidence
        ]
        for field, evidence in reviewed.field_provenance.items()
    }
    updated = reviewed.model_copy(
        update={
            "metadata_verification_status": status,
            "metadata_reviewed_by": reviewer.strip(),
            "metadata_reviewed_at": datetime.now(timezone.utc),
            "metadata_review_notes": notes.strip() if notes and notes.strip() else None,
            "field_provenance": provenance,
            "deadlines": [
                item.model_copy(update={"verification_status": status})
                for item in reviewed.deadlines
            ],
            "document_relations": [
                item.model_copy(update={"verification_status": status})
                for item in reviewed.document_relations
            ],
        }
    )
    records[index] = updated

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(record.model_dump_json() + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return updated


def save_metadata_review_and_reindex(
    reviewed: DocumentMetadata,
    *,
    reviewer: str,
    status: str,
    notes: str | None = None,
) -> DocumentMetadata:
    if not settings.processed_dir.is_dir():
        raise FileNotFoundError("Processed corpus chưa tồn tại.")
    settings.project_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".metadata-review-transaction-",
        dir=settings.project_root,
    ) as temporary:
        transaction_root = Path(temporary)
        staged_processed = transaction_root / "processed"
        staged_indexes = transaction_root / "indexes"
        shutil.copytree(settings.processed_dir, staged_processed)
        updated = save_metadata_review(
            staged_processed / "documents.jsonl",
            reviewed,
            reviewer=reviewer,
            status=status,
            notes=notes,
        )
        for strategy in ("fixed_size", "recursive"):
            build_strategy(
                strategy,
                pages_path=staged_processed / "pages_clean.jsonl",
                force=True,
                output_root=staged_indexes,
                processed_dir=staged_processed,
            )
            load_strategy(strategy, index_root=staged_indexes)
        _publish_directories(
            staged_processed,
            staged_indexes,
            transaction_root,
        )
    return updated
