"""Validate, ingest, and transactionally publish user-uploaded PDFs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Callable, Iterator

from src.config import settings
from src.indexing.index_manager import build_strategy, load_strategy
from src.ingestion.ingestion_pipeline import run_ingestion
from src.ingestion.pdf_loader import document_id_for, sha256_file
from src.ingestion.validator import validate_pdf

LOGGER = logging.getLogger(__name__)
_THREAD_LOCK = threading.Lock()


@dataclass(frozen=True)
class UploadPayload:
    filename: str
    content: bytes


@dataclass
class UploadResult:
    filename: str
    status: str
    message: str
    document_id: str | None = None
    page_count: int | None = None
    uploaded_at: str | None = None
    ocr_pages_needing_review: list[int] = field(default_factory=list)


def upload_registry_path() -> Path:
    return settings.project_root / "data" / "uploaded_documents.jsonl"


def load_upload_registry() -> list[dict[str, object]]:
    path = upload_registry_path()
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
    return records


def _write_registry(records: list[dict[str, object]]) -> None:
    path = upload_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _save_upload_records(updates: list[dict[str, object]]) -> None:
    records = {
        str(record.get("upload_id")): record
        for record in load_upload_registry()
        if record.get("upload_id")
    }
    for record in updates:
        records[str(record["upload_id"])] = record
    _write_registry(list(records.values()))


@contextmanager
def _upload_lock() -> Iterator[None]:
    settings.project_root.mkdir(parents=True, exist_ok=True)
    lock_path = settings.project_root / ".upload-index.lock"
    with _THREAD_LOCK, lock_path.open("a+b") as lock_file:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _safe_filename(filename: str) -> str:
    basename = PureWindowsPath(filename).name
    basename = Path(basename).name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    return (safe or "uploaded.pdf")[:120]


def _display_filename(filename: str) -> str:
    return PureWindowsPath(filename).name or "uploaded.pdf"


def _existing_documents() -> tuple[dict[str, str], dict[str, str]]:
    hashes: dict[str, str] = {}
    names: dict[str, str] = {}
    root = settings.pdf_dir
    if root.exists():
        for path in root.rglob("*.pdf"):
            digest = sha256_file(path)
            hashes[digest] = document_id_for(path, root)
            names[digest] = path.name
    return hashes, names


def _make_record(
    payload: UploadPayload,
    status: str,
    message: str,
    *,
    digest: str | None = None,
    document_id: str | None = None,
    page_count: int | None = None,
    uploaded_at: str | None = None,
) -> dict[str, object]:
    return {
        "upload_id": digest or uuid.uuid4().hex,
        "sha256": digest,
        "filename": _display_filename(payload.filename),
        "document_id": document_id,
        "page_count": page_count,
        "uploaded_at": uploaded_at or datetime.now(timezone.utc).isoformat(),
        "processing_status": status,
        "message": message,
    }


def _publish_directories(
    staged_processed: Path,
    staged_indexes: Path,
    transaction_root: Path,
) -> None:
    targets = (
        (staged_processed, settings.processed_dir, transaction_root / "backup-processed"),
        (staged_indexes, settings.index_dir, transaction_root / "backup-indexes"),
    )
    for _, target, _ in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.parent.stat().st_dev != transaction_root.stat().st_dev:
            raise OSError(
                f"Không thể cập nhật an toàn: {target} nằm trên phân vùng khác thư mục staging."
            )

    backed_up: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for staged, target, backup in targets:
            backup.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                os.replace(target, backup)
                backed_up.append((backup, target))
            os.replace(staged, target)
            published.append(target)
    except BaseException:
        for target in reversed(published):
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
        for backup, target in reversed(backed_up):
            os.replace(backup, target)
        raise


def _restore_published_directories(transaction_root: Path) -> None:
    rollback_root = transaction_root / "rollback"
    rollback_root.mkdir(parents=True, exist_ok=True)
    for target, backup in (
        (settings.index_dir, transaction_root / "backup-indexes"),
        (settings.processed_dir, transaction_root / "backup-processed"),
    ):
        if target.exists():
            os.replace(target, rollback_root / target.name)
        if backup.exists():
            os.replace(backup, target)


def process_pdf_uploads(
    payloads: list[UploadPayload],
    on_stage: Callable[[str, str], None] | None = None,
) -> list[UploadResult]:
    if not payloads:
        return []
    stage_callback = on_stage or (lambda _stage, _detail: None)
    max_bytes = settings.max_pdf_upload_mb * 1024 * 1024
    results: list[UploadResult | None] = [None] * len(payloads)
    accepted_paths: list[tuple[int, Path, str, str, str]] = []
    validation_records: list[dict[str, object]] = []
    upload_root = settings.pdf_dir / "uploaded"
    upload_root.mkdir(parents=True, exist_ok=True)

    with _upload_lock():
        existing_hashes, existing_names = _existing_documents()
        batch_documents: dict[str, tuple[str, str]] = {}
        transaction_root = Path(
            tempfile.mkdtemp(prefix=".upload-transaction-", dir=settings.project_root)
        )
        try:
            stage_callback("validating", "Kiểm tra định dạng, dung lượng, nội dung và file trùng.")
            for index, payload in enumerate(payloads):
                filename = _safe_filename(payload.filename)
                display_name = _display_filename(payload.filename)
                if Path(filename).suffix.casefold() != ".pdf":
                    results[index] = UploadResult(display_name, "failed", "Chỉ chấp nhận tệp PDF.")
                    record = _make_record(payload, "failed", "Chỉ chấp nhận tệp PDF.")
                    record["upload_id"] = uuid.uuid4().hex
                    validation_records.append(record)
                    continue
                if not payload.content:
                    results[index] = UploadResult(display_name, "failed", "Tệp rỗng.")
                    record = _make_record(payload, "failed", "Tệp rỗng.")
                    record["upload_id"] = uuid.uuid4().hex
                    validation_records.append(record)
                    continue
                if len(payload.content) > max_bytes:
                    message = f"Tệp vượt giới hạn {settings.max_pdf_upload_mb} MB."
                    results[index] = UploadResult(
                        display_name, "failed", message
                    )
                    record = _make_record(payload, "failed", message)
                    record["upload_id"] = uuid.uuid4().hex
                    validation_records.append(record)
                    continue
                if not payload.content.startswith(b"%PDF-"):
                    message = "Nội dung tệp không có chữ ký PDF hợp lệ."
                    results[index] = UploadResult(
                        display_name, "failed", message
                    )
                    record = _make_record(payload, "failed", message)
                    record["upload_id"] = uuid.uuid4().hex
                    validation_records.append(record)
                    continue

                digest = hashlib.sha256(payload.content).hexdigest()
                if digest in existing_hashes or digest in batch_documents:
                    batch_document = batch_documents.get(digest)
                    duplicate_id = existing_hashes.get(digest) or (
                        batch_document[0] if batch_document else None
                    )
                    duplicate_name = existing_names.get(
                        digest, batch_document[1] if batch_document else filename
                    )
                    results[index] = UploadResult(
                        display_name,
                        "duplicate",
                        f"Tài liệu trùng nội dung đã tồn tại: {duplicate_name}"
                        + (f" ({duplicate_id})." if duplicate_id else "."),
                        document_id=duplicate_id,
                    )
                    record = _make_record(
                        payload,
                        "duplicate",
                        f"Tài liệu trùng nội dung đã tồn tại: {duplicate_name}.",
                        digest=digest,
                        document_id=duplicate_id,
                    )
                    record["upload_id"] = uuid.uuid4().hex
                    validation_records.append(record)
                    continue

                pending = transaction_root / f"{uuid.uuid4().hex}.pdf"
                pending.write_bytes(payload.content)
                valid, reason = validate_pdf(
                    pending, transaction_root, set(), inspect_content=False
                )
                if not valid:
                    pending.unlink(missing_ok=True)
                    results[index] = UploadResult(
                        display_name, "failed", reason or "Tài liệu không hợp lệ."
                    )
                    record = _make_record(
                        payload, "failed", reason or "Tài liệu không hợp lệ."
                    )
                    record["upload_id"] = uuid.uuid4().hex
                    validation_records.append(record)
                    continue

                final_path = upload_root / f"{digest[:16]}_{filename}"
                document_id = document_id_for(final_path, settings.pdf_dir)
                batch_documents[digest] = (document_id, filename)
                accepted_paths.append(
                    (
                        index,
                        final_path,
                        digest,
                        filename,
                        datetime.now(timezone.utc).isoformat(),
                    )
                )
                os.replace(pending, final_path)

            if not accepted_paths:
                _save_upload_records(validation_records)
                return [result for result in results if result is not None]

            pending_records = [
                _make_record(
                    payloads[index],
                    "processing",
                    "Tệp đã kiểm tra; đang tích hợp vào chỉ mục.",
                    digest=digest,
                    document_id=document_id_for(path, settings.pdf_dir),
                    uploaded_at=timestamp,
                )
                for index, path, digest, _, timestamp in accepted_paths
            ]
            _save_upload_records([*validation_records, *pending_records])

            staged_processed = transaction_root / "processed"
            staged_indexes = transaction_root / "indexes"
            try:
                stage_callback("extracting", "Đang trích xuất văn bản, bảng và metadata.")
                run_ingestion(pdf_dir=settings.pdf_dir, output_dir=staged_processed)
                metadata_path = staged_processed / "documents.jsonl"
                metadata_by_id = {
                    item["document_id"]: item
                    for item in (
                        json.loads(line)
                        for line in metadata_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
                }
                uploaded_ids = {
                    document_id_for(path, settings.pdf_dir)
                    for _, path, _, _, _ in accepted_paths
                }
                if not uploaded_ids.issubset(metadata_by_id):
                    raise ValueError("Một hoặc nhiều PDF tải lên không được ingestion chấp nhận.")

                review_pages_by_document: dict[str, list[int]] = {}
                clean_pages_path = staged_processed / "pages_clean.jsonl"
                for line in clean_pages_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    page = json.loads(line)
                    quality = page.get("ocr_quality")
                    if (
                        isinstance(quality, dict)
                        and quality.get("status")
                        in {"needs_review", "failed", "unreadable"}
                    ):
                        review_pages_by_document.setdefault(
                            str(page["document_id"]), []
                        ).append(int(page["page_no"]))

                for index, path, digest, filename, timestamp in accepted_paths:
                    metadata = metadata_by_id[document_id_for(path, settings.pdf_dir)]
                    review_pages = review_pages_by_document.get(
                        document_id_for(path, settings.pdf_dir), []
                    )
                    message = "Tài liệu đã được tích hợp vào cả hai chỉ mục."
                    if review_pages:
                        message += (
                            " Cần kiểm tra chất lượng OCR tại trang "
                            + ", ".join(str(page) for page in review_pages)
                            + "; nội dung các trang này không được đưa vào chỉ mục."
                        )
                    results[index] = UploadResult(
                        filename,
                        "indexing",
                        message,
                        document_id=str(metadata["document_id"]),
                        page_count=int(metadata["page_count"]),
                        uploaded_at=timestamp,
                        ocr_pages_needing_review=review_pages,
                    )

                stage_callback("chunking", "Đang tạo chunks theo Fixed-size và Recursive.")
                for strategy in ("fixed_size", "recursive"):
                    stage_callback(
                        "embedding",
                        f"Đang tạo embeddings và BM25/FAISS cho {strategy}.",
                    )
                    build_strategy(
                        strategy,
                        pages_path=staged_processed / "pages_clean.jsonl",
                        force=True,
                        output_root=staged_indexes,
                        processed_dir=staged_processed,
                    )

                for strategy in ("fixed_size", "recursive"):
                    bundle = load_strategy(strategy, index_root=staged_indexes)
                    if not uploaded_ids.issubset(
                        {chunk.document_id for chunk in bundle.chunks}
                    ):
                        raise ValueError(
                            f"Chỉ mục {strategy} thiếu chunks của PDF mới."
                        )

                completed_records = [
                    _make_record(
                        payloads[index],
                        "completed",
                        (
                            "Đã ingestion và lập chỉ mục thành công cho cả hai chiến lược."
                            if not review_pages_by_document.get(
                                document_id_for(path, settings.pdf_dir)
                            )
                            else (
                                "Đã lập chỉ mục; cần kiểm tra OCR trang "
                                + ", ".join(
                                    str(page)
                                    for page in review_pages_by_document[
                                        document_id_for(path, settings.pdf_dir)
                                    ]
                                )
                                + ". Nội dung các trang này đã bị loại khỏi index."
                            )
                        ),
                        digest=digest,
                        document_id=document_id_for(path, settings.pdf_dir),
                        page_count=int(
                            metadata_by_id[document_id_for(path, settings.pdf_dir)][
                                "page_count"
                            ]
                        ),
                        uploaded_at=timestamp,
                    )
                    for index, path, digest, _, timestamp in accepted_paths
                ]
                stage_callback("publishing", "Đang kiểm tra và công bố bộ chỉ mục mới.")
                _publish_directories(
                    staged_processed, staged_indexes, transaction_root
                )
                try:
                    _save_upload_records(completed_records)
                except OSError:
                    _restore_published_directories(transaction_root)
                    raise

                stage_callback("completed", "Tài liệu đã có thể tìm kiếm.")
                for index, path, _, filename, timestamp in accepted_paths:
                    document_id = document_id_for(path, settings.pdf_dir)
                    review_pages = review_pages_by_document.get(document_id, [])
                    message = "Đã tích hợp vào BM25, FAISS và RRF."
                    if review_pages:
                        message += (
                            " Cần kiểm tra chất lượng OCR tại trang "
                            + ", ".join(str(page) for page in review_pages)
                            + "; nội dung các trang này không được đưa vào chỉ mục."
                        )
                    results[index] = UploadResult(
                        _display_filename(payloads[index].filename),
                        "completed",
                        message,
                        document_id=document_id,
                        page_count=int(
                            metadata_by_id[document_id]["page_count"]
                        ),
                        uploaded_at=timestamp,
                        ocr_pages_needing_review=review_pages,
                    )
            except BaseException:
                for _, path, _, _, _ in accepted_paths:
                    path.unlink(missing_ok=True)
                raise
        except Exception as exc:
            LOGGER.exception("Không thể tích hợp PDF tải lên")
            failed_records = [
                _make_record(
                    payloads[index],
                    "failed",
                    str(exc),
                    digest=digest,
                    document_id=document_id_for(path, settings.pdf_dir),
                    uploaded_at=timestamp,
                )
                for index, path, digest, _, timestamp in accepted_paths
            ]
            for index, path, digest, filename, _ in accepted_paths:
                path.unlink(missing_ok=True)
                results[index] = UploadResult(
                    _display_filename(payloads[index].filename),
                    "failed",
                    f"Tích hợp thất bại; các chỉ mục hiện tại được giữ nguyên. {exc}",
                    document_id=document_id_for(path, settings.pdf_dir),
                )
            _save_upload_records(failed_records)
        finally:
            shutil.rmtree(transaction_root, ignore_errors=True)

    return [
        result
        for result in results
        if result is not None
    ]
