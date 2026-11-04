from pathlib import Path

import numpy as np
import pymupdf

from src.config import settings
from src.indexing.index_manager import build_strategy, load_strategy
from src.ingestion.ingestion_pipeline import run_ingestion
from src.ingestion.pdf_loader import document_id_for
from src.ingestion.upload_service import (
    UploadPayload,
    load_upload_registry,
    process_pdf_uploads,
)
from src.rag.source_viewer import build_pdf_view, resolve_pdf_path
from src.retrieval.hybrid_retriever import HybridRetriever


def _pdf_bytes(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((50, 60), "Số 321/TB-TEST")
    page.insert_text((50, 85), "Hà Nội, ngày 02/01/2026")
    page.insert_text((50, 120), text)
    content = document.tobytes()
    document.close()
    return content


def test_upload_publishes_both_indexes_and_rolls_back_on_rebuild_failure(
    tmp_path: Path, monkeypatch
):
    import src.indexing.index_manager as index_manager
    import src.ingestion.metadata_extractor as metadata_extractor
    import src.ingestion.upload_service as upload_service

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    old_pdf = raw_dir / "existing.pdf"
    old_pdf.write_bytes(_pdf_bytes("ORIGINAL_LIBRARY_CONTENT"))
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    monkeypatch.setattr(settings, "project_root", tmp_path)
    monkeypatch.setattr(settings, "pdf_dir", raw_dir)
    monkeypatch.setattr(settings, "web_dir", web_dir)
    monkeypatch.setattr(settings, "processed_dir", tmp_path / "processed")
    monkeypatch.setattr(settings, "index_dir", tmp_path / "indexes")
    monkeypatch.setattr(settings, "max_pdf_upload_mb", 1)
    monkeypatch.setattr(metadata_extractor, "needs_header_ocr", lambda pages: False)

    class TestEmbedder:
        def __init__(self, model_name=None, batch_size=None):
            self.model_name = model_name

        def encode(self, texts):
            return np.tile(
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                (len(texts), 1),
            )

    monkeypatch.setattr(index_manager, "SentenceTransformerEmbedder", TestEmbedder)
    monkeypatch.setattr(
        "src.retrieval.hybrid_retriever.SentenceTransformerEmbedder",
        TestEmbedder,
    )
    run_ingestion()
    for strategy in ("fixed_size", "recursive"):
        build_strategy(strategy, force=True)

    new_bytes = _pdf_bytes("UNIQUE_UPLOAD_CONTENT")
    second_bytes = _pdf_bytes("SECOND_UNIQUE_UPLOAD_CONTENT")
    results = process_pdf_uploads(
        [
            UploadPayload("same filename.pdf", new_bytes),
            UploadPayload("same filename.pdf", second_bytes),
            UploadPayload("alias.pdf", new_bytes),
        ]
    )

    assert [result.status for result in results] == [
        "completed",
        "completed",
        "duplicate",
    ]
    assert [result.page_count for result in results[:2]] == [1, 1]
    assert results[0].document_id != results[1].document_id
    assert results[2].document_id == results[0].document_id
    assert len(list((raw_dir / "uploaded").glob("*.pdf"))) == 2
    assert {
        row["processing_status"] for row in load_upload_registry()
    } == {"completed", "duplicate"}
    active_manifests = {
        strategy: (settings.index_dir / strategy / "manifest.json").read_text(
            encoding="utf-8"
        )
        for strategy in ("fixed_size", "recursive")
    }

    for strategy in ("fixed_size", "recursive"):
        bundle = load_strategy(strategy)
        document_ids = {chunk.document_id for chunk in bundle.chunks}
        assert {result.document_id for result in results[:2]}.issubset(document_ids)
        assert document_id_for(old_pdf, raw_dir) in document_ids
        for result, query in zip(
            results[:2],
            ("UNIQUE_UPLOAD_CONTENT", "SECOND_UNIQUE_UPLOAD_CONTENT"),
            strict=True,
        ):
            query_results = HybridRetriever(bundle).search(query)
            assert any(
                item["chunk"].document_id == result.document_id
                for item in query_results["results"]
            )

    source_path = next(
        chunk.metadata["source_path"]
        for chunk in load_strategy("fixed_size").chunks
        if chunk.document_id == results[0].document_id
    )
    assert resolve_pdf_path(source_path).is_file()
    _, highlight_rect, highlight_status = build_pdf_view(
        resolve_pdf_path(source_path),
        1,
        "UNIQUE_UPLOAD_CONTENT",
        evidence_segments=["UNIQUE_UPLOAD_CONTENT"],
    )
    assert highlight_status == "matched"
    assert highlight_rect is not None

    duplicate = process_pdf_uploads(
        [UploadPayload("renamed copy.pdf", new_bytes)]
    )[0]
    assert duplicate.status == "duplicate"
    assert duplicate.document_id == results[0].document_id
    assert len(list((raw_dir / "uploaded").glob("*.pdf"))) == 2

    failed_bytes = _pdf_bytes("ROLLBACK_UPLOAD_CONTENT")
    original_builder = upload_service.build_strategy

    def fail_recursive(strategy, *args, **kwargs):
        if strategy == "recursive":
            raise RuntimeError("simulated recursive index failure")
        return original_builder(strategy, *args, **kwargs)

    monkeypatch.setattr(upload_service, "build_strategy", fail_recursive)
    failed = process_pdf_uploads([UploadPayload("rollback.pdf", failed_bytes)])[0]

    assert failed.status == "failed"
    assert "giữ nguyên" in failed.message
    assert len(list((raw_dir / "uploaded").glob("*.pdf"))) == 2
    assert {
        strategy: (settings.index_dir / strategy / "manifest.json").read_text(
            encoding="utf-8"
        )
        for strategy in ("fixed_size", "recursive")
    } == active_manifests
    assert "ROLLBACK_UPLOAD_CONTENT" not in (
        settings.processed_dir / "pages_clean.jsonl"
    ).read_text(encoding="utf-8")
    registry = load_upload_registry()
    assert any(
        row["processing_status"] == "failed"
        and "simulated recursive index failure" in row["message"]
        for row in registry
    )


def test_upload_rejects_spoofed_pdf_and_records_failure(tmp_path, monkeypatch):
    from src.ingestion.upload_service import UploadPayload, load_upload_registry
    import src.ingestion.upload_service as upload_service

    monkeypatch.setattr(settings, "project_root", tmp_path)
    monkeypatch.setattr(settings, "pdf_dir", tmp_path / "raw")
    monkeypatch.setattr(settings, "max_pdf_upload_mb", 1)

    results = upload_service.process_pdf_uploads(
        [
            UploadPayload("not-really.pdf", b"not a pdf"),
            UploadPayload("corrupt.pdf", b"%PDF-1.7\nnot a real PDF"),
            UploadPayload("empty.pdf", b""),
            UploadPayload("oversized.pdf", b"%PDF-" + b"x" * (1024 * 1024)),
            UploadPayload("not-a-pdf.txt", b"plain text"),
        ]
    )

    assert [result.status for result in results] == ["failed"] * 5
    assert "chữ ký PDF" in results[0].message
    assert "Không thể mở PDF" in results[1].message
    assert "rỗng" in results[2].message
    assert "vượt giới hạn" in results[3].message
    assert "Chỉ chấp nhận" in results[4].message
    records = load_upload_registry()
    assert len(records) == 5
    assert all(record["processing_status"] == "failed" for record in records)


def test_upload_restores_active_directories_when_completion_record_fails(
    tmp_path, monkeypatch
):
    import json
    from types import SimpleNamespace

    import src.ingestion.upload_service as upload_service

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    (processed_dir / "previous.jsonl").write_text("previous", encoding="utf-8")
    index_dir = tmp_path / "indexes"
    for strategy in ("fixed_size", "recursive"):
        strategy_dir = index_dir / strategy
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "previous.manifest").write_text(
            "previous", encoding="utf-8"
        )
    monkeypatch.setattr(settings, "project_root", tmp_path)
    monkeypatch.setattr(settings, "pdf_dir", raw_dir)
    monkeypatch.setattr(settings, "processed_dir", processed_dir)
    monkeypatch.setattr(settings, "index_dir", index_dir)
    monkeypatch.setattr(settings, "max_pdf_upload_mb", 1)

    def fake_ingestion(pdf_dir, output_dir):
        uploaded_pdf = next((Path(pdf_dir) / "uploaded").glob("*.pdf"))
        document_id = document_id_for(uploaded_pdf, Path(pdf_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "documents.jsonl").write_text(
            json.dumps({"document_id": document_id, "page_count": 1}) + "\n",
            encoding="utf-8",
        )
        (output_dir / "pages_clean.jsonl").write_text(
            json.dumps(
                {
                    "document_id": document_id,
                    "page_no": 1,
                    "text": "NEW_CONTENT",
                    "ocr_quality": None,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def fake_build(strategy, output_root, **_kwargs):
        (Path(output_root) / strategy).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(upload_service, "run_ingestion", fake_ingestion)
    monkeypatch.setattr(upload_service, "build_strategy", fake_build)
    monkeypatch.setattr(
        upload_service,
        "load_strategy",
        lambda strategy, **_kwargs: SimpleNamespace(
            chunks=[
                SimpleNamespace(
                    document_id=document_id_for(
                        next((raw_dir / "uploaded").glob("*.pdf")), raw_dir
                    )
                )
            ]
        ),
    )
    original_save = upload_service._save_upload_records
    failed_completion_write = False

    def fail_completed_record(records):
        nonlocal failed_completion_write
        if (
            not failed_completion_write
            and any(record["processing_status"] == "completed" for record in records)
        ):
            failed_completion_write = True
            raise OSError("simulated registry write failure")
        original_save(records)

    monkeypatch.setattr(upload_service, "_save_upload_records", fail_completed_record)

    result = upload_service.process_pdf_uploads(
        [UploadPayload("publish-failure.pdf", _pdf_bytes("NEW_CONTENT"))]
    )[0]

    assert result.status == "failed"
    assert "registry write failure" in result.message
    assert (processed_dir / "previous.jsonl").read_text(encoding="utf-8") == "previous"
    for strategy in ("fixed_size", "recursive"):
        assert (
            index_dir / strategy / "previous.manifest"
        ).read_text(encoding="utf-8") == "previous"
    assert not list((raw_dir / "uploaded").glob("*.pdf"))
    assert load_upload_registry()[0]["processing_status"] == "failed"
