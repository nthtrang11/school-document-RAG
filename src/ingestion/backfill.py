"""Reprocess the current corpus and publish it only after integrity checks pass."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings
from src.indexing.index_manager import build_strategy, load_strategy
from src.ingestion.ingestion_pipeline import run_ingestion
from src.ingestion.upload_service import _publish_directories, _upload_lock
from src.models import DocumentMetadata, PageRecord
from src.retrieval.hybrid_retriever import HybridRetriever

LOGGER = logging.getLogger(__name__)
_STRATEGIES = ("fixed_size", "recursive")
_SMOKE_QUERIES = (
    "Thông báo nào về đăng ký bổ sung lớp học phần?",
    "Gia hạn thời gian nộp học phí trong học kỳ",
    "Thời hạn nộp học phí là khi nào?",
    "Đăng ký bổ sung lớp học phần có quy mô nhỏ",
    "Hạn đăng ký bổ sung lớp học phần là khi nào?",
    "Thông báo nào gia hạn thời gian nộp học phí?",
    "Văn bản nào được viện dẫn trong thông báo?",
    "Hạn đóng bảo hiểm y tế là khi nào?",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _page_issues(page: PageRecord) -> list[str]:
    issues = []
    dimensions = page.page_dimensions
    for region in page.ocr_regions:
        if region.bbox is None:
            continue
        if dimensions is None:
            issues.append("OCR bbox without page dimensions")
            break
        x0, y0, x1, y1 = region.bbox
        if x0 < 0 or y0 < 0 or x1 > dimensions[0] or y1 > dimensions[1]:
            issues.append("OCR bbox outside page dimensions")
            break
        if x0 >= x1 or y0 >= y1:
            issues.append("Invalid OCR bbox")
            break
    if page.ocr_quality and page.ocr_quality.indexable and not page.text.strip():
        issues.append("Page marked indexable without indexed text")
    return issues


def _corpus_snapshot(processed_dir: Path, index_root: Path) -> dict[str, Any]:
    metadata_records = [
        DocumentMetadata.model_validate(item)
        for item in _read_jsonl(processed_dir / "documents.jsonl")
    ]
    pages = [
        PageRecord.model_validate(item)
        for item in _read_jsonl(processed_dir / "pages_clean.jsonl")
    ]
    documents = {item.document_id: item for item in metadata_records}
    pages_by_document: dict[str, list[PageRecord]] = {}
    for page in pages:
        pages_by_document.setdefault(page.document_id, []).append(page)

    chunk_counts: dict[str, Counter[str]] = {
        strategy: Counter(
            item["document_id"]
            for item in _read_jsonl(index_root / strategy / "chunks.jsonl")
        )
        for strategy in _STRATEGIES
    }
    index_stats: dict[str, Any] = {}
    integrity_issues: list[str] = []
    for strategy in _STRATEGIES:
        try:
            bundle = load_strategy(strategy, index_root=index_root)
            index_stats[strategy] = {
                "chunks": len(bundle.chunks),
                "embeddings": bundle.faiss.index.ntotal,
                "bm25_records": len(bundle.bm25.tokens),
            }
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            index_stats[strategy] = {"error": str(exc)}
            integrity_issues.append(f"{strategy}: {exc}")

    per_document = {}
    for document_id, metadata in documents.items():
        document_pages = pages_by_document.get(document_id, [])
        missing_fields = [
            field
            for field in (
                "document_type",
                "number_symbol",
                "issue_date",
                "title_subject",
                "issuing_authority",
            )
            if getattr(metadata, field) is None
        ]
        page_ocr_status = Counter(
            page.ocr_quality.status
            for page in document_pages
            if page.ocr_quality is not None
        )
        per_document[document_id] = {
            "source_path": metadata.source_path,
            "page_count_expected": metadata.page_count,
            "page_records": len(document_pages),
            "text_layer_pages": sum(
                page.text_source == "text_layer" for page in document_pages
            ),
            "ocr_pages": sum(
                page.text_source in {"ocr", "hybrid"} for page in document_pages
            ),
            "ocr_status_counts": dict(page_ocr_status),
            "low_quality_pages": [
                page.page_no
                for page in document_pages
                if page.ocr_quality
                and page.ocr_quality.status in {"needs_review", "failed", "unreadable"}
            ],
            "empty_pages": [
                page.page_no for page in document_pages if not page.text.strip()
            ],
            "missing_metadata_fields": missing_fields,
            "metadata_verification_status": metadata.metadata_verification_status,
            "chunks": {
                strategy: chunk_counts[strategy].get(document_id, 0)
                for strategy in _STRATEGIES
            },
            "index_status": {
                strategy: (
                    index_stats[strategy].get("chunks")
                    == index_stats[strategy].get("embeddings")
                    == index_stats[strategy].get("bm25_records")
                    and chunk_counts[strategy].get(document_id, 0) > 0
                )
                for strategy in _STRATEGIES
            },
            "page_issues": [
                {"page": page.page_no, "issues": _page_issues(page)}
                for page in document_pages
                if _page_issues(page)
            ],
        }

    return {
        "documents": len(metadata_records),
        "pages": len(pages),
        "text_pages": sum(bool(page.text.strip()) for page in pages),
        "ocr_pages": sum(
            page.text_source in {"ocr", "hybrid"} for page in pages
        ),
        "ocr_success_pages": sum(
            page.ocr_quality is not None
            and page.ocr_quality.status == "completed"
            for page in pages
        ),
        "ocr_review_or_failed_pages": sum(
            page.ocr_quality is not None
            and page.ocr_quality.status in {"needs_review", "failed", "unreadable"}
            for page in pages
        ),
        "chunks_by_strategy": {
            strategy: sum(chunk_counts[strategy].values())
            for strategy in _STRATEGIES
        },
        "indexes": index_stats,
        "integrity_issues": integrity_issues,
        "documents_by_id": per_document,
    }


def _ranked_results(
    results: list[tuple[str, float]],
    chunks_by_id: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "document_id": chunks_by_id[chunk_id].document_id,
            "chunk_id": chunk_id,
            "page": chunks_by_id[chunk_id].page,
            "score": float(score),
        }
        for rank, (chunk_id, score) in enumerate(results[:3], start=1)
        if chunk_id in chunks_by_id
    ]


def _retrieval_smoke_tests(index_root: Path) -> dict[str, Any]:
    strategies = {}
    for strategy in _STRATEGIES:
        bundle = load_strategy(strategy, index_root=index_root)
        retriever = HybridRetriever(bundle)
        chunks_by_id = {chunk.chunk_id: chunk for chunk in bundle.chunks}
        query_results = []
        for query_number, query in enumerate(_SMOKE_QUERIES, start=1):
            result = retriever.search(query, top_k=3)
            bm25 = _ranked_results(result.get("bm25", []), chunks_by_id)
            faiss = _ranked_results(result.get("faiss", []), chunks_by_id)
            hybrid = [
                {
                    "rank": rank,
                    "document_id": item["chunk"].document_id,
                    "chunk_id": item["chunk"].chunk_id,
                    "page": item["chunk"].page,
                    "score": float(item.get("rrf_score", 0.0)),
                }
                for rank, item in enumerate(result.get("results", [])[:3], start=1)
            ]
            query_results.append(
                {
                    "query_id": f"Q{query_number}",
                    "query": query,
                    "bm25": bm25,
                    "faiss": faiss,
                    "hybrid": hybrid,
                }
            )
        strategies[strategy] = query_results
    return {
        "queries": strategies,
        "query_count": len(_SMOKE_QUERIES),
    }


def _retrieval_regressions(
    before: dict[str, Any], after: dict[str, Any]
) -> list[str]:
    regressions = []
    for strategy in _STRATEGIES:
        before_queries = {
            item["query_id"]: item
            for item in before["queries"].get(strategy, [])
        }
        for result in after["queries"].get(strategy, []):
            query_id = result["query_id"]
            baseline = before_queries.get(query_id)
            if baseline is None:
                regressions.append(f"{strategy} {query_id}: no baseline result")
                continue
            if not baseline["hybrid"]:
                continue
            baseline_top_document = baseline["hybrid"][0]["document_id"]
            old_rank = next(
                (
                    item["rank"]
                    for item in baseline["hybrid"]
                    if item["document_id"] == baseline_top_document
                ),
                None,
            )
            new_rank = next(
                (
                    item["rank"]
                    for item in result["hybrid"]
                    if item["document_id"] == baseline_top_document
                ),
                None,
            )
            if old_rank is not None and (new_rank is None or new_rank > old_rank):
                regressions.append(
                    f"{strategy} {query_id}: top-document rank regressed "
                    f"from {old_rank} to {new_rank}"
                )
            if not result["hybrid"]:
                regressions.append(f"{strategy} {query_id}: no hybrid results")
    return regressions


def _verify_staging(
    snapshot: dict[str, Any],
    smoke: dict[str, Any],
    smoke_before: dict[str, Any],
) -> list[str]:
    issues = list(snapshot["integrity_issues"])
    if not snapshot["documents"] or snapshot["pages"] == 0:
        issues.append("Staging corpus is empty")
    for document_id, item in snapshot["documents_by_id"].items():
        if item["page_count_expected"] != item["page_records"]:
            issues.append(f"{document_id}: page count mismatch")
        for strategy in _STRATEGIES:
            if not item["index_status"][strategy]:
                issues.append(
                    f"{document_id}: {strategy} index is incomplete or has no chunks"
                )
        for page in item["page_issues"]:
            issues.append(f"{document_id} page {page['page']}: {page['issues']}")
    if smoke["query_count"] != len(_SMOKE_QUERIES):
        issues.append("Retrieval smoke query count does not match the configured set")
    issues.extend(_retrieval_regressions(smoke_before, smoke))
    return issues


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_backfill(*, apply_changes: bool = False) -> dict[str, Any]:
    settings.project_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "apply_requested": apply_changes,
        "source_directory": str(settings.pdf_dir),
        "processing_version": "2",
    }
    report_path = settings.results_dir / "backfill" / "backfill_report.json"
    with _upload_lock():
        report["before"] = _corpus_snapshot(settings.processed_dir, settings.index_dir)
        report["retrieval_before"] = _retrieval_smoke_tests(settings.index_dir)
        with tempfile.TemporaryDirectory(
            prefix=".ocr-backfill-",
            dir=settings.project_root,
        ) as temporary:
            transaction_root = Path(temporary)
            staged_processed = transaction_root / "processed"
            staged_indexes = transaction_root / "indexes"
            try:
                report["ingestion"] = run_ingestion(
                    pdf_dir=settings.pdf_dir,
                    output_dir=staged_processed,
                )
                for strategy in _STRATEGIES:
                    build_strategy(
                        strategy,
                        pages_path=staged_processed / "pages_clean.jsonl",
                        force=True,
                        output_root=staged_indexes,
                        processed_dir=staged_processed,
                    )
                    load_strategy(strategy, index_root=staged_indexes)
                report["staged"] = _corpus_snapshot(
                    staged_processed, staged_indexes
                )
                report["retrieval_after"] = _retrieval_smoke_tests(staged_indexes)
                issues = _verify_staging(
                    report["staged"],
                    report["retrieval_after"],
                    report["retrieval_before"],
                )
                report["verification_issues"] = issues
                if issues:
                    report["status"] = "verification_failed"
                elif apply_changes:
                    _publish_directories(
                        staged_processed,
                        staged_indexes,
                        transaction_root,
                    )
                    report["status"] = "published"
                    report["after"] = _corpus_snapshot(
                        settings.processed_dir, settings.index_dir
                    )
                    report["retrieval_after_publish"] = _retrieval_smoke_tests(
                        settings.index_dir
                    )
                else:
                    report["status"] = "verified_not_published"
            except Exception as exc:
                report["status"] = "failed"
                report["error"] = f"{type(exc).__name__}: {exc}"
                report["completed_at"] = datetime.now(timezone.utc).isoformat()
                _write_report(report_path, report)
                raise
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_report(report_path, report)
    if report["status"] not in {"published", "verified_not_published"}:
        raise RuntimeError(
            f"Backfill không được publish: {report.get('verification_issues')}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Publish staged processed data and indexes after verification.",
    )
    args = parser.parse_args()
    report = run_backfill(apply_changes=args.apply)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
