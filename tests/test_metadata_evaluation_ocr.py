import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pymupdf
import pytest

from src.config import settings
from src.evaluation.dataset import is_verified, question_skip_reason
from src.evaluation.report import run_evaluation
from src.evaluation.retrieval_metrics import ndcg_at_k
from src.ingestion.metadata_extractor import (
    extract_metadata,
    resolve_document_relations,
)
from src.ingestion.metadata_review import save_metadata_review
from src.ingestion.pdf_loader import read_pdf
from src.models import (
    ChunkRecord,
    DocumentMetadata,
    DocumentRelation,
    MetadataEvidence,
    PageRecord,
)


def _metadata(text: str) -> DocumentMetadata:
    page = PageRecord(document_id="doc-test", page_no=1, text=text)
    return extract_metadata(
        Path("test.pdf"),
        Path("."),
        "doc-test",
        "a" * 64,
        [page],
    )


def _pdf(path: Path, text: str = "") -> Path:
    document = pymupdf.open()
    page = document.new_page()
    if text:
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()
    return path


def test_ndcg_binary_and_graded_relevance():
    assert ndcg_at_k(["b", "a", "x"], {"a", "b"}, 3) == pytest.approx(
        (1 + 1 / 2) / (1 + 1 / 2)
    )
    assert ndcg_at_k(["b", "a"], {"a": 3.0, "b": 1.0}, 3) < 1.0
    assert ndcg_at_k([], {"a"}, 3) == 0.0
    assert ndcg_at_k(["a"], set(), 3) == 0.0


def test_evaluation_requires_manual_reviewer_and_document_page_ground_truth():
    row = pd.Series(
        {
            "question_id": "Q1",
            "question": "Tìm nội dung",
            "verification_status": "verified",
            "relevant_document_ids": "doc-1",
            "relevant_pages": "2",
            "verified_by": "",
            "verified_at": "2026-10-01",
        }
    )
    assert not is_verified(row)
    assert question_skip_reason(row) == "ground_truth_invalid:missing_verified_by"
    row["verification_status"] = "needs_manual_verification"
    assert question_skip_reason(row) == "needs_manual_ground_truth"


def test_effective_date_deadlines_and_relations_keep_page_evidence():
    metadata = _metadata(
        "THÔNG BÁO\n"
        "Số: 123/TB-XYZ\n"
        "Hà Nội, ngày 01 tháng 02 năm 2026\n"
        "Văn bản này có hiệu lực kể từ ngày 05/02/2026.\n"
        "Đối tượng A hoàn thành trước ngày 10/02/2026.\n"
        "Đối tượng B nộp hồ sơ chậm nhất vào ngày 20/02/2026.\n"
        "Thay thế Thông báo số 900/TB-XYZ.\n"
        "Căn cứ Thông báo số 500/TB-XYZ."
    )

    assert metadata.issue_date == "2026-02-01"
    assert metadata.effective_date == "2026-02-05"
    assert metadata.effective_date_raw == "05/02/2026"
    assert [item.value for item in metadata.deadlines] == [
        "2026-02-10",
        "2026-02-20",
    ]
    assert all(item.source_page == 1 and item.evidence for item in metadata.deadlines)
    assert metadata.deadline == "2026-02-10"
    assert metadata.replaces == "900/TB-XYZ"
    assert metadata.amends is None
    assert [
        (item.relation_type, item.referenced_number)
        for item in metadata.document_relations
    ] == [
        ("replaces", "900/TB-XYZ"),
        ("references", "500/TB-XYZ"),
    ]
    assert metadata.field_provenance["effective_date"][0].source_page == 1


def test_metadata_does_not_infer_effective_date_or_relations_from_citations():
    metadata = _metadata(
        "QUYẾT ĐỊNH\n"
        "Số: 123/QĐ-XYZ\n"
        "Hà Nội, ngày 01/02/2026\n"
        "Căn cứ Thông báo số 900/TB-XYZ."
    )

    assert metadata.issue_date == "2026-02-01"
    assert metadata.effective_date is None
    assert [
        (item.relation_type, item.referenced_number)
        for item in metadata.document_relations
    ] == [("references", "900/TB-XYZ")]


def test_effective_date_in_a_multiline_law_citation_is_not_document_effective_date():
    metadata = _metadata(
        "KẾ HOẠCH\n"
        "Về việc thu phí bảo hiểm y tế\n"
        "Căn cứ Luật Bảo hiểm y tế số 25/2008/QH12 ngày 14/11/2008 của Quốc hội\n"
        "khóa 12 kỳ họp thứ 10 thông qua và có hiệu lực từ ngày 01/07/2009; Luật số\n"
        "46/2014/QH13 ngày 13/6/2014 sửa đổi, bổ sung một số điều của luật BHYT;\n"
        "Căn cứ Nghị định số 188/2025/NĐ-CP ngày 01/07/2025 của Chính phủ."
    )

    assert metadata.effective_date is None


def test_document_relations_link_unique_references_and_flag_ambiguous_numbers():
    relation = DocumentRelation(
        relation_type="extends",
        referenced_number="923/TB-ĐHGTVT",
        raw_value="923/TB-DHGTVT",
        source_page=1,
        evidence="Gia hạn thời hạn theo Thông báo số 923/TB-DHGTVT.",
    )
    source = DocumentMetadata(
        document_id="source",
        number_symbol="939/TB-ĐHGTVT",
        document_relations=[relation],
        source_path="source.pdf",
        accessed_at="2026-01-01T00:00:00Z",
        file_hash="a" * 64,
        page_count=1,
    )
    target = DocumentMetadata(
        document_id="target",
        number_symbol="923/TB-ĐHGTVT",
        source_path="target.pdf",
        accessed_at="2026-01-01T00:00:00Z",
        file_hash="b" * 64,
        page_count=1,
    )

    linked_source = resolve_document_relations([source, target])[0]
    assert linked_source.document_relations[0].target_document_id == "target"
    assert linked_source.document_relations[0].target_match_status == "matched"

    duplicate = target.model_copy(update={"document_id": "duplicate"})
    ambiguous_source = resolve_document_relations(
        [source, target, duplicate]
    )[0]
    assert ambiguous_source.document_relations[0].target_document_id is None
    assert ambiguous_source.document_relations[0].target_match_status == "ambiguous"
    assert ambiguous_source.metadata_verification_status == "needs_manual_review"


def test_effective_date_from_cited_law_is_not_document_effective_date_across_lines():
    metadata = _metadata(
        "THÔNG BÁO\n"
        "Số: 123/TB-XYZ\n"
        "Hà Nội, ngày 01/02/2026\n"
        "Căn cứ Luật Bảo hiểm y tế được Quốc hội thông qua\n"
        "và có hiệu lực từ ngày 01/07/2009."
    )

    assert metadata.effective_date is None


def test_effective_date_is_extracted_when_current_document_is_subject():
    metadata = _metadata(
        "THÔNG BÁO\n"
        "Số: 123/TB-XYZ\n"
        "Hà Nội, ngày 01/02/2026\n"
        "Văn bản này có hiệu lực kể từ ngày 05/02/2026."
    )

    assert metadata.effective_date == "2026-02-05"
    evidence = metadata.field_provenance["effective_date"][0]
    assert evidence.source_page == 1
    assert evidence.raw_value == "05/02/2026"
    assert "Văn bản này" in evidence.evidence


def test_metadata_relation_extractor_distinguishes_amendment_from_replacement():
    metadata = _metadata(
        "Sửa đổi, bổ sung một số nội dung tại Thông báo số 900/TB-XYZ.\n"
        "Bãi bỏ Công văn số 123/CV-ABC."
    )

    assert metadata.amends == "900/TB-XYZ"
    assert metadata.replaces is None
    assert [item.relation_type for item in metadata.document_relations] == [
        "amends",
        "revokes",
    ]


def test_metadata_does_not_treat_routine_supplementary_registration_as_amendment():
    metadata = _metadata(
        "Đăng ký bổ sung lớp học phần theo quy định tại mục 1.\n"
        "Thực hiện theo Thông báo số 923/TB-XYZ."
    )

    assert [
        (item.relation_type, item.referenced_number)
        for item in metadata.document_relations
    ] == [("references", "923/TB-XYZ")]


def test_metadata_detects_wrapped_deadline_amendment_reference():
    page = PageRecord(
        document_id="doc-wrapped-relation",
        page_no=1,
        text=(
            "Trường Đại học GTVT\n"
            "Thay cho\n"
            "thdi han trước ngày 20/10/2026 quy định tại\n"
            "Thông báo số 923/TB-DHGTVT."
        ),
    )
    metadata = extract_metadata(
        Path("notice.pdf"),
        Path("."),
        page.document_id,
        "c" * 64,
        [page],
    )

    assert len(metadata.document_relations) == 1
    assert metadata.document_relations[0].relation_type == "extends"
    assert metadata.document_relations[0].referenced_number == "923/TB-ĐHGTVT"


def test_duplicate_deadline_evidence_keeps_the_more_informative_description():
    metadata = _metadata(
        "dupe gia han den het ngay 01/11/2026.\n"
        "hoc phi du'oc gia han den het ngay 01/11/2026."
    )

    assert len(metadata.deadlines) == 1
    assert metadata.deadlines[0].value == "2026-11-01"
    assert "hoc phi" in metadata.deadlines[0].deadline_type
    assert "dupe gia han" in metadata.deadlines[0].evidence
    assert "hoc phi du'oc gia han" in metadata.deadlines[0].evidence


def test_duplicate_deadline_contexts_merge_without_merging_distinct_tasks():
    duplicate = _metadata(
        "Bao gồm sinh viên đăng ký bổ sung: thời hạn nộp học phí được gia hạn "
        "đến hết ngày 01/11/2026.\n"
        "Học phí được gia hạn đến hết ngày 01/11/2026."
    )
    distinct = _metadata(
        "Thu học phí trước ngày 01/11/2026.\n"
        "Hoàn thành đăng ký môn học trước ngày 01/11/2026."
    )

    assert len(duplicate.deadlines) == 1
    assert len(distinct.deadlines) == 2


def test_deadline_type_keeps_the_full_action_context_and_range_dates():
    metadata = _metadata(
        "Phòng Đào tạo hướng dẫn sinh viên đăng ký và nộp hồ sơ trực tuyến "
        "trước ngày\n24/07/2026."
    )

    assert len(metadata.deadlines) == 1
    assert "Phòng Đào tạo hướng dẫn sinh viên đăng ký" in (
        metadata.deadlines[0].deadline_type
    )
    assert metadata.deadlines[0].value == "2026-07-24"
    assert metadata.deadline == metadata.deadlines[0].value
    assert metadata.field_provenance["deadline"][0].value == metadata.deadline


def test_hybrid_metadata_uses_native_fallback_for_dates_and_relations():
    page = PageRecord(
        document_id="doc-hybrid",
        page_no=1,
        text=(
            "S: 939/TB-ĐHGTVT\n"
            "Hà Nội, ngày 20/10/2026\n"
            "THÔNG BÁO\n"
            "V vic đăng ký bổ sung và gia hạn học phí"
        ),
        text_source="hybrid",
        text_layer_text=(
            "Số: 939/TB-DHGTVT\n"
            "Trường Đại học GTVT\n"
            "Hà Nội, ngày 20/10/2026\n"
            "THÔNG BÁO\n"
            "Về việc đăng ký bổ sung học phần\n"
            "Sinh viên đăng ký bổ sung đến hết ngày 22/10/2026.\n"
            "Thời hạn nộp học phí được gia hạn đến hết ngày 01/11/2026, "
            "thay cho thời hạn trước ngày 20/10/2026 quy định tại mục 5 "
            "Thông báo số 923/TB-DHGTVT."
        ),
    )
    metadata = extract_metadata(
        Path("notice.pdf"),
        Path("."),
        page.document_id,
        "b" * 64,
        [page],
    )

    assert metadata.number_symbol == "939/TB-ĐHGTVT"
    assert metadata.issue_date == "2026-10-20"
    assert metadata.document_type == "Thông báo"
    assert metadata.title_subject == "V vic đăng ký bổ sung và gia hạn học phí"
    assert metadata.issuing_authority == "Trường Đại học GTVT"
    assert {"2026-10-22", "2026-11-01", "2026-10-20"}.issubset(
        {deadline.value for deadline in metadata.deadlines}
    )
    assert metadata.extends == "923/TB-ĐHGTVT"
    assert metadata.effective_date is None
    assert metadata.field_provenance["issuing_authority"][0].source_type == "text_layer"


def test_effective_date_from_cited_regulation_is_not_document_effective_date():
    metadata = _metadata(
        "THÔNG BÁO\n"
        "Số: 123/TB-XYZ\n"
        "Hà Nội, ngày 01/02/2026\n"
        "Căn cứ Nghị định số 1/2020/NĐ-CP có hiệu lực từ ngày 01/07/2020."
    )

    assert metadata.effective_date is None


def test_metadata_review_persists_reviewer_and_field_status(tmp_path: Path):
    metadata = DocumentMetadata(
        document_id="doc-review",
        document_type="Thông báo",
        source_path="notice.pdf",
        accessed_at="2026-01-01T00:00:00Z",
        file_hash="a" * 64,
        page_count=1,
        field_provenance={
            "document_type": [
                MetadataEvidence(
                    value="Thông báo",
                    raw_value="THÔNG BÁO",
                    evidence="THÔNG BÁO",
                    source_page=1,
                )
            ]
        },
    )
    path = tmp_path / "documents.jsonl"
    path.write_text(metadata.model_dump_json() + "\n", encoding="utf-8")

    reviewed = save_metadata_review(
        path,
        metadata,
        reviewer="Reviewer",
        status="verified",
        notes="Đã đối chiếu PDF gốc",
    )
    stored = DocumentMetadata.model_validate_json(path.read_text(encoding="utf-8"))
    assert reviewed.metadata_verification_status == "verified"
    assert stored.metadata_reviewed_by == "Reviewer"
    assert stored.metadata_review_notes == "Đã đối chiếu PDF gốc"
    assert stored.field_provenance["document_type"][0].verification_status == "verified"


def test_native_text_page_skips_ocr(tmp_path: Path, monkeypatch):
    from src.ingestion import pdf_loader

    path = _pdf(tmp_path / "native.pdf", "Văn bản có text layer đầy đủ.")

    def unexpected_ocr():
        pytest.fail("A good native text page should not invoke OCR")

    monkeypatch.setattr(pdf_loader, "_ocr_engine", unexpected_ocr)
    _, _, pages = read_pdf(path, tmp_path)

    assert pages[0].page_classification == "native_text"
    assert pages[0].text_source == "text_layer"
    assert pages[0].ocr_quality.status == "not_needed"


def test_scan_ocr_preserves_page_bbox_and_excludes_low_quality_text(
    tmp_path: Path, monkeypatch
):
    from src.ingestion import pdf_loader

    path = _pdf(tmp_path / "scan.pdf")

    class OCR:
        def __call__(self, _image):
            return SimpleNamespace(
                txts=["Văn bản scan tiếng Việt đủ dài để lập chỉ mục."],
                scores=[0.98],
                boxes=[[[20, 20], [220, 20], [220, 60], [20, 60]]],
            )

    monkeypatch.setattr(pdf_loader, "_ocr_engine", lambda: OCR())
    _, _, pages = read_pdf(path, tmp_path)
    page = pages[0]
    assert page.page_classification == "ocr_required"
    assert page.text_source == "ocr"
    assert page.ocr_quality.status == "completed"
    assert "tiếng Việt" in page.text
    assert page.ocr_regions[0].bbox == pytest.approx((10.0, 10.0, 110.0, 30.0))

    monkeypatch.setattr(settings, "ocr_min_text_characters", 100)
    _, _, low_quality_pages = read_pdf(path, tmp_path)
    low_quality = low_quality_pages[0]
    assert low_quality.ocr_quality.status == "needs_review"
    assert not low_quality.ocr_quality.indexable
    assert low_quality.text == ""
    assert low_quality.ocr_regions[0].text


def test_ocr_failure_is_recorded_without_fabricated_text_or_bbox(
    tmp_path: Path, monkeypatch
):
    from src.ingestion import pdf_loader

    path = _pdf(tmp_path / "unreadable.pdf")

    class FailingOCR:
        def __call__(self, _image):
            raise RuntimeError("engine failed")

    monkeypatch.setattr(pdf_loader, "_ocr_engine", lambda: FailingOCR())
    _, _, pages = read_pdf(path, tmp_path)

    assert pages[0].page_classification == "unreadable"
    assert pages[0].text == ""
    assert pages[0].ocr_quality.status == "failed"
    assert pages[0].ocr_quality.indexable is False
    assert pages[0].ocr_regions == []


def test_evaluation_exports_grouped_real_rankings_and_keeps_status_blocked(
    tmp_path: Path, monkeypatch
):
    from src.evaluation import report

    dataset_path = tmp_path / "questions.csv"
    pd.DataFrame(
        [
            {
                "question_id": "Q1",
                "question": "Tìm thông báo",
                "question_type": "Keyword",
                "ground_truth_answer": "Nội dung thật",
                "relevant_document_ids": "doc-1",
                "relevant_pages": "1",
                "relevance_scores": "",
                "verified_by": "Reviewer",
                "verified_at": "2026-11-01T00:00:00Z",
                "verification_status": "verified",
            }
        ]
    ).to_csv(dataset_path, index=False)
    monkeypatch.setattr(settings, "evaluation_file", dataset_path)
    monkeypatch.setattr(settings, "results_dir", tmp_path / "results")
    monkeypatch.setattr(settings, "evaluation_min_verified_questions", 2)
    chunks = [
        ChunkRecord(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="Relevant",
            page=1,
            end_page=1,
            strategy="fixed_size",
        ),
        ChunkRecord(
            chunk_id="chunk-2",
            document_id="doc-2",
            text="Other",
            page=1,
            end_page=1,
            strategy="fixed_size",
        ),
    ]

    class FakeBundle:
        def __init__(self, strategy):
            self.strategy = strategy
            self.chunks = chunks

    class FakeRetriever:
        def __init__(self, _bundle):
            pass

        def search(self, _question, top_k):
            assert top_k == 3
            return {
                "bm25": [("chunk-1", 1.0), ("chunk-2", 0.5)],
                "faiss": [("chunk-2", 0.8), ("chunk-1", 0.4)],
                "results": [
                    {"chunk": chunks[0], "rrf_score": 0.03},
                    {"chunk": chunks[1], "rrf_score": 0.02},
                ],
                "latency_seconds": {
                    "bm25": 0.01,
                    "faiss": 0.02,
                    "hybrid": 0.04,
                },
            }

    monkeypatch.setattr(report, "load_strategy", lambda strategy: FakeBundle(strategy))
    monkeypatch.setattr(report, "HybridRetriever", FakeRetriever)
    summary = run_evaluation()

    assert summary["status"] == "insufficient_verified_ground_truth"
    assert summary["verified_questions"] == 1
    assert summary["evaluated_rows"] == 6
    assert {row["method"] for row in summary["comparison"]} == {
        "bm25",
        "faiss",
        "hybrid",
    }
    assert len(summary["by_query_type"]) == 6
    report_rows = pd.read_csv(
        settings.results_dir / "metrics" / "retrieval_metrics.csv"
    )
    assert set(report_rows["evaluation_unit"]) == {"document"}
    assert "ndcg_at_3" in report_rows.columns
    assert json.loads(
        (settings.results_dir / "cases" / "case_1.json").read_text(
            encoding="utf-8"
        )
    )["status"] == "retrieved_analysis_pending"
