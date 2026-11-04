from pathlib import Path

import pytest

from src.ingestion.cleaner import clean_pages
from src.ingestion.pdf_loader import document_id_for, read_pdf
from src.ingestion.metadata_extractor import extract_metadata, extract_pdf_metadata
from src.models import DocumentMetadata, PageRecord


def _extract_metadata(*page_texts: str) -> DocumentMetadata:
    pages = [
        PageRecord(document_id="doc-test", page_no=index, text=text)
        for index, text in enumerate(page_texts, start=1)
    ]
    return extract_metadata(
        Path("test.pdf"), Path("."), "doc-test", "a" * 64, pages
    )


def test_cleaner_preserves_vietnamese_identifiers_dates_and_tables():
    page = PageRecord(
        document_id="doc-a",
        page_no=1,
        text="  Số: 406/TB-ĐHGTVT  \r\n\r\nNgày 28/5/2026\r\nHọc phí | 1.250.000 đồng ",
    )
    raw, cleaned = clean_pages([page])
    assert "406/TB-ĐHGTVT" in raw[0].text
    assert "406/TB-ĐHGTVT" in cleaned[0].text
    assert "1.250.000 đồng" in cleaned[0].text


def test_read_pdf_uses_original_one_based_pages(tmp_path: Path):
    import pymupdf

    path = tmp_path / "source.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Trang mot")
    document.new_page().insert_text((72, 72), "Trang hai")
    document.save(path)
    document.close()
    document_id, file_hash, pages = read_pdf(path, tmp_path)
    assert document_id == document_id_for(path, tmp_path)
    assert len(file_hash) == 64
    assert [page.page_no for page in pages] == [1, 2]


def test_table_parts_on_adjacent_pages_share_group_only_when_headers_match():
    from src.ingestion.pdf_loader import _link_continued_tables
    from src.models import TableRowRecord

    def page(page_no, table_id, headers, cells):
        return PageRecord(
            document_id="doc-multipage",
            page_no=page_no,
            text="",
            table_rows=[
                TableRowRecord(
                    table_id=table_id,
                    row_index=1,
                    headers=headers,
                    header_rows=[headers],
                    cells=cells,
                    bboxes=[None, None],
                )
            ],
        )

    pages = _link_continued_tables(
        [
            page(1, "doc-multipage:p1:t1", ["Đối tượng", "Mức"], ["Nhóm A", "100"]),
            page(2, "doc-multipage:p2:t1", ["Đối tượng", "Mức"], ["Nhóm B", "200"]),
            page(3, "doc-multipage:p3:t1", ["Khóa", "Thời hạn"], ["Khóa C", "1 năm"]),
        ]
    )

    assert pages[0].table_rows[0].table_id != pages[1].table_rows[0].table_id
    assert pages[0].table_rows[0].table_group_id == (
        pages[1].table_rows[0].table_group_id
    )
    assert pages[2].table_rows[0].table_group_id is None


def test_real_faculty_feedback_notice_extracts_activity_deadline_and_issue_date():
    source_dir = Path(__file__).parents[1] / "data" / "raw"
    source = source_dir / "danh_gia_giang_vien.pdf"
    if not source.is_file():
        pytest.skip("Project source PDF is unavailable")

    document_id, file_hash, pages = read_pdf(source, source_dir)
    metadata = extract_pdf_metadata(
        source, source_dir, document_id, file_hash, pages
    )

    assert len(pages) == 1
    assert "Từ 5/7/2026" in pages[0].text
    assert "đến hết ngày" in pages[0].text
    assert "20/07/2026" in pages[0].text
    assert metadata.number_symbol == "516/TB-ĐHGTVT"
    assert metadata.issue_date == "2026-07-01"


def test_real_project_table_extraction_keeps_complete_row_cells_and_bboxes():
    source_dir = Path(__file__).parents[1] / "data" / "raw"
    source = source_dir / "ke_hoach_hk1_2026.pdf"
    if not source.is_file():
        pytest.skip("Project source PDF is unavailable")

    _, _, pages = read_pdf(source, source_dir)
    row = next(
        row for row in pages[6].table_rows
        if any("Đợt học 26" in cell for cell in row.cells)
    )

    assert row.headers[1] == "Lớp học"
    assert "Đợt học 26" in row.cells[1]
    assert "VPK 303-B3" in row.cells[3]
    assert "Từ 10/8 đến 30/8/2026" in row.cells[3]
    assert all(
        bbox is not None
        for cell, bbox in zip(row.cells, row.bboxes)
        if cell
    )


def test_real_schedule_table_titles_keep_distinct_audiences_separate():
    source_dir = Path(__file__).parents[1] / "data" / "raw"
    source = source_dir / "ke_hoach_hk1_2026.pdf"
    if not source.is_file():
        pytest.skip("Project source PDF is unavailable")

    _, _, pages = read_pdf(source, source_dir)
    regular_rows = pages[5].table_rows
    advanced_rows = pages[6].table_rows

    assert len(regular_rows) == 6
    assert len(advanced_rows) == 6
    assert "LỊCH ĐĂNG KÝ HỌC" in regular_rows[0].table_title
    assert "CÁC LỚP CHƯƠNG TRÌNH TIÊN TIẾN" in advanced_rows[0].table_title
    assert {
        row.table_group_id for row in regular_rows
    }.isdisjoint({row.table_group_id for row in advanced_rows})


def test_real_bhyt_table_repeats_merged_cells_on_each_applicable_row():
    source_dir = Path(__file__).parents[1] / "data" / "raw"
    source = source_dir / "bao_hiem_y_te.pdf"
    if not source.is_file():
        pytest.skip("Project source PDF is unavailable")

    _, _, pages = read_pdf(source, source_dir)
    rows = pages[0].table_rows
    assert all(row.table_title is None for row in rows)
    assert rows[0].header_cell_spans[0].column_end == 2
    assert rows[0].header_cell_spans[1] == rows[0].header_cell_spans[0]
    keyed_rows = {
        (row.cells[0], row.cells[1]): row
        for row in rows
    }

    assert keyed_rows[("Khóa 66", "")].cells[2:] == [
        "683.100",
        "12 tháng (Từ 01/01/2027 đến\n31/12/2027)",
    ]
    assert keyed_rows[("Khóa 66", "")].cell_states == [
        "explicit_value",
        "covered_by_merged_cell",
        "explicit_value",
        "explicit_value",
    ]
    assert keyed_rows[("Khóa 65", "")].cells[2:] == [
        "683.100",
        "12 tháng (Từ 01/01/2027 đến\n31/12/2027)",
    ]
    assert keyed_rows[("Khóa 65", "")].cell_states[2:] == [
        "inherited_from_merged_cell",
        "inherited_from_merged_cell",
    ]
    assert keyed_rows[("Khoá 64", "Hệ kỹ sư")].cells[2:] == [
        "683.100",
        "12 tháng (Từ 01/01/2027 đến\n31/12/2027)",
    ]
    assert keyed_rows[("Khoá 64", "Hệ cử nhân")].cells[2:] == [
        "455.400",
        "08 tháng (Từ 01/01/2027 đến\n31/08/2027)",
    ]
    assert keyed_rows[("Khoá 64", "Hệ cử nhân")].cell_states[0] == (
        "inherited_from_merged_cell"
    )
    assert keyed_rows[("Khóa 63", "Hệ kỹ sư")].cells[2:] == [
        "455.400",
        "08 tháng (Từ 01/01/2027 đến\n31/08/2027)",
    ]
    assert all(
        row.bboxes[column] is not None
        for row in rows
        for column in (2, 3)
    )
    assert keyed_rows[("Khóa 65", "")].cell_spans[2].row_start == 1
    assert keyed_rows[("Khóa 65", "")].cell_spans[2].row_end == 3
    assert keyed_rows[("Khóa 66", "")].cell_spans[1].column_end == 2
    assert keyed_rows[("Khóa 66", "")].table_id == keyed_rows[
        ("Khóa 65", "")
    ].table_id
    assert keyed_rows[("Khóa 66", "")].bboxes[2:] == keyed_rows[
        ("Khóa 65", "")
    ].bboxes[2:]
    assert all(not keyed_rows[(key, "")].cells[1] for key in ("Khóa 66", "Khóa 65"))
    assert "Từ ngày 22/10/2026 đến hết ngày 15/11/2026" in pages[0].text
    assert "56,925 đồng/tháng/SV" in pages[0].text
    assert "không nộp phí BHYT tại trường" in pages[1].text
    assert "hoàn thành trước ngày 15/11/2026" in pages[1].text


def test_both_chunk_strategies_keep_all_bhyt_table_rows_and_cell_provenance(
    monkeypatch,
):
    from src.chunking.fixed_size import chunk_pages as fixed_chunks
    from src.chunking.recursive import chunk_pages as recursive_chunks
    from src.indexing.bm25_index import BM25Index
    from src.indexing.index_manager import IndexBundle
    from src.models import ChunkRecord
    from src.rag.context_builder import build_context
    from src.retrieval.hybrid_retriever import HybridRetriever

    source_dir = Path(__file__).parents[1] / "data" / "raw"
    source = source_dir / "bao_hiem_y_te.pdf"
    if not source.is_file():
        pytest.skip("Project source PDF is unavailable")

    _, _, pages = read_pdf(source, source_dir)
    questions = [
        "Mức đóng bảo hiểm y tế là bao nhiêu tiền cho mỗi khóa?",
        "Khóa 64 hệ kỹ sư đóng bao nhiêu tiền?",
        "Khóa 64 hệ cử nhân đóng bao nhiêu tiền?",
        "Khóa 63 đóng bao nhiêu tiền?",
        "Khóa 65 và Khóa 66 có phân biệt hệ đào tạo trong bảng không?",
        "Những khóa nào có nhiều mức đóng khác nhau?",
        "Vì sao Khóa 64 có hai mức đóng khác nhau?",
    ]

    for chunker in (fixed_chunks, recursive_chunks):
        chunks = chunker(pages)
        table_chunks = [chunk for chunk in chunks if chunk.metadata.get("table_id")]
        assert len(table_chunks) == 5
        assert len({chunk.metadata["table_id"] for chunk in table_chunks}) == 1
        fee_table_id = table_chunks[0].metadata["table_id"]
        chunks.append(
            ChunkRecord(
                chunk_id=f"unrelated-schedule-{table_chunks[0].strategy}",
                document_id="unrelated-schedule",
                text=(
                    "Sinh viên các khóa 63, 64, 65, 66 đăng ký học; "
                    "nộp số mức học phần theo hệ đào tạo."
                ),
                page=2,
                end_page=2,
                strategy=table_chunks[0].strategy,
                search_text=(
                    "Đối tượng: Sinh viên các khóa 63, 64, 65, 66; "
                    "Hệ đào tạo: hệ chính quy; nộp số mức học phần"
                ),
                metadata={
                    "table_id": "unrelated-schedule:p2:t1",
                    "table_row_index": 1,
                    "table_headers": ["Đối tượng", "Hệ đào tạo"],
                    "table_cells": [
                        "Sinh viên các khóa 63, 64, 65, 66",
                        "hệ chính quy",
                    ],
                    "table_bboxes": [None, None],
                },
            )
        )
        class StaticFaiss:
            def search(self, vector, top_k):
                return [
                    (index, float(len(chunks) - index))
                    for index in range(len(chunks))
                ]

        class StaticEmbedder:
            def __init__(self, model_name):
                pass

            def encode(self, texts):
                return [[0.0] for _ in texts]

        monkeypatch.setattr(
            "src.retrieval.hybrid_retriever.SentenceTransformerEmbedder",
            StaticEmbedder,
        )
        retriever = HybridRetriever(
            IndexBundle(
                strategy=table_chunks[0].strategy,
                chunks=chunks,
                bm25=BM25Index(chunks),
                faiss=StaticFaiss(),
                model_name="test-model",
            )
        )
        for question in questions:
            result = retriever.search(question, top_k=1)
            retrieved_table_rows = [
                item for item in result["results"]
                if item["chunk"].metadata.get("table_id")
            ]
            assert len(retrieved_table_rows) == 5
            assert {
                item["chunk"].metadata["table_id"]
                for item in retrieved_table_rows
            } == {fee_table_id}
            evidence = build_context(result["results"], query=question)
            assert len(evidence) == 5
            assert len({item["chunk_id"] for item in evidence}) == 5
            assert all(item["page"] == 1 for item in evidence)
        engineering_row = next(
            chunk for chunk in table_chunks
            if chunk.metadata["table_cells"][1] == "Hệ kỹ sư"
        )
        assert engineering_row.metadata["table_headers"][1] == (
            "Đối tượng > [nhãn cột chưa xác định]"
        )

        assert "683.100" in "\n".join(
            item["text"] for item in build_context(
                retriever.search(questions[5], top_k=1)["results"],
                query=questions[5],
            )
        )
        amount_summary_context = build_context(
            retriever.search(questions[6], top_k=1)["results"],
            query=questions[6],
        )
        amount_summary_text = "\n".join(
            item["text"] for item in amount_summary_context
        )
        assert "Hệ kỹ sư" in amount_summary_text
        assert "Hệ cử nhân" in amount_summary_text

        per_row = {
            tuple(chunk.metadata["table_cells"]): chunk
            for chunk in table_chunks
        }
        no_system_row = next(
            chunk for chunk in table_chunks
            if chunk.metadata["table_cells"][0] == "Khóa 66"
        )
        no_system_evidence = build_context(
            [{"chunk": no_system_row}],
            query="Khóa 66 có phân biệt hệ đào tạo không?",
        )[0]
        assert "phạm vi của ô gộp" in no_system_evidence["text"]
        assert "Khóa 66" in no_system_evidence["text"]
        assert "683.100" in no_system_evidence["text"]
        assert "Hệ đào tạo" not in no_system_evidence["text"]
        assert len(per_row) == 5


def test_validator_accepts_text_pdf_and_rejects_duplicate_hash(tmp_path: Path):
    import pymupdf
    from src.ingestion.validator import validate_pdf

    original = tmp_path / "valid.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Văn bản có thể đọc được")
    document.save(original)
    document.close()
    duplicate = tmp_path / "duplicate.pdf"
    duplicate.write_bytes(original.read_bytes())
    seen = set()
    assert validate_pdf(original, tmp_path, seen) == (True, None)
    valid, reason = validate_pdf(duplicate, tmp_path, seen)
    assert not valid
    assert reason == "Tệp trùng SHA-256 với tài liệu đã xử lý"


def test_validator_rejects_pdf_without_extractable_text(tmp_path: Path):
    import pymupdf
    from src.ingestion.validator import validate_pdf

    path = tmp_path / "image-only.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(path)
    document.close()
    valid, reason = validate_pdf(path, tmp_path, set())
    assert not valid
    assert reason == "Không có trang văn bản hoặc OCR đạt chất lượng để lập chỉ mục"


def test_validator_accepts_structured_table_without_page_text(tmp_path, monkeypatch):
    import pymupdf
    import src.ingestion.validator as validator
    from src.models import PageRecord, TableRowRecord

    path = tmp_path / "table-only.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(path)
    document.close()
    page = PageRecord(
        document_id="doc-table-only",
        page_no=1,
        text="",
        table_rows=[
            TableRowRecord(
                table_id="doc-table-only:p1:t1",
                row_index=1,
                headers=["Đối tượng", "Mức"],
                cells=["Nhóm A", "100"],
                bboxes=[None, None],
            )
        ],
    )
    monkeypatch.setattr(
        validator,
        "read_pdf",
        lambda source, root: ("doc-table-only", "a" * 64, [page]),
    )

    assert validator.validate_pdf(path, tmp_path, set()) == (True, None)


def test_empty_or_malformed_pdf_is_rejected(tmp_path: Path):
    from src.ingestion.validator import validate_pdf

    bad = tmp_path / "bad.pdf"
    bad.write_text("not a pdf", encoding="utf-8")
    valid, reason = validate_pdf(bad, tmp_path, set())
    assert not valid
    assert reason == "Thiếu chữ ký PDF hợp lệ"


def test_personal_data_scan_flags_identifiers_and_preserves_source_contacts():
    from src.ingestion.validator import personal_data_warning

    assert personal_data_warning("Mã số: 012345678901")
    assert personal_data_warning("Không có dữ liệu nhạy cảm.") is None
    page = PageRecord(
        document_id="doc-a",
        page_no=1,
        text="Vui lòng liên hệ: 090 123 4567 (Cô A) hoặc person@example.edu.vn",
    )
    _, cleaned = clean_pages([page])
    assert "090 123 4567" in cleaned[0].text
    assert "person@example.edu.vn" in cleaned[0].text
    assert "Vui lòng liên hệ" in cleaned[0].text


def test_metadata_does_not_mistake_cited_laws_for_document_number(tmp_path: Path):
    page = PageRecord(
        document_id="doc-a",
        page_no=1,
        text=(
            "BỘ GIÁO DỤC VÀ ĐÀO TẠO\nTRƯỜNG ĐẠI HỌC GTVT\n"
            "Số: /TB-ĐHGTVT\nTHÔNG BÁO\nVề việc tổ chức học kỳ\n"
            "Căn cứ Luật số 25/2008/QH12 ngày 14/11/2008"
        ),
    )
    metadata = extract_metadata(
        tmp_path / "notice.pdf", tmp_path, "doc-a", "a" * 64, [page]
    )
    assert metadata.number_symbol is None
    assert metadata.document_type == "Thông báo"
    assert metadata.title_subject == "Về việc tổ chức học kỳ"
    assert metadata.issuing_authority == "Trường Đại học GTVT"


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("123/QĐ-ĐHGTVT", "123/QĐ-ĐHGTVT"),
        ("123/QĐ-ĐHGTVT-HN", "123/QĐ-ĐHGTVT-HN"),
        ("45/TB-ĐHGTVT", "45/TB-ĐHGTVT"),
        ("12/KH-ĐHGTVT", "12/KH-ĐHGTVT"),
        ("123/2026/QĐ-ĐHGTVT", "123/2026/QĐ-ĐHGTVT"),
        ("123 / 2026 / QĐ - ĐHGTVT - HN", "123/2026/QĐ-ĐHGTVT-HN"),
    ],
)
def test_extracts_document_number_formats_and_spacing(symbol, expected):
    metadata = _extract_metadata(f"TRƯỜNG ĐẠI HỌC GTVT\nSố: {symbol}\nQUYẾT ĐỊNH")
    assert metadata.number_symbol == expected


def test_extracts_document_number_split_across_lines():
    metadata = _extract_metadata("TRƯỜNG ĐẠI HỌC GTVT\nSố:\n 123 / QĐ-\n ĐHGTVT")
    assert metadata.number_symbol == "123/QĐ-ĐHGTVT"
    assert metadata.number_symbol_raw == "123 / QĐ-\nĐHGTVT"


def test_incomplete_document_number_is_kept_as_raw_without_inference():
    metadata = _extract_metadata("TRƯỜNG ĐẠI HỌC GTVT\nSố:\n /QĐ-ĐHGTVT")
    assert metadata.number_symbol is None
    assert metadata.number_symbol_raw == "/QĐ-ĐHGTVT"


@pytest.mark.parametrize(
    ("date_text", "expected"),
    [
        ("Ngày 15 tháng 03 năm 2026", "2026-03-15"),
        ("15/03/2026", "2026-03-15"),
        ("15-03-2026", "2026-03-15"),
        ("Hà Nội, ngày 15 tháng 03 năm 2026", "2026-03-15"),
        ("Ngày 15/03/2026", "2026-03-15"),
        ("15 tháng 3 năm 2026", "2026-03-15"),
        ("Hà Nội, ngày\n15 tháng\n03 năm 2026", "2026-03-15"),
    ],
)
def test_extracts_supported_issue_date_formats(date_text, expected):
    metadata = _extract_metadata(f"TRƯỜNG ĐẠI HỌC GTVT\n{date_text}\nQUYẾT ĐỊNH")
    assert metadata.issue_date == expected


def test_issue_date_uses_issue_context_not_first_date_or_effective_date():
    metadata = _extract_metadata(
        "QUYẾT ĐỊNH\nCó hiệu lực từ ngày 01/02/2026\n"
        "Căn cứ văn bản ngày 03/02/2026\nHà Nội, ngày 15/03/2026"
    )
    assert metadata.issue_date == "2026-03-15"


def test_issue_date_does_not_use_schedule_dates():
    metadata = _extract_metadata(
        "Lịch học từ ngày 21/09 đến ngày 22/11/2026, "
        "thi từ ngày 23/11 đến ngày 29/11/2026"
    )
    assert metadata.issue_date is None


def test_issue_date_is_not_inferred_when_incomplete_or_invalid():
    partial = _extract_metadata("Hà Nội, ngày\n tháng năm 2026")
    invalid = _extract_metadata("Hà Nội, ngày 31 tháng 02 năm 2026")
    assert partial.issue_date is None
    assert partial.issue_date_raw == "Hà Nội, ngày tháng năm 2026"
    assert partial.issue_year == 2026
    assert invalid.issue_date is None
    assert invalid.issue_year == 2026


def test_metadata_search_includes_later_pages():
    metadata = _extract_metadata(
        "QUYẾT ĐỊNH\n" + "\n".join(f"Nội dung dòng {n}" for n in range(35)),
        "Số: 123/QĐ-ĐHGTVT\nHà Nội, ngày 15/03/2026",
    )
    assert metadata.number_symbol == "123/QĐ-ĐHGTVT"
    assert metadata.issue_date == "2026-03-15"
    assert metadata.number_symbol_page == 2
    assert metadata.issue_date_page == 2


def test_complete_native_header_skips_ocr(tmp_path: Path, monkeypatch):
    from src.ingestion import metadata_extractor

    def unexpected_ocr(*_args, **_kwargs):
        assert _args or _kwargs
        pytest.fail("OCR should not run when header metadata is complete")

    monkeypatch.setattr(metadata_extractor, "read_pdf_header_ocr", unexpected_ocr)
    page = PageRecord(
        document_id="doc-test",
        page_no=1,
        text=(
            "Số: 123/QĐ-ĐHGTVT\n"
            "Hà Nội, ngày 15 tháng 03 năm 2026"
        ),
    )
    metadata = metadata_extractor.extract_pdf_metadata(
        tmp_path / "source.pdf",
        tmp_path,
        "doc-test",
        "a" * 64,
        [page],
    )
    assert metadata.number_symbol_source == "text_layer"
    assert metadata.issue_date_source == "text_layer"


def test_pdf_header_ocr_discards_low_confidence_lines(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    import pymupdf
    from src.ingestion import pdf_loader

    path = tmp_path / "header.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(path)
    document.close()

    class MixedConfidenceOCR:
        def __call__(self, _image):
            assert _image is not None
            return SimpleNamespace(
                txts=(
                    "S: 123/TB-ĐHGTVT",
                    "Hà Nội, ngày 15 tháng 03 năm 2026",
                ),
                scores=(0.95, 0.89),
            )

    monkeypatch.setattr(pdf_loader, "_ocr_engine", lambda: MixedConfidenceOCR())
    page = pdf_loader.read_pdf_header_ocr(path, "doc-test")
    native_page = PageRecord(document_id="doc-test", page_no=1, text="")
    metadata = extract_metadata(
        path,
        tmp_path,
        "doc-test",
        "a" * 64,
        [native_page],
        ocr_pages=[page],
    )
    assert metadata.number_symbol == "123/TB-ĐHGTVT"
    assert metadata.issue_date is None
    assert metadata.issue_date_raw == "Hà Nội, ngày 15 tháng 03 năm 2026"


def test_metadata_can_use_repeated_header_in_an_appendix():
    metadata = _extract_metadata(
        "THÔNG BÁO\nSố: /TB-ĐHGTVT",
        "Phụ lục\nSố: 574/TB-ĐHGTVT\nHà Nội, ngày 24 tháng 7 năm 2026",
    )
    assert metadata.number_symbol == "574/TB-ĐHGTVT"
    assert metadata.issue_date == "2026-07-24"
    assert metadata.number_symbol_page == 2
    assert metadata.issue_date_page == 2


def test_same_hash_existing_metadata_is_not_replaced_by_nulls():
    previous = DocumentMetadata(
        document_id="doc-test",
        source_path="test.pdf",
        accessed_at="2026-01-01T00:00:00Z",
        file_hash="a" * 64,
        page_count=1,
        number_symbol="123/QĐ-ĐHGTVT",
        issue_date="2026-03-15",
    )
    current = DocumentMetadata(
        document_id="doc-test",
        source_path="test.pdf",
        accessed_at="2026-02-01T00:00:00Z",
        file_hash="a" * 64,
        page_count=1,
    )
    from src.ingestion.ingestion_pipeline import _preserve_existing_metadata

    merged = _preserve_existing_metadata(current, {"doc-test": previous})
    assert merged.number_symbol == previous.number_symbol
    assert merged.issue_date == previous.issue_date
    changed_source = current.model_copy(update={"file_hash": "b" * 64})
    unchanged = _preserve_existing_metadata(changed_source, {"doc-test": previous})
    assert unchanged.number_symbol is None
    assert unchanged.issue_date is None


def test_low_trust_ocr_does_not_replace_previously_native_metadata():
    from src.ingestion.ingestion_pipeline import _preserve_existing_metadata

    previous = DocumentMetadata(
        document_id="doc-test",
        source_path="test.pdf",
        accessed_at="2026-01-01T00:00:00Z",
        file_hash="a" * 64,
        page_count=1,
        number_symbol="123/QĐ-ĐHGTVT",
        number_symbol_source="text_layer",
        number_symbol_page=1,
        issue_date="2026-03-15",
        issue_date_source="text_layer",
        issue_date_page=1,
    )
    current = previous.model_copy(
        update={
            "number_symbol": "128/QĐ-ĐHGTVT",
            "number_symbol_raw": "128/QĐ-ĐHGTVT",
            "number_symbol_source": "ocr",
            "issue_date": "2026-03-18",
            "issue_date_source": "ocr",
            "metadata_verification_status": "needs_manual_review",
        }
    )
    merged = _preserve_existing_metadata(current, {"doc-test": previous})
    assert merged.number_symbol == previous.number_symbol
    assert merged.issue_date == previous.issue_date
    assert merged.number_symbol_source == "text_layer"
    assert merged.issue_date_source == "text_layer"

    legacy_previous = previous.model_copy(
        update={
            "number_symbol_source": None,
            "number_symbol_page": None,
            "issue_date_source": None,
            "issue_date_page": None,
        }
    )
    legacy_merged = _preserve_existing_metadata(
        current, {"doc-test": legacy_previous}
    )
    assert legacy_merged.number_symbol == legacy_previous.number_symbol
    assert legacy_merged.issue_date == legacy_previous.issue_date


def test_legacy_metadata_without_provenance_fields_remains_loadable():
    legacy = DocumentMetadata.model_validate(
        {
            "document_id": "doc-old",
            "source_path": "old.pdf",
            "accessed_at": "2026-01-01T00:00:00Z",
            "file_hash": "a" * 64,
            "page_count": 1,
        }
    )
    assert legacy.number_symbol_source is None
    assert legacy.issue_date_page is None
    assert legacy.metadata_verification_status == "unverified"


def test_legacy_document_status_is_discarded_but_verification_status_is_kept():
    legacy = DocumentMetadata.model_validate(
        {
            "document_id": "doc-old",
            "source_path": "old.pdf",
            "accessed_at": "2026-01-01T00:00:00Z",
            "file_hash": "a" * 64,
            "page_count": 1,
            "status": "chua_xac_minh",
            "metadata_verification_status": "needs_manual_review",
        }
    )

    serialized = legacy.model_dump(mode="json")

    assert "status" not in serialized
    assert serialized["metadata_verification_status"] == "needs_manual_review"


def test_project_pdf_header_order_recovers_native_metadata():
    root = Path(__file__).parents[1] / "data" / "raw"
    path = root / "bao_hiem_y_te.pdf"
    if not path.exists():
        pytest.skip("Project source PDF is unavailable")
    document_id, file_hash, pages = read_pdf(path, root)
    raw_text = "\n".join(page.text for page in pages)
    assert "Hà Nội, ngày" in raw_text
    metadata = extract_metadata(path, root, document_id, file_hash, pages)
    assert metadata.number_symbol == "904/KH-ĐHGTVT"
    assert metadata.issue_date == "2026-10-21"
    assert metadata.number_symbol_source == "text_layer"
    assert metadata.issue_date_source == "text_layer"
    assert metadata.issue_date_page == 1


@pytest.mark.parametrize(
    ("filename", "expected_number", "expected_date", "expected_source"),
    [
        ("bao_hiem_y_te.pdf", "904/KH-ĐHGTVT", "2026-10-21", "text_layer"),
        (
            "dang_ky_bs_15_va_gia_han_hoc_phi.pdf",
            "939/TB-ĐHGTVT",
            "2026-10-20",
            "ocr",
        ),
        (
            "danh_gia_giang_vien.pdf",
            "516/TB-ĐHGTVT",
            "2026-07-01",
            "text_layer",
        ),
        (
            "ke_hoach_hk1_2026.pdf",
            "574/TB-ĐHGTVT",
            "2026-07-24",
            "text_layer",
        ),
    ],
)
def test_project_pdfs_extract_header_metadata_with_provenance(
    filename, expected_number, expected_date, expected_source
):
    root = Path(__file__).parents[1] / "data" / "raw"
    path = root / filename
    if not path.exists():
        pytest.skip("Project source PDF is unavailable")
    doc_id, file_hash, pages = read_pdf(path, root)
    metadata = extract_pdf_metadata(path, root, doc_id, file_hash, pages)
    assert metadata.number_symbol == expected_number
    assert metadata.issue_date == expected_date
    assert metadata.number_symbol_source == expected_source
    assert metadata.issue_date_source == expected_source
    assert metadata.number_symbol_page == 1
    assert metadata.issue_date_page == 1
    assert metadata.metadata_verification_status == (
        "needs_manual_review" if expected_source == "ocr" else "unverified"
    )

    if filename == "danh_gia_giang_vien.pdf":
        assert metadata.number_symbol != "406/TB-ĐHGTVT"
        assert metadata.issue_date != "2026-05-28"
    if filename == "dang_ky_bs_15_va_gia_han_hoc_phi.pdf":
        assert metadata.number_symbol != "238/2025/ND-CP"
        assert metadata.issue_date != "2026-09-24"


def test_local_web_loader_keeps_source_url(tmp_path: Path):
    from src.ingestion.web_loader import load_web_file

    web_root = tmp_path / "web"
    web_root.mkdir()
    page_path = web_root / "notice.html"
    page_path.write_text("<h1>Thông báo</h1><p>Nội dung</p><script>hidden()</script>", encoding="utf-8")
    page_path.with_suffix(".html.url").write_text("https://school.example/notice", encoding="utf-8")
    document_id, file_hash, page, source_url = load_web_file(page_path, tmp_path)
    assert document_id.startswith("doc_")
    assert len(file_hash) == 64
    assert "hidden" not in page.text
    assert source_url == "https://school.example/notice"
