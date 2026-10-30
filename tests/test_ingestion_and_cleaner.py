from pathlib import Path

from src.ingestion.cleaner import clean_pages
from src.ingestion.pdf_loader import document_id_for, read_pdf
from src.ingestion.metadata_extractor import extract_metadata
from src.models import PageRecord


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
    assert reason == "Không trích xuất được văn bản; cần OCR trước khi index"


def test_empty_or_malformed_pdf_is_rejected(tmp_path: Path):
    from src.ingestion.validator import validate_pdf

    bad = tmp_path / "bad.pdf"
    bad.write_text("not a pdf", encoding="utf-8")
    valid, reason = validate_pdf(bad, tmp_path, set())
    assert not valid
    assert reason == "Thiếu chữ ký PDF hợp lệ"


def test_personal_data_scan_flags_identifiers_and_redacts_contact_lines():
    from src.ingestion.validator import personal_data_warning

    assert personal_data_warning("Mã số: 012345678901")
    assert personal_data_warning("Không có dữ liệu nhạy cảm.") is None
    page = PageRecord(
        document_id="doc-a",
        page_no=1,
        text="Vui lòng liên hệ: 090 123 4567 (Cô A) hoặc person@example.edu.vn",
    )
    _, cleaned = clean_pages([page])
    assert "090 123 4567" not in cleaned[0].text
    assert "person@example.edu.vn" not in cleaned[0].text
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
