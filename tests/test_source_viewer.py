from pathlib import Path

import pymupdf
import pytest

from src.rag.source_viewer import build_pdf_view, resolve_pdf_path


def _make_pdf(path: Path) -> str:
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Trang đầu không chứa bằng chứng.")
    document.new_page().insert_text(
        (72, 72), "Student pays tuition before 01/11/2026 according to notice."
    )
    document.save(path)
    document.close()
    return "Student pays tuition before 01/11/2026 according to notice."


def test_pdf_view_opens_on_correct_page_and_highlights_exact_text(tmp_path: Path):
    source = tmp_path / "source.pdf"
    evidence = _make_pdf(source)
    output, bbox, status = build_pdf_view(source, 2, evidence)

    opened = pymupdf.open(stream=output, filetype="pdf")
    assert opened.page_count == 2
    catalog = opened.pdf_catalog()
    open_action_type, open_action = opened.xref_get_key(catalog, "OpenAction")
    assert open_action_type == "array"
    assert f"{opened[1].xref} 0 R" in open_action
    assert bbox is not None
    assert status == "matched"
    assert len(list(opened[1].annots())) == 1
    opened.close()


def test_pdf_view_does_not_highlight_unmatched_evidence(tmp_path: Path):
    source = tmp_path / "source.pdf"
    _make_pdf(source)
    output, bbox, status = build_pdf_view(source, 2, "Đoạn văn bản không có trong tài liệu này.")

    opened = pymupdf.open(stream=output, filetype="pdf")
    assert bbox is None
    assert status == "not_found"
    assert list(opened[1].annots()) == []
    assert opened.page_count == 2
    opened.close()


def test_pdf_view_does_not_highlight_when_evidence_text_is_not_found(tmp_path: Path):
    source = tmp_path / "source.pdf"
    _make_pdf(source)
    output, highlighted_bbox, status = build_pdf_view(source, 2, "Evidence without a text match.")

    opened = pymupdf.open(stream=output, filetype="pdf")
    assert highlighted_bbox is None
    assert status == "not_found"
    assert list(opened[1].annots()) == []
    opened.close()


def test_pdf_view_does_not_highlight_partial_words(tmp_path: Path):
    source = tmp_path / "source.pdf"
    _make_pdf(source)
    output, highlighted_bbox, status = build_pdf_view(source, 2, "uition pays tuition")

    opened = pymupdf.open(stream=output, filetype="pdf")
    assert highlighted_bbox is None
    assert status == "not_found"
    assert list(opened[1].annots()) == []
    opened.close()


def test_pdf_view_does_not_highlight_ambiguous_repeated_evidence(tmp_path: Path):
    source = tmp_path / "duplicate.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Repeated source evidence.")
    page.insert_text((72, 120), "Repeated source evidence.")
    document.save(source)
    document.close()

    output, highlighted_bbox, status = build_pdf_view(
        source, 1, "Repeated source evidence."
    )

    opened = pymupdf.open(stream=output, filetype="pdf")
    assert highlighted_bbox is None
    assert status == "not_found"
    assert list(opened[0].annots()) == []
    opened.close()


def test_pdf_view_maps_whitespace_variants_across_lines(tmp_path: Path):
    source = tmp_path / "multiline.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Student pays tuition")
    page.insert_text((72, 92), "before 01/11/2026 according to notice.")
    document.save(source)
    document.close()

    output, bbox, status = build_pdf_view(
        source,
        1,
        "Student pays tuition\n\t  before 01/11/2026 according to notice.",
    )

    opened = pymupdf.open(stream=output, filetype="pdf")
    assert bbox is not None
    assert status == "matched"
    assert len(list(opened[0].annots())) == 1
    opened.close()


def test_pdf_view_highlights_all_selected_evidence_segments(tmp_path: Path):
    source = tmp_path / "segments.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "First specific evidence.")
    page.insert_text((72, 100), "Second specific evidence.")
    document.save(source)
    document.close()

    output, bbox, status = build_pdf_view(
        source,
        1,
        "First specific evidence.\nSecond specific evidence.",
        evidence_segments=[
            "First specific evidence.",
            "Second specific evidence.",
        ],
    )

    opened = pymupdf.open(stream=output, filetype="pdf")
    annotations = list(opened[0].annots())
    assert status == "matched"
    assert bbox is not None
    assert len(annotations) == 1
    assert bbox[3] - bbox[1] > 20
    opened.close()


def test_pdf_view_does_not_highlight_partial_set_of_evidence_segments(tmp_path: Path):
    source = tmp_path / "partial-segments.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "First specific evidence.")
    document.save(source)
    document.close()

    output, bbox, status = build_pdf_view(
        source,
        1,
        "First specific evidence.\nMissing second evidence.",
        evidence_segments=[
            "First specific evidence.",
            "Missing second evidence.",
        ],
    )

    opened = pymupdf.open(stream=output, filetype="pdf")
    assert bbox is None
    assert status == "partial_match"
    assert list(opened[0].annots()) == []
    opened.close()


def test_project_pdf_vietnamese_evidence_is_highlighted():
    source = Path(__file__).parents[1] / "data" / "raw" / "bao_hiem_y_te.pdf"
    if not source.is_file():
        pytest.skip("Project source PDF is unavailable")

    output, bbox, status = build_pdf_view(
        source, 1, "Hà Nội, ngày 21 tháng 10 năm 2026"
    )

    opened = pymupdf.open(stream=output, filetype="pdf")
    assert bbox is not None
    assert status == "matched"
    assert len(list(opened[0].annots())) == 1
    opened.close()


def test_table_evidence_highlights_all_cells_from_the_real_project_pdf():
    from src.ingestion.pdf_loader import read_pdf

    raw_dir = Path(__file__).parents[1] / "data" / "raw"
    source = raw_dir / "ke_hoach_hk1_2026.pdf"
    if not source.is_file():
        pytest.skip("Project source PDF is unavailable")

    _, _, pages = read_pdf(source, raw_dir)
    page = pages[6]
    row = next(
        row for row in page.table_rows
        if any("Đợt học 26" in cell for cell in row.cells)
    )
    evidence = "\n".join(row.cells).strip()
    locations = [
        {"text": cell, "bbox": bbox}
        for cell, bbox in zip(row.cells, row.bboxes)
    ]

    output, bbox, status = build_pdf_view(
        source, page.page_no, evidence, locations
    )

    opened = pymupdf.open(stream=output, filetype="pdf")
    assert status == "matched"
    assert bbox is not None
    assert len(list(opened[page.page_no - 1].annots())) == 1
    opened.close()


def test_table_evidence_does_not_highlight_when_only_some_cells_match(tmp_path: Path):
    source = tmp_path / "table.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "A real cell")
    page.insert_text((72, 100), "Another real cell")
    first = page.search_for("A real cell")[0]
    second = page.search_for("Another real cell")[0]
    document.save(source)
    document.close()

    output, bbox, status = build_pdf_view(
        source,
        1,
        "A real cell\nAnother real cell",
        [
            {"text": "A real cell", "bbox": list(first)},
            {"text": "Wrong expected cell", "bbox": list(second)},
        ],
    )

    opened = pymupdf.open(stream=output, filetype="pdf")
    assert bbox is None
    assert status == "partial_match"
    assert list(opened[0].annots()) == []
    opened.close()


def test_all_table_rows_in_project_pdfs_highlight_their_full_mapped_evidence():
    from src.ingestion.pdf_loader import read_pdf

    raw_dir = Path(__file__).parents[1] / "data" / "raw"
    sources = sorted(raw_dir.glob("*.pdf"))
    if not sources:
        pytest.skip("Project source PDFs are unavailable")

    checked_rows = 0
    for source in sources:
        _, _, pages = read_pdf(source, raw_dir)
        for page in pages:
            for row in page.table_rows:
                locations = [
                    {"text": cell, "bbox": bbox}
                    for cell, bbox in zip(row.cells, row.bboxes)
                    if cell
                ]
                _, bbox, status = build_pdf_view(
                    source,
                    page.page_no,
                    "\n".join(row.cells).strip(),
                    locations,
                )
                assert status == "matched", (
                    source.name,
                    page.page_no,
                    row.row_index,
                    status,
                )
                assert bbox is not None
                checked_rows += 1

    assert checked_rows > 0


def test_pdf_view_validates_page_number(tmp_path: Path):
    source = tmp_path / "source.pdf"
    _make_pdf(source)
    with pytest.raises(ValueError, match="Trang citation không hợp lệ"):
        build_pdf_view(source, 3, "nội dung")


def test_source_path_must_stay_inside_configured_raw_directory():
    with pytest.raises(ValueError, match="ngoài thư mục"):
        resolve_pdf_path("..\\..\\Windows\\win.ini")


def test_missing_source_pdf_reports_file_not_found():
    with pytest.raises(FileNotFoundError, match="Không tìm thấy PDF nguồn"):
        resolve_pdf_path("missing-citation-source.pdf")
