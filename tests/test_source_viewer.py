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
    output, bbox = build_pdf_view(source, 2, evidence)

    opened = pymupdf.open(stream=output, filetype="pdf")
    assert opened.page_count == 2
    catalog = opened.pdf_catalog()
    open_action_type, open_action = opened.xref_get_key(catalog, "OpenAction")
    assert open_action_type == "array"
    assert f"{opened[1].xref} 0 R" in open_action
    assert bbox is not None
    assert len(list(opened[1].annots())) == 1
    opened.close()


def test_pdf_view_does_not_highlight_unmatched_evidence(tmp_path: Path):
    source = tmp_path / "source.pdf"
    _make_pdf(source)
    output, bbox = build_pdf_view(source, 2, "Đoạn văn bản không có trong tài liệu này.")

    opened = pymupdf.open(stream=output, filetype="pdf")
    assert bbox is None
    assert list(opened[1].annots()) == []
    assert opened.page_count == 2
    opened.close()


def test_pdf_view_uses_only_valid_supplied_bbox_when_text_is_not_found(tmp_path: Path):
    source = tmp_path / "source.pdf"
    _make_pdf(source)
    with pymupdf.open(source) as document:
        rect = document[1].search_for("Student pays tuition")[0]
        bbox = list(rect)

    output, highlighted_bbox = build_pdf_view(
        source, 2, "Evidence without a text match.", bbox
    )

    opened = pymupdf.open(stream=output, filetype="pdf")
    assert highlighted_bbox == [round(value, 2) for value in rect]
    assert len(list(opened[1].annots())) == 1
    opened.close()


@pytest.mark.parametrize(
    "bbox",
    [
        [0, 0, 5000, 5000],
        [0, 0, 100, float("nan")],
        [0, 0, 30],
        [0, 0, True, 30],
    ],
)
def test_pdf_view_ignores_invalid_bbox(tmp_path: Path, bbox):
    source = tmp_path / "source.pdf"
    _make_pdf(source)

    output, highlighted_bbox = build_pdf_view(
        source, 2, "Evidence without a text match.", bbox
    )

    opened = pymupdf.open(stream=output, filetype="pdf")
    assert highlighted_bbox is None
    assert list(opened[1].annots()) == []
    opened.close()


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
