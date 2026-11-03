from src.chunking.fixed_size import chunk_pages as fixed_chunks
from src.chunking.recursive import chunk_pages as recursive_chunks
from src.models import PageRecord, TableRowRecord


def test_fixed_size_overlap_and_page_mapping():
    page = PageRecord(
        document_id="doc-a", page_no=4, text="alpha beta gamma delta"
    )
    chunks = fixed_chunks([page], chunk_size=11, overlap=5)
    assert [chunk.text for chunk in chunks] == ["alpha beta gamma", "gamma delta"]
    assert all(chunk.page == 4 and chunk.end_page == 4 for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_recursive_keeps_section_and_respects_size():
    page = PageRecord(
        document_id="doc-a",
        page_no=2,
        text="Điều 1. Phạm vi\n\nĐoạn thứ nhất có nội dung.\n\nĐiều 2. Điều kiện\n\nĐoạn thứ hai có nội dung.",
    )
    chunks = recursive_chunks([page], chunk_size=35, overlap=5)
    assert chunks
    assert all(len(chunk.text) <= 35 + max(map(len, page.text.split())) for chunk in chunks)
    assert {
        word for chunk in chunks for word in chunk.text.split()
    } == set(page.text.split())
    assert any(chunk.section and chunk.section.startswith("Điều 1") for chunk in chunks)


def test_recursive_preserves_numbered_section_metadata():
    page = PageRecord(
        document_id="doc-a",
        page_no=1,
        text="1.2. Thời hạn nộp học phí\n\nNội dung của mục này nói về hạn nộp.",
    )
    chunks = recursive_chunks([page], chunk_size=100, overlap=10)
    assert chunks[0].section == "1.2. Thời hạn nộp học phí"


def test_roman_section_metadata_is_not_overwritten_by_numbered_items():
    page = PageRecord(
        document_id="doc-bhyt",
        page_no=1,
        text=(
            "III/ Thời gian thực hiện và mức nộp BHYT.\n"
            "1. Thời gian: Từ ngày 22/10/2026 đến hết ngày 15/11/2026\n"
            "2. Mức nộp: 56,925 đồng/tháng/SV."
        ),
    )

    for chunks in (
        fixed_chunks([page], chunk_size=300, overlap=20),
        recursive_chunks([page], chunk_size=300, overlap=20),
    ):
        assert chunks[0].section == "III/ Thời gian thực hiện và mức nộp BHYT."


def test_both_strategies_keep_table_rows_verbatim_and_index_column_labels():
    page = PageRecord(
        document_id="doc-table",
        page_no=7,
        text="Some surrounding page text.",
        table_rows=[
            TableRowRecord(
                table_id="doc-table:p7:t1",
                row_index=1,
                headers=["Đợt", "Đăng ký"],
                cells=["Đợt học 26", "Từ 10/8 đến 30/8/2026"],
                bboxes=[(10, 10, 20, 20), (20, 10, 30, 20)],
            )
        ],
    )

    for chunks in (
        fixed_chunks([page], chunk_size=100, overlap=10),
        recursive_chunks([page], chunk_size=100, overlap=10),
    ):
        row = next(chunk for chunk in chunks if chunk.metadata.get("table_id"))
        assert row.text == "Đợt học 26\nTừ 10/8 đến 30/8/2026"
        assert row.search_text == (
            "Đợt: Đợt học 26\nĐăng ký: Từ 10/8 đến 30/8/2026"
        )
        assert row.metadata["table_bboxes"] == [
            (10.0, 10.0, 20.0, 20.0),
            (20.0, 10.0, 30.0, 20.0),
        ]
