from src.chunking.fixed_size import chunk_pages as fixed_chunks
from src.chunking.recursive import chunk_pages as recursive_chunks
from src.models import PageRecord


def test_fixed_size_overlap_and_page_mapping():
    page = PageRecord(document_id="doc-a", page_no=4, text="abcdefghij")
    chunks = fixed_chunks([page], chunk_size=6, overlap=2)
    assert [chunk.text for chunk in chunks] == ["abcdef", "efghij"]
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
    assert all(len(chunk.text) <= 35 for chunk in chunks)
    assert any(chunk.section and chunk.section.startswith("Điều 1") for chunk in chunks)


def test_recursive_preserves_numbered_section_metadata():
    page = PageRecord(
        document_id="doc-a",
        page_no=1,
        text="1.2. Thời hạn nộp học phí\n\nNội dung của mục này nói về hạn nộp.",
    )
    chunks = recursive_chunks([page], chunk_size=100, overlap=10)
    assert chunks[0].section == "1.2. Thời hạn nộp học phí"
