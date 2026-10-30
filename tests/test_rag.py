from src.rag.citation_validator import validate_citations
from src.rag.context_builder import build_context
from src.models import ChunkRecord


def test_context_and_citation_registry_validation():
    chunk = ChunkRecord(
        chunk_id="c1",
        document_id="doc1",
        text="Nội dung có căn cứ.",
        page=2,
        end_page=2,
        section="Điều 1",
        strategy="fixed_size",
        metadata={
            "title_subject": "Thông báo",
            "number_symbol": "01/TB",
            "source_path": "notice.pdf",
            "page_count": 3,
        },
    )
    evidence = build_context([{"chunk": chunk}], token_budget=100)
    assert evidence[0]["evidence_id"] == "E1"
    assert evidence[0]["citation_id"] == "E1"
    assert evidence[0]["page"] == 2
    assert evidence[0]["chunk_id"] == "c1"
    assert evidence[0]["evidence_text"] == "Nội dung có căn cứ."
    assert evidence[0]["bbox"] is None
    checked = validate_citations("Thông tin [E1]. Bịa [E9].", evidence)
    assert checked["sources"][0]["document_id"] == "doc1"
    assert checked["sources"][0]["citation_id"] == "E1"
    assert checked["sources"][0]["chunk_id"] == "c1"
    assert checked["invalid_citations"] == ["E9"]
    assert "[E9]" not in checked["answer"]
    assert "[E1]" in checked["answer"]


def test_citation_without_marker_does_not_invent_a_citation():
    checked = validate_citations("Không tìm thấy thông tin phù hợp.", [])
    assert checked["sources"] == []
    assert checked["answer"] == "Không tìm thấy thông tin phù hợp."


def test_citation_without_source_or_with_invalid_page_is_removed():
    citation = {
        "evidence_id": "E1",
        "document_id": "doc1",
        "chunk_id": "c1",
        "page": 4,
        "page_count": 3,
        "evidence_text": "Evidence",
        "source_path": "notice.pdf",
    }
    checked = validate_citations("Dữ liệu [E1].", [citation])
    assert checked["invalid_citations"] == ["E1"]
    assert checked["sources"] == []
    assert "[E1]" not in checked["answer"]


def test_multiple_citations_keep_their_own_document_and_page_mapping():
    first = ChunkRecord(
        chunk_id="first",
        document_id="doc-a",
        text="First source evidence.",
        page=2,
        end_page=2,
        strategy="fixed_size",
        metadata={"source_path": "first.pdf", "page_count": 4},
    )
    second = ChunkRecord(
        chunk_id="second",
        document_id="doc-b",
        text="Second source evidence.",
        page=3,
        end_page=3,
        strategy="recursive",
        metadata={"source_path": "second.pdf", "page_count": 5},
    )
    evidence = build_context([{"chunk": first}, {"chunk": second}])
    checked = validate_citations("Một ý [E1], ý khác [E2].", evidence)
    assert [
        (source["citation_id"], source["document_id"], source["page"])
        for source in checked["sources"]
    ] == [("E1", "doc-a", 2), ("E2", "doc-b", 3)]


def test_context_respects_small_token_budget():
    chunk = ChunkRecord(
        chunk_id="c1",
        document_id="doc1",
        text="a" * 1000,
        page=1,
        end_page=1,
        strategy="recursive",
        metadata={},
    )
    evidence = build_context([{"chunk": chunk}], token_budget=100)
    assert evidence
    assert evidence[0]["estimated_tokens"] <= 100


def test_context_preserves_ingested_bbox_for_citation_mapping():
    bbox = [72.0, 100.0, 420.0, 140.0]
    chunk = ChunkRecord(
        chunk_id="c1",
        document_id="doc1",
        text="Evidence with recorded geometry.",
        page=1,
        end_page=1,
        strategy="recursive",
        metadata={"source_path": "source.pdf", "bbox": bbox},
    )

    evidence = build_context([{"chunk": chunk}])

    assert evidence[0]["bbox"] == bbox
