import pytest

from src.rag.citation_validator import retrieval_only_response, validate_citations
from src.rag.context_builder import build_context
from src.rag.prompt import SYSTEM_PROMPT, user_prompt
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
    assert "[E9]" in checked["answer"]
    assert "[E1]" in checked["answer"]


def test_prompt_distinguishes_current_document_metadata_from_cited_document():
    chunk = ChunkRecord(
        chunk_id="notice",
        document_id="doc1",
        text=(
            "Số: 516/TB-ĐHGTVT. Thông báo này đã viện dẫn "
            "Thông báo 406/TB-ĐHGTVT ngày 28/5/2026."
        ),
        page=1,
        end_page=1,
        strategy="fixed_size",
        metadata={
            "title_subject": "Thông báo về hoạt động giảng dạy",
            "number_symbol": "516/TB-ĐHGTVT",
            "issue_date": "2026-07-01",
            "source_path": "notice.pdf",
            "page_count": 1,
        },
    )
    evidence = build_context([{"chunk": chunk}])
    prompt = user_prompt("Số hiệu và ngày ban hành là gì?", evidence)

    assert "Số: 516/TB-ĐHGTVT" in prompt
    assert "Ngày: 2026-07-01" in prompt
    assert "406/TB-ĐHGTVT" in prompt
    normalized_prompt = " ".join(SYSTEM_PROMPT.split())
    assert "số hiệu các văn bản được viện dẫn trong nội dung không thay thế" in normalized_prompt


@pytest.mark.parametrize(
    "citation",
    ["[E1]", "【E1】", "【E1†L1-L3】"],
)
def test_citation_spellings_normalize_only_against_real_evidence(citation):
    evidence = [
        {
            "evidence_id": "E1",
            "document_id": "doc1",
            "chunk_id": "chunk1",
            "page": 7,
            "page_count": 8,
            "evidence_text": "Bằng chứng gốc.",
            "source_path": "source.pdf",
        }
    ]

    checked = validate_citations(f"Nội dung {citation}.", evidence)

    assert checked["answer"] == "Nội dung [E1]."
    assert checked["sources"][0]["page"] == 7
    assert checked["sources"][0]["citation_id"] == "E1"


def test_unmapped_extended_citation_is_preserved_and_not_assigned_a_page():
    citation = "【E9†L1-L3】"

    checked = validate_citations(f"Nội dung {citation}.", [])

    assert checked["answer"] == f"Nội dung {citation}."
    assert checked["sources"] == []
    assert checked["invalid_citations"] == ["E9"]


def test_citation_without_marker_does_not_invent_a_citation():
    checked = validate_citations("Không tìm thấy thông tin phù hợp.", [])
    assert checked["sources"] == []
    assert checked["answer"] == "Không tìm thấy thông tin phù hợp."


def test_retrieval_only_response_keeps_sources_clickable_when_llm_is_unavailable():
    evidence = [
        {
            "evidence_id": "E1",
            "document_id": "doc1",
            "chunk_id": "c1",
            "page": 1,
            "page_count": 1,
            "evidence_text": "Nguyên văn của tài liệu.",
            "text": "Nguyên văn",
            "source_path": "notice.pdf",
        }
    ]

    response = retrieval_only_response(evidence)

    assert response["generated"] is False
    assert response["answer"] == "LLM chưa khả dụng. Bằng chứng đã truy xuất: [E1]"
    assert response["sources"][0]["evidence_text"] == "Nguyên văn của tài liệu."


def test_retrieval_only_response_without_evidence_does_not_add_citations():
    response = retrieval_only_response([])

    assert response["generated"] is False
    assert response["sources"] == []
    assert response["answer"] == "Chưa tìm thấy bằng chứng phù hợp trong chỉ mục hiện tại."


def test_citation_without_source_or_with_invalid_page_stays_unlinked():
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
    assert "[E1]" in checked["answer"]
    assert checked["answer"] == "Dữ liệu [E1]."


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
    assert evidence == []


def test_registration_schedule_keeps_all_table_groups_over_token_budget():
    chunks = [
        ChunkRecord(
            chunk_id=f"schedule-row-{index}",
            document_id="schedule",
            text=f"Đợt {index}\nThời gian đăng ký: {index} tháng 8",
            page=6 + index // 2,
            end_page=6 + index // 2,
            strategy="fixed_size",
            metadata={
                "table_id": f"schedule:p{6 + index // 2}:t1",
                "table_group_id": f"schedule:p{6 + index // 2}:t1",
                "table_headers": ["Đợt", "Thời gian đăng ký"],
                "table_cells": [f"Đợt {index}", f"{index} tháng 8"],
                "table_bboxes": [None, None],
            },
        )
        for index in range(4)
    ]
    results = [{"chunk": chunk} for chunk in chunks]

    evidence = build_context(
        results,
        token_budget=10,
        query="Lịch đăng ký học kỳ I gồm những đợt nào?",
    )

    assert [item["chunk_id"] for item in evidence] == [
        chunk.chunk_id for chunk in chunks
    ]
    assert {item["page"] for item in evidence} == {6, 7}


def test_table_context_keeps_verbatim_cells_and_page_geometry():
    chunk = ChunkRecord(
        chunk_id="table-row",
        document_id="doc1",
        text="5\nLớp học phần ngoại ngữ\nĐăng ký trực tiếp tại VPK 303-B3",
        page=7,
        end_page=7,
        strategy="recursive",
        search_text="TT: 5\nLớp học: Lớp học phần ngoại ngữ",
        metadata={
            "table_id": "doc1:p7:t1",
            "table_headers": ["TT", "Lớp học", "Thời gian đăng ký"],
            "table_cells": [
                "5",
                "Lớp học phần ngoại ngữ",
                "Đăng ký trực tiếp tại VPK 303-B3",
            ],
            "table_bboxes": [
                [10.0, 10.0, 20.0, 20.0],
                [20.0, 10.0, 40.0, 20.0],
                [40.0, 10.0, 60.0, 20.0],
            ],
            "page_count": 8,
        },
    )

    evidence = build_context(
        [{"chunk": chunk}],
        token_budget=100,
        query="Đợt học 26 đăng ký ở đâu và thời gian nào?",
    )

    assert evidence[0]["evidence_text"] == (
        "Lớp học phần ngoại ngữ\nĐăng ký trực tiếp tại VPK 303-B3"
    )
    assert evidence[0]["text"] == (
        "Lớp học: Lớp học phần ngoại ngữ\n"
        "Thời gian đăng ký: Đăng ký trực tiếp tại VPK 303-B3"
    )
    assert evidence[0]["table_locations"] == [
        {"text": cell, "bbox": bbox}
        for cell, bbox in zip(
            chunk.metadata["table_cells"][1:],
            chunk.metadata["table_bboxes"][1:],
        )
    ]


def test_context_selects_only_the_requested_table_fee_column():
    chunk = ChunkRecord(
        chunk_id="table-row",
        document_id="doc1",
        text="Đợt học 26\n31/8/2026\n07/9/2026",
        page=7,
        end_page=7,
        strategy="recursive",
        metadata={
            "table_id": "doc1:p7:t1",
            "table_headers": [
                "Lớp học",
                "Thời gian tính học phí",
                "Thời gian thu học phí",
            ],
            "table_cells": [
                "Đợt học 26",
                "31/8/2026 ÷ 04/9/2026",
                "07/9/2026 ÷ 18/10/2026",
            ],
            "table_bboxes": [
                [10.0, 10.0, 20.0, 20.0],
                [20.0, 10.0, 40.0, 20.0],
                [40.0, 10.0, 60.0, 20.0],
            ],
        },
    )

    evidence = build_context(
        [{"chunk": chunk}], query="Thời gian thu học phí đợt học 26?"
    )[0]

    assert evidence["table_headers"] == [
        "Lớp học",
        "Thời gian thu học phí",
    ]
    assert evidence["evidence_text"] == (
        "Đợt học 26\n07/9/2026 ÷ 18/10/2026"
    )


def test_context_keeps_key_fee_and_exact_term_together_for_each_key_query():
    chunk = ChunkRecord(
        chunk_id="bhyt-row",
        document_id="doc1",
        text=(
            "Khóa 64\nHệ kỹ sư\n683.100\n"
            "12 tháng (Từ 01/01/2027 đến\n31/12/2027)"
        ),
        page=1,
        end_page=1,
        strategy="recursive",
        metadata={
            "table_id": "doc1:p1:t1",
            "table_headers": ["Khóa", "Hệ đào tạo", "Mức đóng", "Thời hạn"],
            "table_cells": [
                "Khóa 64",
                "Hệ kỹ sư",
                "683.100",
                "12 tháng (Từ 01/01/2027 đến\n31/12/2027)",
            ],
            "table_bboxes": [None, None, None, None],
        },
    )

    evidence = build_context(
        [{"chunk": chunk}],
        query="Mỗi khóa và hệ đào tạo đóng mức nào, thời hạn bao lâu?",
    )[0]

    assert evidence["evidence_text"] == (
        "Khóa 64\nHệ kỹ sư\n683.100\n"
        "12 tháng (Từ 01/01/2027 đến\n31/12/2027)"
    )
    assert "đến\n31/12/2027" in evidence["text"]


@pytest.mark.parametrize(
    ("query", "cells", "expected_cells"),
    [
        (
            "Mức đóng BHYT của khóa 65 là bao nhiêu?",
            ["Khóa 65", "", "683.100", "12 tháng"],
            ["Khóa 65", "683.100"],
        ),
        (
            "Mức đóng BHYT của sinh viên khóa 64 hệ kỹ sư?",
            ["Khoá 64", "Hệ kỹ sư", "683.100", "12 tháng"],
            ["Khoá 64", "Hệ kỹ sư", "683.100"],
        ),
    ],
)
def test_table_context_keeps_the_requested_key_and_training_system(
    query, cells, expected_cells
):
    headers = [
        "Đối tượng",
        "Cột 2",
        "Số tiền đóng BHYT (VNĐ)",
        "Thời gian sử dụng thẻ",
    ]
    chunk = ChunkRecord(
        chunk_id="bhyt-row",
        document_id="doc1",
        text="\n".join(cells),
        page=1,
        end_page=1,
        strategy="recursive",
        metadata={
            "table_id": "doc1:p1:t1",
            "table_headers": headers,
            "table_cells": cells,
            "table_bboxes": [None, None, [1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 3.0, 6.0]],
        },
    )

    evidence = build_context([{"chunk": chunk}], query=query)[0]

    assert evidence["evidence_segments"] == expected_cells
    assert "683.100" in evidence["text"]
    if "Khóa 65" in cells:
        assert "Hệ đào tạo" not in evidence["text"]


def test_payment_deadline_context_keeps_section_and_exact_cutoff_wording():
    chunk = ChunkRecord(
        chunk_id="bhyt-section-iii",
        document_id="doc1",
        text=(
            "III/ Thời gian thực hiện và mức nộp BHYT.\n"
            "1. Thời gian: Từ ngày 22/10/2026 đến hết ngày 15/11/2026\n"
            "2. Mức nộp: 56,925 đồng/tháng/SV."
        ),
        page=1,
        end_page=1,
        section="2. Mức nộp: 56,925 đồng/tháng/SV.",
        strategy="fixed_size",
        metadata={"source_path": "bao_hiem_y_te.pdf", "page_count": 2},
    )
    card_duration = ChunkRecord(
        chunk_id="bhyt-card-duration",
        document_id="doc1",
        text="Từ 01/01/2027 đến 31/12/2027",
        page=1,
        end_page=1,
        strategy="fixed_size",
        metadata={
            "table_id": "doc1:p1:t1",
            "table_headers": ["Đối tượng", "Thời gian sử dụng thẻ"],
            "table_cells": ["Khóa 66", "12 tháng (Từ 01/01/2027 đến 31/12/2027)"],
            "table_bboxes": [None, None],
        },
    )

    evidence = build_context(
        [{"chunk": card_duration}, {"chunk": chunk}],
        query="Hạn đóng bảo hiểm y tế?",
    )

    assert len(evidence) == 1
    assert evidence[0]["chunk_id"] == "bhyt-section-iii"
    assert "III/ Thời gian thực hiện" in evidence[0]["text"]
    assert "Từ ngày 22/10/2026 đến hết ngày 15/11/2026" in evidence[0]["text"]
    assert "12 tháng" not in evidence[0]["text"]
    assert "56,925 đồng/tháng/SV" in evidence[0]["text"]


def test_policy_card_exception_keeps_its_distinct_before_date_wording():
    chunk = ChunkRecord(
        chunk_id="bhyt-policy-note",
        document_id="doc1",
        text=(
            "Lưu ý: Sinh viên có thẻ BHYT chính sách không nộp phí BHYT tại trường, "
            "thực hiện kê khai và hoàn thành trước ngày 15/11/2026."
        ),
        page=2,
        end_page=2,
        strategy="recursive",
        metadata={"source_path": "bao_hiem_y_te.pdf", "page_count": 2},
    )

    evidence = build_context(
        [{"chunk": chunk}],
        query="Sinh viên có thẻ BHYT chính sách có phải nộp phí tại trường không?",
    )

    assert len(evidence) == 1
    assert "không nộp phí BHYT tại trường" in evidence[0]["text"]
    assert "hoàn thành trước ngày 15/11/2026" in evidence[0]["text"]


def test_system_prompt_preserves_deadlines_and_row_relationships():
    normalized_prompt = " ".join(SYSTEM_PROMPT.split())
    assert "đến hết ngày" in normalized_prompt
    assert "trước ngày" in normalized_prompt
    assert "thời hạn đóng/nộp" in normalized_prompt
    assert "đúng đối tượng/hệ/khóa trên cùng dòng" in normalized_prompt
    assert "thay vì tự suy diễn hoặc gộp các mức và thời hạn" in normalized_prompt
    assert "không dùng ngày ban hành thay cho thời hạn hoạt động" in normalized_prompt
    assert "chỉ khẳng định không có thông tin khi evidence thực sự không liên quan" in normalized_prompt


def test_context_uses_exact_relevant_sentences_for_text_evidence():
    chunk = ChunkRecord(
        chunk_id="paragraph-chunk",
        document_id="doc1",
        text=(
            "General information unrelated to enrollment.\n\n"
            "Enrollment starts on 25 July 2026 for students in K63.\n\n"
            "Tuition payments are due later."
        ),
        page=2,
        end_page=2,
        strategy="recursive",
        metadata={"source_path": "notice.pdf", "page_count": 4},
    )

    evidence = build_context(
        [{"chunk": chunk}], query="enrollment starts K63"
    )[0]

    assert evidence["evidence_text"] == (
        "Enrollment starts on 25 July 2026 for students in K63."
    )
    assert evidence["evidence_segments"] == [evidence["evidence_text"]]
    assert evidence["source_filename"] == "notice.pdf"


def test_context_keeps_date_for_relevant_sentence_in_a_long_question():
    chunk = ChunkRecord(
        chunk_id="deadline",
        document_id="doc1",
        text=(
            "Thông báo 516/TB.\n\n"
            "Từ 5/7/2026\n"
            "Sinh viên các khóa thực hiện gửi ý kiến phản hồi về hoạt động "
            "giảng dạy của giảng viên.\n"
            "20/07/2026\n"
            "- Phòng tổng hợp ý kiến phản hồi về hoạt động giảng dạy "
            "của giảng viên."
        ),
        page=1,
        end_page=1,
        strategy="recursive",
        metadata={"source_path": "notice.pdf", "page_count": 1},
    )

    evidence = build_context(
        [{"chunk": chunk}],
        query=(
            "Tài liệu có quy định thời hạn đánh giá giảng viên không?"
        ),
    )[0]

    assert "20/07/2026" in evidence["text"]


def test_context_does_not_inherit_document_bbox_as_chunk_geometry():
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

    assert bbox not in evidence[0].values()
    assert evidence[0]["bbox"] is None


def test_context_keeps_citation_text_identical_to_original_chunk():
    chunk = ChunkRecord(
        chunk_id="c1",
        document_id="doc1",
        text="Ngày 24 tháng 7 năm 2026 về việc tổ chức",
        page=1,
        end_page=1,
        strategy="fixed_size",
        metadata={"source_path": "notice.pdf"},
    )

    evidence = build_context([{"chunk": chunk}])

    assert evidence[0]["text"] == chunk.text
    assert evidence[0]["evidence_text"] == chunk.text


def test_citation_mapping_keeps_full_source_text_when_prompt_text_is_shortened():
    citation = {
        "evidence_id": "E1",
        "document_id": "doc1",
        "chunk_id": "c1",
        "page": 1,
        "page_count": 1,
        "evidence_text": "Câu nguồn đầy đủ có ngữ cảnh.",
        "text": "Câu nguồn",
        "source_path": "notice.pdf",
    }

    checked = validate_citations("Thông tin [E1].", [citation])

    assert checked["sources"][0]["evidence_text"] == citation["evidence_text"]
