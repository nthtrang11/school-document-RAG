from src.ui.components import (
    build_inline_answer_html,
    lookup_citation,
    register_citations,
)


def _source(citation_id: str, page: int, title: str) -> dict:
    return {
        "citation_id": citation_id,
        "document_id": f"doc-{citation_id}",
        "chunk_id": f"chunk-{citation_id}",
        "source_path": f"{citation_id}.pdf",
        "source": f"{citation_id}.pdf",
        "document_title": title,
        "page": page,
        "page_count": 8,
        "evidence_text": f"Đoạn bằng chứng của {citation_id}.",
        "bbox": None,
    }


def test_inline_citations_keep_position_and_map_each_answer_source():
    answer = "**Ý thứ nhất [E1]**, ý tiếp theo [E2]."
    rendered, linked = build_inline_answer_html(
        answer,
        [_source("E1", 2, "Quy chế"), _source("E2", 7, "Thông báo")],
        "token-1",
    )

    assert "<strong>Ý thứ nhất " in rendered
    assert 'data-citation-token="token-1" data-citation-id="E1"' in rendered
    assert 'data-citation-token="token-1" data-citation-id="E2"' in rendered
    assert "[E1]</a>" in rendered and "[E2]</a>" in rendered
    assert list(linked) == ["E1", "E2"]
    assert linked["E1"]["pdf_path"] == "E1.pdf"
    assert linked["E2"]["page"] == 7


def test_preview_preserves_unicode_newlines_and_escapes_source_html():
    source = {
        **_source("E1", 5, '<Tài liệu "gốc">'),
        "evidence_text": "Dòng tiếng Việt thứ nhất.\nDòng thứ hai <script>alert(1)</script> & dữ liệu.",
    }

    rendered, linked = build_inline_answer_html("Trích dẫn [E1].", [source], "token-2")

    assert '<strong class="citation-preview__title">&lt;Tài liệu &quot;gốc&quot;&gt;</strong>' in rendered
    assert "Trang 05" in rendered
    assert "Dòng tiếng Việt thứ nhất.\nDòng thứ hai" in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; dữ liệu." in rendered
    assert "<script>" not in rendered
    assert linked["E1"]["evidence_text"] == source["evidence_text"]


def test_answer_html_is_escaped_without_losing_markdown():
    rendered, _ = build_inline_answer_html(
        "Đoạn **đậm** có <img src=x onerror=alert(1)>.", [], "token-3"
    )

    assert "<strong>đậm</strong>" in rendered
    assert "&lt;img" in rendered
    assert "onerror=alert" in rendered
    assert "<img" not in rendered


def test_unknown_or_invalid_citations_are_not_registered_or_linked():
    answer = "Nguồn hợp lệ [E1], mã không rõ [E9]."
    incomplete = {**_source("E1", 2, "Quy chế"), "evidence_text": ""}

    rendered, linked = build_inline_answer_html(answer, [incomplete], "token-4")

    assert "[E1]" in rendered and "[E9]" in rendered
    assert "citation-marker" not in rendered
    assert linked == {}


def test_mapping_survives_rerender_and_new_answers_do_not_overwrite_old_records():
    registry: dict[str, dict[str, dict]] = {}
    first_html, first_records = build_inline_answer_html(
        "Ý một [E1].", [_source("E1", 2, "Nguồn một")], "answer-one"
    )
    register_citations(registry, "answer-one", first_records)

    rerendered_html, rerendered_records = build_inline_answer_html(
        "Ý một [E1].", [_source("E1", 2, "Nguồn một")], "answer-one"
    )
    register_citations(registry, "answer-one", rerendered_records)
    _, second_records = build_inline_answer_html(
        "Ý hai [E1].", [_source("E1", 6, "Nguồn hai")], "answer-two"
    )
    register_citations(registry, "answer-two", second_records)

    assert first_html == rerendered_html
    assert lookup_citation(registry, "answer-one", "E1")["page"] == 2
    assert lookup_citation(registry, "answer-one", "[E1]")["document_title"] == "Nguồn một"
    assert lookup_citation(registry, "answer-two", "E1")["page"] == 6
    assert lookup_citation(registry, "answer-two", "E9") is None


def test_multiple_citations_same_pdf_keep_independent_pages():
    first = _source("E1", 2, "Cùng tài liệu")
    second = {
        **_source("E2", 5, "Cùng tài liệu"),
        "source_path": first["source_path"],
    }
    rendered, linked = build_inline_answer_html(
        "Hai trang [E1] và [E2].", [first, second], "answer-pages"
    )
    registry: dict[str, dict[str, dict]] = {}
    register_citations(registry, "answer-pages", linked)

    assert 'data-citation-id="E1"' in rendered
    assert 'data-citation-id="E2"' in rendered
    assert lookup_citation(registry, "answer-pages", "E1")["page"] == 2
    assert lookup_citation(registry, "answer-pages", "E2")["page"] == 5


def test_invalid_page_is_not_clickable():
    rendered, linked = build_inline_answer_html(
        "Nội dung [E1].", [_source("E1", 9, "Quy chế")], "token-5"
    )

    assert "[E1]" in rendered
    assert "citation-marker" not in rendered
    assert linked == {}
