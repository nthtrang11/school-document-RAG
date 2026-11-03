from src.evaluation.retrieval_metrics import precision_at_k, recall_at_k
from src.indexing.bm25_index import tokenize
from src.retrieval.query_processor import process_query
from src.retrieval.rrf import reciprocal_rank_fusion


def test_tokenizer_and_exact_document_number():
    assert "406" in tokenize("406/TB-ĐHGTVT")
    query = process_query("Thông báo 406/TB-ĐHGTVT ngày 28/5/2026")
    assert "406/TB-ĐHGTVT" in query["exact_symbols"]
    assert "28/5/2026" not in query["exact_symbols"]


def test_query_expansion_preserves_original_terms_and_adds_equivalent_phrases():
    query = process_query("Thời hạn đánh giá giảng viên")

    assert query["bm25_query"].startswith("Thời hạn đánh giá giảng viên")
    assert "thời gian" in query["bm25_query"]
    assert "đến hết ngày" in query["bm25_query"]
    assert "lấy ý kiến phản hồi" in query["bm25_query"]


def test_bhyt_payment_deadline_queries_expand_to_document_wording():
    deadline = process_query("Hạn đóng bảo hiểm y tế?")
    last_day = process_query("Ngày cuối cùng đóng BHYT là ngày nào?")
    amount = process_query("Mức đóng BHYT của khóa 65 là bao nhiêu?")

    assert "Hạn đóng bảo hiểm y tế?" in deadline["bm25_query"]
    assert "thời gian thực hiện" in deadline["bm25_query"]
    assert "nộp phí bảo hiểm y tế" in deadline["bm25_query"]
    assert "đến hết ngày" in last_day["bm25_query"]
    assert "số tiền đóng" in amount["bm25_query"]
    from src.retrieval.query_processor import is_monthly_rate_query

    assert is_monthly_rate_query("Mức nộp BHYT mỗi tháng là bao nhiêu?")
    assert not is_monthly_rate_query("Thời hạn sử dụng thẻ BHYT là bao lâu?")


def test_registration_schedule_queries_expand_to_related_schedule_terms():
    from src.retrieval.query_processor import is_registration_schedule_query

    queries = (
        "Lịch đăng ký học kỳ I năm học 2026–2027?",
        "Các đợt đăng ký học phần gồm những đợt nào?",
        "Khi nào được đăng ký học phần tự chọn?",
        "Có lịch đăng ký bổ sung không?",
    )
    for query in queries:
        processed = process_query(query)
        assert is_registration_schedule_query(query)
        assert query in processed["bm25_query"]
        assert "thời gian đăng ký học phần" in processed["bm25_query"]
        assert "đăng ký bổ sung học phần" in processed["bm25_query"]


def test_registration_schedule_retrieval_keeps_all_matching_year_groups(monkeypatch):
    from src.indexing.bm25_index import BM25Index
    from src.indexing.index_manager import IndexBundle
    from src.models import ChunkRecord
    from src.retrieval.hybrid_retriever import HybridRetriever

    chunks = []
    rows = (
        ("doc-2026", "2026-2027", "chính quy", 6),
        ("doc-2026", "2026-2027", "tiên tiến", 7),
        ("doc-2025", "2025-2026", "chính quy", 6),
        ("doc-fees", "2027", "BHYT", 1),
    )
    for index, (document_id, year, audience, page) in enumerate(rows):
        table_id = f"{document_id}:p{page}:t1"
        chunks.append(
            ChunkRecord(
                chunk_id=f"schedule-{year}-{audience}",
                document_id=document_id,
                text=f"Lớp {audience}\nTừ 10/8 đến 30/8",
                page=page,
                end_page=page,
                strategy="fixed_size",
                search_text=(
                    f"Tiêu đề: Lịch đăng ký học kỳ I năm học {year} {audience}\n"
                    "Lớp học: Đợt học 26\nThời gian đăng ký: Từ 10/8 đến 30/8"
                ),
                metadata={
                    "table_id": table_id,
                    "table_group_id": table_id,
                    "table_row_index": 1,
                    "table_title": (
                        f"Lịch đăng ký học kỳ I năm học {year} {audience}"
                        if audience != "BHYT"
                        else None
                    ),
                    "table_headers": (
                        ["Lớp học", "Thời gian đăng ký"]
                        if audience != "BHYT"
                        else [
                            "Đối tượng",
                            "Số tiền đóng BHYT",
                            "Thời gian sử dụng thẻ",
                        ]
                    ),
                    "table_cells": (
                        [f"Lớp {audience}", "Từ 10/8 đến 30/8"]
                        if audience != "BHYT"
                        else ["Khóa 64", "455.400", "08 tháng"]
                    ),
                },
            )
        )

    class StaticFaiss:
        def search(self, query_vector, top_k):
            return [(index, float(len(chunks) - index)) for index in range(len(chunks))]

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
            strategy="fixed_size",
            chunks=chunks,
            bm25=BM25Index(chunks),
            faiss=StaticFaiss(),
            model_name="test-model",
        )
    )

    results = retriever.search(
        "Lịch đăng ký học kỳ I năm học 2026–2027?", top_k=1
    )["results"]

    assert {
        item["chunk"].metadata["table_title"] for item in results
    } == {
        "Lịch đăng ký học kỳ I năm học 2026-2027 chính quy",
        "Lịch đăng ký học kỳ I năm học 2026-2027 tiên tiến",
    }


def test_rrf_fuses_by_chunk_id_and_ignores_raw_scores():
    fused = reciprocal_rank_fusion(
        [[("a", 900.0), ("b", 1.0)], [("b", 0.1), ("a", 0.0)]], k=60
    )
    assert fused[0][0] == "a"
    assert round(fused[0][1], 6) == round(1 / 61 + 1 / 62, 6)
    assert round(fused[1][1], 6) == round(1 / 62 + 1 / 61, 6)


def test_precision_recall_for_multiple_relevant_chunks():
    retrieved = ["a", "x", "b"]
    relevant = {"a", "b", "c"}
    assert precision_at_k(retrieved, relevant, 3) == 2 / 3
    assert recall_at_k(retrieved, relevant, 3) == 2 / 3


def test_metadata_filters_are_applied_before_returning_results():
    from src.models import ChunkRecord
    from src.retrieval.hybrid_retriever import _matches

    chunk = ChunkRecord(
        chunk_id="c",
        document_id="d",
        text="Nội dung học phí",
        page=1,
        end_page=1,
        strategy="fixed_size",
        metadata={"document_type": "Thông báo", "issue_date": "2026-10-14"},
    )
    assert _matches(chunk, {"document_type": "Thông báo"})
    assert not _matches(chunk, {"document_type": "Kế hoạch"})
    assert _matches(chunk, {"issue_date_from": "2026-01-01", "issue_date_to": "2026-12-31"})


def test_exact_table_phrase_promotes_the_matching_row(monkeypatch):
    from src.indexing.bm25_index import BM25Index
    from src.indexing.index_manager import IndexBundle
    from src.models import ChunkRecord
    from src.retrieval.hybrid_retriever import HybridRetriever

    chunks = [
        ChunkRecord(
            chunk_id=f"unrelated-{index}",
            document_id="doc",
            text="Nội dung chung về lịch học và học phí.",
            page=1,
            end_page=1,
            strategy="fixed_size",
        )
        for index in range(5)
    ]
    chunks.append(
        ChunkRecord(
            chunk_id="target-table-row",
            document_id="doc",
            text="Đợt học 26\nTừ 10/8 đến 30/8/2026",
            page=7,
            end_page=7,
            strategy="fixed_size",
            search_text="Đợt: Đợt học 26\nThời gian đăng ký: Từ 10/8 đến 30/8/2026",
            metadata={"table_id": "doc:p7:t1"},
        )
    )

    class StaticFaiss:
        def search(self, query_vector, top_k):
            return [(index, float(len(chunks) - index)) for index in range(len(chunks))]

    class StaticEmbedder:
        def __init__(self, model_name):
            pass

        def encode(self, texts):
            return [[0.0]]

    monkeypatch.setattr(
        "src.retrieval.hybrid_retriever.SentenceTransformerEmbedder",
        StaticEmbedder,
    )
    bundle = IndexBundle(
        strategy="fixed_size",
        chunks=chunks,
        bm25=BM25Index(chunks),
        faiss=StaticFaiss(),
        model_name="test-model",
    )

    results = HybridRetriever(bundle).search(
        "Tìm thông tin đợt học 26", top_k=1
    )["results"]

    assert results[0]["chunk"].chunk_id == "target-table-row"


def test_table_title_retrieval_expands_only_rows_from_the_same_table(monkeypatch):
    from src.indexing.bm25_index import BM25Index
    from src.indexing.index_manager import IndexBundle
    from src.models import ChunkRecord
    from src.retrieval.hybrid_retriever import HybridRetriever

    headers = ["Khóa", "Mức đóng", "Thời hạn"]
    chunks = [
        ChunkRecord(
            chunk_id="bhyt-row-1",
            document_id="doc",
            text="Khóa 66\n683.100\n12 tháng",
            page=1,
            end_page=1,
            strategy="fixed_size",
            search_text="Khóa: Khóa 66\nMức đóng: 683.100\nThời hạn: 12 tháng",
            metadata={
                "title_subject": "Thông báo bảo hiểm y tế",
                "table_id": "doc:p1:t1",
                "table_row_index": 1,
                "table_headers": headers,
                "table_cells": ["Khóa 66", "683.100", "12 tháng"],
            },
        ),
        ChunkRecord(
            chunk_id="bhyt-row-2",
            document_id="doc",
            text="Khóa 65\n683.100\n12 tháng",
            page=1,
            end_page=1,
            strategy="fixed_size",
            search_text="Khóa: Khóa 65\nMức đóng: 683.100\nThời hạn: 12 tháng",
            metadata={
                "title_subject": "Thông báo bảo hiểm y tế",
                "table_id": "doc:p1:t1",
                "table_row_index": 2,
                "table_headers": headers,
                "table_cells": ["Khóa 65", "683.100", "12 tháng"],
            },
        ),
        ChunkRecord(
            chunk_id="unrelated-table-row",
            document_id="doc",
            text="Khóa 64\n455.400\n08 tháng",
            page=1,
            end_page=1,
            strategy="fixed_size",
            search_text="Khóa: Khóa 64\nMức đóng: 455.400\nThời hạn: 08 tháng",
            metadata={
                "title_subject": "Thông báo bảo hiểm y tế",
                "table_id": "doc:p1:t2",
                "table_row_index": 1,
                "table_headers": headers,
                "table_cells": ["Khóa 64", "455.400", "08 tháng"],
            },
        ),
    ]

    class StaticFaiss:
        def search(self, query_vector, top_k):
            return [(index, float(len(chunks) - index)) for index in range(len(chunks))]

    class StaticEmbedder:
        def __init__(self, model_name):
            pass

        def encode(self, texts):
            return [[0.0]]

    monkeypatch.setattr(
        "src.retrieval.hybrid_retriever.SentenceTransformerEmbedder",
        StaticEmbedder,
    )
    bundle = IndexBundle(
        strategy="fixed_size",
        chunks=chunks,
        bm25=BM25Index(chunks),
        faiss=StaticFaiss(),
        model_name="test-model",
    )

    retriever = HybridRetriever(bundle)
    results = retriever.search("bảo hiểm y tế", top_k=1)["results"]

    assert [item["chunk"].chunk_id for item in results] == [
        "bhyt-row-1",
        "bhyt-row-2",
    ]
    bm25_ids = {
        chunks[index].chunk_id
        for index, _ in retriever._search_bm25.search(
            "bảo hiểm y tế", len(chunks)
        )
    }
    assert "bhyt-row-1" in bm25_ids


def test_table_row_expansion_includes_continuation_pages(monkeypatch):
    from src.chunking.fixed_size import chunk_pages
    from src.indexing.bm25_index import BM25Index
    from src.indexing.index_manager import IndexBundle
    from src.ingestion.pdf_loader import _link_continued_tables
    from src.models import PageRecord, TableRowRecord
    from src.retrieval.hybrid_retriever import HybridRetriever

    pages = _link_continued_tables(
        [
            PageRecord(
                document_id="multi-page-doc",
                page_no=1,
                text="",
                table_rows=[
                    TableRowRecord(
                        table_id="multi-page-doc:p1:t1",
                        row_index=1,
                        headers=["Đối tượng", "Mức đóng"],
                        header_rows=[["Đối tượng", "Mức đóng"]],
                        cells=["Nhóm A", "100"],
                        bboxes=[None, None],
                    )
                ],
            ),
            PageRecord(
                document_id="multi-page-doc",
                page_no=2,
                text="",
                table_rows=[
                    TableRowRecord(
                        table_id="multi-page-doc:p2:t1",
                        row_index=1,
                        headers=["Đối tượng", "Mức đóng"],
                        header_rows=[["Đối tượng", "Mức đóng"]],
                        cells=["Nhóm B", "200"],
                        bboxes=[None, None],
                    )
                ],
            ),
        ]
    )
    chunks = chunk_pages(pages)

    class StaticFaiss:
        def search(self, query_vector, top_k):
            return [(index, float(len(chunks) - index)) for index in range(len(chunks))]

    class StaticEmbedder:
        def __init__(self, model_name):
            pass

        def encode(self, texts):
            return [[0.0]]

    monkeypatch.setattr(
        "src.retrieval.hybrid_retriever.SentenceTransformerEmbedder",
        StaticEmbedder,
    )
    bundle = IndexBundle(
        strategy="fixed_size",
        chunks=chunks,
        bm25=BM25Index(chunks),
        faiss=StaticFaiss(),
        model_name="test-model",
    )

    results = HybridRetriever(bundle).search("Mức đóng", top_k=1)["results"]

    assert [item["chunk"].page for item in results] == [1, 2]
    assert [item["chunk"].metadata["table_cells"][0] for item in results] == [
        "Nhóm A",
        "Nhóm B",
    ]


def test_fused_text_match_precedes_broad_table_fallback(monkeypatch):
    from src.indexing.bm25_index import BM25Index
    from src.indexing.index_manager import IndexBundle
    from src.models import ChunkRecord
    from src.retrieval.hybrid_retriever import HybridRetriever

    chunks = [
        ChunkRecord(
            chunk_id="feedback-notice",
            document_id="notice",
            text=(
                "Thông báo lấy ý kiến phản hồi về hoạt động giảng dạy "
                "của giảng viên đến hết ngày 20/07/2026."
            ),
            page=1,
            end_page=1,
            strategy="fixed_size",
            metadata={"title_subject": "Thông báo hoạt động giảng dạy"},
        ),
        ChunkRecord(
            chunk_id="unrelated-table-row",
            document_id="schedule",
            text="Lớp học phần\nTháng 8",
            page=1,
            end_page=1,
            strategy="fixed_size",
            search_text="Thời gian thu: Tháng 8",
            metadata={
                "table_id": "schedule:p1:t1",
                "table_row_index": 1,
                "table_headers": ["Lớp học phần", "Thời gian thu"],
                "table_cells": ["Lớp học phần", "Tháng 8"],
            },
        ),
    ]

    class StaticFaiss:
        def search(self, query_vector, top_k):
            return [(0, 0.9), (1, 0.4)]

    class StaticEmbedder:
        def __init__(self, model_name):
            pass

        def encode(self, texts):
            return [[0.0]]

    monkeypatch.setattr(
        "src.retrieval.hybrid_retriever.SentenceTransformerEmbedder",
        StaticEmbedder,
    )
    bundle = IndexBundle(
        strategy="fixed_size",
        chunks=chunks,
        bm25=BM25Index(chunks),
        faiss=StaticFaiss(),
        model_name="test-model",
    )

    results = HybridRetriever(bundle).search(
        "Thời hạn đánh giá giảng viên là khi nào?", top_k=1
    )["results"]

    assert results[0]["chunk"].chunk_id == "feedback-notice"


def test_retrieval_adds_adjacent_same_page_chunk_with_deadline(monkeypatch):
    from src.indexing.bm25_index import BM25Index
    from src.indexing.index_manager import IndexBundle
    from src.models import ChunkRecord
    from src.rag.context_builder import build_context
    from src.retrieval.hybrid_retriever import HybridRetriever

    chunks = [
        ChunkRecord(
            chunk_id="activity",
            document_id="notice",
            text=(
                "Sinh viên gửi ý kiến phản hồi về hoạt động giảng dạy "
                "của giảng viên. Thời gian thực hiện:"
            ),
            page=1,
            end_page=1,
            strategy="recursive",
            metadata={
                "title_subject": "Thông báo lấy ý kiến phản hồi về giảng dạy",
                "source_path": "notice.pdf",
                "page_count": 1,
            },
        ),
        ChunkRecord(
            chunk_id="split-deadline",
            document_id="notice",
            text=(
                "20/07/2026\n"
                "- Giảng viên theo dõi kết quả trên hệ thống."
            ),
            page=1,
            end_page=1,
            strategy="recursive",
            metadata={
                "title_subject": "Thông báo lấy ý kiến phản hồi về giảng dạy",
                "source_path": "notice.pdf",
                "page_count": 1,
            },
        ),
        ChunkRecord(
            chunk_id="other-document",
            document_id="other",
            text="Thông tin về giảng viên không liên quan.",
            page=1,
            end_page=1,
            strategy="recursive",
        ),
    ]

    class StaticFaiss:
        def search(self, query_vector, top_k):
            return [(0, 0.9), (1, 0.5), (2, 0.1)]

    class StaticEmbedder:
        def __init__(self, model_name):
            pass

        def encode(self, texts):
            return [[0.0]]

    monkeypatch.setattr(
        "src.retrieval.hybrid_retriever.SentenceTransformerEmbedder",
        StaticEmbedder,
    )
    bundle = IndexBundle(
        strategy="recursive",
        chunks=chunks,
        bm25=BM25Index(chunks),
        faiss=StaticFaiss(),
        model_name="test-model",
    )

    results = HybridRetriever(bundle).search(
        "Thời hạn đánh giá giảng viên là khi nào?", top_k=1
    )["results"]
    evidence = build_context(results, query="Thời hạn đánh giá giảng viên là khi nào?")

    result_ids = [item["chunk"].chunk_id for item in results]
    assert result_ids == ["activity", "split-deadline"]
    assert any("20/07/2026" in item["text"] for item in evidence)


def test_payment_deadline_retrieval_excludes_card_validity_table(monkeypatch):
    from src.indexing.bm25_index import BM25Index
    from src.indexing.index_manager import IndexBundle
    from src.models import ChunkRecord
    from src.rag.context_builder import build_context
    from src.retrieval.hybrid_retriever import HybridRetriever

    chunks = [
        ChunkRecord(
            chunk_id="card-validity",
            document_id="bhyt",
            text="Khóa 66\n12 tháng (Từ 01/01/2027 đến 31/12/2027)",
            page=1,
            end_page=1,
            strategy="fixed_size",
            metadata={
                "title_subject": "Kế hoạch thu phí gia hạn bảo hiểm y tế",
                "table_id": "bhyt:p1:t1",
                "table_headers": ["Đối tượng", "Thời gian sử dụng thẻ"],
                "table_cells": [
                    "Khóa 66",
                    "12 tháng (Từ 01/01/2027 đến 31/12/2027)",
                ],
            },
        ),
        ChunkRecord(
            chunk_id="payment-deadline",
            document_id="bhyt",
            text=(
                "III/ Thời gian thực hiện và mức nộp BHYT. "
                "Thời gian: Từ ngày 22/10/2026 đến hết ngày 15/11/2026."
            ),
            page=1,
            end_page=1,
            strategy="fixed_size",
            metadata={"title_subject": "Kế hoạch thu phí gia hạn bảo hiểm y tế"},
        ),
    ]

    class StaticFaiss:
        def search(self, query_vector, top_k):
            return [(0, 0.99), (1, 0.8)]

    class StaticEmbedder:
        def __init__(self, model_name):
            pass

        def encode(self, texts):
            return [[0.0]]

    monkeypatch.setattr(
        "src.retrieval.hybrid_retriever.SentenceTransformerEmbedder",
        StaticEmbedder,
    )
    bundle = IndexBundle(
        strategy="fixed_size",
        chunks=chunks,
        bm25=BM25Index(chunks),
        faiss=StaticFaiss(),
        model_name="test-model",
    )

    result = HybridRetriever(bundle).search(
        "Thời hạn đóng bảo hiểm y tế?", top_k=1
    )
    evidence = build_context(result["results"], query=result["query"])

    assert [item["chunk"].chunk_id for item in result["results"]] == [
        "payment-deadline"
    ]
    assert "đến hết ngày 15/11/2026" in evidence[0]["text"]


def test_monthly_rate_retrieval_excludes_annual_card_fee_table(monkeypatch):
    from src.indexing.bm25_index import BM25Index
    from src.indexing.index_manager import IndexBundle
    from src.models import ChunkRecord
    from src.rag.context_builder import build_context
    from src.retrieval.hybrid_retriever import HybridRetriever

    chunks = [
        ChunkRecord(
            chunk_id="annual-card-fee",
            document_id="bhyt",
            text="Khóa 66\n683.100\n12 tháng",
            page=1,
            end_page=1,
            strategy="recursive",
            metadata={
                "title_subject": "Kế hoạch thu phí gia hạn bảo hiểm y tế",
                "table_id": "bhyt:p1:t1",
                "table_headers": ["Đối tượng", "Số tiền đóng BHYT", "Thời gian sử dụng thẻ"],
                "table_cells": ["Khóa 66", "683.100", "12 tháng"],
            },
        ),
        ChunkRecord(
            chunk_id="monthly-rate",
            document_id="bhyt",
            text="III/ Thời gian thực hiện và mức nộp BHYT. Mức nộp: 56,925 đồng/tháng/SV.",
            page=1,
            end_page=1,
            strategy="recursive",
            metadata={"title_subject": "Kế hoạch thu phí gia hạn bảo hiểm y tế"},
        ),
    ]

    class StaticFaiss:
        def search(self, query_vector, top_k):
            return [(0, 0.99), (1, 0.8)]

    class StaticEmbedder:
        def __init__(self, model_name):
            pass

        def encode(self, texts):
            return [[0.0]]

    monkeypatch.setattr(
        "src.retrieval.hybrid_retriever.SentenceTransformerEmbedder",
        StaticEmbedder,
    )
    bundle = IndexBundle(
        strategy="recursive",
        chunks=chunks,
        bm25=BM25Index(chunks),
        faiss=StaticFaiss(),
        model_name="test-model",
    )

    result = HybridRetriever(bundle).search(
        "Mức nộp BHYT mỗi tháng là bao nhiêu?", top_k=1
    )
    evidence = build_context(result["results"], query=result["query"])

    assert [item["chunk"].chunk_id for item in result["results"]] == ["monthly-rate"]
    assert "56,925 đồng/tháng/SV" in evidence[0]["text"]
    assert "683.100" not in "\n".join(item["text"] for item in evidence)
