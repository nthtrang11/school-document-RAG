from src.evaluation.retrieval_metrics import precision_at_k, recall_at_k
from src.indexing.bm25_index import tokenize
from src.retrieval.query_processor import process_query
from src.retrieval.rrf import reciprocal_rank_fusion


def test_tokenizer_and_exact_document_number():
    assert "406" in tokenize("406/TB-ĐHGTVT")
    query = process_query("Thông báo 406/TB-ĐHGTVT ngày 28/5/2026")
    assert "406/TB-ĐHGTVT" in query["exact_symbols"]
    assert "28/5/2026" not in query["exact_symbols"]


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
