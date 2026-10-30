from src.indexing.bm25_index import BM25Index
from src.models import ChunkRecord


def test_bm25_search_and_token_mapping_round_trip(tmp_path):
    chunks = [
        ChunkRecord(
            chunk_id="fee",
            document_id="doc-a",
            text="gia hạn học phí học phần",
            page=1,
            end_page=1,
            strategy="fixed_size",
        ),
        ChunkRecord(
            chunk_id="health",
            document_id="doc-b",
            text="bảo hiểm y tế sinh viên",
            page=1,
            end_page=1,
            strategy="fixed_size",
        ),
    ]
    index = BM25Index(chunks)
    assert index.search("học phí", 2)[0][0] == 0
    tokens_path = tmp_path / "tokens.json"
    index.save_tokens(tokens_path)
    restored = BM25Index.load(chunks, tokens_path)
    assert restored.search("bảo hiểm y tế", 1)[0][0] == 1
