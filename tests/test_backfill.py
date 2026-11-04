from src.ingestion.backfill import _retrieval_regressions, _verify_staging


def _snapshot(index_status: dict[str, bool]) -> dict:
    return {
        "documents": 1,
        "pages": 1,
        "integrity_issues": [],
        "documents_by_id": {
            "doc-1": {
                "page_count_expected": 1,
                "page_records": 1,
                "index_status": index_status,
                "page_issues": [],
            }
        },
    }


def _smoke_results(hybrid_documents: list[str]) -> dict:
    return {
        "query_count": 8,
        "queries": {
            strategy: [
                {
                    "query_id": "Q1",
                    "hybrid": [
                        {"rank": rank, "document_id": document_id}
                        for rank, document_id in enumerate(
                            hybrid_documents, start=1
                        )
                    ],
                }
            ]
            for strategy in ("fixed_size", "recursive")
        },
    }


def test_backfill_requires_both_index_strategies():
    issues = _verify_staging(
        _snapshot({"fixed_size": True, "recursive": False}),
        _smoke_results(["doc-1"]),
        _smoke_results(["doc-1"]),
    )

    assert any("recursive index is incomplete" in issue for issue in issues)


def test_backfill_rejects_retrieval_rank_regressions():
    before = _smoke_results(["doc-1"])
    after = _smoke_results(["doc-2"])

    assert len(_retrieval_regressions(before, after)) == 2
