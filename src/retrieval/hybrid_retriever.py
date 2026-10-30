"""Metadata-aware hybrid retrieval with exact document-number prioritization."""

from __future__ import annotations

from datetime import date
from typing import Any

from src.config import settings
from src.embedding.embedder import SentenceTransformerEmbedder
from src.indexing.index_manager import IndexBundle
from src.models import SearchFilter
from src.retrieval.query_processor import process_query
from src.retrieval.rrf import reciprocal_rank_fusion


def _matches(chunk: Any, filters: SearchFilter | dict[str, Any] | None) -> bool:
    if filters is None:
        return True
    values = filters.model_dump(exclude_none=True) if isinstance(filters, SearchFilter) else filters
    meta = chunk.metadata
    checks = {
        "document_type": meta.get("document_type"),
        "issuing_authority": meta.get("issuing_authority"),
        "status": meta.get("status"),
    }
    for key in ("document_type", "issuing_authority", "status"):
        if values.get(key) and checks[key] != values[key]:
            return False
    if values.get("topic") and values["topic"].casefold() not in (
        f"{meta.get('title_subject') or ''} {chunk.section or ''} {chunk.text}".casefold()
    ):
        return False
    issue_date = meta.get("issue_date")
    try:
        parsed_date = date.fromisoformat(issue_date) if issue_date else None
        if values.get("issue_date_from") and (
            parsed_date is None or parsed_date < date.fromisoformat(values["issue_date_from"])
        ):
            return False
        if values.get("issue_date_to") and (
            parsed_date is None or parsed_date > date.fromisoformat(values["issue_date_to"])
        ):
            return False
    except ValueError:
        return False
    return True


class HybridRetriever:
    def __init__(self, bundle: IndexBundle):
        self.bundle = bundle
        self.embedder: SentenceTransformerEmbedder | None = None
        self._by_id = {chunk.chunk_id: chunk for chunk in bundle.chunks}

    def search(
        self,
        query: str,
        filters: SearchFilter | dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        processed = process_query(query)
        valid_indexes = [
            index for index, chunk in enumerate(self.bundle.chunks) if _matches(chunk, filters)
        ]
        allowed = set(valid_indexes)
        bm25_all = self.bundle.bm25.search(query, len(self.bundle.chunks))
        bm25 = [(index, score) for index, score in bm25_all if index in allowed][
            : settings.bm25_top_k
        ]
        if self.embedder is None:
            self.embedder = SentenceTransformerEmbedder(self.bundle.model_name)
        query_vector = self.embedder.encode([query])[0]
        faiss_all = self.bundle.faiss.search(query_vector, len(self.bundle.chunks))
        faiss_results = [(index, score) for index, score in faiss_all if index in allowed][
            : settings.faiss_top_k
        ]
        bm25_ranked = [(self.bundle.chunks[i].chunk_id, score) for i, score in bm25]
        faiss_ranked = [(self.bundle.chunks[i].chunk_id, score) for i, score in faiss_results]
        fused = reciprocal_rank_fusion([bm25_ranked, faiss_ranked], settings.rrf_k)

        exact_symbols = processed["exact_symbols"]
        exact_ids = {
            chunk.chunk_id
            for chunk in self.bundle.chunks
            if chunk.document_id
            and _matches(chunk, filters)
            and any(
                symbol.casefold()
                in str(chunk.metadata.get("number_symbol") or chunk.text).replace(" ", "").casefold()
                for symbol in exact_symbols
            )
        }
        ordered_ids = [
            chunk.chunk_id for chunk in self.bundle.chunks if chunk.chunk_id in exact_ids
        ]
        ordered_ids.extend(chunk_id for chunk_id, _ in fused if chunk_id not in exact_ids)
        limit = top_k or settings.final_top_k
        by_rank = {chunk_id: score for chunk_id, score in fused}
        results = []
        for chunk_id in ordered_ids[:limit]:
            chunk = self._by_id[chunk_id]
            results.append(
                {
                    "chunk": chunk,
                    "rrf_score": by_rank.get(chunk_id, 0.0),
                    "exact_match": chunk_id in exact_ids,
                    "bm25_rank": next(
                        (rank for rank, (candidate, _) in enumerate(bm25_ranked, 1)
                         if candidate == chunk_id), None
                    ),
                    "faiss_rank": next(
                        (rank for rank, (candidate, _) in enumerate(faiss_ranked, 1)
                         if candidate == chunk_id), None
                    ),
                }
            )
        return {
            "query": query,
            "bm25": bm25_ranked,
            "faiss": faiss_ranked,
            "rrf": fused,
            "results": results,
            "exact_symbols": exact_symbols,
        }
