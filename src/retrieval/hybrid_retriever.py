"""Metadata-aware hybrid retrieval with exact document-number prioritization."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from datetime import date
from time import perf_counter
from typing import Any

from src.config import settings
from src.embedding.embedder import SentenceTransformerEmbedder
from src.indexing.bm25_index import BM25Index, tokenize
from src.indexing.index_manager import IndexBundle
from src.models import SearchFilter
from src.retrieval.query_processor import (
    is_document_identity_query,
    is_health_insurance_payment_query,
    is_monthly_rate_query,
    is_payment_deadline_query,
    is_registration_schedule_query,
    is_tuition_deadline_query,
    process_query,
)
from src.retrieval.rrf import reciprocal_rank_fusion

_DATE_IN_TEXT = re.compile(
    r"\b\d{1,2}\s*(?:[/\-]\s*\d{1,2}\s*[/\-]\s*\d{4}|"
    r"tháng\s*\d{1,2}\s*năm\s*\d{4})\b",
    re.IGNORECASE,
)
_DEADLINE_IN_TEXT = re.compile(
    r"\b(?:th(?:oi|di) han|han|den het|truoc ngay|"
    r"cham nhat|gia han|nop truoc|thoi gian thu|thoi gian nop)\b",
    re.IGNORECASE,
)


def _fold_search_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold()).replace("đ", "d")
    return "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )


def _matches(chunk: Any, filters: SearchFilter | dict[str, Any] | None) -> bool:
    if filters is None:
        return True
    values = filters.model_dump(exclude_none=True) if isinstance(filters, SearchFilter) else filters
    meta = chunk.metadata
    checks = {
        "document_type": meta.get("document_type"),
        "issuing_authority": meta.get("issuing_authority"),
    }
    for key in ("document_type", "issuing_authority"):
        if values.get(key) and checks[key] != values[key]:
            return False
    if values.get("topic") and values["topic"].casefold() not in (
        f"{meta.get('title_subject') or ''} {chunk.section or ''} "
        f"{chunk.search_text or chunk.text}".casefold()
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


def _matches_query(chunk: Any, query: str) -> bool:
    headers = chunk.metadata.get("table_headers")
    has_card_duration = isinstance(headers, list) and any(
        "thời gian sử dụng thẻ" in str(header).casefold()
        for header in headers
    )
    if not has_card_duration:
        return True
    if is_payment_deadline_query(query):
        return False
    if is_monthly_rate_query(query):
        cells = chunk.metadata.get("table_cells")
        return isinstance(cells, list) and any(
            "/tháng" in str(cell).casefold()
            or "đồng/tháng" in str(cell).casefold()
            for cell in cells
        )
    return True


def _matches_payment_topic(query: str, text: str) -> bool:
    folded_text = text.casefold()
    if is_tuition_deadline_query(query) and not any(
        phrase in folded_text for phrase in ("học phí", "hoc phi", "tiền học", "tien hoc")
    ):
        return False
    if is_health_insurance_payment_query(query) and not any(
        phrase in folded_text
        for phrase in ("bảo hiểm y tế", "bao hiem y te", "bhyt")
    ):
        return False
    return True


def _matches_payment_deadline_chunk(query: str, chunk: Any) -> bool:
    table_headers = chunk.metadata.get("table_headers") or []
    table_cells = chunk.metadata.get("table_cells") or []
    text = " ".join(
        str(value)
        for value in (
            chunk.search_text or chunk.text,
            chunk.metadata.get("title_subject"),
            chunk.metadata.get("table_title"),
            " ".join(str(value) for value in table_headers),
            " ".join(str(value) for value in table_cells),
        )
        if value
    )
    return (
        _matches_payment_topic(query, text)
        and bool(_DATE_IN_TEXT.search(text))
        and bool(_DEADLINE_IN_TEXT.search(_fold_search_text(text)))
    )


def _asks_for_table_values(query: str) -> bool:
    query_text = " ".join(query.casefold().split())
    return bool(
        is_registration_schedule_query(query)
        or
        re.search(r"\bkh[oó]a\s+\d+\b", query_text)
        or any(
            phrase in query_text
            for phrase in (
                "mỗi khóa",
                "mỗi khoá",
                "theo từng",
                "từng khóa",
                "từng khoá",
                "mức đóng",
                "mức tiền",
                "mức phí",
                "hệ đào tạo",
            )
        )
    )


class HybridRetriever:
    def __init__(self, bundle: IndexBundle):
        self.bundle = bundle
        self.embedder: SentenceTransformerEmbedder | None = None
        self._by_id = {chunk.chunk_id: chunk for chunk in bundle.chunks}
        self._text_chunk_positions: dict[
            tuple[str, int], list[str]
        ] = {}
        self._text_chunk_position_by_id: dict[str, int] = {}
        for chunk in bundle.chunks:
            if chunk.metadata.get("table_id"):
                continue
            group_key = (chunk.document_id, chunk.page)
            group = self._text_chunk_positions.setdefault(group_key, [])
            self._text_chunk_position_by_id[chunk.chunk_id] = len(group)
            group.append(chunk.chunk_id)
        self._search_bm25 = BM25Index(
            bundle.chunks,
            tokenized=[
                tokenize(
                    " ".join(
                        value
                        for value in (
                            str(chunk.metadata.get("title_subject") or ""),
                            str(chunk.metadata.get("number_symbol") or ""),
                            str(chunk.section or ""),
                            chunk.search_text or chunk.text,
                        )
                        if value
                    )
                )
                for chunk in bundle.chunks
            ],
        )
        self._table_chunk_indexes = [
            index
            for index, chunk in enumerate(bundle.chunks)
            if chunk.metadata.get("table_id")
        ]
        self._table_search_bm25 = BM25Index(
            [bundle.chunks[index] for index in self._table_chunk_indexes],
            tokenized=[
                tokenize(
                    " ".join(
                        value
                        for value in (
                            str(chunk.metadata.get("title_subject") or ""),
                            chunk.search_text or chunk.text,
                            " ".join(
                                str(value)
                                for value in (
                                    chunk.metadata.get("table_headers") or []
                                )
                            ),
                            " ".join(
                                str(value)
                                for value in (
                                    chunk.metadata.get("table_cells") or []
                                )
                            ),
                        )
                        if value
                    )
                )
                for chunk in (
                    bundle.chunks[index] for index in self._table_chunk_indexes
                )
            ],
        )
        self._table_groups: dict[tuple[str, str], list[str]] = {}
        self._table_group_by_chunk: dict[str, tuple[str, str]] = {}
        for chunk in bundle.chunks:
            table_id = chunk.metadata.get("table_id")
            if not table_id:
                continue
            table_group_id = chunk.metadata.get("table_group_id") or table_id
            table_key = (chunk.document_id, str(table_group_id))
            self._table_groups.setdefault(table_key, []).append(chunk.chunk_id)
            self._table_group_by_chunk[chunk.chunk_id] = table_key
        self._related_documents: dict[str, set[str]] = {}
        for chunk in bundle.chunks:
            relations = chunk.metadata.get("document_relations")
            if not isinstance(relations, list):
                continue
            for relation in relations:
                if not isinstance(relation, dict):
                    continue
                target_id = relation.get("target_document_id")
                if not target_id:
                    continue
                if relation.get("relation_type") not in {
                    "amends",
                    "extends",
                    "replaces",
                }:
                    continue
                self._related_documents.setdefault(chunk.document_id, set()).add(
                    str(target_id)
                )
                self._related_documents.setdefault(str(target_id), set()).add(
                    chunk.document_id
                )

        table_text_by_id = {
            chunk.chunk_id: " ".join(
                value
                for value in (
                    str(chunk.metadata.get("title_subject") or ""),
                    str(chunk.metadata.get("number_symbol") or ""),
                    chunk.search_text or chunk.text,
                )
                if value
            )
            for chunk in bundle.chunks
            if chunk.metadata.get("table_id")
        }
        self._table_phrases = {
            chunk_id: self._ngrams(tokenize(table_text_by_id[chunk_id]))
            for chunk_id in table_text_by_id
        }
        self._table_phrase_frequency = Counter(
            phrase
            for phrases in self._table_phrases.values()
            for phrase in phrases
        )

    @staticmethod
    def _ngrams(tokens: list[str], maximum: int = 6) -> set[tuple[str, ...]]:
        return {
            tuple(tokens[start : start + size])
            for size in range(3, min(maximum, len(tokens)) + 1)
            for start in range(len(tokens) - size + 1)
        }

    def _related_table_chunk_ids(
        self, selected_chunk_ids: list[str]
    ) -> list[str]:
        selected_groups = {
            self._table_group_by_chunk[chunk_id]
            for chunk_id in selected_chunk_ids
            if chunk_id in self._table_group_by_chunk
        }
        return [
            chunk_id
            for group in sorted(selected_groups)
            for chunk_id in sorted(
                self._table_groups[group],
                key=lambda value: (
                    self._by_id[value].page,
                    self._by_id[value].metadata.get("table_row_index", 0),
                    value,
                ),
            )
        ]

    def _neighbor_text_chunk_ids(
        self,
        selected_chunk_ids: list[str],
        filters: SearchFilter | dict[str, Any] | None,
    ) -> list[str]:
        neighbors: list[str] = []
        selected = set(selected_chunk_ids)
        for chunk_id in selected_chunk_ids:
            group_key = (
                self._by_id[chunk_id].document_id,
                self._by_id[chunk_id].page,
            )
            group = self._text_chunk_positions.get(group_key, [])
            position = self._text_chunk_position_by_id.get(chunk_id)
            if position is None:
                continue
            for neighbor_position in (position - 1, position + 1):
                if 0 <= neighbor_position < len(group):
                    neighbor_id = group[neighbor_position]
                    if (
                        neighbor_id not in selected
                        and _matches(self._by_id[neighbor_id], filters)
                        and neighbor_id not in neighbors
                    ):
                        neighbors.append(neighbor_id)
        return neighbors

    def search(
        self,
        query: str,
        filters: SearchFilter | dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        total_started = perf_counter()
        processed = process_query(query)
        valid_indexes = [
            index
            for index, chunk in enumerate(self.bundle.chunks)
            if _matches(chunk, filters) and _matches_query(chunk, query)
        ]
        allowed = set(valid_indexes)
        allowed_chunk_ids = {
            self.bundle.chunks[index].chunk_id for index in valid_indexes
        }
        bm25_query = str(processed["bm25_query"])
        bm25_started = perf_counter()
        bm25_all = self._search_bm25.search(
            bm25_query, len(self.bundle.chunks)
        )
        bm25 = [(index, score) for index, score in bm25_all if index in allowed][
            : settings.bm25_top_k
        ]
        bm25_latency = perf_counter() - bm25_started
        faiss_started = perf_counter()
        if self.embedder is None:
            self.embedder = SentenceTransformerEmbedder(self.bundle.model_name)
        query_vector = self.embedder.encode([query])[0]
        faiss_all = self.bundle.faiss.search(query_vector, len(self.bundle.chunks))
        faiss_results = [(index, score) for index, score in faiss_all if index in allowed][
            : settings.faiss_top_k
        ]
        faiss_latency = perf_counter() - faiss_started
        bm25_ranked = [(self.bundle.chunks[i].chunk_id, score) for i, score in bm25]
        faiss_ranked = [(self.bundle.chunks[i].chunk_id, score) for i, score in faiss_results]
        rrf_started = perf_counter()
        fused = reciprocal_rank_fusion([bm25_ranked, faiss_ranked], settings.rrf_k)
        rrf_latency = perf_counter() - rrf_started
        fused_score_by_id = dict(fused)

        exact_symbols = processed["exact_symbols"]
        exact_ids = {
            chunk.chunk_id
            for chunk in self.bundle.chunks
            if chunk.document_id
            and _matches(chunk, filters)
            and _matches_query(chunk, query)
            and any(
                symbol.casefold()
                in str(chunk.metadata.get("number_symbol") or chunk.text).replace(" ", "").casefold()
                for symbol in exact_symbols
            )
        }
        if is_payment_deadline_query(query):
            exact_ids = {
                chunk_id
                for chunk_id in exact_ids
                if _matches_payment_deadline_chunk(
                    query, self._by_id[chunk_id]
                )
            }
        query_phrases = self._ngrams(tokenize(query))
        table_count = len(self._table_phrases)
        table_phrase_scores = {
            chunk_id: sum(
                len(phrase)
                * math.log(
                    (table_count + 1)
                    / (self._table_phrase_frequency[phrase] + 0.5)
                )
                for phrase in phrases.intersection(query_phrases)
            )
            for chunk_id, phrases in self._table_phrases.items()
            if chunk_id in allowed_chunk_ids
        }
        table_phrase_ids = [
            chunk_id
            for chunk_id, score in sorted(
                table_phrase_scores.items(),
                key=lambda entry: (
                    -entry[1],
                    -fused_score_by_id.get(entry[0], 0.0),
                    entry[0],
                ),
            )
            if score > 0
            and chunk_id not in exact_ids
            and _matches(self._by_id[chunk_id], filters)
            and chunk_id in allowed_chunk_ids
        ]
        table_search_results = self._table_search_bm25.search(
            bm25_query, len(self._table_chunk_indexes)
        )
        table_search_ids = [
            self.bundle.chunks[self._table_chunk_indexes[index]].chunk_id
            for index, _ in table_search_results
            if self.bundle.chunks[self._table_chunk_indexes[index]].chunk_id
            not in exact_ids
            and self.bundle.chunks[self._table_chunk_indexes[index]].chunk_id
            in allowed_chunk_ids
            and _matches(
                self.bundle.chunks[self._table_chunk_indexes[index]], filters
            )
        ]
        if is_payment_deadline_query(query):
            table_phrase_ids = [
                chunk_id
                for chunk_id in table_phrase_ids
                if _matches_payment_deadline_chunk(
                    query, self._by_id[chunk_id]
                )
            ]
            table_search_ids = [
                chunk_id
                for chunk_id in table_search_ids
                if _matches_payment_deadline_chunk(
                    query, self._by_id[chunk_id]
                )
            ]
        payment_deadline_ids: list[str] = []
        if is_payment_deadline_query(query):
            bm25_score_by_index = {index: score for index, score in bm25}
            candidates_by_document: dict[
                str, tuple[tuple[int, float, str], str]
            ] = {}
            for index in valid_indexes:
                chunk = self.bundle.chunks[index]
                if not _matches_payment_deadline_chunk(query, chunk):
                    continue
                text = " ".join(
                    (
                        chunk.search_text or chunk.text,
                        str(chunk.metadata.get("title_subject") or ""),
                    )
                )
                folded_text = text.casefold()
                if not any(
                    phrase in folded_text
                    for phrase in ("học phí", "hoc phi", "nộp phí", "nop phi")
                ):
                    continue
                relations = chunk.metadata.get("document_relations") or []
                has_update_relation = any(
                    isinstance(relation, dict)
                    and relation.get("relation_type") in {"amends", "extends"}
                    for relation in relations
                )
                issue_date = str(chunk.metadata.get("issue_date") or "")
                score = (
                    int(has_update_relation),
                    bm25_score_by_index.get(index, 0.0),
                    issue_date,
                )
                previous = candidates_by_document.get(chunk.document_id)
                if previous is None or score > previous[0]:
                    candidates_by_document[chunk.document_id] = (
                        score,
                        chunk.chunk_id,
                    )
            payment_deadline_ids = [
                candidate[1]
                for candidate in sorted(
                    candidates_by_document.values(),
                    key=lambda item: item[0],
                    reverse=True,
                )
            ]
        ordered_ids = []
        if is_payment_deadline_query(query):
            ordered_ids.extend(payment_deadline_ids)
        ordered_ids.extend(
            chunk.chunk_id for chunk in self.bundle.chunks if chunk.chunk_id in exact_ids
        )
        ordered_ids.extend(
            chunk_id for chunk_id in payment_deadline_ids if chunk_id not in ordered_ids
        )
        schedule_group_keys: set[tuple[str, str]] = set()
        if _asks_for_table_values(query):
            table_candidates = list(
                dict.fromkeys([*table_search_ids, *table_phrase_ids])
            )
            if table_candidates:
                mentioned_keys = re.findall(
                    r"\bkh[oó]a\s+(\d+)\b", query.casefold()
                )
                group_key_scores: dict[tuple[str, str], int] = Counter()
                query_terms = set(tokenize(bm25_query))
                group_header_scores: dict[tuple[str, str], int] = {}
                group_header_terms: dict[tuple[str, str], set[str]] = {}
                group_term_scores: dict[tuple[str, str], int] = {}
                group_title_scores: dict[tuple[str, str], int] = {}
                for group_key, chunk_ids in self._table_groups.items():
                    exact_keys = {
                        key
                        for chunk_id in chunk_ids
                        for cell in (
                            self._by_id[chunk_id].metadata.get("table_cells") or []
                        )
                        if isinstance(cell, str)
                        for key in mentioned_keys
                        if re.match(
                            rf"\bkh[oó]a\s+{re.escape(key)}\b",
                            cell.casefold(),
                        )
                    }
                    group_key_scores[group_key] = len(exact_keys)
                    group_headers = {
                        token
                        for chunk_id in chunk_ids
                        for token in tokenize(
                            " ".join(
                                str(value)
                                for value in (
                                    self._by_id[chunk_id].metadata.get("table_headers")
                                    or []
                                )
                            )
                        )
                    }
                    group_header_terms[group_key] = group_headers
                    group_header_scores[group_key] = len(
                        query_terms.intersection(group_headers)
                    )
                    group_titles = {
                        token
                        for chunk_id in chunk_ids
                        for token in tokenize(
                            str(self._by_id[chunk_id].metadata.get("table_title") or "")
                        )
                    }
                    group_title_scores[group_key] = len(
                        query_terms.intersection(group_titles)
                    )
                    group_terms = {
                        token
                        for chunk_id in chunk_ids
                        for token in tokenize(
                            " ".join(
                                value
                                for value in (
                                    str(self._by_id[chunk_id].metadata.get("title_subject") or ""),
                                    str(self._by_id[chunk_id].metadata.get("table_title") or ""),
                                    self._by_id[chunk_id].search_text
                                    or self._by_id[chunk_id].text,
                                    " ".join(
                                        str(value)
                                        for value in (
                                            self._by_id[chunk_id].metadata.get("table_headers")
                                            or []
                                        )
                                    ),
                                    " ".join(
                                        str(value)
                                        for value in (
                                            self._by_id[chunk_id].metadata.get("table_cells")
                                            or []
                                        )
                                    ),
                                )
                                if value
                            )
                        )
                    }
                    group_term_scores[group_key] = len(
                        query_terms.intersection(group_terms)
                    )
                candidate_groups = {
                    self._table_group_by_chunk[chunk_id]
                    for chunk_id in table_candidates
                    if chunk_id in self._table_group_by_chunk
                }
                preferred_group = max(
                    candidate_groups,
                    key=lambda group: (
                        group_key_scores.get(group, 0),
                        group_header_scores.get(group, 0),
                        group_term_scores.get(group, 0),
                        -next(
                            (
                                rank
                                for rank, chunk_id in enumerate(
                                    table_candidates, start=1
                                )
                                if self._table_group_by_chunk.get(chunk_id) == group
                            ),
                            len(table_candidates) + 1,
                        ),
                        group,
                    ),
                )
                if is_registration_schedule_query(query):
                    mentioned_years = set(re.findall(r"\b(?:19|20)\d{2}\b", query))
                    for group in candidate_groups:
                        if group[0] != preferred_group[0]:
                            continue
                        group_text = " ".join(
                            " ".join(
                                str(value)
                                for value in (
                                    self._by_id[chunk_id].metadata.get("title_subject"),
                                    self._by_id[chunk_id].metadata.get("table_title"),
                                )
                                if value
                            )
                            for chunk_id in self._table_groups[group]
                        )
                        group_years = set(
                            re.findall(r"\b(?:19|20)\d{2}\b", group_text)
                        )
                        header_terms = group_header_terms.get(group, set())
                        has_registration_time_headers = (
                            {"đăng", "ký"}.issubset(header_terms)
                            and bool({"thời", "gian"}.intersection(header_terms))
                        )
                        if (
                            has_registration_time_headers
                            or group_title_scores.get(group, 0) >= 2
                        ) and (
                            not mentioned_years
                            or mentioned_years.issubset(group_years)
                        ):
                            schedule_group_keys.add(group)
                    if not schedule_group_keys:
                        schedule_group_keys.add(preferred_group)
                else:
                    schedule_group_keys.add(preferred_group)
                ordered_ids.extend(
                    chunk_id
                    for chunk_id in table_candidates
                    if self._table_group_by_chunk.get(chunk_id) in schedule_group_keys
                    and chunk_id not in ordered_ids
                )
        elif not is_document_identity_query(query):
            ordered_ids.extend(table_phrase_ids)
        ordered_ids.extend(
            chunk_id
            for chunk_id, _ in fused
            if chunk_id not in ordered_ids
            and (
                not is_payment_deadline_query(query)
                or _matches_payment_deadline_chunk(query, self._by_id[chunk_id])
            )
        )
        ordered_ids.extend(
            chunk_id
            for chunk_id in table_search_ids
            if chunk_id not in ordered_ids
        )
        limit = top_k or settings.final_top_k
        selected_ids = ordered_ids[:limit]
        neighbor_ids = self._neighbor_text_chunk_ids(selected_ids, filters)
        expanded_ids = self._related_table_chunk_ids(selected_ids)
        related_document_ids = {
            related_id
            for chunk_id in selected_ids
            for related_id in self._related_documents.get(
                self._by_id[chunk_id].document_id, set()
            )
        }
        expanded_ids.extend(
            chunk.chunk_id
            for chunk in self.bundle.chunks
            if chunk.document_id in related_document_ids
            and _matches(chunk, filters)
            and chunk.chunk_id not in expanded_ids
        )
        if schedule_group_keys:
            expanded_ids.extend(
                chunk_id
                for group in sorted(schedule_group_keys)
                for chunk_id in sorted(
                    self._table_groups[group],
                    key=lambda value: (
                        self._by_id[value].page,
                        self._by_id[value].metadata.get("table_row_index", 0),
                        value,
                    ),
                )
                if chunk_id not in expanded_ids
            )
        if is_payment_deadline_query(query):
            expanded_ids = [
                chunk_id
                for chunk_id in expanded_ids
                if _matches_payment_deadline_chunk(query, self._by_id[chunk_id])
            ]
        result_ids = list(
            dict.fromkeys([*selected_ids, *neighbor_ids, *expanded_ids])
        )
        by_rank = fused_score_by_id
        results = []
        for chunk_id in result_ids:
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
            "latency_seconds": {
                "bm25": bm25_latency,
                "faiss": faiss_latency,
                "rrf": rrf_latency,
                "hybrid": perf_counter() - total_started,
            },
        }
