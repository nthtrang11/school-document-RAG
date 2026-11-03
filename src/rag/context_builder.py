"""Build a bounded, labelled context from retrieved evidence."""

from __future__ import annotations

import re
from typing import Any

from src.config import settings
from src.indexing.bm25_index import tokenize
from src.retrieval.query_processor import (
    is_monthly_rate_query,
    is_payment_deadline_query,
    is_registration_schedule_query,
)


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_EXPLICIT_DATE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_TIME_CUE = re.compile(
    r"\b(?:thời gian|thời hạn|hạn|đến hết|trước ngày|chậm nhất|"
    r"hoàn thành|từ ngày|kết thúc)\b",
    re.IGNORECASE,
)
_SECTION_HEADING = re.compile(
    r"^\s*(?:(?:[IVXLCDM]+)\s*[/.)]|(?:mục|điều)\s+\d+|"
    r"\d+(?:\.\d+){0,3}\.?)\s*",
    re.IGNORECASE,
)
_GENERIC_QUERY_TERMS = {
    "bao", "bằng", "biết", "các", "cho", "chưa", "có", "của", "đã",
    "đâu", "để", "đến", "điều", "được", "gì", "hãy", "khi", "không",
    "là", "nào", "này", "nêu", "như", "nội", "ở", "phần", "ra", "sao",
    "thế", "thì", "thông", "tin", "trong", "từ", "và", "về", "với",
    "học", "phí", "thời", "gian", "đăng", "ký",
}


def _select_text_segments(text: str, query: str | None) -> list[str]:
    if not query:
        return [text.strip()] if text.strip() else []
    terms = {
        token
        for token in tokenize(query)
        if token not in _GENERIC_QUERY_TERMS and len(token) > 1
    }
    if len(terms) < 2:
        return [text.strip()] if text.strip() else []

    segments = [
        segment.strip()
        for segment in _SENTENCE_BOUNDARY.split(text)
        if segment.strip()
    ]
    scored = [
        (sum(term in set(tokenize(segment)) for term in terms), index)
        for index, segment in enumerate(segments)
    ]
    matching = [
        index
        for score, index in scored
        if score >= 2 and score / len(terms) >= 0.15
    ]
    if is_payment_deadline_query(query):
        temporal_query_terms = {
            "hạn", "thời", "gian", "ngày", "cuối", "cùng",
            "đóng", "nộp", "thu", "phí", "khi", "nào",
        }
        topic_terms = terms - temporal_query_terms
        text_terms = set(tokenize(text))
        if not topic_terms or topic_terms.intersection(text_terms):
            for index, segment in enumerate(segments):
                if not _EXPLICIT_DATE.search(segment):
                    continue
                nearby = range(max(0, index - 1), min(len(segments), index + 2))
                cue_indexes = [
                    nearby_index
                    for nearby_index in nearby
                    if _TIME_CUE.search(segments[nearby_index])
                ]
                if not cue_indexes:
                    continue
                matching.append(index)
                matching.extend(cue_indexes)
                for cue_index in cue_indexes:
                    if cue_index > 0 and _SECTION_HEADING.match(
                        segments[cue_index - 1]
                    ):
                        matching.append(cue_index - 1)
    if not matching:
        return [text.strip()] if text.strip() else []
    return [segments[index] for index in sorted(set(matching))]


def _table_columns_for_query(
    headers: list[str],
    cells: list[str],
    query: str | None,
    cell_states: list[str] | None = None,
) -> list[int]:
    populated = [
        index
        for index, cell in enumerate(cells)
        if cell.strip()
        or (
            cell_states is not None
            and index < len(cell_states)
            and cell_states[index] in {
                "empty_or_unknown",
                "covered_by_merged_cell",
                "text_not_extracted",
            }
        )
    ]
    if not query:
        return populated

    query_text = " ".join(query.casefold().split())
    if any(
        phrase in query_text
        for phrase in (
            "mỗi khóa",
            "mỗi khoá",
            "theo khóa",
            "theo khoá",
            "từng khóa",
            "từng khoá",
            "mỗi đối tượng",
            "theo đối tượng",
            "từng đối tượng",
        )
    ):
        return populated

    amount_query = any(
        phrase in query_text
        for phrase in ("mức đóng", "mức tiền", "mức phí", "số tiền", "bao nhiêu tiền")
    )
    if amount_query:
        return [
            index
            for index in populated
            if "thời gian" not in headers[index].casefold()
        ]

    query_terms = set(tokenize(query))
    selected: set[int] = set()
    for index, header in enumerate(headers):
        normalized_header = " ".join(header.casefold().split())
        if not normalized_header:
            continue
        if "thời gian" in normalized_header:
            if "đăng ký" in normalized_header and any(
                phrase in query_text
                for phrase in ("đăng ký", "địa điểm", "ở đâu", "nơi đăng ký")
            ):
                selected.add(index)
            if "tính học phí" in query_text and "tính học phí" in normalized_header:
                selected.add(index)
            if "thu học phí" in query_text and "thu học phí" in normalized_header:
                selected.add(index)
            if (
                "học phí" in query_text
                and "tính học phí" not in query_text
                and "thu học phí" not in query_text
                and "học phí" in normalized_header
            ):
                selected.add(index)
            continue
        if any(
            phrase in query_text
            for phrase in ("đối tượng", "điều kiện", "sinh viên", "áp dụng")
        ) and any(term in normalized_header for term in ("đối tượng", "điều kiện")):
            selected.add(index)
            continue
        header_terms = {
            term
            for term in tokenize(header)
            if term not in _GENERIC_QUERY_TERMS and len(term) > 1
        }
        if header_terms.intersection(query_terms):
            selected.add(index)

    if not selected:
        return populated

    mentions_key = re.search(r"\bkh[oó]a\s+\d+\b", query_text) is not None
    mentions_system = any(
        phrase in query_text
        for phrase in ("hệ", "kỹ sư", "cử nhân", "kỹ thuật", "công nghệ")
    )
    if mentions_key or mentions_system:
        for index in range(len(cells)):
            cell_text = " ".join(cells[index].casefold().split())
            if mentions_key and re.search(r"\bkh[oó]a\s+\d+\b", cell_text):
                selected.add(index)
            if mentions_system and any(
                phrase in cell_text
                for phrase in ("hệ ", "kỹ sư", "cử nhân", "kỹ thuật", "công nghệ")
            ):
                selected.add(index)
            if (
                mentions_system
                and cell_states is not None
                and index < len(cell_states)
                and cell_states[index]
                in {
                    "empty_or_unknown",
                    "covered_by_merged_cell",
                    "text_not_extracted",
                }
            ):
                selected.add(index)

    label_column = next(
        (
            index
            for index in populated
            if any(
                term in " ".join(tokenize(cells[index]))
                for term in (
                    "đợt", "lớp", "học phần", "thực tập", "đồ án",
                    "khóa", "khoá", "đối tượng",
                )
            )
        ),
        None,
    )
    if label_column is not None:
        selected.add(label_column)
    return [index for index in populated if index in selected]


def _table_row_context(
    headers: list[str],
    cells: list[str],
    states: list[str | None] | None,
    query: str | None,
) -> tuple[list[int], str, list[str]]:
    indices = _table_columns_for_query(headers, cells, query, states)
    selected_headers = [headers[index] for index in indices]
    selected_cells = [cells[index] for index in indices]
    selected_states = [
        states[index] if states is not None and index < len(states) else None
        for index in indices
    ]
    lines = []
    for header, cell, state in zip(
        selected_headers, selected_cells, selected_states
    ):
        line = f"{header}: {cell}"
        if state == "inherited_from_merged_cell":
            line += " (giá trị kế thừa từ ô gộp; không phải giá trị ghi trực tiếp)"
        elif not cell and state == "empty_or_unknown":
            line += " [ô trống hoặc chưa xác định trong bảng]"
        elif not cell and state == "covered_by_merged_cell":
            line += " [phạm vi của ô gộp; không phải ô giá trị độc lập]"
        elif not cell and state == "text_not_extracted":
            line += " [có thể có nội dung nhưng chưa trích xuất được]"
        if cell or state in {
            "empty_or_unknown",
            "covered_by_merged_cell",
            "text_not_extracted",
        }:
            lines.append(line)
    return indices, "\n".join(lines), [cell for cell in selected_cells if cell]


def build_context(
    results: list[dict[str, Any]],
    token_budget: int | None = None,
    query: str | None = None,
) -> list[dict[str, Any]]:
    budget = token_budget or settings.token_budget
    used = 0
    evidence: list[dict[str, Any]] = []
    table_group_costs: dict[tuple[str, str], int] = {}
    table_group_seen: dict[tuple[str, str], set[str]] = {}
    for item in results:
        chunk = item["chunk"]
        table_id = chunk.metadata.get("table_id")
        if not table_id:
            continue
        group_id = chunk.metadata.get("table_group_id") or table_id
        group_key = (chunk.document_id, str(group_id))
        if chunk.chunk_id in table_group_seen.setdefault(group_key, set()):
            continue
        table_group_seen[group_key].add(chunk.chunk_id)
        headers = chunk.metadata.get("table_headers") or []
        cells = chunk.metadata.get("table_cells")
        states = chunk.metadata.get("table_cell_states")
        if (
            isinstance(headers, list)
            and isinstance(cells, list)
            and len(headers) == len(cells)
        ):
            _, row_text, _ = _table_row_context(
                headers,
                cells,
                states if isinstance(states, list) else None,
                query,
            )
        else:
            row_text = chunk.search_text or chunk.text
        title = str(chunk.metadata.get("table_title") or "")
        table_group_costs[group_key] = table_group_costs.get(group_key, 0) + max(
            1,
            (len(row_text) + len(title) + 32 + 3) // 4,
        )
    table_group_admitted: dict[tuple[str, str], bool] = {}
    payment_deadline_query = is_payment_deadline_query(query)
    registration_schedule_query = is_registration_schedule_query(query)
    for item in results:
        chunk = item["chunk"]
        table_id = chunk.metadata.get("table_id")
        table_group = (
            (
                chunk.document_id,
                str(chunk.metadata.get("table_group_id") or table_id),
            )
            if table_id
            else None
        )
        table_headers = chunk.metadata.get("table_headers")
        table_cells = chunk.metadata.get("table_cells")
        table_bboxes = chunk.metadata.get("table_bboxes")
        table_cell_states = chunk.metadata.get("table_cell_states")
        table_cell_spans = chunk.metadata.get("table_cell_spans")
        is_card_duration_table = isinstance(table_headers, list) and any(
            "thời gian sử dụng thẻ" in str(header).casefold()
            for header in table_headers
        )
        if is_card_duration_table and payment_deadline_query:
            continue
        if is_card_duration_table and is_monthly_rate_query(query):
            if not isinstance(table_cells, list) or not any(
                "/tháng" in str(cell).casefold() for cell in table_cells
            ):
                continue
        table_indices: list[int] | None = None
        if (
            isinstance(table_headers, list)
            and isinstance(table_cells, list)
            and len(table_headers) == len(table_cells)
        ):
            table_indices, text, evidence_segments = _table_row_context(
                table_headers,
                table_cells,
                table_cell_states if isinstance(table_cell_states, list) else None,
                query,
            )
            selected_headers = [table_headers[index] for index in table_indices]
            selected_cells = [table_cells[index] for index in table_indices]
            selected_states = [
                table_cell_states[index]
                if isinstance(table_cell_states, list)
                and index < len(table_cell_states)
                else None
                for index in table_indices
            ]
            selected_spans = [
                table_cell_spans[index]
                if isinstance(table_cell_spans, list)
                and index < len(table_cell_spans)
                else None
                for index in table_indices
            ]
            selected_bboxes = [
                table_bboxes[index]
                if isinstance(table_bboxes, list) and index < len(table_bboxes)
                else None
                for index in table_indices
            ]
            if chunk.metadata.get("table_title"):
                text = f"Tiêu đề bảng: {chunk.metadata['table_title']}\n{text}"
        else:
            evidence_segments = _select_text_segments(chunk.text, query)
            text = "\n".join(evidence_segments)
            selected_headers = table_headers
            selected_cells = table_cells
            selected_states = table_cell_states
            selected_spans = table_cell_spans
            selected_bboxes = table_bboxes
        if not text.strip():
            continue
        estimated_tokens = max(1, len(text) // 4)
        if table_group:
            if table_group not in table_group_admitted:
                table_group_admitted[table_group] = (
                    used + table_group_costs.get(table_group, estimated_tokens)
                    <= budget
                    or registration_schedule_query
                )
                if table_group_admitted[table_group]:
                    used += table_group_costs.get(table_group, estimated_tokens)
            if not table_group_admitted[table_group]:
                continue
        elif used + estimated_tokens > budget:
            continue
        evidence.append(
            {
                "evidence_id": f"E{len(evidence) + 1}",
                "citation_id": f"E{len(evidence) + 1}",
                "document_id": chunk.document_id,
                "title": chunk.metadata.get("title_subject") or "Chưa xác định tiêu đề",
                "document_title": chunk.metadata.get("title_subject") or "Chưa xác định tiêu đề",
                "number_symbol": chunk.metadata.get("number_symbol"),
                "issue_date": chunk.metadata.get("issue_date"),
                "page": chunk.page,
                "section": chunk.section,
                "source_url": chunk.metadata.get("source_url"),
                "source_path": chunk.metadata.get("source_path"),
                "source_filename": str(
                    chunk.metadata.get("source_path") or ""
                ).replace("\\", "/").rsplit("/", 1)[-1],
                "source": chunk.metadata.get("source_url") or chunk.metadata.get("source_path"),
                "text": text,
                "evidence_text": "\n".join(evidence_segments),
                "evidence_segments": evidence_segments,
                "chunk_id": chunk.chunk_id,
                "page_count": chunk.metadata.get("page_count"),
                "bbox": None,
                "table_headers": selected_headers,
                "table_cells": selected_cells,
                "table_bboxes": selected_bboxes,
                "table_cell_states": selected_states,
                "table_cell_spans": selected_spans,
                "table_title": chunk.metadata.get("table_title"),
                "table_id": chunk.metadata.get("table_id"),
                "table_group_id": chunk.metadata.get("table_group_id"),
                "table_row_index": chunk.metadata.get("table_row_index"),
                "table_bbox": chunk.metadata.get("table_bbox"),
                "table_header_rows": chunk.metadata.get("table_header_rows"),
                "table_header_spans": chunk.metadata.get("table_header_spans"),
                "table_locations": (
                    [
                        {
                            "text": cell,
                            "bbox": bbox,
                            **({"state": state} if state is not None else {}),
                            **({"span": span} if span is not None else {}),
                        }
                        for cell, bbox, state, span in zip(
                            selected_cells,
                            selected_bboxes or [],
                            selected_states or [None] * len(selected_cells or []),
                            selected_spans or [None] * len(selected_cells or []),
                        )
                        if cell
                    ]
                    if chunk.metadata.get("table_id")
                    else None
                ),
                "estimated_tokens": estimated_tokens,
            }
        )
        if not table_group:
            used += estimated_tokens
    return evidence
