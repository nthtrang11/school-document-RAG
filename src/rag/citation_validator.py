"""Validate citation references against the evidence registry."""

from __future__ import annotations

import re
from typing import Any

_CITATION = re.compile(
    r"\[(?P<square>E\d+)\]"
    r"|【(?P<reference>E\d+)(?:†L\d+(?:-L\d+)?)?】"
)


def normalize_citations(
    answer: str, valid_citation_ids: set[str]
) -> tuple[str, list[str]]:
    """Normalize supported citation spellings only when their IDs are mapped."""
    citation_ids: list[str] = []

    def replace(match: re.Match[str]) -> str:
        citation_id = match.group("square") or match.group("reference")
        if citation_id is None:
            return match.group(0)
        citation_ids.append(citation_id)
        if citation_id in valid_citation_ids:
            return f"[{citation_id}]"
        return match.group(0)

    return _CITATION.sub(replace, answer), citation_ids


def _has_valid_source(item: dict[str, Any]) -> bool:
    page = item.get("page")
    page_count = item.get("page_count")
    has_source = bool(item.get("source_path") or item.get("source_url"))
    has_valid_page = isinstance(page, int) and page >= 1
    if page_count is not None:
        has_valid_page = has_valid_page and isinstance(page_count, int) and page <= page_count
    return bool(
        has_source
        and has_valid_page
        and item.get("document_id")
        and item.get("chunk_id")
        and str(item.get("evidence_text") or "").strip()
    )


def validate_citations(answer: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    known = {
        item["evidence_id"]: item
        for item in evidence
        if item.get("evidence_id") and _has_valid_source(item)
    }
    normalized_answer, cited_ids = normalize_citations(answer, set(known))
    invalid = sorted({citation for citation in cited_ids if citation not in known})
    valid_ids = list(dict.fromkeys(citation for citation in cited_ids if citation in known))
    return {
        "answer": normalized_answer,
        "sources": [
            {
                **known[citation],
                "citation_id": citation,
                "source": known[citation].get("source_path")
                or known[citation].get("document_id"),
                "document_title": known[citation].get("title"),
                "evidence_text": known[citation].get("evidence_text", ""),
            }
            for citation in valid_ids
        ],
        "invalid_citations": invalid,
        "citation_registry_valid": not invalid,
    }


def retrieval_only_response(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep retrieved sources navigable when answer generation is unavailable."""
    if not evidence:
        answer = "Chưa tìm thấy bằng chứng phù hợp trong chỉ mục hiện tại."
    else:
        citations = " ".join(
            f"[{item['evidence_id']}]"
            for item in evidence
            if item.get("evidence_id")
        )
        answer = f"LLM chưa khả dụng. Bằng chứng đã truy xuất: {citations}"
    return {
        **validate_citations(answer, evidence),
        "generated": False,
    }
