"""Validate citation references against the evidence registry."""

from __future__ import annotations

import re
from typing import Any

_CITATION = re.compile(r"\[(E\d+)\]")


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
        and item.get("evidence_text")
    )


def validate_citations(answer: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    known = {
        item["evidence_id"]: item
        for item in evidence
        if item.get("evidence_id") and _has_valid_source(item)
    }
    cited_ids = _CITATION.findall(answer)
    invalid = sorted({citation for citation in cited_ids if citation not in known})
    valid_ids = list(dict.fromkeys(citation for citation in cited_ids if citation in known))
    sanitized = _CITATION.sub(
        lambda match: match.group(0) if match.group(1) in known else "",
        answer,
    )
    return {
        "answer": re.sub(r" {2,}", " ", sanitized).strip(),
        "sources": [
            {
                **known[citation],
                "citation_id": citation,
                "source": known[citation].get("source_path")
                or known[citation].get("document_id"),
                "document_title": known[citation].get("title"),
                "evidence_text": known[citation].get("text", ""),
            }
            for citation in valid_ids
        ],
        "invalid_citations": invalid,
        "citation_registry_valid": not invalid,
    }
