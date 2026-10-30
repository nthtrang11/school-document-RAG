"""Build a bounded, labelled context from retrieved evidence."""

from __future__ import annotations

from typing import Any

from src.config import settings


def build_context(results: list[dict[str, Any]], token_budget: int | None = None) -> list[dict[str, Any]]:
    budget = token_budget or settings.token_budget
    used = 0
    evidence: list[dict[str, Any]] = []
    for item in results:
        chunk = item["chunk"]
        text = chunk.text.strip()
        estimated_tokens = max(1, len(text) // 4)
        if used + estimated_tokens > budget:
            remaining_chars = max(0, (budget - used) * 4)
            if remaining_chars < 80:
                break
            text = text[:remaining_chars]
            estimated_tokens = max(1, len(text) // 4)
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
                "source": chunk.metadata.get("source_url") or chunk.metadata.get("source_path"),
                "text": text,
                "evidence_text": text,
                "chunk_id": chunk.chunk_id,
                "page_count": chunk.metadata.get("page_count"),
                "bbox": chunk.metadata.get("bbox"),
                "estimated_tokens": estimated_tokens,
            }
        )
        used += estimated_tokens
    return evidence
