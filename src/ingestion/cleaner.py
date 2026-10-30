"""Conservative text normalization that retains Vietnamese content."""

from __future__ import annotations

import re
from collections import Counter

from src.models import PageRecord

_SPACE_RUN = re.compile(r"[^\S\r\n]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_CONTACT_DETAIL = re.compile(
    r"(?<!\d)(?:\+?84|0)(?:[\s.-]?\d){8,10}(?!\d)|"
    r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"
)


def _redact_contact_lines(text: str) -> str:
    safe_lines = []
    for line in text.splitlines():
        match = _CONTACT_DETAIL.search(line)
        if not match:
            safe_lines.append(line)
            continue
        prefix = line[:match.start()].rstrip(" :-,;")
        if prefix:
            safe_lines.append(f"{prefix} [đã lược bỏ thông tin liên hệ]")
    return "\n".join(safe_lines)


def clean_pages(pages: list[PageRecord]) -> tuple[list[PageRecord], list[PageRecord]]:
    normalized: list[PageRecord] = []
    for page in pages:
        text = page.text.replace("\ufffd", " ")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [_SPACE_RUN.sub(" ", line).strip() for line in text.split("\n")]
        text = _MANY_NEWLINES.sub("\n\n", "\n".join(lines)).strip()
        normalized.append(page.model_copy(update={"text": text}))

    line_counts = Counter(
        line
        for page in normalized
        for line in set(page.text.splitlines())
        if len(line) >= 8
    )
    threshold = max(2, (len(normalized) + 1) // 2)
    repeated = {line for line, count in line_counts.items() if count >= threshold}
    cleaned = [
        page.model_copy(
            update={
                "text": _redact_contact_lines("\n".join(
                    line for line in page.text.splitlines() if line not in repeated
                )).strip()
            }
        )
        for page in normalized
    ]
    return normalized, cleaned
