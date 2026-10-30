"""Preserve exact identifiers and normalize text for hybrid search."""

from __future__ import annotations

import re
import unicodedata

from src.indexing.bm25_index import tokenize

_NUMBER_SYMBOL = re.compile(
    r"(?<!\d)(?!\d{1,2}/\d{1,2}/\d{4}\b)\d{1,5}\s*/\s*[A-ZĐ0-9][A-ZĐ0-9./-]*\b",
    re.I,
)


def process_query(query: str) -> dict[str, object]:
    exact_symbols = [re.sub(r"\s+", "", value).upper() for value in _NUMBER_SYMBOL.findall(query)]
    normalized = unicodedata.normalize("NFC", query).strip()
    return {
        "original": query,
        "normalized": normalized,
        "tokens": tokenize(normalized),
        "exact_symbols": exact_symbols,
    }
