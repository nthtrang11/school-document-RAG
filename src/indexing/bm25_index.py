"""BM25 corpus tokenization and persistence-safe token storage."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.models import ChunkRecord

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN.findall(text)]


class BM25Index:
    def __init__(self, chunks: list[ChunkRecord], tokenized: list[list[str]] | None = None):
        self.chunks = chunks
        self.tokens = tokenized if tokenized is not None else [tokenize(c.text) for c in chunks]
        if len(self.tokens) != len(self.chunks):
            raise ValueError("BM25 corpus token mapping không khớp chunks")
        self._index = BM25Okapi(self.tokens) if self.tokens else None

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if not self._index:
            return []
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        scores = self._index.get_scores(list(query_tokens))
        ranked = sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))
        matched = [
            (index, float(score))
            for index, score in ranked
            if query_tokens.intersection(self.tokens[index])
        ]
        return matched[:top_k]

    def save_tokens(self, path: Path) -> None:
        path.write_text(json.dumps(self.tokens, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, chunks: list[ChunkRecord], path: Path) -> "BM25Index":
        tokens = json.loads(path.read_text(encoding="utf-8"))
        return cls(chunks, tokenized=tokens)
