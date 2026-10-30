"""FAISS inner-product index over L2-normalized embeddings."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class FaissIndex:
    def __init__(self, vectors: np.ndarray, index: faiss.Index | None = None):
        if vectors.ndim != 2:
            raise ValueError("Vectors phải có 2 chiều")
        self.dimension = int(vectors.shape[1])
        self.index = index or faiss.IndexFlatIP(self.dimension)
        if index is None and len(vectors):
            normalized = np.asarray(vectors, dtype=np.float32).copy()
            faiss.normalize_L2(normalized)
            self.index.add(normalized)

    def search(self, vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1).copy()
        if query.shape[1] != self.dimension:
            raise ValueError("Embedding truy vấn không cùng chiều index")
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))
        return [
            (int(index), float(score))
            for index, score in zip(indices[0], scores[0], strict=True)
            if index >= 0
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "FaissIndex":
        index = faiss.read_index(str(path))
        return cls(np.empty((0, index.d), dtype=np.float32), index=index)
