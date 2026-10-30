"""CPU-compatible SentenceTransformer embeddings with normalized vectors."""

from __future__ import annotations

import logging

import numpy as np

from src.config import settings

LOGGER = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str | None = None, batch_size: int | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu sentence-transformers; cài các phụ thuộc bằng pip install -r requirements.txt"
            ) from exc
        self._model = SentenceTransformer(self.model_name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self.batch_size,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError("Embedding trả về vector sai chiều hoặc chứa giá trị không hữu hạn")
        return vectors
