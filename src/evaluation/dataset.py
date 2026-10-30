"""Read and validate the manually verified retrieval evaluation corpus."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "question_id",
    "question",
    "question_type",
    "ground_truth_answer",
    "relevant_document_id",
    "relevant_chunk_ids",
    "relevant_page",
    "relevant_section",
    "relevant_chunk_ids_fixed_size",
    "relevant_chunk_ids_recursive",
}


def load_questions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy bộ câu hỏi: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Thiếu cột evaluation: {', '.join(sorted(missing))}")
    return frame


def is_verified(row: pd.Series) -> bool:
    return bool(
        str(row.get("ground_truth_answer", "")).strip()
        and str(row.get("relevant_document_id", "")).strip()
        and str(row.get("relevant_chunk_ids_fixed_size", "")).strip()
        and str(row.get("relevant_chunk_ids_recursive", "")).strip()
        and str(row.get("verification_status", "")).casefold() == "verified"
    )
