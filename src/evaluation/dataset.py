"""Load evaluation questions without promoting draft answers to ground truth."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {"question_id", "question"}


def load_questions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy bộ câu hỏi: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Thiếu cột evaluation: {', '.join(sorted(missing))}")
    return frame


def parse_identifiers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return list(dict.fromkeys(str(item).strip() for item in parsed if str(item).strip()))
    if isinstance(parsed, dict):
        return list(dict.fromkeys(str(item).strip() for item in parsed if str(item).strip()))
    return list(
        dict.fromkeys(
            item.strip()
            for item in text.replace(";", "|").split("|")
            if item.strip()
        )
    )


def relevant_document_ids(row: pd.Series) -> set[str]:
    values = parse_identifiers(row.get("relevant_document_ids"))
    values.extend(parse_identifiers(row.get("expected_document_ids")))
    values.extend(parse_identifiers(row.get("relevant_document_id")))
    return set(values)


def _relevance_scores(row: pd.Series, document_ids: set[str]) -> dict[str, float] | None:
    raw = row.get("relevance_scores", "")
    if raw is None or not str(raw).strip():
        return None
    try:
        values = json.loads(str(raw))
    except json.JSONDecodeError:
        return None
    if not isinstance(values, dict):
        return None
    try:
        scores = {str(key): float(value) for key, value in values.items()}
    except (TypeError, ValueError):
        return None
    if any(score < 0 for score in scores.values()) or not set(scores).issubset(document_ids):
        return None
    return scores


def question_skip_reason(row: pd.Series) -> str | None:
    if not str(row.get("question_id", "")).strip():
        return "missing_question_id"
    if not str(row.get("question", "")).strip():
        return "missing_question"
    if str(row.get("verification_status", "")).casefold() != "verified":
        return "needs_manual_ground_truth"
    if not str(row.get("verified_by", "")).strip():
        return "ground_truth_invalid:missing_verified_by"
    verified_at = str(row.get("verified_at", "")).strip()
    if not verified_at:
        return "ground_truth_invalid:missing_verified_at"
    try:
        datetime.fromisoformat(verified_at.replace("Z", "+00:00"))
    except ValueError:
        return "ground_truth_invalid:invalid_verified_at"
    if not relevant_document_ids(row):
        return "ground_truth_invalid:missing_relevant_document_ids"
    pages = parse_identifiers(
        row.get("relevant_pages")
        or row.get("expected_pages")
        or row.get("relevant_page")
    )
    if not pages:
        return "ground_truth_invalid:missing_relevant_pages"
    raw_scores = row.get("relevance_scores", "")
    if raw_scores is not None and str(raw_scores).strip():
        scores = _relevance_scores(row, relevant_document_ids(row))
        if scores is None:
            return "ground_truth_invalid:invalid_relevance_scores"
    return None


def is_verified(row: pd.Series) -> bool:
    return question_skip_reason(row) is None


def query_type(row: pd.Series) -> str:
    return str(row.get("query_type") or row.get("question_type") or "unspecified").strip()


def relevance_scores(row: pd.Series) -> dict[str, float]:
    ids = relevant_document_ids(row)
    scores = _relevance_scores(row, ids)
    if scores is None:
        return {identifier: 1.0 for identifier in ids}
    return {identifier: scores.get(identifier, 0.0) for identifier in ids}
