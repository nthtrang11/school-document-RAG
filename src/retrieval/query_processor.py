"""Preserve exact identifiers and normalize text for hybrid search."""

from __future__ import annotations

import re
import unicodedata

from src.indexing.bm25_index import tokenize

_NUMBER_SYMBOL = re.compile(
    r"(?<!\d)(?!\d{1,2}/\d{1,2}/\d{4}\b)\d{1,5}\s*/\s*[A-ZĐ0-9][A-ZĐ0-9./-]*\b",
    re.I,
)
_QUERY_EQUIVALENTS = {
    "đánh giá": ("lấy ý kiến phản hồi",),
    "thời hạn": ("thời gian", "đến hết ngày"),
    "thời gian": ("thời hạn", "đến hết ngày"),
    "hạn đóng": ("thời gian thực hiện", "thời gian nộp", "nộp phí"),
    "thời hạn đóng": ("thời gian thực hiện", "thời gian nộp", "nộp phí"),
    "ngày cuối cùng": ("đến hết ngày", "trước ngày", "kết thúc"),
    "đóng bảo hiểm y tế": ("nộp phí bảo hiểm y tế", "thu phí BHYT"),
    "mức đóng": ("số tiền đóng", "mức nộp"),
    "mức nộp": ("mức đóng", "số tiền đóng"),
    "bảo hiểm y tế": ("BHYT",),
    "hoàn thành": ("thực hiện",),
    "kết thúc": ("đến hết ngày",),
}
_REGISTRATION_SCHEDULE_EXPANSIONS = (
    "lịch đăng ký học",
    "thời gian đăng ký học phần",
    "đợt đăng ký học phần",
    "lớp học phần chính khóa",
    "môn tự chọn",
    "đăng ký bổ sung học phần",
    "hủy học phần",
    "đăng ký học vượt",
)
_TUITION_DEADLINE_EXPANSIONS = (
    "thời hạn nộp học phí",
    "hạn đóng học phí",
    "gia hạn thời gian nộp học phí",
    "học phí được gia hạn đến hết ngày",
    "thời gian thu học phí",
)


def is_registration_schedule_query(query: str | None) -> bool:
    if not query:
        return False
    query_text = " ".join(query.casefold().split())
    asks_registration = any(
        phrase in query_text for phrase in ("đăng ký", "đăng ky", "registration")
    )
    asks_schedule_detail = any(
        phrase in query_text
        for phrase in (
            "lịch",
            "thời gian",
            "đợt",
            "học kỳ",
            "học phần",
            "tự chọn",
            "bổ sung",
            "theo từng khóa",
            "theo từng khoá",
        )
    )
    return asks_registration and asks_schedule_detail


def is_document_identity_query(query: str | None) -> bool:
    if not query:
        return False
    query_text = " ".join(query.casefold().split())
    return any(
        phrase in query_text
        for phrase in ("văn bản nào", "số hiệu", "ký hiệu văn bản", "ban hành theo")
    )


def _expanded_query(query: str) -> str:
    normalized = query.strip()
    folded = normalized.casefold()
    expansions = [
        equivalent
        for phrase, equivalents in _QUERY_EQUIVALENTS.items()
        if phrase in folded
        for equivalent in equivalents
    ]
    if is_registration_schedule_query(normalized):
        expansions.extend(_REGISTRATION_SCHEDULE_EXPANSIONS)
    if is_tuition_deadline_query(normalized):
        expansions.extend(_TUITION_DEADLINE_EXPANSIONS)
    return " ".join(dict.fromkeys([normalized, *expansions]))


def is_payment_deadline_query(query: str | None) -> bool:
    if not query:
        return False
    query_text = " ".join(query.casefold().split())
    asks_when = any(
        phrase in query_text
        for phrase in ("hạn", "ngày cuối", "ngày nào", "khi nào", "kết thúc", "thời gian")
    )
    asks_payment = any(
        phrase in query_text for phrase in ("đóng", "nộp", "thu phí", "thanh toán")
    )
    return asks_when and asks_payment


def is_tuition_deadline_query(query: str | None) -> bool:
    if not query:
        return False
    query_text = " ".join(query.casefold().split())
    return any(
        phrase in query_text
        for phrase in ("học phí", "hoc phi", "tiền học", "tien hoc")
    )


def is_health_insurance_payment_query(query: str | None) -> bool:
    if not query:
        return False
    query_text = " ".join(query.casefold().split())
    return any(
        phrase in query_text
        for phrase in ("bảo hiểm y tế", "bao hiem y te", "bhyt")
    )


def is_monthly_rate_query(query: str | None) -> bool:
    if not query:
        return False
    query_text = " ".join(query.casefold().split())
    asks_rate = any(
        phrase in query_text
        for phrase in ("mức đóng", "mức nộp", "số tiền", "mức phí")
    )
    asks_monthly = any(
        phrase in query_text
        for phrase in ("mỗi tháng", "theo tháng", "hàng tháng", "đồng/tháng")
    )
    return asks_rate and asks_monthly


def process_query(query: str) -> dict[str, object]:
    exact_symbols = [re.sub(r"\s+", "", value).upper() for value in _NUMBER_SYMBOL.findall(query)]
    normalized = unicodedata.normalize("NFC", query).strip()
    return {
        "original": query,
        "normalized": normalized,
        "tokens": tokenize(normalized),
        "exact_symbols": exact_symbols,
        "bm25_query": _expanded_query(normalized),
    }
