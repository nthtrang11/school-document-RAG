"""OpenAI-compatible LLM client with explicit missing-key and HTTP errors."""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from src.config import settings
from src.rag.citation_validator import validate_citations
from src.rag.prompt import SYSTEM_PROMPT, user_prompt

LOGGER = logging.getLogger(__name__)


def _safe_error_text(value: str) -> str:
    text = " ".join(value.split())
    if settings.llm_api_key:
        text = text.replace(settings.llm_api_key, "[redacted]")
    text = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [redacted]", text)
    text = re.sub(r"\bgsk_[A-Za-z0-9_-]+", "[redacted]", text)
    return text[:500]


def _http_error_context(
    response_body: str, headers: Any | None = None
) -> str:
    error_fields: list[str] = []
    try:
        response_data = json.loads(response_body)
    except json.JSONDecodeError:
        response_data = None
    error_data = response_data.get("error") if isinstance(response_data, dict) else None
    if isinstance(error_data, dict):
        for field in ("type", "code", "message"):
            value = error_data.get(field)
            if isinstance(value, (str, int, float)):
                safe_value = _safe_error_text(str(value))
                if safe_value:
                    error_fields.append(f"{field}={safe_value}")
    elif response_body:
        error_fields.append(f"body={_safe_error_text(response_body)}")

    endpoint_host = urlsplit(settings.llm_endpoint).hostname or "unknown"
    request_id = None
    if headers:
        request_id = headers.get("x-request-id") or headers.get("request-id")
    context = (
        f"provider={settings.llm_provider}, model={settings.llm_model}, "
        f"endpoint_host={endpoint_host}"
    )
    if error_fields:
        context += ", " + ", ".join(error_fields)
    if request_id:
        request_id = _safe_error_text(request_id)
        context += f", request_id={request_id}"
    return context


def _http_error_detail(status_code: int, response_body: str) -> str:
    if status_code == 403 and re.search(
        r"\berror code:\s*1010\b", response_body, flags=re.IGNORECASE
    ):
        return (
            "Cloudflare chặn request theo client signature (1010); lỗi này "
            "không chứng minh API key hoặc model bị từ chối."
        )
    if status_code in {401, 403}:
        return "Kiểm tra LLM_API_KEY và quyền truy cập model."
    if status_code == 404:
        return "Kiểm tra LLM_ENDPOINT và LLM_MODEL."
    if status_code == 429:
        return "Nhà cung cấp đang giới hạn yêu cầu; hãy thử lại sau."
    if status_code >= 500:
        return "Nhà cung cấp có thể đang gián đoạn; hãy thử lại sau."
    return "Kiểm tra cấu hình endpoint và yêu cầu LLM."


def _request_groq(request_payload: dict[str, Any]) -> dict[str, Any]:
    base_url = settings.llm_endpoint.removesuffix("/chat/completions")
    transient_retries = 0
    with OpenAI(
        api_key=settings.llm_api_key,
        base_url=base_url,
        timeout=settings.llm_timeout,
        max_retries=0,
        default_headers={"User-Agent": "school-document-rag/1.0"},
    ) as client:
        while True:
            try:
                response = client.chat.completions.create(**request_payload)
                return response.model_dump()
            except APIStatusError as exc:
                if exc.status_code in {502, 503, 504} and transient_retries == 0:
                    transient_retries += 1
                    time.sleep(1)
                    continue
                response_body = exc.response.text
                context = _http_error_context(response_body, exc.response.headers)
                LOGGER.warning(
                    "LLM HTTP request failed: status=%s, %s",
                    exc.status_code,
                    context,
                )
                detail = _http_error_detail(exc.status_code, response_body)
                reason = exc.response.reason_phrase
                raise RuntimeError(
                    f"LLM trả về HTTP {exc.status_code} ({reason}). {detail} "
                    f"Chi tiết: {context}"
                ) from exc
            except APITimeoutError as exc:
                raise RuntimeError(
                    f"LLM không phản hồi trong {settings.llm_timeout} giây. "
                    "Kiểm tra kết nối mạng, LLM_ENDPOINT và trạng thái nhà cung cấp."
                ) from exc
            except APIConnectionError as exc:
                raise RuntimeError(
                    f"Không thể kết nối tới LLM tại "
                    f"{urlsplit(settings.llm_endpoint).hostname or 'unknown'}."
                ) from exc


def generate_answer(question: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    if not evidence:
        return {
            "answer": "Chưa tìm thấy bằng chứng phù hợp trong chỉ mục hiện tại.",
            "sources": [],
            "citation_registry_valid": True,
            "invalid_citations": [],
            "generated": False,
        }
    if not settings.llm_api_key or not settings.llm_model:
        raise RuntimeError("Cần cấu hình LLM_API_KEY và LLM_MODEL để tạo câu trả lời.")
    if settings.llm_provider.casefold() not in {"openai_compatible", "groq"}:
        raise ValueError(f"LLM_PROVIDER chưa được hỗ trợ: {settings.llm_provider}")
    request_payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(question, evidence)},
        ],
        "temperature": 0.1,
    }
    if settings.llm_reasoning_effort:
        request_payload["reasoning_effort"] = settings.llm_reasoning_effort
    if settings.llm_provider.casefold() == "groq":
        response_data = _request_groq(request_payload)
        try:
            answer = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Phản hồi LLM không đúng cấu trúc chat completion") from exc
        validated = validate_citations(str(answer), evidence)
        return {**validated, "generated": True}

    payload = json.dumps(request_payload).encode("utf-8")
    request = urllib.request.Request(
        settings.llm_endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    transient_retries = 0
    while True:
        try:
            with urllib.request.urlopen(
                request, timeout=settings.llm_timeout
            ) as response:
                response_body = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            if exc.code in {502, 503, 504} and transient_retries == 0:
                transient_retries += 1
                exc.close()
                time.sleep(1)
                continue
            try:
                response_body = exc.read().decode("utf-8", errors="replace")
            except OSError:
                response_body = ""
            finally:
                exc.close()
            context = _http_error_context(response_body, exc.headers)
            LOGGER.warning(
                "LLM HTTP request failed: status=%s, %s",
                exc.code,
                context,
            )
            detail = _http_error_detail(exc.code, response_body)
            raise RuntimeError(
                f"LLM trả về HTTP {exc.code} ({exc.reason}). {detail} "
                f"Chi tiết: {context}"
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise RuntimeError(
                    f"LLM không phản hồi trong {settings.llm_timeout} giây. "
                    "Kiểm tra kết nối mạng, LLM_ENDPOINT và trạng thái nhà cung cấp."
                ) from exc
            raise RuntimeError(f"Không thể gọi LLM: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"LLM không phản hồi trong {settings.llm_timeout} giây. "
                "Kiểm tra kết nối mạng, LLM_ENDPOINT và trạng thái nhà cung cấp."
            ) from exc
    try:
        response_data = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Không thể gọi LLM: {exc}") from exc
    try:
        answer = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Phản hồi LLM không đúng cấu trúc chat completion") from exc
    validated = validate_citations(str(answer), evidence)
    return {**validated, "generated": True}
