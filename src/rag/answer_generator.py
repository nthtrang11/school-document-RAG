"""OpenAI-compatible LLM client with explicit missing-key and HTTP errors."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from src.config import settings
from src.rag.citation_validator import validate_citations
from src.rag.prompt import SYSTEM_PROMPT, user_prompt


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
    if settings.llm_provider != "openai_compatible":
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
    try:
        with urllib.request.urlopen(request, timeout=settings.llm_timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
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
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Không thể gọi LLM: {exc}") from exc
    try:
        answer = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Phản hồi LLM không đúng cấu trúc chat completion") from exc
    validated = validate_citations(str(answer), evidence)
    return {**validated, "generated": True}
