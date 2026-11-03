import json
import urllib.error
from unittest.mock import MagicMock

import pytest

from src.config import settings
from src.rag import answer_generator
from src.rag.prompt import SYSTEM_PROMPT, user_prompt


_EVIDENCE = [
    {
        "evidence_id": "E1",
        "citation_id": "E1",
        "document_id": "doc-1",
        "chunk_id": "chunk-1",
        "page": 1,
        "page_count": 1,
        "source_path": "source.pdf",
        "evidence_text": "Bằng chứng kiểm thử.",
        "text": "Bằng chứng kiểm thử.",
        "title": "Tài liệu",
        "number_symbol": None,
        "issue_date": None,
        "section": None,
    }
]


def _configure_llm(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(settings, "llm_api_key", "test-api-key")
    monkeypatch.setattr(settings, "llm_model", "test-model")
    monkeypatch.setattr(settings, "llm_timeout", 180)
    monkeypatch.setattr(settings, "llm_reasoning_effort", "low")


def test_prompt_places_citations_on_supported_atomic_claims_not_group_headings():
    assert "trình bày mỗi thông tin thực tế thành một ý riêng" in SYSTEM_PROMPT
    assert "Không đặt citation ở heading" in SYSTEM_PROMPT
    assert "không gom citation ở cuối câu trả lời" in SYSTEM_PROMPT
    assert "theo dạng [E1]" in SYSTEM_PROMPT
    assert "Đánh số nhóm theo thứ tự logic, thời gian hoặc chủ đề" in SYSTEM_PROMPT
    assert "Với câu hỏi đơn giản, trả lời trực tiếp, ngắn gọn" in SYSTEM_PROMPT
    assert "không đưa hoạt động ngoài phạm vi" in SYSTEM_PROMPT
    assert "không dùng số dòng" in SYSTEM_PROMPT


def test_synthesis_answer_keeps_hierarchical_markdown_and_inline_citations(monkeypatch):
    _configure_llm(monkeypatch)
    evidence = [
        {**_EVIDENCE[0], "evidence_id": "E1", "citation_id": "E1"},
        {
            **_EVIDENCE[0],
            "evidence_id": "E2",
            "citation_id": "E2",
            "document_id": "doc-2",
            "chunk_id": "chunk-2",
            "source_path": "source-2.pdf",
            "page": 2,
            "page_count": 2,
        },
    ]
    answer = (
        "### 1. **Đăng ký chính thức**\n\n"
        "- **Sinh viên đúng tiến độ:**\n"
        "  - Từ **25/07/2026 đến 29/07/2026** [E1].\n"
        "- **Đăng ký bổ sung:** Từ **10/08/2026 đến 18/08/2026** [E2]."
    )
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(
        {"choices": [{"message": {"content": answer}}]}
    ).encode()
    monkeypatch.setattr(
        answer_generator.urllib.request, "urlopen", lambda *args, **kwargs: response
    )

    result = answer_generator.generate_answer("Tổng hợp lịch?", evidence)

    assert result["answer"] == answer
    assert [source["citation_id"] for source in result["sources"]] == ["E1", "E2"]


def test_context_provides_evidence_id_and_pdf_page_without_line_references():
    context = user_prompt("Câu hỏi?", _EVIDENCE)

    assert "[E1] Tài liệu" in context
    assert "Trang 1" in context
    assert "L1-L3" not in context


def test_generate_answer_sends_bearer_key_and_configured_timeout(monkeypatch):
    _configure_llm(monkeypatch)
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(
        {"choices": [{"message": {"content": "Câu trả lời 【E1】."}}]}
    ).encode()
    calls = {}
    payloads = []

    def fake_urlopen(request, timeout):
        calls["authorization"] = request.get_header("Authorization")
        calls["timeout"] = timeout
        payloads.append(json.loads(request.data))
        return response

    monkeypatch.setattr(answer_generator.urllib.request, "urlopen", fake_urlopen)

    result = answer_generator.generate_answer("Câu hỏi?", _EVIDENCE)

    assert payloads[0]["reasoning_effort"] == "low"

    assert calls == {"authorization": "Bearer test-api-key", "timeout": 180}
    assert result["answer"] == "Câu trả lời [E1]."
    assert result["sources"][0]["citation_id"] == "E1"


def test_gemini_openai_compatible_response_uses_shared_citation_normalizer(monkeypatch):
    _configure_llm(monkeypatch)
    monkeypatch.setattr(
        settings,
        "llm_endpoint",
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    )
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(
        {"choices": [{"message": {"content": "Câu trả lời 【E1†L1-L3】."}}]}
    ).encode()
    monkeypatch.setattr(
        answer_generator.urllib.request,
        "urlopen",
        lambda *args, **kwargs: response,
    )

    result = answer_generator.generate_answer("Câu hỏi?", _EVIDENCE)

    assert result["answer"] == "Câu trả lời [E1]."
    assert result["sources"][0]["citation_id"] == "E1"


def test_groq_citation_line_style_is_normalized_before_rendering(monkeypatch):
    _configure_llm(monkeypatch)
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(
        settings,
        "llm_endpoint",
        "https://api.groq.com/openai/v1/chat/completions",
    )
    response = MagicMock()
    response.model_dump.return_value = {
        "choices": [{"message": {"content": "Lịch bắt đầu sau đó 【E1†L1-L3】."}}]
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.chat.completions.create.return_value = response
    monkeypatch.setattr(answer_generator, "OpenAI", lambda **kwargs: client)
    evidence = [{**_EVIDENCE[0], "page": 7, "page_count": 8}]

    result = answer_generator.generate_answer("Khi nào?", evidence)

    assert result["answer"] == "Lịch bắt đầu sau đó [E1]."
    assert result["sources"][0]["page"] == 7


def test_generate_answer_accepts_groq_provider(monkeypatch):
    _configure_llm(monkeypatch)
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(
        settings,
        "llm_endpoint",
        "https://api.groq.com/openai/v1/chat/completions",
    )
    response = MagicMock()
    response.model_dump.return_value = {
        "choices": [{"message": {"content": "Câu trả lời [E1]."}}]
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.chat.completions.create.return_value = response
    client_options = {}
    monkeypatch.setattr(
        answer_generator,
        "OpenAI",
        lambda **kwargs: client_options.update(kwargs) or client,
    )

    result = answer_generator.generate_answer("Câu hỏi?", _EVIDENCE)

    assert client_options == {
        "api_key": "test-api-key",
        "base_url": "https://api.groq.com/openai/v1",
        "timeout": 180,
        "max_retries": 0,
        "default_headers": {"User-Agent": "school-document-rag/1.0"},
    }
    assert client.chat.completions.create.call_args.kwargs["model"] == "test-model"
    assert result["generated"] is True
    assert result["answer"] == "Câu trả lời [E1]."


def test_groq_http_403_preserves_safe_provider_error_details(monkeypatch, caplog):
    _configure_llm(monkeypatch)
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(
        settings,
        "llm_endpoint",
        "https://api.groq.com/openai/v1/chat/completions",
    )
    headers = {"x-request-id": "request-123"}
    response_body = "error code: 1010"
    from httpx2 import Request, Response

    from openai import APIStatusError

    http_response = Response(
        403,
        headers=headers,
        text=response_body,
        request=Request(
            "POST", "https://api.groq.com/openai/v1/chat/completions"
        ),
    )
    sdk_error = APIStatusError(
        "403 Forbidden", response=http_response, body=response_body
    )
    client = MagicMock()
    client.__enter__.return_value = client
    client.chat.completions.create.side_effect = sdk_error
    monkeypatch.setattr(answer_generator, "OpenAI", lambda **kwargs: client)

    with pytest.raises(RuntimeError, match="HTTP 403") as exc:
        answer_generator.generate_answer("Câu hỏi?", _EVIDENCE)

    message = str(exc.value)
    assert "provider=groq" in message
    assert "model=test-model" in message
    assert "endpoint_host=api.groq.com" in message
    assert "body=error code: 1010" in message
    assert "Cloudflare chặn request theo client signature (1010)" in message
    assert "không chứng minh API key hoặc model bị từ chối" in message
    assert "request_id=request-123" in message
    assert "test-api-key" not in message
    assert "error code: 1010" in caplog.text
    assert "test-api-key" not in caplog.text


def test_generate_answer_omits_provider_specific_reasoning_when_unset(monkeypatch):
    _configure_llm(monkeypatch)
    monkeypatch.setattr(settings, "llm_reasoning_effort", None)
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(
        {"choices": [{"message": {"content": "Câu trả lời [E1]."}}]}
    ).encode()
    payloads = []

    def fake_urlopen(request, timeout):
        payloads.append(json.loads(request.data))
        return response

    monkeypatch.setattr(answer_generator.urllib.request, "urlopen", fake_urlopen)

    answer_generator.generate_answer("Câu hỏi?", _EVIDENCE)

    assert "reasoning_effort" not in payloads[0]


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("The read operation timed out"),
        urllib.error.URLError(TimeoutError("The read operation timed out")),
    ],
)
def test_generate_answer_reports_actionable_timeout(monkeypatch, error):
    _configure_llm(monkeypatch)

    def fake_urlopen(request, timeout):
        raise error

    monkeypatch.setattr(answer_generator.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="LLM không phản hồi trong 180 giây") as exc:
        answer_generator.generate_answer("Câu hỏi?", _EVIDENCE)

    assert "LLM_ENDPOINT" in str(exc.value)


@pytest.mark.parametrize(
    ("status", "reason", "message"),
    [
        (503, "Service Unavailable", "đang gián đoạn"),
        (401, "Unauthorized", "LLM_API_KEY"),
        (429, "Too Many Requests", "giới hạn yêu cầu"),
    ],
)
def test_generate_answer_reports_actionable_http_errors(
    monkeypatch, status, reason, message
):
    _configure_llm(monkeypatch)
    monkeypatch.setattr(answer_generator.time, "sleep", lambda _seconds: None)

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            "https://provider.invalid/chat/completions",
            status,
            reason,
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(answer_generator.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match=f"HTTP {status}") as exc:
        answer_generator.generate_answer("Câu hỏi?", _EVIDENCE)

    assert message in str(exc.value)


def test_generate_answer_retries_transient_service_unavailable(monkeypatch):
    _configure_llm(monkeypatch)
    monkeypatch.setattr(answer_generator.time, "sleep", lambda _seconds: None)
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(
        {"choices": [{"message": {"content": "Câu trả lời [E1]."}}]}
    ).encode()
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(
                "https://provider.invalid/chat/completions",
                503,
                "Service Unavailable",
                hdrs=None,
                fp=None,
            )
        return response

    monkeypatch.setattr(answer_generator.urllib.request, "urlopen", fake_urlopen)

    result = answer_generator.generate_answer("Câu hỏi?", _EVIDENCE)

    assert calls == 2
    assert result["answer"] == "Câu trả lời [E1]."
