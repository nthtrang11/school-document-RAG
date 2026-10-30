import json
import urllib.error
from unittest.mock import MagicMock

import pytest

from src.config import settings
from src.rag import answer_generator


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
    monkeypatch.setattr(settings, "llm_api_key", "test-api-key")
    monkeypatch.setattr(settings, "llm_model", "test-model")
    monkeypatch.setattr(settings, "llm_timeout", 180)
    monkeypatch.setattr(settings, "llm_reasoning_effort", "low")


def test_generate_answer_sends_bearer_key_and_configured_timeout(monkeypatch):
    _configure_llm(monkeypatch)
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(
        {"choices": [{"message": {"content": "Câu trả lời [E1]."}}]}
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
