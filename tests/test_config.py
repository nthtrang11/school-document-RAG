from src.config import Settings


def test_google_gemini_endpoint_defaults_to_low_reasoning_effort(monkeypatch):
    monkeypatch.setenv(
        "LLM_ENDPOINT",
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    )
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)

    settings = Settings.from_env()

    assert settings.llm_reasoning_effort == "low"


def test_custom_reasoning_effort_overrides_gemini_default(monkeypatch):
    monkeypatch.setenv(
        "LLM_ENDPOINT",
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    )
    monkeypatch.setenv("LLM_REASONING_EFFORT", "medium")

    settings = Settings.from_env()

    assert settings.llm_reasoning_effort == "medium"


def test_non_google_endpoint_does_not_get_provider_specific_reasoning(monkeypatch):
    monkeypatch.setenv("LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)

    settings = Settings.from_env()

    assert settings.llm_reasoning_effort is None
