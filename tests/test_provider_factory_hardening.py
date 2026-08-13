from promptlens.models.config import ModelConfig
from promptlens.providers.factory import get_provider
from promptlens.providers.http import HTTPProvider


def test_http_provider_supports_legacy_endpoint_in_additional_params() -> None:
    provider = get_provider(
        ModelConfig(
            name="Local model",
            provider="http",
            model="llama3.1:8b",
            additional_params={
                "endpoint": "http://localhost:11434/api/generate",
                "temperature": 0.2,
            },
        )
    )

    assert isinstance(provider, HTTPProvider)
    assert provider.endpoint == "http://localhost:11434/api/generate"
    assert "endpoint" not in provider.config.additional_params
    assert provider.config.additional_params["temperature"] == 0.2


def test_http_provider_accepts_top_level_endpoint() -> None:
    provider = get_provider(
        ModelConfig(
            name="Local model",
            provider="http",
            model="llama3.1:8b",
            endpoint="http://localhost:11434/api/generate",
        )
    )

    assert isinstance(provider, HTTPProvider)
    assert provider.endpoint == "http://localhost:11434/api/generate"


def test_get_provider_normalizes_whitespace_and_case(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = ModelConfig(
        name="OpenAI",
        provider="  OpenAI  ",
        model="gpt-4o-mini",
    )

    provider = get_provider(model)

    assert provider.config.name == "openai"


def test_get_provider_rejects_blank_provider_name():
    model = ModelConfig(
        name="Bad",
        provider="   ",
        model="gpt-4o-mini",
    )

    try:
        get_provider(model)
        assert False, "Expected ValueError for blank provider name"
    except ValueError as exc:
        msg = str(exc)
        assert "cannot be empty" in msg
        assert "Available providers:" in msg
