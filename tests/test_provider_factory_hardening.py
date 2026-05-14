from promptlens.models.config import ModelConfig
from promptlens.providers.factory import get_provider


def test_get_provider_normalizes_whitespace_and_case():
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
