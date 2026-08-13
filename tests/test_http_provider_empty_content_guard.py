from promptlens.models.config import ProviderConfig
from promptlens.providers.http import HTTPProvider


def _provider() -> HTTPProvider:
    return HTTPProvider(
        ProviderConfig(
            name="http",
            model="test-model",
            endpoint="http://localhost:11434/api/generate",
        )
    )


def test_extract_error_message_from_nested_error() -> None:
    provider = _provider()

    assert (
        provider._extract_error_message({"error": {"message": "model overloaded"}})
        == "model overloaded"
    )


def test_extract_error_message_from_top_level_detail() -> None:
    provider = _provider()

    assert provider._extract_error_message({"detail": "service unavailable"}) == "service unavailable"


def test_extract_error_message_returns_empty_when_absent() -> None:
    provider = _provider()

    assert provider._extract_error_message({"foo": "bar"}) == ""
