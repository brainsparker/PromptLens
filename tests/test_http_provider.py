from promptlens.models.config import ProviderConfig
from promptlens.providers.http import HTTPProvider


def _provider() -> HTTPProvider:
    return HTTPProvider(
        ProviderConfig(name="http", model="test-model", endpoint="http://localhost:11434/api/generate")
    )


def test_extract_content_ollama_shape() -> None:
    provider = _provider()
    assert provider._extract_content({"response": "hello"}) == "hello"


def test_extract_content_openai_text_completion_shape() -> None:
    provider = _provider()
    payload = {"choices": [{"text": "hello from choices"}]}
    assert provider._extract_content(payload) == "hello from choices"


def test_extract_content_openai_chat_message_string_shape() -> None:
    provider = _provider()
    payload = {"choices": [{"message": {"content": "chat reply"}}]}
    assert provider._extract_content(payload) == "chat reply"


def test_extract_content_openai_chat_message_parts_shape() -> None:
    provider = _provider()
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "Part A"},
                        {"type": "text", "text": " + Part B"},
                    ]
                }
            }
        ]
    }
    assert provider._extract_content(payload) == "Part A + Part B"


def test_extract_content_top_level_content_parts_shape() -> None:
    provider = _provider()
    payload = {"content": [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}]}
    assert provider._extract_content(payload) == "foobar"
