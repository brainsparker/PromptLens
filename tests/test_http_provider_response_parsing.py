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


def test_extract_content_from_ollama_response() -> None:
    provider = _provider()
    data = {"response": "Hello from Ollama"}

    assert provider._extract_content(data) == "Hello from Ollama"


def test_extract_content_from_openai_chat_string_content() -> None:
    provider = _provider()
    data = {
        "choices": [
            {
                "message": {
                    "content": "Hello from OpenAI chat",
                }
            }
        ]
    }

    assert provider._extract_content(data) == "Hello from OpenAI chat"


def test_extract_content_from_openai_chat_content_parts() -> None:
    provider = _provider()
    data = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "Part 1 "},
                        {"type": "text", "text": "Part 2"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                    ]
                }
            }
        ]
    }

    assert provider._extract_content(data) == "Part 1 Part 2"


def test_extract_content_from_openai_responses_api_output_text() -> None:
    provider = _provider()
    data = {"output_text": "Hello from Responses API"}

    assert provider._extract_content(data) == "Hello from Responses API"


def test_extract_content_from_openai_responses_api_output_array() -> None:
    provider = _provider()
    data = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Hello "},
                    {"type": "output_text", "text": "world"},
                ],
            }
        ]
    }

    assert provider._extract_content(data) == "Hello world"


def test_extract_content_returns_empty_for_unknown_shape() -> None:
    provider = _provider()

    assert provider._extract_content({"foo": "bar"}) == ""
