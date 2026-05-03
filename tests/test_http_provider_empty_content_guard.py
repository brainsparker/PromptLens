from unittest.mock import patch

import pytest

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


@pytest.mark.asyncio
async def test_generate_returns_error_when_content_unparseable() -> None:
    provider = _provider()

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        async def json(self):
            return {"foo": "bar"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class _FakeSession:
        def post(self, *args, **kwargs):
            return _FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    with patch("promptlens.providers.http.aiohttp.ClientSession", return_value=_FakeSession()):
        result = await provider.generate("hello")

    assert result.error is not None
    assert "did not contain parseable text content" in result.error
    assert result.content == ""
