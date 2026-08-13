"""Tests for runtime retry/timeout settings resolution on providers."""

from promptlens.models.config import ProviderConfig
from promptlens.models.result import ModelResponse
from promptlens.providers.base import BaseProvider


class DummyProvider(BaseProvider):
    """Minimal provider used to test BaseProvider helper methods."""

    async def generate(self, prompt: str, tools=None, **kwargs):  # pragma: no cover
        return ModelResponse(
            content=prompt,
            model=self.config.model,
            provider=self.provider_name,
            latency_ms=0.0,
        )

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return 0.0

    @property
    def provider_name(self) -> str:
        return "dummy"


def test_get_retry_settings_uses_runtime_values() -> None:
    provider = DummyProvider(ProviderConfig(name="dummy", model="dummy-model"))

    attempts, delay = provider.get_retry_settings(
        {"retry_attempts": 5, "retry_delay_seconds": 2.5}
    )

    assert attempts == 5
    assert delay == 2.5


def test_get_retry_settings_clamps_invalid_values() -> None:
    provider = DummyProvider(ProviderConfig(name="dummy", model="dummy-model"))

    attempts, delay = provider.get_retry_settings(
        {"retry_attempts": 0, "retry_delay_seconds": -1}
    )

    assert attempts == 1
    assert delay == 0.0


def test_get_timeout_seconds_prefers_runtime_override() -> None:
    provider = DummyProvider(
        ProviderConfig(name="dummy", model="dummy-model", timeout=42)
    )

    assert provider.get_timeout_seconds({"timeout_seconds": 12}) == 12


def test_get_timeout_seconds_falls_back_to_provider_config() -> None:
    provider = DummyProvider(
        ProviderConfig(name="dummy", model="dummy-model", timeout=42)
    )

    assert provider.get_timeout_seconds({"timeout_seconds": 0}) == 42
    assert provider.get_timeout_seconds({}) == 42
