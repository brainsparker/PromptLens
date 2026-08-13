import pytest
from pydantic import ValidationError

from promptlens.models.config import ProviderConfig


def test_provider_config_accepts_http_endpoint() -> None:
    config = ProviderConfig(name="http", model="m", endpoint="http://localhost:11434/api/generate")
    assert config.endpoint == "http://localhost:11434/api/generate"


def test_provider_config_rejects_non_http_scheme() -> None:
    with pytest.raises(ValidationError, match="endpoint must use http or https scheme"):
        ProviderConfig(name="http", model="m", endpoint="file:///tmp/model.sock")


def test_provider_config_rejects_missing_host() -> None:
    with pytest.raises(ValidationError, match="endpoint must include a host"):
        ProviderConfig(name="http", model="m", endpoint="https:///api/generate")
