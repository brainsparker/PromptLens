import pytest
from pydantic import ValidationError

from promptlens.models.config import ModelConfig, OutputConfig, RunConfig


def test_model_config_rejects_invalid_temperature():
    with pytest.raises(ValidationError):
        ModelConfig(
            name="bad-temp",
            provider="openai",
            model="gpt-4o",
            temperature=2.5,
            max_tokens=100,
        )


def test_model_config_rejects_non_positive_max_tokens():
    with pytest.raises(ValidationError):
        ModelConfig(
            name="bad-tokens",
            provider="openai",
            model="gpt-4o",
            temperature=0.5,
            max_tokens=0,
        )


def test_output_config_normalizes_and_deduplicates_formats():
    cfg = OutputConfig(formats=["HTML", "json", "markdown", "md", "JSON"])
    assert cfg.formats == ["html", "json", "md"]


def test_output_config_rejects_unsupported_format():
    with pytest.raises(ValidationError):
        OutputConfig(formats=["xml"])


def test_run_config_requires_at_least_one_model():
    with pytest.raises(ValidationError):
        RunConfig(golden_set="./tests.yaml", models=[])
