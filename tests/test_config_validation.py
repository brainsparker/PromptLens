"""Tests for configuration hardening and validation."""

import pytest
from pydantic import ValidationError

from promptlens.models.config import ModelConfig, OutputConfig, RunConfig


def test_model_config_rejects_invalid_temperature() -> None:
    with pytest.raises(ValidationError, match="temperature must be between 0 and 2"):
        ModelConfig(
            name="Bad Temp",
            provider="openai",
            model="gpt-4o-mini",
            temperature=2.5,
        )


def test_model_config_rejects_non_positive_max_tokens() -> None:
    with pytest.raises(ValidationError, match="max_tokens must be greater than 0"):
        ModelConfig(
            name="Bad Tokens",
            provider="openai",
            model="gpt-4o-mini",
            max_tokens=0,
        )


def test_output_formats_are_normalized_and_deduplicated() -> None:
    output = OutputConfig(formats=["JSON", "html", "json"])

    assert output.formats == ["json", "html"]


def test_output_formats_reject_unknown_values() -> None:
    with pytest.raises(ValidationError, match="unsupported output format"):
        OutputConfig(formats=["pdf"])


def test_run_config_requires_at_least_one_model() -> None:
    with pytest.raises(ValidationError, match="models must include at least one model configuration"):
        RunConfig(golden_set="tests.yaml", models=[])
