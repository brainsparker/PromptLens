import pytest
from pydantic import ValidationError

from promptlens.models.config import ExecutionConfig, ModelConfig, OutputConfig, RunConfig


def test_model_config_temperature_bounds() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(name="m", provider="openai", model="gpt", temperature=2.5)


def test_execution_config_parallel_requests_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ExecutionConfig(parallel_requests=0)


def test_output_formats_are_normalized_to_lowercase() -> None:
    cfg = OutputConfig(formats=["HTML", "JSON"])
    assert cfg.formats == ["html", "json"]


def test_output_formats_reject_unsupported_values() -> None:
    with pytest.raises(ValidationError):
        OutputConfig(formats=["html", "xml"])


def test_run_config_requires_at_least_one_model() -> None:
    with pytest.raises(ValidationError):
        RunConfig(golden_set="./tests.yaml", models=[])
