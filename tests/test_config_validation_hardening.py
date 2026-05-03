from pydantic import ValidationError

from promptlens.models.config import ModelConfig, OutputConfig


def test_model_temperature_must_be_between_0_and_2() -> None:
    try:
        ModelConfig(name="m", provider="openai", model="gpt-4", temperature=2.1)
        assert False, "Expected ValidationError"
    except ValidationError:
        assert True


def test_model_max_tokens_must_be_positive() -> None:
    try:
        ModelConfig(name="m", provider="openai", model="gpt-4", max_tokens=0)
        assert False, "Expected ValidationError"
    except ValidationError:
        assert True


def test_output_formats_must_not_be_empty() -> None:
    try:
        OutputConfig(formats=[])
        assert False, "Expected ValidationError"
    except ValidationError:
        assert True
