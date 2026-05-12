import pytest
from pydantic import ValidationError

from promptlens.models.config import JudgeConfig, ModelConfig


def test_model_config_rejects_out_of_range_temperature() -> None:
    with pytest.raises(ValidationError, match="temperature must be between 0.0 and 2.0"):
        ModelConfig(
            name="Test",
            provider="openai",
            model="gpt-4o-mini",
            temperature=2.5,
        )


def test_model_config_rejects_non_positive_max_tokens() -> None:
    with pytest.raises(ValidationError, match="max_tokens must be >= 1"):
        ModelConfig(
            name="Test",
            provider="openai",
            model="gpt-4o-mini",
            max_tokens=0,
        )


def test_model_config_rejects_blank_provider() -> None:
    with pytest.raises(ValidationError, match="must be a non-empty string"):
        ModelConfig(name="Test", provider="  ", model="gpt-4o-mini")


def test_judge_config_rejects_empty_criteria() -> None:
    with pytest.raises(ValidationError, match="criteria must contain at least one item"):
        JudgeConfig(criteria=[])


def test_judge_config_normalizes_criteria_whitespace() -> None:
    cfg = JudgeConfig(criteria=[" accuracy ", "helpfulness"])

    assert cfg.criteria == ["accuracy", "helpfulness"]
