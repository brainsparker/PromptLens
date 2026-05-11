from pydantic import ValidationError

from promptlens.models.config import RunConfig


def _base_config(models):
    return {
        "golden_set": "./examples/golden_sets/customer_support.yaml",
        "models": models,
    }


def test_run_config_rejects_duplicate_model_names_case_insensitive():
    config = _base_config(
        [
            {"name": "Claude Fast", "provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
            {"name": "claude fast", "provider": "openai", "model": "gpt-4o-mini"},
        ]
    )

    try:
        RunConfig(**config)
        raise AssertionError("Expected ValidationError for duplicate model names")
    except ValidationError as exc:
        assert "Model names must be unique" in str(exc)


def test_run_config_allows_unique_model_names():
    config = _base_config(
        [
            {"name": "Claude Fast", "provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
            {"name": "GPT Fast", "provider": "openai", "model": "gpt-4o-mini"},
        ]
    )

    run_config = RunConfig(**config)
    assert len(run_config.models) == 2
