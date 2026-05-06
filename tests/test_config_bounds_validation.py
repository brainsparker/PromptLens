import pytest

from promptlens.models.config import RunConfig


def _base_config() -> dict:
    return {
        "golden_set": "./examples/golden_sets/customer_support.yaml",
        "models": [
            {
                "name": "GPT",
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 512,
            }
        ],
    }


@pytest.mark.parametrize(
    "path,value",
    [
        (("models", 0, "temperature"), -0.1),
        (("models", 0, "temperature"), 2.1),
        (("models", 0, "max_tokens"), 0),
        (("judge", "temperature"), 9),
        (("execution", "parallel_requests"), 0),
        (("execution", "retry_attempts"), -1),
        (("execution", "retry_delay_seconds"), -0.5),
        (("execution", "timeout_seconds"), 0),
    ],
)
def test_run_config_rejects_invalid_bounds(path, value):
    config = _base_config()
    target = config
    for key in path[:-1]:
        if isinstance(key, int):
            target = target[key]
        else:
            target = target.setdefault(key, {})
    target[path[-1]] = value

    with pytest.raises(Exception):
        RunConfig(**config)
