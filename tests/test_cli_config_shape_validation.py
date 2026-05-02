import pytest

from promptlens.cli import _normalize_config_data


def test_normalize_config_data_rejects_empty_config() -> None:
    with pytest.raises(ValueError, match="Configuration file is empty"):
        _normalize_config_data(None, "config.yaml")


def test_normalize_config_data_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="expected a YAML mapping at top level"):
        _normalize_config_data([{"golden_set": "tests.yaml"}], "config.yaml")


def test_normalize_config_data_accepts_mapping() -> None:
    data = {"golden_set": "tests.yaml", "models": []}
    assert _normalize_config_data(data, "config.yaml") == data
