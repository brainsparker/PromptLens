"""Tests for CLI config loading and override hardening."""

from pathlib import Path

import pytest

from promptlens.config_loading import apply_run_overrides, load_config_data


def test_load_config_data_rejects_empty_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Configuration file is empty"):
        load_config_data(str(config_path))


def test_load_config_data_rejects_non_mapping_root(tmp_path: Path) -> None:
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Configuration root must be a YAML mapping"):
        load_config_data(str(config_path))


def test_apply_run_overrides_creates_output_mapping_when_missing() -> None:
    config = {
        "golden_set": "tests.yaml",
        "models": [{"name": "m", "provider": "openai", "model": "gpt-4"}],
    }

    merged = apply_run_overrides(config, golden_set=None, output_dir="./out")

    assert merged["output"]["directory"] == "./out"


def test_apply_run_overrides_rejects_invalid_output_section_type() -> None:
    config = {
        "golden_set": "tests.yaml",
        "models": [{"name": "m", "provider": "openai", "model": "gpt-4"}],
        "output": "bad-type",
    }

    with pytest.raises(ValueError, match="'output' section must be a mapping"):
        apply_run_overrides(config, golden_set=None, output_dir="./out")
