from pathlib import Path

import pytest

from promptlens.cli import _load_config_data


def test_load_config_data_rejects_empty_file(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("")

    with pytest.raises(ValueError, match="empty"):
        _load_config_data(str(config))


def test_load_config_data_rejects_non_mapping_top_level(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("- item\n- item2\n")

    with pytest.raises(ValueError, match="top level"):
        _load_config_data(str(config))


def test_load_config_data_accepts_mapping(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("golden_set: ./tests.yaml\nmodels: []\n")

    loaded = _load_config_data(str(config))

    assert loaded["golden_set"] == "./tests.yaml"
