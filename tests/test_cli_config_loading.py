import click
import pytest

from promptlens.cli import _apply_cli_overrides, _load_config_data


def test_load_config_data_rejects_empty_file(tmp_path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("")

    with pytest.raises(click.ClickException, match="Configuration file is empty"):
        _load_config_data(str(config_path))


def test_load_config_data_rejects_non_mapping_root(tmp_path):
    config_path = tmp_path / "list_root.yaml"
    config_path.write_text("- just\n- a\n- list\n")

    with pytest.raises(
        click.ClickException, match="Configuration root must be a mapping/object"
    ):
        _load_config_data(str(config_path))


def test_apply_cli_overrides_creates_output_mapping_when_missing():
    config_data = {"golden_set": "tests.yaml", "models": []}

    updated = _apply_cli_overrides(
        config_data=config_data,
        golden_set=None,
        output_dir="./custom_results",
    )

    assert updated["output"]["directory"] == "./custom_results"


def test_apply_cli_overrides_rejects_non_mapping_output():
    config_data = {"golden_set": "tests.yaml", "models": [], "output": "bad"}

    with pytest.raises(
        click.ClickException, match="Invalid configuration: 'output' must be an object"
    ):
        _apply_cli_overrides(
            config_data=config_data,
            golden_set=None,
            output_dir="./custom_results",
        )
