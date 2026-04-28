from pathlib import Path

from click.testing import CliRunner

from promptlens.cli import cli


def test_run_rejects_empty_config_file(tmp_path: Path) -> None:
    config = tmp_path / "empty.yaml"
    config.write_text("", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["run", str(config), "--dry-run"])

    assert result.exit_code == 1
    assert "configuration file is empty" in result.output


def test_run_rejects_non_mapping_top_level_config(tmp_path: Path) -> None:
    config = tmp_path / "list.yaml"
    config.write_text("- a\n- b\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["run", str(config), "--dry-run"])

    assert result.exit_code == 1
    assert "top-level YAML must be a mapping/object" in result.output
