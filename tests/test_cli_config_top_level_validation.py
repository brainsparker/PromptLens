from click.testing import CliRunner

from promptlens.cli import cli


def test_run_rejects_empty_config_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["run", str(config_path), "--dry-run"])

    assert result.exit_code == 1
    assert "empty YAML file" in result.output


def test_run_rejects_non_mapping_top_level_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["run", str(config_path), "--dry-run"])

    assert result.exit_code == 1
    assert "expected a YAML mapping at top level" in result.output
