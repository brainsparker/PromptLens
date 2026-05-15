from click.testing import CliRunner

from promptlens.cli import cli


def test_run_rejects_empty_config_file(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("", encoding="utf-8")

    result = CliRunner().invoke(cli, ["run", str(config), "--dry-run"])

    assert result.exit_code == 1
    assert "config file is empty" in result.output


def test_run_rejects_non_mapping_top_level_yaml(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    result = CliRunner().invoke(cli, ["run", str(config), "--dry-run"])

    assert result.exit_code == 1
    assert "top-level YAML must be a mapping/object" in result.output
