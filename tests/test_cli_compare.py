"""Tests for the promptlens compare CLI command."""

import pytest
from click.testing import CliRunner

from promptlens.cli import cli
from promptlens.models.result import (
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)


def _make_eval(test_case_id, model="model-a", score=None, error=None):
    return EvaluationResult(
        test_case_id=test_case_id,
        query="What is the answer?",
        expected_behavior="Answers correctly",
        model_response=ModelResponse(
            content="The answer is 42." if not error else "",
            model=model,
            provider="anthropic",
            latency_ms=1000.0,
            cost_usd=0.002,
            error=error,
        ),
        judge_score=(
            JudgeScore(
                score=score,
                explanation="Judged.",
                judge_model="judge-model",
                judge_provider="anthropic",
            )
            if score is not None
            else None
        ),
    )


def _write_run(tmp_path, name, run_id, evals):
    run = RunResult(
        run_id=run_id,
        golden_set_name="golden-set",
        models_tested=sorted({e.model_response.model for e in evals}),
        results=evals,
    )
    path = tmp_path / name
    path.write_text(run.model_dump_json(), encoding="utf-8")
    return str(path)


@pytest.fixture
def runner():
    return CliRunner()


class TestCompareCommand:
    def test_clean_comparison_exits_zero(self, runner, tmp_path):
        baseline = _write_run(tmp_path, "base.json", "base", [_make_eval("tc-1", score=4)])
        current = _write_run(tmp_path, "cur.json", "cur", [_make_eval("tc-1", score=4)])

        result = runner.invoke(cli, ["compare", baseline, current])

        assert result.exit_code == 0
        assert "No regressions" in result.output

    def test_regression_without_gate_exits_zero(self, runner, tmp_path):
        baseline = _write_run(tmp_path, "base.json", "base", [_make_eval("tc-1", score=5)])
        current = _write_run(tmp_path, "cur.json", "cur", [_make_eval("tc-1", score=2)])

        result = runner.invoke(cli, ["compare", baseline, current])

        assert result.exit_code == 0
        assert "regression(s)" in result.output

    def test_regression_with_gate_exits_three(self, runner, tmp_path):
        baseline = _write_run(tmp_path, "base.json", "base", [_make_eval("tc-1", score=5)])
        current = _write_run(tmp_path, "cur.json", "cur", [_make_eval("tc-1", score=2)])

        result = runner.invoke(
            cli, ["compare", baseline, current, "--fail-on-regression"]
        )

        assert result.exit_code == 3

    def test_gate_passes_on_clean_run(self, runner, tmp_path):
        baseline = _write_run(tmp_path, "base.json", "base", [_make_eval("tc-1", score=4)])
        current = _write_run(tmp_path, "cur.json", "cur", [_make_eval("tc-1", score=5)])

        result = runner.invoke(
            cli, ["compare", baseline, current, "--fail-on-regression"]
        )

        assert result.exit_code == 0

    def test_threshold_option_relaxes_gate(self, runner, tmp_path):
        baseline = _write_run(tmp_path, "base.json", "base", [_make_eval("tc-1", score=5)])
        current = _write_run(tmp_path, "cur.json", "cur", [_make_eval("tc-1", score=4)])

        result = runner.invoke(
            cli,
            ["compare", baseline, current, "--fail-on-regression", "--threshold", "2"],
        )

        assert result.exit_code == 0

    def test_markdown_report_written(self, runner, tmp_path):
        baseline = _write_run(tmp_path, "base.json", "base", [_make_eval("tc-1", score=5)])
        current = _write_run(tmp_path, "cur.json", "cur", [_make_eval("tc-1", score=2)])
        md_path = tmp_path / "diff.md"

        result = runner.invoke(
            cli, ["compare", baseline, current, "--markdown", str(md_path)]
        )

        assert result.exit_code == 0
        content = md_path.read_text(encoding="utf-8")
        assert "# PromptLens Run Comparison" in content
        assert "tc-1" in content

    def test_json_report_written(self, runner, tmp_path):
        baseline = _write_run(tmp_path, "base.json", "base", [_make_eval("tc-1", score=5)])
        current = _write_run(tmp_path, "cur.json", "cur", [_make_eval("tc-1", score=2)])
        json_path = tmp_path / "diff.json"

        result = runner.invoke(
            cli, ["compare", baseline, current, "--json", str(json_path)]
        )

        assert result.exit_code == 0
        assert '"baseline_run_id"' in json_path.read_text(encoding="utf-8")

    def test_invalid_input_file_exits_one(self, runner, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json", encoding="utf-8")
        other = _write_run(tmp_path, "cur.json", "cur", [_make_eval("tc-1", score=4)])

        result = runner.invoke(cli, ["compare", str(bad), other])

        assert result.exit_code == 1
        assert "Comparison failed" in result.output

    def test_missing_file_rejected_by_click(self, runner, tmp_path):
        other = _write_run(tmp_path, "cur.json", "cur", [_make_eval("tc-1", score=4)])

        result = runner.invoke(cli, ["compare", str(tmp_path / "nope.json"), other])

        assert result.exit_code == 2
