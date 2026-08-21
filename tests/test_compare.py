"""Tests for run-over-run comparison and the compare CLI command."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from promptlens.cli import cli
from promptlens.comparison import (
    STATUS_ADDED,
    STATUS_IMPROVED,
    STATUS_REGRESSED,
    STATUS_REMOVED,
    STATUS_UNCHANGED,
    STATUS_UNSCORED,
    compare_runs,
    comparison_to_markdown,
)
from promptlens.models.result import (
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)


def _make_response(model="model-a", error=None, latency_ms=1000.0, cost_usd=0.01):
    return ModelResponse(
        content="" if error else "The answer is 42.",
        model=model,
        provider="anthropic",
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        error=error,
    )


def _make_eval(
    test_case_id,
    model="model-a",
    score=None,
    error=None,
    latency_ms=1000.0,
    cost_usd=0.01,
    explanation="Judged.",
):
    return EvaluationResult(
        test_case_id=test_case_id,
        query=f"Query for {test_case_id}",
        expected_behavior="Answers correctly",
        model_response=_make_response(
            model=model, error=error, latency_ms=latency_ms, cost_usd=cost_usd
        ),
        judge_score=(
            JudgeScore(
                score=score,
                explanation=explanation,
                judge_model="judge-model",
                judge_provider="anthropic",
            )
            if score is not None
            else None
        ),
    )


def _make_run(results, run_id="run-1", run_name=None, models=None):
    models = models or sorted({r.model_response.model for r in results})
    return RunResult(
        run_id=run_id,
        run_name=run_name,
        timestamp=datetime(2026, 8, 21, 12, 0, 0),
        golden_set_name="golden-set",
        models_tested=models,
        results=results,
        total_cost_usd=sum(r.model_response.cost_usd or 0.0 for r in results),
        total_time_ms=sum(r.model_response.latency_ms for r in results),
    )


class TestCompareRuns:
    def test_classifies_improvements_regressions_and_unchanged(self):
        baseline = _make_run(
            [
                _make_eval("tc1", score=3),
                _make_eval("tc2", score=4),
                _make_eval("tc3", score=5),
            ],
            run_id="base",
        )
        candidate = _make_run(
            [
                _make_eval("tc1", score=5),
                _make_eval("tc2", score=2),
                _make_eval("tc3", score=5),
            ],
            run_id="cand",
        )

        result = compare_runs(baseline, candidate)

        assert result.improved == 1
        assert result.regressed == 1
        assert result.unchanged == 1
        by_id = {c.test_case_id: c for c in result.cases}
        assert by_id["tc1"].status == STATUS_IMPROVED
        assert by_id["tc1"].score_delta == 2
        assert by_id["tc2"].status == STATUS_REGRESSED
        assert by_id["tc2"].score_delta == -2
        assert by_id["tc3"].status == STATUS_UNCHANGED
        assert result.has_regressions is True

    def test_regressions_sorted_first(self):
        baseline = _make_run(
            [_make_eval("a-improves", score=2), _make_eval("z-regresses", score=5)],
            run_id="base",
        )
        candidate = _make_run(
            [_make_eval("a-improves", score=4), _make_eval("z-regresses", score=3)],
            run_id="cand",
        )

        result = compare_runs(baseline, candidate)

        assert result.cases[0].test_case_id == "z-regresses"
        assert result.cases[0].status == STATUS_REGRESSED

    def test_threshold_mutes_small_drops(self):
        baseline = _make_run([_make_eval("tc1", score=5)], run_id="base")
        candidate = _make_run([_make_eval("tc1", score=4)], run_id="cand")

        strict = compare_runs(baseline, candidate, regression_threshold=0.0)
        lenient = compare_runs(baseline, candidate, regression_threshold=2.0)

        assert strict.regressed == 1
        assert lenient.regressed == 0
        assert lenient.cases[0].status == STATUS_UNCHANGED

    def test_error_transition_is_regression_even_without_scores(self):
        baseline = _make_run([_make_eval("tc1", score=4)], run_id="base")
        candidate = _make_run([_make_eval("tc1", error="429 rate limited")], run_id="cand")

        result = compare_runs(baseline, candidate)

        assert result.regressed == 1
        assert "429" in result.cases[0].detail
        assert result.cases[0].score_delta is None

    def test_error_fixed_is_improvement(self):
        baseline = _make_run([_make_eval("tc1", error="timeout")], run_id="base")
        candidate = _make_run([_make_eval("tc1", score=4)], run_id="cand")

        result = compare_runs(baseline, candidate)

        assert result.improved == 1
        assert result.cases[0].status == STATUS_IMPROVED

    def test_both_errored_is_unchanged(self):
        baseline = _make_run([_make_eval("tc1", error="timeout")], run_id="base")
        candidate = _make_run([_make_eval("tc1", error="timeout")], run_id="cand")

        result = compare_runs(baseline, candidate)

        assert result.unchanged == 1

    def test_missing_scores_are_unscored(self):
        baseline = _make_run([_make_eval("tc1", score=4)], run_id="base")
        candidate = _make_run([_make_eval("tc1")], run_id="cand")

        result = compare_runs(baseline, candidate)

        assert result.unscored == 1
        assert result.cases[0].status == STATUS_UNSCORED

    def test_added_and_removed_cases(self):
        baseline = _make_run(
            [_make_eval("tc1", score=4), _make_eval("tc-old", score=3)], run_id="base"
        )
        candidate = _make_run(
            [_make_eval("tc1", score=4), _make_eval("tc-new", score=5)], run_id="cand"
        )

        result = compare_runs(baseline, candidate)

        assert result.added == 1
        assert result.removed == 1
        by_id = {c.test_case_id: c for c in result.cases}
        assert by_id["tc-new"].status == STATUS_ADDED
        assert by_id["tc-old"].status == STATUS_REMOVED

    def test_matches_per_model_by_default(self):
        baseline = _make_run(
            [_make_eval("tc1", model="model-a", score=4), _make_eval("tc1", model="model-b", score=2)],
            run_id="base",
        )
        candidate = _make_run(
            [_make_eval("tc1", model="model-a", score=2), _make_eval("tc1", model="model-b", score=4)],
            run_id="cand",
        )

        result = compare_runs(baseline, candidate)

        statuses = {
            (c.test_case_id, c.candidate_model): c.status for c in result.cases
        }
        assert statuses[("tc1", "model-a")] == STATUS_REGRESSED
        assert statuses[("tc1", "model-b")] == STATUS_IMPROVED

    def test_cross_model_comparison(self):
        baseline = _make_run(
            [_make_eval("tc1", model="gpt-4o", score=3)], run_id="base"
        )
        candidate = _make_run(
            [_make_eval("tc1", model="claude-sonnet", score=5)], run_id="cand"
        )

        result = compare_runs(
            baseline,
            candidate,
            baseline_model="gpt-4o",
            candidate_model="claude-sonnet",
        )

        assert result.improved == 1
        case = result.cases[0]
        assert case.baseline_model == "gpt-4o"
        assert case.candidate_model == "claude-sonnet"

    def test_cross_model_requires_both_flags(self):
        baseline = _make_run([_make_eval("tc1", score=3)], run_id="base")
        candidate = _make_run([_make_eval("tc1", score=4)], run_id="cand")

        with pytest.raises(ValueError):
            compare_runs(baseline, candidate, baseline_model="gpt-4o")

    def test_aggregate_deltas(self):
        baseline = _make_run(
            [
                _make_eval("tc1", score=3, latency_ms=1000.0, cost_usd=0.01),
                _make_eval("tc2", score=3, latency_ms=1000.0, cost_usd=0.01),
            ],
            run_id="base",
        )
        candidate = _make_run(
            [
                _make_eval("tc1", score=5, latency_ms=1500.0, cost_usd=0.02),
                _make_eval("tc2", score=3, latency_ms=500.0, cost_usd=0.02),
            ],
            run_id="cand",
        )

        result = compare_runs(baseline, candidate)

        assert result.baseline_avg_score == pytest.approx(3.0)
        assert result.candidate_avg_score == pytest.approx(4.0)
        assert result.avg_score_delta == pytest.approx(1.0)
        assert result.total_cost_delta_usd == pytest.approx(0.02)
        assert result.avg_latency_delta_ms == pytest.approx(0.0)


class TestComparisonMarkdown:
    def test_markdown_contains_summary_and_sections(self):
        baseline = _make_run(
            [_make_eval("tc1", score=5), _make_eval("tc2", score=2)],
            run_id="base",
            run_name="main",
        )
        candidate = _make_run(
            [
                _make_eval("tc1", score=3, explanation="Missed the edge case"),
                _make_eval("tc2", score=4),
            ],
            run_id="cand",
            run_name="feature",
        )

        md = comparison_to_markdown(compare_runs(baseline, candidate))

        assert "# PromptLens Run Comparison" in md
        assert "`main`" in md and "`feature`" in md
        assert "## Regressions" in md
        assert "## Improvements" in md
        assert "Missed the edge case" in md
        assert "1 regressed" in md

    def test_markdown_escapes_pipes_in_detail(self):
        baseline = _make_run([_make_eval("tc1", score=5)], run_id="base")
        candidate = _make_run(
            [_make_eval("tc1", score=2, explanation="bad | output")], run_id="cand"
        )

        md = comparison_to_markdown(compare_runs(baseline, candidate))

        assert "bad \\| output" in md


class TestCompareCLI:
    def _write_run(self, results_dir, run):
        run_dir = Path(results_dir) / run.run_id
        run_dir.mkdir(parents=True)
        with open(run_dir / "results.json", "w") as f:
            f.write(run.model_dump_json())

    def test_compare_happy_path(self, tmp_path):
        self._write_run(tmp_path, _make_run([_make_eval("tc1", score=3)], run_id="base"))
        self._write_run(tmp_path, _make_run([_make_eval("tc1", score=5)], run_id="cand"))

        runner = CliRunner()
        result = runner.invoke(
            cli, ["compare", "base", "cand", "--output-dir", str(tmp_path)]
        )

        assert result.exit_code == 0
        assert "1 improved" in result.output

    def test_fail_on_regression_exits_2(self, tmp_path):
        self._write_run(tmp_path, _make_run([_make_eval("tc1", score=5)], run_id="base"))
        self._write_run(tmp_path, _make_run([_make_eval("tc1", score=2)], run_id="cand"))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                "base",
                "cand",
                "--output-dir",
                str(tmp_path),
                "--fail-on-regression",
            ],
        )

        assert result.exit_code == 2
        assert "Regression gate failed" in result.output

    def test_fail_on_regression_passes_when_clean(self, tmp_path):
        self._write_run(tmp_path, _make_run([_make_eval("tc1", score=3)], run_id="base"))
        self._write_run(tmp_path, _make_run([_make_eval("tc1", score=3)], run_id="cand"))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                "base",
                "cand",
                "--output-dir",
                str(tmp_path),
                "--fail-on-regression",
            ],
        )

        assert result.exit_code == 0
        assert "Regression gate passed" in result.output

    def test_missing_run_exits_1(self, tmp_path):
        self._write_run(tmp_path, _make_run([_make_eval("tc1", score=3)], run_id="base"))

        runner = CliRunner()
        result = runner.invoke(
            cli, ["compare", "base", "missing", "--output-dir", str(tmp_path)]
        )

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_mismatched_model_flags_exit_1(self, tmp_path):
        self._write_run(tmp_path, _make_run([_make_eval("tc1", score=3)], run_id="base"))
        self._write_run(tmp_path, _make_run([_make_eval("tc1", score=3)], run_id="cand"))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                "base",
                "cand",
                "--output-dir",
                str(tmp_path),
                "--baseline-model",
                "model-a",
            ],
        )

        assert result.exit_code == 1

    def test_markdown_file_output(self, tmp_path):
        self._write_run(tmp_path, _make_run([_make_eval("tc1", score=5)], run_id="base"))
        self._write_run(tmp_path, _make_run([_make_eval("tc1", score=2)], run_id="cand"))
        out_file = tmp_path / "comparison.md"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                "base",
                "cand",
                "--output-dir",
                str(tmp_path),
                "--format",
                "md",
                "--output",
                str(out_file),
            ],
        )

        assert result.exit_code == 0
        content = out_file.read_text()
        assert "# PromptLens Run Comparison" in content
        assert "## Regressions" in content

    def test_json_file_output(self, tmp_path):
        self._write_run(tmp_path, _make_run([_make_eval("tc1", score=3)], run_id="base"))
        self._write_run(tmp_path, _make_run([_make_eval("tc1", score=4)], run_id="cand"))
        out_file = tmp_path / "comparison.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                "base",
                "cand",
                "--output-dir",
                str(tmp_path),
                "--format",
                "json",
                "--output",
                str(out_file),
            ],
        )

        assert result.exit_code == 0
        data = json.loads(out_file.read_text())
        assert data["baseline_run_id"] == "base"
        assert data["improved"] == 1
