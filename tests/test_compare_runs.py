"""Tests for cross-run comparison and the compare CLI command."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from promptlens.cli import cli
from promptlens.comparison import compare_runs, render_markdown
from promptlens.models.comparison import (
    REASON_ERROR_RESOLVED,
    REASON_NEW_ERROR,
    REASON_NO_BASELINE_SCORE,
    REASON_SCORE_MISSING,
    STATUS_IMPROVED,
    STATUS_INCOMPARABLE,
    STATUS_REGRESSED,
    STATUS_UNCHANGED,
)
from promptlens.models.result import (
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)


def _make_response(model="model-a", error=None, cost=0.002, latency=1000.0):
    return ModelResponse(
        content="The answer is 42." if error is None else "",
        model=model,
        provider="anthropic",
        latency_ms=latency,
        cost_usd=cost,
        error=error,
    )


def _make_eval(test_case_id, model="model-a", score=None, error=None, cost=0.002, latency=1000.0):
    return EvaluationResult(
        test_case_id=test_case_id,
        query="What is the answer?",
        expected_behavior="Answers correctly",
        model_response=_make_response(model=model, error=error, cost=cost, latency=latency),
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


def _make_run(results, models=None, run_id="run-1", run_name=None):
    return RunResult(
        run_id=run_id,
        run_name=run_name,
        timestamp=datetime(2026, 8, 17, 12, 0, 0),
        golden_set_name="golden-set",
        models_tested=models or ["model-a"],
        results=results,
    )


def _case(comparison, test_case_id, model="model-a"):
    for case in comparison.cases:
        if case.test_case_id == test_case_id and case.model == model:
            return case
    raise AssertionError(f"case {test_case_id}/{model} not found")


class TestCompareRuns:
    def test_score_drop_is_regression(self):
        baseline = _make_run([_make_eval("tc-1", score=5)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=3)], run_id="cand")

        comparison = compare_runs(baseline, candidate)
        case = _case(comparison, "tc-1")

        assert case.status == STATUS_REGRESSED
        assert case.score_delta == -2
        assert comparison.has_regressions
        assert comparison.regression_count == 1

    def test_score_gain_is_improvement(self):
        baseline = _make_run([_make_eval("tc-1", score=2)])
        candidate = _make_run([_make_eval("tc-1", score=4)])

        comparison = compare_runs(baseline, candidate)

        assert _case(comparison, "tc-1").status == STATUS_IMPROVED
        assert not comparison.has_regressions

    def test_same_score_is_unchanged(self):
        baseline = _make_run([_make_eval("tc-1", score=4)])
        candidate = _make_run([_make_eval("tc-1", score=4)])

        comparison = compare_runs(baseline, candidate)

        assert _case(comparison, "tc-1").status == STATUS_UNCHANGED

    def test_threshold_suppresses_small_drops(self):
        baseline = _make_run([_make_eval("tc-1", score=5)])
        candidate = _make_run([_make_eval("tc-1", score=4)])

        comparison = compare_runs(baseline, candidate, threshold=2.0)

        assert _case(comparison, "tc-1").status == STATUS_UNCHANGED
        assert not comparison.has_regressions

    def test_threshold_must_be_positive(self):
        run = _make_run([_make_eval("tc-1", score=4)])
        with pytest.raises(ValueError):
            compare_runs(run, run, threshold=0)

    def test_new_error_is_regression(self):
        baseline = _make_run([_make_eval("tc-1", score=5)])
        candidate = _make_run([_make_eval("tc-1", error="API timeout")])

        comparison = compare_runs(baseline, candidate)
        case = _case(comparison, "tc-1")

        assert case.status == STATUS_REGRESSED
        assert case.reason == REASON_NEW_ERROR

    def test_resolved_error_is_improvement(self):
        baseline = _make_run([_make_eval("tc-1", error="API timeout")])
        candidate = _make_run([_make_eval("tc-1", score=4)])

        comparison = compare_runs(baseline, candidate)
        case = _case(comparison, "tc-1")

        assert case.status == STATUS_IMPROVED
        assert case.reason == REASON_ERROR_RESOLVED

    def test_error_in_both_runs_is_unchanged(self):
        baseline = _make_run([_make_eval("tc-1", error="boom")])
        candidate = _make_run([_make_eval("tc-1", error="boom again")])

        comparison = compare_runs(baseline, candidate)

        assert _case(comparison, "tc-1").status == STATUS_UNCHANGED

    def test_lost_score_is_regression(self):
        baseline = _make_run([_make_eval("tc-1", score=5)])
        candidate = _make_run([_make_eval("tc-1")])

        comparison = compare_runs(baseline, candidate)
        case = _case(comparison, "tc-1")

        assert case.status == STATUS_REGRESSED
        assert case.reason == REASON_SCORE_MISSING

    def test_missing_baseline_score_is_incomparable(self):
        baseline = _make_run([_make_eval("tc-1")])
        candidate = _make_run([_make_eval("tc-1", score=5)])

        comparison = compare_runs(baseline, candidate)
        case = _case(comparison, "tc-1")

        assert case.status == STATUS_INCOMPARABLE
        assert case.reason == REASON_NO_BASELINE_SCORE
        assert not comparison.has_regressions

    def test_added_and_removed_cases_are_reported_not_compared(self):
        baseline = _make_run([_make_eval("tc-1", score=4), _make_eval("tc-old", score=4)])
        candidate = _make_run([_make_eval("tc-1", score=4), _make_eval("tc-new", score=4)])

        comparison = compare_runs(baseline, candidate)

        assert comparison.cases_added == ["tc-new"]
        assert comparison.cases_removed == ["tc-old"]
        assert [c.test_case_id for c in comparison.cases] == ["tc-1"]

    def test_added_and_removed_models_are_reported_not_compared(self):
        baseline = _make_run(
            [_make_eval("tc-1", model="model-a", score=4), _make_eval("tc-1", model="model-old", score=4)],
            models=["model-a", "model-old"],
        )
        candidate = _make_run(
            [_make_eval("tc-1", model="model-a", score=4), _make_eval("tc-1", model="model-new", score=4)],
            models=["model-a", "model-new"],
        )

        comparison = compare_runs(baseline, candidate)

        assert comparison.models_compared == ["model-a"]
        assert comparison.models_added == ["model-new"]
        assert comparison.models_removed == ["model-old"]
        assert all(c.model == "model-a" for c in comparison.cases)

    def test_model_summary_aggregates(self):
        baseline = _make_run(
            [
                _make_eval("tc-1", score=5, cost=0.01, latency=1000.0),
                _make_eval("tc-2", score=3, cost=0.01, latency=1000.0),
            ]
        )
        candidate = _make_run(
            [
                _make_eval("tc-1", score=3, cost=0.02, latency=2000.0),
                _make_eval("tc-2", score=3, cost=0.02, latency=2000.0),
            ]
        )

        comparison = compare_runs(baseline, candidate)
        summary = comparison.model_summaries[0]

        assert summary.model == "model-a"
        assert summary.baseline_avg_score == pytest.approx(4.0)
        assert summary.candidate_avg_score == pytest.approx(3.0)
        assert summary.avg_score_delta == pytest.approx(-1.0)
        assert summary.regressed == 1
        assert summary.unchanged == 1
        assert summary.cost_delta_usd == pytest.approx(0.02)
        assert summary.latency_delta_ms == pytest.approx(1000.0)

    def test_duplicate_pairs_use_first_occurrence(self):
        baseline = _make_run([_make_eval("tc-1", score=5), _make_eval("tc-1", score=1)])
        candidate = _make_run([_make_eval("tc-1", score=5)])

        comparison = compare_runs(baseline, candidate)

        assert len(comparison.cases) == 1
        assert _case(comparison, "tc-1").baseline_score == 5


class TestRenderMarkdown:
    def test_markdown_contains_verdict_and_regressions(self):
        baseline = _make_run([_make_eval("tc-1", score=5)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=3)], run_id="cand")
        comparison = compare_runs(baseline, candidate)

        markdown = render_markdown(comparison)

        assert "REGRESSED" in markdown
        assert "`base`" in markdown
        assert "`cand`" in markdown
        assert "## Regressions (1)" in markdown
        assert "tc-1" in markdown

    def test_markdown_pass_verdict_without_regressions(self):
        baseline = _make_run([_make_eval("tc-1", score=4)])
        candidate = _make_run([_make_eval("tc-1", score=4)])
        comparison = compare_runs(baseline, candidate)

        markdown = render_markdown(comparison)

        assert "PASS" in markdown
        assert "## Regressions" not in markdown

    def test_markdown_reports_suite_drift(self):
        baseline = _make_run([_make_eval("tc-old", score=4)])
        candidate = _make_run([_make_eval("tc-new", score=4)])
        comparison = compare_runs(baseline, candidate)

        markdown = render_markdown(comparison)

        assert "## Suite Drift" in markdown
        assert "tc-new" in markdown
        assert "tc-old" in markdown


def _write_run(output_dir: Path, run: RunResult) -> None:
    run_dir = output_dir / run.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(run.model_dump_json())


class TestCompareCLI:
    def _invoke(self, tmp_path, baseline, candidate, *extra_args):
        _write_run(tmp_path, baseline)
        _write_run(tmp_path, candidate)
        runner = CliRunner()
        return runner.invoke(
            cli,
            [
                "compare",
                baseline.run_id,
                candidate.run_id,
                "--output-dir",
                str(tmp_path),
                *extra_args,
            ],
        )

    def test_exit_zero_without_gate_even_on_regression(self, tmp_path):
        baseline = _make_run([_make_eval("tc-1", score=5)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=3)], run_id="cand")

        result = self._invoke(tmp_path, baseline, candidate)

        assert result.exit_code == 0
        assert "regression(s) detected" in result.output

    def test_gate_exits_two_on_regression(self, tmp_path):
        baseline = _make_run([_make_eval("tc-1", score=5)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=3)], run_id="cand")

        result = self._invoke(tmp_path, baseline, candidate, "--fail-on-regression")

        assert result.exit_code == 2

    def test_gate_passes_without_regression(self, tmp_path):
        baseline = _make_run([_make_eval("tc-1", score=4)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=4)], run_id="cand")

        result = self._invoke(tmp_path, baseline, candidate, "--fail-on-regression")

        assert result.exit_code == 0
        assert "No regressions detected" in result.output

    def test_missing_run_exits_one(self, tmp_path):
        baseline = _make_run([_make_eval("tc-1", score=4)], run_id="base")
        _write_run(tmp_path, baseline)
        runner = CliRunner()

        result = runner.invoke(
            cli, ["compare", "base", "missing", "--output-dir", str(tmp_path)]
        )

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_writes_markdown_and_json_reports(self, tmp_path):
        baseline = _make_run([_make_eval("tc-1", score=5)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=3)], run_id="cand")
        md_path = tmp_path / "reports" / "comparison.md"
        json_path = tmp_path / "reports" / "comparison.json"

        result = self._invoke(
            tmp_path,
            baseline,
            candidate,
            "--markdown",
            str(md_path),
            "--json",
            str(json_path),
        )

        assert result.exit_code == 0
        assert "REGRESSED" in md_path.read_text()
        data = json.loads(json_path.read_text())
        assert data["baseline_run_id"] == "base"
        assert data["cases"][0]["status"] == "regressed"

    def test_custom_threshold_is_honored(self, tmp_path):
        baseline = _make_run([_make_eval("tc-1", score=5)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=4)], run_id="cand")

        result = self._invoke(
            tmp_path, baseline, candidate, "--threshold", "2", "--fail-on-regression"
        )

        assert result.exit_code == 0
