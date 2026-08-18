"""Tests for run comparison, regression detection, and the compare CLI."""

import json
from datetime import datetime

import pytest
from click.testing import CliRunner

from promptlens.cli import cli
from promptlens.comparison import (
    CaseStatus,
    compare_runs,
    render_markdown,
)
from promptlens.models.result import (
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)


def _make_response(model="model-a", provider="anthropic", error=None, **kwargs):
    defaults = {
        "content": "The answer is 42.",
        "model": model,
        "provider": provider,
        "latency_ms": 1000.0,
        "cost_usd": 0.002,
        "tokens_used": 150,
        "error": error,
    }
    defaults.update(kwargs)
    return ModelResponse(**defaults)


def _make_score(score, explanation="Looks correct."):
    return JudgeScore(
        score=score,
        explanation=explanation,
        judge_model="judge-model",
        judge_provider="anthropic",
    )


def _make_eval(test_case_id, model="model-a", score=None, error=None, **kwargs):
    return EvaluationResult(
        test_case_id=test_case_id,
        query="What is the answer?",
        expected_behavior="Answers correctly",
        model_response=_make_response(model=model, error=error, **kwargs),
        judge_score=_make_score(score) if score is not None else None,
    )


def _make_run(results, models=None, run_id="run-1", run_name=None):
    models = models or ["model-a"]
    return RunResult(
        run_id=run_id,
        run_name=run_name,
        timestamp=datetime(2026, 8, 18, 12, 0, 0),
        golden_set_name="golden-set",
        models_tested=models,
        results=results,
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassification:
    def test_score_drop_is_regression(self):
        baseline = _make_run([_make_eval("tc-1", score=5)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=3)], run_id="cand")

        comparison = compare_runs(baseline, candidate)

        assert len(comparison.cases) == 1
        case = comparison.cases[0]
        assert case.status == CaseStatus.REGRESSED
        assert case.score_delta == -2
        assert comparison.has_regressions()

    def test_score_rise_is_improvement(self):
        baseline = _make_run([_make_eval("tc-1", score=2)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=4)], run_id="cand")

        comparison = compare_runs(baseline, candidate)

        assert comparison.cases[0].status == CaseStatus.IMPROVED
        assert comparison.cases[0].score_delta == 2
        assert not comparison.has_regressions()

    def test_same_score_is_unchanged(self):
        baseline = _make_run([_make_eval("tc-1", score=4)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=4)], run_id="cand")

        comparison = compare_runs(baseline, candidate)

        assert comparison.cases[0].status == CaseStatus.UNCHANGED

    def test_threshold_suppresses_small_changes(self):
        baseline = _make_run([_make_eval("tc-1", score=4)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=3)], run_id="cand")

        comparison = compare_runs(baseline, candidate, score_threshold=1)

        assert comparison.cases[0].status == CaseStatus.UNCHANGED
        assert not comparison.has_regressions()

    def test_threshold_still_flags_large_drops(self):
        baseline = _make_run([_make_eval("tc-1", score=5)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=3)], run_id="cand")

        comparison = compare_runs(baseline, candidate, score_threshold=1)

        assert comparison.cases[0].status == CaseStatus.REGRESSED

    def test_negative_threshold_rejected(self):
        baseline = _make_run([_make_eval("tc-1", score=5)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=5)], run_id="cand")

        with pytest.raises(ValueError):
            compare_runs(baseline, candidate, score_threshold=-1)

    def test_new_error_is_regression_even_with_scores(self):
        baseline = _make_run([_make_eval("tc-1", score=5)], run_id="base")
        candidate = _make_run(
            [_make_eval("tc-1", score=5, error="rate limited")], run_id="cand"
        )

        comparison = compare_runs(baseline, candidate)

        assert comparison.cases[0].status == CaseStatus.REGRESSED

    def test_fixed_error_is_improvement(self):
        baseline = _make_run(
            [_make_eval("tc-1", error="timeout")], run_id="base"
        )
        candidate = _make_run([_make_eval("tc-1", score=4)], run_id="cand")

        comparison = compare_runs(baseline, candidate)

        assert comparison.cases[0].status == CaseStatus.IMPROVED

    def test_error_in_both_runs_is_unchanged(self):
        baseline = _make_run([_make_eval("tc-1", error="timeout")], run_id="base")
        candidate = _make_run([_make_eval("tc-1", error="timeout")], run_id="cand")

        comparison = compare_runs(baseline, candidate)

        assert comparison.cases[0].status == CaseStatus.UNCHANGED

    def test_missing_judge_score_is_unscored(self):
        baseline = _make_run([_make_eval("tc-1", score=4)], run_id="base")
        candidate = _make_run([_make_eval("tc-1")], run_id="cand")

        comparison = compare_runs(baseline, candidate)

        assert comparison.cases[0].status == CaseStatus.UNSCORED
        assert not comparison.has_regressions()


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


class TestPairing:
    def test_pairs_by_test_case_and_model(self):
        baseline = _make_run(
            [
                _make_eval("tc-1", model="model-a", score=5),
                _make_eval("tc-1", model="model-b", score=2),
            ],
            models=["model-a", "model-b"],
            run_id="base",
        )
        candidate = _make_run(
            [
                _make_eval("tc-1", model="model-b", score=4),
                _make_eval("tc-1", model="model-a", score=5),
            ],
            models=["model-a", "model-b"],
            run_id="cand",
        )

        comparison = compare_runs(baseline, candidate)

        by_model = {c.model: c for c in comparison.cases}
        assert by_model["model-a"].status == CaseStatus.UNCHANGED
        assert by_model["model-b"].status == CaseStatus.IMPROVED

    def test_survives_reordered_and_extra_cases(self):
        baseline = _make_run(
            [_make_eval("tc-1", score=4), _make_eval("tc-2", score=4)],
            run_id="base",
        )
        candidate = _make_run(
            [
                _make_eval("tc-3", score=5),
                _make_eval("tc-2", score=4),
                _make_eval("tc-1", score=4),
            ],
            run_id="cand",
        )

        comparison = compare_runs(baseline, candidate)

        statuses = {c.test_case_id: c.status for c in comparison.cases}
        assert statuses["tc-1"] == CaseStatus.UNCHANGED
        assert statuses["tc-2"] == CaseStatus.UNCHANGED
        assert statuses["tc-3"] == CaseStatus.ADDED

    def test_removed_case_reported(self):
        baseline = _make_run(
            [_make_eval("tc-1", score=4), _make_eval("tc-2", score=3)],
            run_id="base",
        )
        candidate = _make_run([_make_eval("tc-1", score=4)], run_id="cand")

        comparison = compare_runs(baseline, candidate)

        statuses = {c.test_case_id: c.status for c in comparison.cases}
        assert statuses["tc-2"] == CaseStatus.REMOVED

    def test_added_and_removed_do_not_trigger_regression(self):
        baseline = _make_run([_make_eval("tc-1", score=4)], run_id="base")
        candidate = _make_run([_make_eval("tc-2", score=1)], run_id="cand")

        comparison = compare_runs(baseline, candidate)

        assert not comparison.has_regressions()

    def test_regressions_sorted_first(self):
        baseline = _make_run(
            [
                _make_eval("tc-1", score=4),
                _make_eval("tc-2", score=5),
                _make_eval("tc-3", score=3),
            ],
            run_id="base",
        )
        candidate = _make_run(
            [
                _make_eval("tc-1", score=4),
                _make_eval("tc-2", score=2),
                _make_eval("tc-3", score=5),
            ],
            run_id="cand",
        )

        comparison = compare_runs(baseline, candidate)

        assert comparison.cases[0].test_case_id == "tc-2"
        assert comparison.cases[0].status == CaseStatus.REGRESSED


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


class TestModelSummaries:
    def test_average_and_cost_deltas(self):
        baseline = _make_run(
            [
                _make_eval("tc-1", score=4, cost_usd=0.01),
                _make_eval("tc-2", score=4, cost_usd=0.01),
            ],
            run_id="base",
        )
        candidate = _make_run(
            [
                _make_eval("tc-1", score=5, cost_usd=0.02),
                _make_eval("tc-2", score=5, cost_usd=0.02),
            ],
            run_id="cand",
        )

        comparison = compare_runs(baseline, candidate)

        assert len(comparison.model_summaries) == 1
        summary = comparison.model_summaries[0]
        assert summary.baseline_avg_score == pytest.approx(4.0)
        assert summary.candidate_avg_score == pytest.approx(5.0)
        assert summary.avg_score_delta == pytest.approx(1.0)
        assert summary.baseline_cost_usd == pytest.approx(0.02)
        assert summary.candidate_cost_usd == pytest.approx(0.04)
        assert summary.improved == 2

    def test_model_only_in_one_run_included(self):
        baseline = _make_run(
            [_make_eval("tc-1", model="model-a", score=4)],
            models=["model-a"],
            run_id="base",
        )
        candidate = _make_run(
            [_make_eval("tc-1", model="model-b", score=4)],
            models=["model-b"],
            run_id="cand",
        )

        comparison = compare_runs(baseline, candidate)

        models = {s.model for s in comparison.model_summaries}
        assert models == {"model-a", "model-b"}


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


class TestMarkdownReport:
    def test_report_contains_verdict_and_changed_cases(self):
        baseline = _make_run(
            [_make_eval("tc-1", score=5), _make_eval("tc-2", score=4)],
            run_id="base",
            run_name="main",
        )
        candidate = _make_run(
            [_make_eval("tc-1", score=2), _make_eval("tc-2", score=4)],
            run_id="cand",
            run_name="pr-42",
        )

        report = render_markdown(compare_runs(baseline, candidate))

        assert "REGRESSIONS DETECTED" in report
        assert "tc-1" in report
        assert "-3" in report
        assert "`base`" in report
        assert "`cand`" in report
        # Unchanged cases stay out of the changed-cases table
        assert report.count("tc-2") == 0

    def test_clean_report_when_no_changes(self):
        baseline = _make_run([_make_eval("tc-1", score=4)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=4)], run_id="cand")

        report = render_markdown(compare_runs(baseline, candidate))

        assert "No regressions detected" in report
        assert "No changed cases" in report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_run(tmp_path, run, dirname):
    run_dir = tmp_path / dirname
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(
        json.dumps(run.model_dump(mode="json"), default=str), encoding="utf-8"
    )
    return run_dir


class TestCompareCli:
    def test_compare_by_run_id(self, tmp_path):
        baseline = _make_run([_make_eval("tc-1", score=4)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=5)], run_id="cand")
        _write_run(tmp_path, baseline, "base")
        _write_run(tmp_path, candidate, "cand")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["compare", "base", "cand", "--output-dir", str(tmp_path)]
        )

        assert result.exit_code == 0
        assert "No regressions detected" in result.output

    def test_compare_by_direct_paths(self, tmp_path):
        baseline = _make_run([_make_eval("tc-1", score=4)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=4)], run_id="cand")
        base_dir = _write_run(tmp_path, baseline, "base")
        cand_dir = _write_run(tmp_path, candidate, "cand")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["compare", str(base_dir / "results.json"), str(cand_dir)],
        )

        assert result.exit_code == 0

    def test_fail_on_regression_sets_exit_code(self, tmp_path):
        baseline = _make_run([_make_eval("tc-1", score=5)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=2)], run_id="cand")
        _write_run(tmp_path, baseline, "base")
        _write_run(tmp_path, candidate, "cand")

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

        assert result.exit_code == 1
        assert "regression" in result.output.lower()

    def test_regression_without_flag_exits_zero(self, tmp_path):
        baseline = _make_run([_make_eval("tc-1", score=5)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=2)], run_id="cand")
        _write_run(tmp_path, baseline, "base")
        _write_run(tmp_path, candidate, "cand")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["compare", "base", "cand", "--output-dir", str(tmp_path)]
        )

        assert result.exit_code == 0
        assert "regression(s) detected" in result.output

    def test_missing_run_reports_error(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["compare", "nope-a", "nope-b", "--output-dir", str(tmp_path)]
        )

        assert result.exit_code == 1
        assert "Comparison failed" in result.output

    def test_writes_markdown_and_json_reports(self, tmp_path):
        baseline = _make_run([_make_eval("tc-1", score=5)], run_id="base")
        candidate = _make_run([_make_eval("tc-1", score=3)], run_id="cand")
        _write_run(tmp_path, baseline, "base")
        _write_run(tmp_path, candidate, "cand")
        md_path = tmp_path / "report.md"
        json_path = tmp_path / "comparison.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                "base",
                "cand",
                "--output-dir",
                str(tmp_path),
                "--output",
                str(md_path),
                "--json-output",
                str(json_path),
            ],
        )

        assert result.exit_code == 0
        assert "REGRESSIONS DETECTED" in md_path.read_text(encoding="utf-8")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["baseline_run_id"] == "base"
        assert data["cases"][0]["status"] == "regressed"
