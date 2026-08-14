"""Tests for run-to-run comparison and the compare CLI command."""

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
        "cost_usd": 0.001,
        "error": error,
    }
    defaults.update(kwargs)
    return ModelResponse(**defaults)


def _make_score(score):
    return JudgeScore(
        score=score,
        explanation="Judged.",
        judge_model="judge-model",
        judge_provider="anthropic",
    )


def _make_eval(test_case_id, model="model-a", score=None, error=None):
    return EvaluationResult(
        test_case_id=test_case_id,
        query="What is the answer?",
        expected_behavior="Answers correctly",
        model_response=_make_response(model=model, error=error),
        judge_score=_make_score(score) if score is not None else None,
    )


def _make_run(results, run_id="run-1", models=None, golden_set="golden-set", run_name=None):
    models = models or ["model-a"]
    return RunResult(
        run_id=run_id,
        run_name=run_name,
        timestamp=datetime(2026, 8, 14, 12, 0, 0),
        golden_set_name=golden_set,
        models_tested=models,
        results=results,
    )


class TestCompareRuns:
    def test_regression_detected(self):
        baseline = _make_run([_make_eval("tc-1", score=5)], run_id="base")
        current = _make_run([_make_eval("tc-1", score=3)], run_id="cur")

        comparison = compare_runs(baseline, current)

        assert len(comparison.case_comparisons) == 1
        case = comparison.case_comparisons[0]
        assert case.status == CaseStatus.REGRESSED
        assert case.delta == -2
        assert comparison.has_regressions

    def test_improvement_detected(self):
        baseline = _make_run([_make_eval("tc-1", score=2)], run_id="base")
        current = _make_run([_make_eval("tc-1", score=4)], run_id="cur")

        comparison = compare_runs(baseline, current)

        case = comparison.case_comparisons[0]
        assert case.status == CaseStatus.IMPROVED
        assert case.delta == 2
        assert not comparison.has_regressions

    def test_unchanged_detected(self):
        baseline = _make_run([_make_eval("tc-1", score=4)], run_id="base")
        current = _make_run([_make_eval("tc-1", score=4)], run_id="cur")

        comparison = compare_runs(baseline, current)

        assert comparison.case_comparisons[0].status == CaseStatus.UNCHANGED
        assert comparison.case_comparisons[0].delta == 0

    def test_added_and_removed_cases(self):
        baseline = _make_run(
            [_make_eval("tc-1", score=4), _make_eval("tc-old", score=5)], run_id="base"
        )
        current = _make_run(
            [_make_eval("tc-1", score=4), _make_eval("tc-new", score=3)], run_id="cur"
        )

        comparison = compare_runs(baseline, current)

        statuses = {c.test_case_id: c.status for c in comparison.case_comparisons}
        assert statuses["tc-old"] == CaseStatus.REMOVED
        assert statuses["tc-new"] == CaseStatus.ADDED
        assert statuses["tc-1"] == CaseStatus.UNCHANGED

        mc = comparison.model_comparisons[0]
        assert mc.added == 1
        assert mc.removed == 1

    def test_unscored_case(self):
        baseline = _make_run([_make_eval("tc-1", score=4)], run_id="base")
        current = _make_run([_make_eval("tc-1", score=None)], run_id="cur")

        comparison = compare_runs(baseline, current)

        case = comparison.case_comparisons[0]
        assert case.status == CaseStatus.UNSCORED
        assert case.delta is None

    def test_multi_model_pairing(self):
        baseline = _make_run(
            [
                _make_eval("tc-1", model="model-a", score=5),
                _make_eval("tc-1", model="model-b", score=3),
            ],
            run_id="base",
            models=["model-a", "model-b"],
        )
        current = _make_run(
            [
                _make_eval("tc-1", model="model-a", score=5),
                _make_eval("tc-1", model="model-b", score=1),
            ],
            run_id="cur",
            models=["model-a", "model-b"],
        )

        comparison = compare_runs(baseline, current)

        by_model = {c.model: c for c in comparison.case_comparisons}
        assert by_model["model-a"].status == CaseStatus.UNCHANGED
        assert by_model["model-b"].status == CaseStatus.REGRESSED
        assert by_model["model-b"].delta == -2

    def test_model_filter(self):
        baseline = _make_run(
            [
                _make_eval("tc-1", model="model-a", score=5),
                _make_eval("tc-1", model="model-b", score=3),
            ],
            run_id="base",
            models=["model-a", "model-b"],
        )
        current = _make_run(
            [
                _make_eval("tc-1", model="model-a", score=4),
                _make_eval("tc-1", model="model-b", score=3),
            ],
            run_id="cur",
            models=["model-a", "model-b"],
        )

        comparison = compare_runs(baseline, current, model="model-b")

        assert all(c.model == "model-b" for c in comparison.case_comparisons)
        assert all(m.model == "model-b" for m in comparison.model_comparisons)

    def test_average_delta(self):
        baseline = _make_run(
            [_make_eval("tc-1", score=4), _make_eval("tc-2", score=4)], run_id="base"
        )
        current = _make_run(
            [_make_eval("tc-1", score=2), _make_eval("tc-2", score=4)], run_id="cur"
        )

        comparison = compare_runs(baseline, current)

        mc = comparison.model_comparisons[0]
        assert mc.baseline_average == pytest.approx(4.0)
        assert mc.current_average == pytest.approx(3.0)
        assert mc.average_delta == pytest.approx(-1.0)

    def test_golden_set_mismatch_flagged(self):
        baseline = _make_run([_make_eval("tc-1", score=4)], run_id="base", golden_set="set-a")
        current = _make_run([_make_eval("tc-1", score=4)], run_id="cur", golden_set="set-b")

        comparison = compare_runs(baseline, current)

        assert comparison.golden_set_mismatch

    def test_regressions_sorted_worst_first(self):
        baseline = _make_run(
            [_make_eval("tc-1", score=5), _make_eval("tc-2", score=4)], run_id="base"
        )
        current = _make_run(
            [_make_eval("tc-1", score=4), _make_eval("tc-2", score=1)], run_id="cur"
        )

        comparison = compare_runs(baseline, current)

        regressions = comparison.regressions
        assert [r.test_case_id for r in regressions] == ["tc-2", "tc-1"]


class TestGates:
    def _comparison(self, baseline_scores, current_scores):
        baseline = _make_run(
            [_make_eval(f"tc-{i}", score=s) for i, s in enumerate(baseline_scores)],
            run_id="base",
        )
        current = _make_run(
            [_make_eval(f"tc-{i}", score=s) for i, s in enumerate(current_scores)],
            run_id="cur",
        )
        return compare_runs(baseline, current)

    def test_average_gate_zero_fails_on_any_drop(self):
        comparison = self._comparison([4, 4], [4, 3])
        failures = comparison.check_gates(max_regression=0)
        assert len(failures) == 1
        assert failures[0].kind == "average"

    def test_average_gate_tolerance_passes_small_drop(self):
        comparison = self._comparison([4, 4], [4, 3])  # avg drop of 0.5
        failures = comparison.check_gates(max_regression=1.0)
        assert failures == []

    def test_case_gate_zero_fails_on_any_case_drop(self):
        comparison = self._comparison([4, 4], [5, 3])  # avg unchanged, one case down
        assert comparison.check_gates(max_regression=0) == []
        failures = comparison.check_gates(max_case_regression=0)
        assert len(failures) == 1
        assert failures[0].kind == "case"
        assert failures[0].test_case_id == "tc-1"

    def test_case_gate_tolerance(self):
        comparison = self._comparison([5], [3])  # drop of 2
        assert comparison.check_gates(max_case_regression=2) == []
        assert len(comparison.check_gates(max_case_regression=1)) == 1

    def test_no_gates_no_failures(self):
        comparison = self._comparison([5], [1])
        assert comparison.check_gates() == []


class TestRenderMarkdown:
    def test_contains_model_summary_and_changed_cases(self):
        baseline = _make_run([_make_eval("tc-1", score=5)], run_id="base")
        current = _make_run([_make_eval("tc-1", score=3)], run_id="cur")
        comparison = compare_runs(baseline, current)

        md = render_markdown(comparison)

        assert "# PromptLens Comparison" in md
        assert "`base`" in md
        assert "`cur`" in md
        assert "| model-a |" in md
        assert "`tc-1`" in md
        assert "regressed" in md

    def test_no_changes_message(self):
        baseline = _make_run([_make_eval("tc-1", score=4)], run_id="base")
        current = _make_run([_make_eval("tc-1", score=4)], run_id="cur")
        comparison = compare_runs(baseline, current)

        md = render_markdown(comparison)

        assert "No score changes between runs." in md

    def test_golden_set_mismatch_warning(self):
        baseline = _make_run([_make_eval("tc-1", score=4)], run_id="base", golden_set="a")
        current = _make_run([_make_eval("tc-1", score=4)], run_id="cur", golden_set="b")
        comparison = compare_runs(baseline, current)

        assert "different golden sets" in render_markdown(comparison)


class TestCompareCLI:
    def _write_run(self, tmp_path, run, subdir=None):
        if subdir:
            run_dir = tmp_path / subdir
        else:
            run_dir = tmp_path / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "results.json").write_text(run.model_dump_json())
        return run_dir

    def test_compare_by_run_id(self, tmp_path):
        baseline = _make_run([_make_eval("tc-1", score=5)], run_id="base")
        current = _make_run([_make_eval("tc-1", score=3)], run_id="cur")
        self._write_run(tmp_path, baseline)
        self._write_run(tmp_path, current)

        runner = CliRunner()
        result = runner.invoke(
            cli, ["compare", "base", "cur", "--output-dir", str(tmp_path)]
        )

        assert result.exit_code == 0
        assert "Regressed cases" in result.output

    def test_compare_by_file_path(self, tmp_path):
        baseline = _make_run([_make_eval("tc-1", score=4)], run_id="base")
        current = _make_run([_make_eval("tc-1", score=4)], run_id="cur")
        base_dir = self._write_run(tmp_path, baseline)
        cur_dir = self._write_run(tmp_path, current)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["compare", str(base_dir / "results.json"), str(cur_dir)],
        )

        assert result.exit_code == 0

    def test_gate_failure_exits_2(self, tmp_path):
        baseline = _make_run([_make_eval("tc-1", score=5)], run_id="base")
        current = _make_run([_make_eval("tc-1", score=3)], run_id="cur")
        self._write_run(tmp_path, baseline)
        self._write_run(tmp_path, current)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                "base",
                "cur",
                "--output-dir",
                str(tmp_path),
                "--max-regression",
                "0",
            ],
        )

        assert result.exit_code == 2
        assert "Regression gate failed" in result.output

    def test_gate_pass_exits_0(self, tmp_path):
        baseline = _make_run([_make_eval("tc-1", score=4)], run_id="base")
        current = _make_run([_make_eval("tc-1", score=4)], run_id="cur")
        self._write_run(tmp_path, baseline)
        self._write_run(tmp_path, current)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                "base",
                "cur",
                "--output-dir",
                str(tmp_path),
                "--max-regression",
                "0",
                "--max-case-regression",
                "0",
            ],
        )

        assert result.exit_code == 0
        assert "Regression gate passed" in result.output

    def test_missing_run_exits_1(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["compare", "nope-a", "nope-b", "--output-dir", str(tmp_path)]
        )

        assert result.exit_code == 1
        assert "Could not resolve run" in result.output

    def test_markdown_and_json_outputs(self, tmp_path):
        baseline = _make_run([_make_eval("tc-1", score=5)], run_id="base")
        current = _make_run([_make_eval("tc-1", score=3)], run_id="cur")
        self._write_run(tmp_path, baseline)
        self._write_run(tmp_path, current)

        md_out = tmp_path / "diff.md"
        json_out = tmp_path / "diff.json"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                "base",
                "cur",
                "--output-dir",
                str(tmp_path),
                "--markdown",
                str(md_out),
                "--json",
                str(json_out),
            ],
        )

        assert result.exit_code == 0
        assert md_out.exists()
        assert "# PromptLens Comparison" in md_out.read_text()
        payload = json.loads(json_out.read_text())
        assert payload["baseline_run_id"] == "base"
        assert payload["case_comparisons"][0]["status"] == "regressed"

    def test_model_filter_option(self, tmp_path):
        baseline = _make_run(
            [
                _make_eval("tc-1", model="model-a", score=5),
                _make_eval("tc-1", model="model-b", score=5),
            ],
            run_id="base",
            models=["model-a", "model-b"],
        )
        current = _make_run(
            [
                _make_eval("tc-1", model="model-a", score=5),
                _make_eval("tc-1", model="model-b", score=1),
            ],
            run_id="cur",
            models=["model-a", "model-b"],
        )
        self._write_run(tmp_path, baseline)
        self._write_run(tmp_path, current)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                "base",
                "cur",
                "--output-dir",
                str(tmp_path),
                "--model",
                "model-a",
                "--max-regression",
                "0",
            ],
        )

        # model-b regressed but is filtered out, so the gate passes
        assert result.exit_code == 0
