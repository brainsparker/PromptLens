"""Tests for run-to-run comparison and regression detection."""

import json

import pytest

from promptlens.comparison import (
    DEFAULT_REGRESSION_THRESHOLD,
    compare_runs,
    load_run_result,
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
        "content": "The answer is 42." if not error else "",
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


def _make_eval(test_case_id, model="model-a", score=None, error=None, **response_kwargs):
    return EvaluationResult(
        test_case_id=test_case_id,
        query="What is the answer?",
        expected_behavior="Answers correctly",
        model_response=_make_response(model=model, error=error, **response_kwargs),
        judge_score=_make_score(score) if score is not None else None,
    )


def _make_run(run_id, evals, models=None):
    models = models or sorted({e.model_response.model for e in evals})
    return RunResult(
        run_id=run_id,
        golden_set_name="golden-set",
        models_tested=models,
        results=evals,
    )


class TestCompareRuns:
    def test_regression_detected_on_score_drop(self):
        baseline = _make_run("base", [_make_eval("tc-1", score=5)])
        current = _make_run("cur", [_make_eval("tc-1", score=3)])

        comparison = compare_runs(baseline, current)

        assert comparison.total_regressions == 1
        assert comparison.has_regressions
        case = comparison.cases[0]
        assert case.status == "regressed"
        assert case.score_delta == -2.0

    def test_improvement_detected_on_score_gain(self):
        baseline = _make_run("base", [_make_eval("tc-1", score=3)])
        current = _make_run("cur", [_make_eval("tc-1", score=5)])

        comparison = compare_runs(baseline, current)

        assert not comparison.has_regressions
        assert comparison.cases[0].status == "improved"

    def test_unchanged_within_threshold(self):
        baseline = _make_run("base", [_make_eval("tc-1", score=4)])
        current = _make_run("cur", [_make_eval("tc-1", score=4)])

        comparison = compare_runs(baseline, current)

        assert comparison.cases[0].status == "unchanged"
        assert not comparison.has_regressions

    def test_threshold_controls_sensitivity(self):
        baseline = _make_run("base", [_make_eval("tc-1", score=5)])
        current = _make_run("cur", [_make_eval("tc-1", score=4)])

        strict = compare_runs(baseline, current, threshold=0.5)
        lenient = compare_runs(baseline, current, threshold=2.0)

        assert strict.cases[0].status == "regressed"
        assert lenient.cases[0].status == "unchanged"

    def test_negative_threshold_rejected(self):
        run = _make_run("base", [_make_eval("tc-1", score=4)])
        with pytest.raises(ValueError):
            compare_runs(run, run, threshold=-1.0)

    def test_new_error_counts_as_regression(self):
        baseline = _make_run("base", [_make_eval("tc-1", score=5)])
        current = _make_run("cur", [_make_eval("tc-1", error="rate limited")])

        comparison = compare_runs(baseline, current)

        assert comparison.cases[0].status == "new_error"
        assert comparison.total_regressions == 1

    def test_fixed_error_not_a_regression(self):
        baseline = _make_run("base", [_make_eval("tc-1", error="rate limited")])
        current = _make_run("cur", [_make_eval("tc-1", score=4)])

        comparison = compare_runs(baseline, current)

        assert comparison.cases[0].status == "fixed_error"
        assert not comparison.has_regressions

    def test_unscored_pair_is_not_a_regression(self):
        baseline = _make_run("base", [_make_eval("tc-1", score=5)])
        current = _make_run("cur", [_make_eval("tc-1")])

        comparison = compare_runs(baseline, current)

        assert comparison.cases[0].status == "unscored"
        assert not comparison.has_regressions

    def test_added_and_removed_cases_reported_not_gated(self):
        baseline = _make_run(
            "base", [_make_eval("tc-1", score=4), _make_eval("tc-2", score=4)]
        )
        current = _make_run(
            "cur", [_make_eval("tc-1", score=4), _make_eval("tc-3", score=2)]
        )

        comparison = compare_runs(baseline, current)

        assert comparison.only_in_baseline == [("tc-2", "model-a")]
        assert comparison.only_in_current == [("tc-3", "model-a")]
        # tc-3 has a low score but no baseline, so it cannot regress.
        assert not comparison.has_regressions

    def test_cases_paired_per_model(self):
        baseline = _make_run(
            "base",
            [
                _make_eval("tc-1", model="model-a", score=5),
                _make_eval("tc-1", model="model-b", score=3),
            ],
        )
        current = _make_run(
            "cur",
            [
                _make_eval("tc-1", model="model-a", score=2),
                _make_eval("tc-1", model="model-b", score=5),
            ],
        )

        comparison = compare_runs(baseline, current)

        by_model = {s.model: s for s in comparison.model_summaries}
        assert by_model["model-a"].regressed == 1
        assert by_model["model-b"].improved == 1
        assert comparison.total_regressions == 1

    def test_model_summary_aggregates(self):
        baseline = _make_run(
            "base",
            [
                _make_eval("tc-1", score=5, latency_ms=1000.0, cost_usd=0.002),
                _make_eval("tc-2", score=5, latency_ms=1000.0, cost_usd=0.002),
            ],
        )
        current = _make_run(
            "cur",
            [
                _make_eval("tc-1", score=3, latency_ms=1500.0, cost_usd=0.003),
                _make_eval("tc-2", score=5, latency_ms=500.0, cost_usd=0.001),
            ],
        )

        comparison = compare_runs(baseline, current)
        summary = comparison.model_summaries[0]

        assert summary.baseline_avg_score == 5.0
        assert summary.current_avg_score == 4.0
        assert summary.avg_score_delta == -1.0
        assert summary.regressed == 1
        assert summary.unchanged == 1
        assert summary.latency_delta_ms == pytest.approx(0.0)
        assert summary.cost_delta_usd == pytest.approx(0.0)


class TestLoadRunResult:
    def test_load_round_trip(self, tmp_path):
        run = _make_run("base", [_make_eval("tc-1", score=4)])
        path = tmp_path / "results.json"
        path.write_text(run.model_dump_json(), encoding="utf-8")

        loaded = load_run_result(str(path))

        assert loaded.run_id == "base"
        assert loaded.results[0].judge_score.score == 4

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_run_result("/nonexistent/results.json")

    def test_invalid_json_raises_value_error(self, tmp_path):
        path = tmp_path / "results.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_run_result(str(path))

    def test_wrong_schema_raises_value_error(self, tmp_path):
        path = tmp_path / "results.json"
        path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
        with pytest.raises(ValueError, match="not a valid PromptLens run result"):
            load_run_result(str(path))


class TestRenderMarkdown:
    def test_markdown_contains_summary_and_regressions(self):
        baseline = _make_run(
            "base", [_make_eval("tc-1", score=5), _make_eval("tc-2", score=4)]
        )
        current = _make_run(
            "cur", [_make_eval("tc-1", score=2), _make_eval("tc-2", score=4)]
        )

        md = render_markdown(compare_runs(baseline, current))

        assert "# PromptLens Run Comparison" in md
        assert "`base`" in md and "`cur`" in md
        assert "1 regression(s) detected" in md
        assert "## Regressions" in md
        assert "tc-1" in md
        assert "-3.00" in md

    def test_markdown_clean_run(self):
        run_a = _make_run("base", [_make_eval("tc-1", score=4)])
        run_b = _make_run("cur", [_make_eval("tc-1", score=4)])

        md = render_markdown(compare_runs(run_a, run_b))

        assert "No regressions detected" in md
        assert "## Regressions" not in md

    def test_markdown_coverage_changes(self):
        baseline = _make_run("base", [_make_eval("tc-1", score=4)])
        current = _make_run(
            "cur", [_make_eval("tc-1", score=4), _make_eval("tc-2", score=4)]
        )

        md = render_markdown(compare_runs(baseline, current))

        assert "## Coverage Changes" in md
        assert "Added: `tc-2`" in md


class TestDefaultThreshold:
    def test_default_flags_full_point_drop(self):
        assert DEFAULT_REGRESSION_THRESHOLD == 0.5
        baseline = _make_run("base", [_make_eval("tc-1", score=4)])
        current = _make_run("cur", [_make_eval("tc-1", score=3)])
        assert compare_runs(baseline, current).cases[0].status == "regressed"
