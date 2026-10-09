"""Tests for multi-sample stability analysis: pass@k, pass^k, flaky cases."""

import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime

import pytest

from promptlens.comparison import compare_runs, format_score, render_markdown
from promptlens.exporters.csv_exporter import CSVExporter
from promptlens.exporters.html_exporter import HTMLExporter
from promptlens.exporters.json_exporter import JSONExporter
from promptlens.exporters.junit_exporter import DEFAULT_FAIL_UNDER, JUnitXMLExporter
from promptlens.exporters.markdown_exporter import MarkdownExporter
from promptlens.models.comparison import STATUS_REGRESSED, STATUS_UNCHANGED
from promptlens.models.result import (
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)
from promptlens.stability import (
    DEFAULT_PASS_THRESHOLD,
    analyze_case,
    compute_stability,
    confidence_interval,
    format_scores,
    group_samples,
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


def _make_eval(test_case_id, score=None, sample_index=0, model="model-a", error=None, cost=0.002):
    return EvaluationResult(
        test_case_id=test_case_id,
        query="What is the answer?",
        expected_behavior="Answers correctly",
        model_response=_make_response(model=model, error=error, cost=cost),
        judge_score=(
            JudgeScore(
                score=score,
                explanation=f"Judged sample {sample_index}.",
                judge_model="judge-model",
                judge_provider="anthropic",
            )
            if score is not None
            else None
        ),
        sample_index=sample_index,
    )


def _sampled(test_case_id, scores, model="model-a"):
    """One EvaluationResult per score, indexed in order. None means errored."""
    results = []
    for index, score in enumerate(scores):
        if score == "error":
            results.append(_make_eval(test_case_id, sample_index=index, model=model, error="boom"))
        else:
            results.append(_make_eval(test_case_id, score=score, sample_index=index, model=model))
    return results


def _make_run(results, models=None, samples=None, run_id="run-1", metadata=None):
    metadata = dict(metadata or {})
    if samples is not None:
        metadata["samples"] = samples
    return RunResult(
        run_id=run_id,
        run_name="stability-run",
        timestamp=datetime(2026, 10, 9, 12, 0, 0),
        golden_set_name="golden-set",
        models_tested=models or ["model-a"],
        results=results,
        metadata=metadata,
    )


class TestRunResultSampleCount:
    def test_defaults_to_one_for_legacy_runs(self):
        run = _make_run([_make_eval("tc-1", score=4)])
        assert run.sample_count == 1
        assert run.test_case_count == 1

    def test_reads_samples_from_metadata(self):
        run = _make_run([_make_eval("tc-1", score=4)], samples=5)
        assert run.sample_count == 5

    def test_falls_back_to_highest_sample_index(self):
        run = _make_run(_sampled("tc-1", [4, 2, 5]))
        assert run.sample_count == 3

    def test_sample_index_defaults_to_zero_when_absent_from_json(self):
        data = _make_eval("tc-1", score=4).model_dump(mode="json")
        del data["sample_index"]
        restored = EvaluationResult(**data)
        assert restored.sample_index == 0

    def test_empty_run_has_one_sample(self):
        assert _make_run([]).sample_count == 1


class TestConfidenceInterval:
    def test_empty_returns_none(self):
        assert confidence_interval([]) == (None, None)

    def test_single_value_is_zero_width(self):
        assert confidence_interval([4.0]) == (4.0, 4.0)

    def test_identical_values_have_zero_width(self):
        low, high = confidence_interval([3.0, 3.0, 3.0])
        assert low == pytest.approx(3.0)
        assert high == pytest.approx(3.0)

    def test_interval_is_symmetric_around_mean(self):
        values = [1.0, 5.0, 3.0, 3.0]
        low, high = confidence_interval(values)
        mean = sum(values) / len(values)
        assert low < mean < high
        assert (mean - low) == pytest.approx(high - mean)
        # Sample std dev of [1,5,3,3] is sqrt(8/3); half-width is 1.96 * sd / 2
        expected_half = 1.96 * math.sqrt(8 / 3) / 2
        assert (high - mean) == pytest.approx(expected_half)


class TestAnalyzeCase:
    def test_always_passing_case_is_stable(self):
        case = analyze_case("tc-1", "model-a", _sampled("tc-1", [4, 5, 4]), 3.0)
        assert case.pass_at_k is True
        assert case.pass_hat_k is True
        assert case.flaky is False
        assert case.passed_samples == 3
        assert case.pass_rate == 1.0
        assert case.mean_score == pytest.approx(13 / 3)
        assert case.min_score == 4
        assert case.max_score == 5

    def test_always_failing_case_is_stable_but_failing(self):
        case = analyze_case("tc-1", "model-a", _sampled("tc-1", [1, 2, 2]), 3.0)
        assert case.pass_at_k is False
        assert case.pass_hat_k is False
        assert case.flaky is False
        assert case.passed_samples == 0

    def test_mixed_case_is_flaky(self):
        case = analyze_case("tc-1", "model-a", _sampled("tc-1", [5, 2, 5]), 3.0)
        assert case.pass_at_k is True
        assert case.pass_hat_k is False
        assert case.flaky is True
        assert case.passed_samples == 2
        assert case.scored_samples == 3
        assert case.pass_rate == pytest.approx(2 / 3)
        assert case.std_dev == pytest.approx(math.sqrt(3.0))

    def test_threshold_is_inclusive(self):
        case = analyze_case("tc-1", "model-a", _sampled("tc-1", [3, 3]), 3.0)
        assert case.pass_hat_k is True

    def test_errored_samples_are_excluded_from_pass_fail(self):
        case = analyze_case("tc-1", "model-a", _sampled("tc-1", [5, "error", 5]), 3.0)
        assert case.samples == 3
        assert case.scored_samples == 2
        assert case.errored_samples == 1
        assert case.pass_hat_k is True
        assert case.flaky is False

    def test_unjudged_samples_are_not_scored(self):
        results = [_make_eval("tc-1", sample_index=0), _make_eval("tc-1", score=4, sample_index=1)]
        case = analyze_case("tc-1", "model-a", results, 3.0)
        assert case.scored_samples == 1
        assert case.scores == [4]

    def test_single_sample_has_zero_std_dev(self):
        case = analyze_case("tc-1", "model-a", _sampled("tc-1", [4]), 3.0)
        assert case.std_dev == 0.0

    def test_nothing_scored_is_neither_pass_nor_flaky(self):
        case = analyze_case("tc-1", "model-a", _sampled("tc-1", ["error", "error"]), 3.0)
        assert case.pass_at_k is False
        assert case.pass_hat_k is False
        assert case.flaky is False
        assert case.pass_rate is None
        assert case.mean_score is None


class TestComputeStability:
    def _run(self):
        results = (
            _sampled("tc-stable", [5, 4, 5])
            + _sampled("tc-flaky", [5, 2, 4])
            + _sampled("tc-bad", [1, 2, 1])
            + _sampled("tc-errored", ["error", "error", "error"])
        )
        return _make_run(results, samples=3)

    def test_model_aggregates(self):
        report = compute_stability(self._run(), pass_threshold=3.0)
        assert report.samples == 3
        assert report.pass_threshold == 3.0
        assert len(report.models) == 1
        model = report.models[0]
        assert model.cases == 4
        assert model.scored_cases == 3
        assert model.pass_at_k_cases == 2
        assert model.pass_hat_k_cases == 1
        assert model.flaky_cases == 1
        assert model.pass_at_k == pytest.approx(2 / 3)
        assert model.pass_hat_k == pytest.approx(1 / 3)
        assert model.flaky_rate == pytest.approx(1 / 3)
        assert model.mean_score == pytest.approx(29 / 9)
        assert model.score_ci_low < model.mean_score < model.score_ci_high

    def test_flaky_cases_listed(self):
        report = compute_stability(self._run())
        assert [c.test_case_id for c in report.flaky_cases] == ["tc-flaky"]
        assert report.has_flaky_cases is True
        assert report.is_multi_sample is True

    def test_default_threshold_is_shared_with_junit(self):
        assert DEFAULT_PASS_THRESHOLD == DEFAULT_FAIL_UNDER == 3.0

    def test_threshold_read_from_run_metadata(self):
        run = _make_run(_sampled("tc-1", [4, 3, 4]), samples=3, metadata={"pass_threshold": 4.0})
        report = compute_stability(run)
        assert report.pass_threshold == 4.0
        assert report.flaky_cases[0].test_case_id == "tc-1"

    def test_explicit_threshold_overrides_metadata(self):
        run = _make_run(_sampled("tc-1", [4, 3, 4]), samples=3, metadata={"pass_threshold": 4.0})
        report = compute_stability(run, pass_threshold=3.0)
        assert report.has_flaky_cases is False

    def test_single_sample_run_is_trivially_stable(self):
        run = _make_run([_make_eval("tc-1", score=4), _make_eval("tc-2", score=1)])
        report = compute_stability(run)
        assert report.is_multi_sample is False
        assert report.has_flaky_cases is False
        assert report.models[0].pass_hat_k == pytest.approx(0.5)

    def test_models_without_results_are_reported_empty(self):
        run = _make_run(_sampled("tc-1", [4, 4]), models=["model-a", "model-b"], samples=2)
        report = compute_stability(run)
        by_model = {m.model: m for m in report.models}
        assert by_model["model-b"].scored_cases == 0
        assert by_model["model-b"].pass_hat_k is None

    def test_group_samples_orders_by_sample_index(self):
        results = list(reversed(_sampled("tc-1", [1, 2, 3])))
        groups = group_samples(_make_run(results))
        assert [r.sample_index for r in groups[("tc-1", "model-a")]] == [0, 1, 2]

    def test_format_scores(self):
        assert format_scores([5, 2, 5]) == "5, 2, 5"
        assert format_scores([]) == "-"


class TestExportersWithSamples:
    def _run(self):
        results = _sampled("tc-stable", [5, 4, 5]) + _sampled("tc-flaky", [5, 2, 4])
        return _make_run(results, samples=3, metadata={"pass_threshold": 3.0})

    def test_json_includes_stability_and_round_trips(self, tmp_path):
        output = tmp_path / "results.json"
        JSONExporter().export(self._run(), str(output))
        data = json.loads(output.read_text())

        assert data["metadata"]["samples"] == 3
        assert data["stability"]["samples"] == 3
        assert data["stability"]["models"][0]["flaky_cases"] == 1
        assert all("sample_index" in r for r in data["results"])

        restored = RunResult(**data)
        assert restored.sample_count == 3
        assert len(restored.results) == 6

    def test_json_omits_stability_for_single_sample_runs(self, tmp_path):
        output = tmp_path / "results.json"
        JSONExporter().export(_make_run([_make_eval("tc-1", score=4)]), str(output))
        assert "stability" not in json.loads(output.read_text())

    def test_csv_has_sample_index_column(self, tmp_path):
        output = tmp_path / "results.csv"
        CSVExporter().export(self._run(), str(output))
        header = output.read_text().splitlines()[0]
        assert "sample_index" in header.split(",")

    def test_markdown_has_stability_section(self, tmp_path):
        output = tmp_path / "results.md"
        MarkdownExporter().export(self._run(), str(output))
        text = output.read_text()
        assert "## Stability (3 samples per test case)" in text
        assert "pass@3" in text and "pass^3" in text
        assert "### Flaky Cases (1)" in text
        assert "`tc-flaky`" in text
        assert "| Samples per Test Case | 3 |" in text
        assert "| Test Cases | 2 |" in text
        assert "| Model | Sample | Score |" in text

    def test_markdown_unchanged_for_single_sample_runs(self, tmp_path):
        output = tmp_path / "results.md"
        MarkdownExporter().export(_make_run([_make_eval("tc-1", score=4)]), str(output))
        text = output.read_text()
        assert "Stability" not in text
        assert "| Model | Score | Latency |" in text

    def test_html_has_stability_section(self, tmp_path):
        output = tmp_path / "report.html"
        HTMLExporter().export(self._run(), str(output))
        html = output.read_text()
        assert "Stability (3 samples per test case)" in html
        assert "Flaky cases (1)" in html
        assert "flaky: model-a" in html
        assert "sample 1/3" in html
        assert "Samples per case" in html

    def test_html_unchanged_for_single_sample_runs(self, tmp_path):
        output = tmp_path / "report.html"
        HTMLExporter().export(_make_run([_make_eval("tc-1", score=4)]), str(output))
        html = output.read_text()
        assert "Stability" not in html
        assert "sample 1/1" not in html


class TestJUnitWithSamples:
    def _export(self, run, tmp_path, fail_under=None):
        output = tmp_path / "junit.xml"
        JUnitXMLExporter(fail_under=fail_under).export(run, str(output))
        return ET.parse(str(output)).getroot()

    def test_one_testcase_per_golden_entry(self, tmp_path):
        run = _make_run(_sampled("tc-1", [5, 4, 5]) + _sampled("tc-2", [4, 4, 4]), samples=3)
        root = self._export(run, tmp_path)
        suite = root.find("testsuite")
        assert suite.get("tests") == "3" or suite.get("tests") == "2"
        assert len(suite.findall("testcase")) == 2
        assert root.get("tests") == "2"
        assert root.get("failures") == "0"

    def test_flaky_case_is_failure_with_flaky_type(self, tmp_path):
        run = _make_run(_sampled("tc-1", [5, 2, 5]), samples=3)
        root = self._export(run, tmp_path)
        testcase = root.find("testsuite").find("testcase")
        failure = testcase.find("failure")
        assert failure is not None
        assert failure.get("type") == "FlakyJudgeScore"
        assert "1 of 3 samples" in failure.get("message")
        assert "5, 2, 5" in failure.get("message")
        assert "flaky: true" in testcase.find("system-out").text
        assert root.get("failures") == "1"

    def test_consistently_failing_case_keeps_threshold_type(self, tmp_path):
        run = _make_run(_sampled("tc-1", [1, 2, 2]), samples=3)
        root = self._export(run, tmp_path)
        failure = root.find("testsuite").find("testcase").find("failure")
        assert failure.get("type") == "JudgeScoreBelowThreshold"
        assert "All 3 samples" in failure.get("message")

    def test_partial_errors_do_not_fail_a_passing_case(self, tmp_path):
        run = _make_run(_sampled("tc-1", [5, "error", 4]), samples=3)
        root = self._export(run, tmp_path)
        testcase = root.find("testsuite").find("testcase")
        assert testcase.find("failure") is None
        assert testcase.find("error") is None
        assert "errored_samples: 1" in testcase.find("system-out").text

    def test_all_errors_is_error(self, tmp_path):
        run = _make_run(_sampled("tc-1", ["error", "error"]), samples=2)
        root = self._export(run, tmp_path)
        error = root.find("testsuite").find("testcase").find("error")
        assert error is not None
        assert "All 2 samples errored" in error.get("message")
        assert root.get("errors") == "1"

    def test_unjudged_samples_are_skipped(self, tmp_path):
        results = [_make_eval("tc-1", sample_index=0), _make_eval("tc-1", sample_index=1)]
        root = self._export(_make_run(results, samples=2), tmp_path)
        assert root.find("testsuite").find("testcase").find("skipped") is not None

    def test_threshold_is_honored(self, tmp_path):
        run = _make_run(_sampled("tc-1", [4, 3, 4]), samples=3)
        assert self._export(run, tmp_path).get("failures") == "0"
        assert self._export(run, tmp_path, fail_under=4.0).get("failures") == "1"

    def test_samples_property_recorded(self, tmp_path):
        run = _make_run(_sampled("tc-1", [4, 4]), samples=2)
        root = self._export(run, tmp_path)
        props = {p.get("name"): p.get("value") for p in root.find("testsuite").find("properties")}
        assert props["samples"] == "2"
        assert props["fail_under"] == "3"


class TestCompareWithSamples:
    def test_multi_sample_runs_compare_on_mean_score(self):
        baseline = _make_run(_sampled("tc-1", [5, 5, 5]), samples=3, run_id="base")
        candidate = _make_run(_sampled("tc-1", [5, 2, 2]), samples=3, run_id="cand")

        comparison = compare_runs(baseline, candidate)

        case = comparison.cases[0]
        assert case.baseline_score == pytest.approx(5.0)
        assert case.candidate_score == pytest.approx(3.0)
        assert case.score_delta == pytest.approx(-2.0)
        assert case.status == STATUS_REGRESSED
        assert case.baseline_samples == 3
        assert case.candidate_samples == 3
        assert comparison.baseline_samples == 3

    def test_small_mean_shift_stays_within_threshold(self):
        baseline = _make_run(_sampled("tc-1", [4, 4, 4]), samples=3, run_id="base")
        candidate = _make_run(_sampled("tc-1", [4, 4, 3]), samples=3, run_id="cand")

        comparison = compare_runs(baseline, candidate)

        assert comparison.cases[0].status == STATUS_UNCHANGED
        assert comparison.cases[0].score_delta == pytest.approx(-1 / 3)

    def test_multi_sample_against_single_sample_baseline(self):
        baseline = _make_run([_make_eval("tc-1", score=4)], run_id="base")
        candidate = _make_run(_sampled("tc-1", [4, 5, 4]), samples=3, run_id="cand")

        comparison = compare_runs(baseline, candidate)

        case = comparison.cases[0]
        assert case.baseline_samples == 1
        assert case.candidate_samples == 3
        assert case.status == STATUS_UNCHANGED

    def test_partial_errors_are_not_a_new_error(self):
        baseline = _make_run(_sampled("tc-1", [4, 4]), samples=2, run_id="base")
        candidate = _make_run(_sampled("tc-1", [4, "error"]), samples=2, run_id="cand")

        comparison = compare_runs(baseline, candidate)

        assert comparison.cases[0].candidate_error is None
        assert comparison.cases[0].status == STATUS_UNCHANGED

    def test_per_sample_cost_is_averaged_for_comparability(self):
        baseline = _make_run([_make_eval("tc-1", score=4, cost=0.01)], run_id="base")
        candidate = _make_run(
            [_make_eval("tc-1", score=4, sample_index=i, cost=0.01) for i in range(4)],
            samples=4,
            run_id="cand",
        )

        comparison = compare_runs(baseline, candidate)

        assert comparison.model_summaries[0].cost_delta_usd == pytest.approx(0.0)

    def test_markdown_renders_fractional_scores_and_sample_note(self):
        baseline = _make_run(_sampled("tc-1", [5, 5, 5]), samples=3, run_id="base")
        candidate = _make_run(_sampled("tc-1", [5, 2, 2]), samples=3, run_id="cand")
        markdown = render_markdown(compare_runs(baseline, candidate))
        assert "**Samples per Test Case:** baseline 3, candidate 3" in markdown
        assert "| tc-1 | model-a | 5 | 3 | -2 | score_drop |" in markdown

    def test_format_score(self):
        assert format_score(None) == "-"
        assert format_score(4.0) == "4"
        assert format_score(3.6666) == "3.67"
        assert format_score(-2.0, signed=True) == "-2"
        assert format_score(0.5, signed=True) == "+0.50"
