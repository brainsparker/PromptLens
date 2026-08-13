"""Tests for the JUnit XML exporter and the --fail-under quality gate."""

import xml.etree.ElementTree as ET
from datetime import datetime

import pytest

from promptlens.cli import _check_fail_under
from promptlens.exporters.junit_exporter import (
    DEFAULT_FAIL_UNDER,
    JUnitXMLExporter,
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
        "latency_ms": 1234.5,
        "cost_usd": 0.0021,
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


def _make_eval(test_case_id, model="model-a", score=None, error=None):
    return EvaluationResult(
        test_case_id=test_case_id,
        query="What is the answer?",
        expected_behavior="Answers correctly",
        model_response=_make_response(model=model, error=error),
        judge_score=_make_score(score) if score is not None else None,
    )


def _make_run(results, models=None, run_name="ci-run"):
    models = models or ["model-a"]
    return RunResult(
        run_id="run-123",
        run_name=run_name,
        timestamp=datetime(2026, 8, 13, 12, 0, 0),
        golden_set_name="golden-set",
        models_tested=models,
        results=results,
    )


def _export(run_result, tmp_path, fail_under=None):
    exporter = JUnitXMLExporter(fail_under=fail_under)
    output = tmp_path / "junit.xml"
    exporter.export(run_result, str(output))
    return ET.parse(str(output)).getroot()


class TestJUnitXMLExporter:
    def test_file_extension(self):
        assert JUnitXMLExporter().file_extension == ".xml"

    def test_passing_case_has_no_failure_children(self, tmp_path):
        run = _make_run([_make_eval("tc-1", score=5)])
        root = _export(run, tmp_path)

        assert root.tag == "testsuites"
        assert root.get("tests") == "1"
        assert root.get("failures") == "0"
        assert root.get("errors") == "0"
        testcase = root.find("./testsuite/testcase")
        assert testcase.get("name") == "tc-1"
        assert testcase.find("failure") is None
        assert testcase.find("error") is None
        assert testcase.find("skipped") is None

    def test_low_score_marked_as_failure(self, tmp_path):
        run = _make_run([_make_eval("tc-1", score=1)])
        root = _export(run, tmp_path)

        assert root.get("failures") == "1"
        failure = root.find("./testsuite/testcase/failure")
        assert failure is not None
        assert failure.get("type") == "JudgeScoreBelowThreshold"
        assert "below" in failure.get("message")
        assert "Explanation:" in failure.text

    def test_custom_fail_under_threshold(self, tmp_path):
        # Score of 4 passes the default gate but fails a 4.5 gate
        run = _make_run([_make_eval("tc-1", score=4)])
        root = _export(run, tmp_path, fail_under=4.5)
        assert root.get("failures") == "1"

        root = _export(run, tmp_path)
        assert root.get("failures") == "0"

    def test_default_threshold_constant(self):
        assert JUnitXMLExporter().fail_under == DEFAULT_FAIL_UNDER

    def test_model_error_marked_as_error(self, tmp_path):
        run = _make_run([_make_eval("tc-1", error="API timeout")])
        root = _export(run, tmp_path)

        assert root.get("errors") == "1"
        assert root.get("failures") == "0"
        error = root.find("./testsuite/testcase/error")
        assert error is not None
        assert error.get("type") == "ModelResponseError"
        assert "API timeout" in error.get("message")

    def test_unjudged_case_marked_as_skipped(self, tmp_path):
        run = _make_run([_make_eval("tc-1")])
        root = _export(run, tmp_path)

        assert root.get("skipped") == "1"
        assert root.get("failures") == "0"
        skipped = root.find("./testsuite/testcase/skipped")
        assert skipped is not None

    def test_one_suite_per_model(self, tmp_path):
        results = [
            _make_eval("tc-1", model="model-a", score=5),
            _make_eval("tc-1", model="model-b", score=2),
        ]
        run = _make_run(results, models=["model-a", "model-b"])
        root = _export(run, tmp_path)

        suites = root.findall("testsuite")
        assert [s.get("name") for s in suites] == ["model-a", "model-b"]
        assert suites[0].get("failures") == "0"
        assert suites[1].get("failures") == "1"
        assert root.get("tests") == "2"
        assert root.get("failures") == "1"

    def test_suite_properties_include_run_metadata(self, tmp_path):
        run = _make_run([_make_eval("tc-1", score=4)])
        root = _export(run, tmp_path)

        props = {
            p.get("name"): p.get("value")
            for p in root.findall("./testsuite/properties/property")
        }
        assert props["model"] == "model-a"
        assert props["provider"] == "anthropic"
        assert props["run_id"] == "run-123"
        assert props["golden_set"] == "golden-set"
        assert props["average_judge_score"] == "4.00"
        assert "total_cost_usd" in props

    def test_testcase_time_uses_latency_seconds(self, tmp_path):
        run = _make_run([_make_eval("tc-1", score=5)])
        root = _export(run, tmp_path)

        testcase = root.find("./testsuite/testcase")
        assert testcase.get("time") == "1.234"
        assert testcase.get("classname") == "golden-set.model-a"

    def test_special_characters_are_escaped(self, tmp_path):
        result = EvaluationResult(
            test_case_id="tc-<&>",
            query='Query with <tags> & "quotes"',
            expected_behavior="Behaves & renders <b>bold</b>",
            model_response=_make_response(),
            judge_score=_make_score(1, explanation='Bad <output> & "wrong"'),
        )
        run = _make_run([result])
        root = _export(run, tmp_path)

        # Parsing succeeded, and content round-trips intact
        testcase = root.find("./testsuite/testcase")
        assert testcase.get("name") == "tc-<&>"
        failure = testcase.find("failure")
        assert 'Bad <output> & "wrong"' in failure.text

    def test_creates_output_directory(self, tmp_path):
        run = _make_run([_make_eval("tc-1", score=5)])
        nested = tmp_path / "deep" / "nested" / "junit.xml"
        JUnitXMLExporter().export(run, str(nested))
        assert nested.exists()

    def test_system_out_contains_metrics(self, tmp_path):
        run = _make_run([_make_eval("tc-1", score=5)])
        root = _export(run, tmp_path)

        system_out = root.find("./testsuite/testcase/system-out")
        assert "provider: anthropic" in system_out.text
        assert "cost_usd: 0.0021" in system_out.text
        assert "judge_score: 5" in system_out.text


class TestFailUnderGate:
    def test_all_models_above_gate(self):
        run = _make_run([_make_eval("tc-1", score=4), _make_eval("tc-2", score=5)])
        assert _check_fail_under(run, 3.0) == []

    def test_model_below_gate_is_reported(self):
        run = _make_run([_make_eval("tc-1", score=2), _make_eval("tc-2", score=2)])
        failing = _check_fail_under(run, 3.0)
        assert len(failing) == 1
        model, avg = failing[0]
        assert model == "model-a"
        assert avg == pytest.approx(2.0)

    def test_average_exactly_at_gate_passes(self):
        run = _make_run([_make_eval("tc-1", score=3)])
        assert _check_fail_under(run, 3.0) == []

    def test_model_with_no_scores_fails_gate(self):
        run = _make_run([_make_eval("tc-1")])
        failing = _check_fail_under(run, 3.0)
        assert failing == [("model-a", None)]

    def test_mixed_models_only_failing_reported(self):
        results = [
            _make_eval("tc-1", model="model-a", score=5),
            _make_eval("tc-1", model="model-b", score=1),
        ]
        run = _make_run(results, models=["model-a", "model-b"])
        failing = _check_fail_under(run, 3.0)
        assert [m for m, _ in failing] == ["model-b"]
