"""Integration tests for deterministic checks: loader, results, exporters, CLI gate."""

import xml.etree.ElementTree as ET

import pytest

from promptlens.cli import _collect_check_failures
from promptlens.exporters.csv_exporter import CSVExporter
from promptlens.exporters.junit_exporter import JUnitXMLExporter
from promptlens.loaders.yaml_loader import YAMLLoader
from promptlens.models.checks import Check, CheckResult
from promptlens.models.config import JudgeConfig, RunConfig
from promptlens.models.result import (
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)


def _make_response(model="model-a", provider="anthropic", error=None):
    return ModelResponse(
        content="The answer is 42.",
        model=model,
        provider=provider,
        latency_ms=1234.5,
        cost_usd=0.0021,
        tokens_used=150,
        error=error,
    )


def _make_score(score, explanation="Looks correct."):
    return JudgeScore(
        score=score,
        explanation=explanation,
        judge_model="judge-model",
        judge_provider="anthropic",
    )


def _make_eval(test_case_id, model="model-a", score=None, error=None, check_results=None):
    return EvaluationResult(
        test_case_id=test_case_id,
        query="What is the answer?",
        expected_behavior="Answers correctly",
        model_response=_make_response(model=model, error=error),
        judge_score=_make_score(score) if score is not None else None,
        check_results=check_results or [],
    )


def _make_run(results, models=("model-a",)):
    return RunResult(
        run_id="testrun1",
        golden_set_name="Golden Set",
        models_tested=list(models),
        results=results,
    )


def _passed_check(description="contains: 'x'"):
    return CheckResult(
        check_type="contains", description=description, passed=True, detail="ok"
    )


def _failed_check(description="regex: 'ORD-\\d{5}'", detail="Pattern did not match"):
    return CheckResult(
        check_type="regex", description=description, passed=False, detail=detail
    )


class TestGoldenSetLoading:
    def test_yaml_loader_parses_checks(self, tmp_path):
        golden = tmp_path / "golden.yaml"
        golden.write_text(
            """
name: Checked Set
version: "1.0"
test_cases:
  - id: t1
    query: Classify sentiment
    expected_behavior: One word answer
    checks:
      - type: exact_match
        value: positive
      - type: not_contains
        value: ["negative"]
""",
            encoding="utf-8",
        )
        golden_set = YAMLLoader().load(str(golden))
        assert len(golden_set.test_cases[0].checks) == 2
        assert golden_set.test_cases[0].checks[0].type == "exact_match"

    def test_yaml_loader_rejects_bad_check_type(self, tmp_path):
        golden = tmp_path / "golden.yaml"
        golden.write_text(
            """
name: Bad Set
version: "1.0"
test_cases:
  - id: t1
    query: q
    expected_behavior: b
    checks:
      - type: made_up_check
        value: x
""",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            YAMLLoader().load(str(golden))

    def test_test_case_without_checks_still_loads(self, tmp_path):
        golden = tmp_path / "golden.yaml"
        golden.write_text(
            """
name: Plain Set
version: "1.0"
test_cases:
  - id: t1
    query: q
    expected_behavior: b
""",
            encoding="utf-8",
        )
        golden_set = YAMLLoader().load(str(golden))
        assert golden_set.test_cases[0].checks == []


class TestEvaluationResultChecks:
    def test_checks_passed_none_without_checks(self):
        assert _make_eval("t1").checks_passed is None

    def test_checks_passed_true_when_all_pass(self):
        result = _make_eval("t1", check_results=[_passed_check(), _passed_check()])
        assert result.checks_passed is True
        assert result.failed_checks == []

    def test_checks_passed_false_when_any_fail(self):
        result = _make_eval("t1", check_results=[_passed_check(), _failed_check()])
        assert result.checks_passed is False
        assert len(result.failed_checks) == 1

    def test_run_result_check_stats(self):
        run = _make_run(
            [
                _make_eval("t1", check_results=[_passed_check()]),
                _make_eval("t2", check_results=[_passed_check(), _failed_check()]),
                _make_eval("t3"),
            ]
        )
        stats = run.get_check_stats("model-a")
        assert stats["cases_with_checks"] == 2
        assert stats["cases_passed"] == 1
        assert stats["cases_failed"] == 1
        assert stats["checks_total"] == 3
        assert stats["checks_failed"] == 1


class TestJudgeConfig:
    def test_judge_enabled_by_default(self):
        assert JudgeConfig().enabled is True

    def test_junit_output_format_accepted(self):
        from promptlens.models.config import OutputConfig

        assert OutputConfig(formats=["junit"]).formats == ["junit"]

    def test_judge_can_be_disabled_via_run_config(self):
        config = RunConfig(
            golden_set="golden.yaml",
            models=[{"name": "M", "provider": "anthropic", "model": "m"}],
            judge={"enabled": False},
        )
        assert config.judge.enabled is False


class TestJUnitCheckMapping:
    def _export(self, results, fail_under=None):
        run = _make_run(results)
        exporter = JUnitXMLExporter(fail_under=fail_under)

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "junit.xml"
            exporter.export(run, str(path))
            return ET.parse(str(path)).getroot()

    def test_failed_check_reported_as_failure(self):
        root = self._export(
            [_make_eval("t1", score=5, check_results=[_failed_check()])]
        )
        testcase = root.find("testsuite/testcase")
        failure = testcase.find("failure")
        assert failure is not None
        assert failure.get("type") == "CheckFailure"
        assert root.get("failures") == "1"

    def test_passing_checks_without_judge_score_is_a_pass(self):
        root = self._export([_make_eval("t1", check_results=[_passed_check()])])
        testcase = root.find("testsuite/testcase")
        assert testcase.find("failure") is None
        assert testcase.find("skipped") is None
        assert testcase.find("error") is None
        assert root.get("skipped") == "0"

    def test_no_judge_and_no_checks_still_skipped(self):
        root = self._export([_make_eval("t1")])
        testcase = root.find("testsuite/testcase")
        assert testcase.find("skipped") is not None

    def test_check_failure_takes_precedence_over_judge_score(self):
        root = self._export(
            [_make_eval("t1", score=5, check_results=[_failed_check()])],
            fail_under=3.0,
        )
        failure = root.find("testsuite/testcase/failure")
        assert failure.get("type") == "CheckFailure"

    def test_check_summary_in_system_out(self):
        root = self._export(
            [_make_eval("t1", check_results=[_passed_check(), _failed_check()])]
        )
        system_out = root.find("testsuite/testcase/system-out")
        assert "checks: 1/2 passed" in system_out.text


class TestCSVCheckColumns:
    def test_csv_includes_check_columns(self, tmp_path):
        run = _make_run(
            [_make_eval("t1", score=4, check_results=[_passed_check(), _failed_check()])]
        )
        path = tmp_path / "results.csv"
        CSVExporter().export(run, str(path))
        content = path.read_text(encoding="utf-8")
        header = content.splitlines()[0]
        assert "checks_total" in header
        assert "checks_failed" in header
        assert "checks_passed" in header
        assert "false" in content


class TestCheckGate:
    def test_collect_check_failures_finds_failing_results(self):
        run = _make_run(
            [
                _make_eval("t1", check_results=[_passed_check()]),
                _make_eval("t2", check_results=[_failed_check()]),
                _make_eval("t3"),
            ]
        )
        failing = _collect_check_failures(run)
        assert [r.test_case_id for r in failing] == ["t2"]

    def test_collect_check_failures_empty_when_all_pass(self):
        run = _make_run([_make_eval("t1", check_results=[_passed_check()])])
        assert _collect_check_failures(run) == []


class TestBackwardCompatibility:
    def test_old_result_json_without_check_results_still_parses(self):
        data = {
            "test_case_id": "t1",
            "query": "q",
            "expected_behavior": "b",
            "model_response": {
                "content": "c",
                "model": "m",
                "provider": "p",
                "latency_ms": 1.0,
            },
        }
        result = EvaluationResult(**data)
        assert result.check_results == []
        assert result.checks_passed is None
