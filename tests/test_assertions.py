"""Tests for deterministic assertions: models, evaluation, loader parsing,
runner integration, JUnit mapping, and the --fail-under gate."""

from datetime import datetime

import pytest
import xml.etree.ElementTree as ET

from promptlens.cli import _check_fail_under
from promptlens.exporters.junit_exporter import JUnitXMLExporter
from promptlens.judges.assertions import evaluate_assertions
from promptlens.loaders.yaml_loader import YAMLLoader
from promptlens.models.result import (
    AssertionResult,
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)
from promptlens.models.test_case import Assertion, TestCase


def _assertion(type_, value=None):
    return Assertion(type=type_, value=value)


def _make_response(model="model-a", provider="anthropic", error=None, content="hello"):
    return ModelResponse(
        content=content,
        model=model,
        provider=provider,
        latency_ms=100.0,
        cost_usd=0.001,
        error=error,
    )


def _make_eval(
    test_case_id="case-1",
    model="model-a",
    judge_score=None,
    assertion_results=None,
    error=None,
):
    return EvaluationResult(
        test_case_id=test_case_id,
        query="q",
        expected_behavior="e",
        model_response=_make_response(model=model, error=error),
        judge_score=judge_score,
        assertion_results=assertion_results or [],
    )


def _make_run(results, models=None):
    return RunResult(
        run_id="run-1",
        golden_set_name="gs",
        models_tested=models or ["model-a"],
        results=results,
        timestamp=datetime(2026, 8, 23, 12, 0, 0),
    )


def _judge(score):
    return JudgeScore(
        score=score,
        explanation="x",
        judge_model="judge",
        judge_provider="anthropic",
    )


class TestAssertionModel:
    def test_valid_types_accepted(self):
        assert _assertion("is_json").type == "is_json"
        assert _assertion("contains", "x").value == "x"
        assert _assertion("json_schema", {"type": "object"}).value == {"type": "object"}

    def test_unknown_type_rejected(self):
        with pytest.raises(ValueError, match="Unsupported assertion type"):
            _assertion("fuzzy_match", "x")

    def test_value_required_for_typed_assertions(self):
        for type_ in ("json_schema", "contains", "not_contains", "regex", "starts_with"):
            with pytest.raises(ValueError, match="requires a"):
                _assertion(type_)

    def test_value_type_enforced(self):
        with pytest.raises(ValueError, match="requires a str"):
            _assertion("contains", 42)
        with pytest.raises(ValueError, match="requires a dict"):
            _assertion("json_schema", "not-a-schema")

    def test_is_json_needs_no_value(self):
        assert _assertion("is_json").value is None


class TestEvaluateAssertions:
    def test_is_json_pass_and_fail(self):
        (ok,) = evaluate_assertions('{"a": 1}', [_assertion("is_json")])
        assert ok.passed is True
        (bad,) = evaluate_assertions("not json", [_assertion("is_json")])
        assert bad.passed is False
        assert "not valid JSON" in bad.message

    def test_json_schema_pass(self):
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        (res,) = evaluate_assertions('{"name": "Jane"}', [_assertion("json_schema", schema)])
        assert res.passed is True

    def test_json_schema_fail_reports_missing_required(self):
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"age": {"type": "integer"}},
        }
        (res,) = evaluate_assertions('{"age": 30}', [_assertion("json_schema", schema)])
        assert res.passed is False
        assert "name" in res.message

    def test_json_schema_fail_reports_path_for_wrong_type(self):
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer"}},
        }
        (res,) = evaluate_assertions('{"age": "old"}', [_assertion("json_schema", schema)])
        assert res.passed is False
        assert "age" in res.message

    def test_json_schema_on_non_json_response(self):
        (res,) = evaluate_assertions("plain text", [_assertion("json_schema", {"type": "object"})])
        assert res.passed is False
        assert "cannot check schema" in res.message

    def test_contains_and_not_contains(self):
        (res,) = evaluate_assertions("hello world", [_assertion("contains", "world")])
        assert res.passed is True
        (res,) = evaluate_assertions("hello world", [_assertion("contains", "World")])
        assert res.passed is False  # case-sensitive
        (res,) = evaluate_assertions("hello world", [_assertion("not_contains", "moon")])
        assert res.passed is True
        (res,) = evaluate_assertions("hello world", [_assertion("not_contains", "world")])
        assert res.passed is False

    def test_regex(self):
        (res,) = evaluate_assertions("score: 42", [_assertion("regex", r"score: \d+")])
        assert res.passed is True
        (res,) = evaluate_assertions("score: n/a", [_assertion("regex", r"score: \d+")])
        assert res.passed is False

    def test_invalid_regex_fails_with_message(self):
        (res,) = evaluate_assertions("x", [_assertion("regex", "(unclosed")])
        assert res.passed is False
        assert "Invalid regex" in res.message

    def test_starts_with_ignores_leading_whitespace(self):
        (res,) = evaluate_assertions("\n  Answer: yes", [_assertion("starts_with", "Answer:")])
        assert res.passed is True
        (res,) = evaluate_assertions("Well, Answer: yes", [_assertion("starts_with", "Answer:")])
        assert res.passed is False

    def test_all_assertions_evaluated_no_short_circuit(self):
        results = evaluate_assertions(
            "not json",
            [_assertion("is_json"), _assertion("contains", "json")],
        )
        assert len(results) == 2
        assert results[0].passed is False
        assert results[1].passed is True


class TestYamlAssertKey:
    def test_assert_key_parses_into_assertions(self, tmp_path):
        golden = tmp_path / "golden.yaml"
        golden.write_text(
            """
name: "Assert Set"
version: "1.0"
test_cases:
  - id: "a-001"
    query: "Return JSON"
    expected_behavior: "Valid JSON"
    assert:
      - type: is_json
      - type: contains
        value: "name"
"""
        )
        golden_set = YAMLLoader().load(str(golden))
        case = golden_set.test_cases[0]
        assert len(case.assertions) == 2
        assert case.assertions[0].type == "is_json"
        assert case.assertions[1].value == "name"

    def test_cases_without_assert_key_still_load(self, tmp_path):
        golden = tmp_path / "golden.yaml"
        golden.write_text(
            """
name: "Plain Set"
version: "1.0"
test_cases:
  - id: "p-001"
    query: "Hello"
    expected_behavior: "Greeting"
"""
        )
        golden_set = YAMLLoader().load(str(golden))
        assert golden_set.test_cases[0].assertions == []

    def test_invalid_assertion_type_rejected_at_load(self, tmp_path):
        golden = tmp_path / "golden.yaml"
        golden.write_text(
            """
name: "Bad Set"
version: "1.0"
test_cases:
  - id: "b-001"
    query: "Hello"
    expected_behavior: "Greeting"
    assert:
      - type: vibes
"""
        )
        with pytest.raises(ValueError, match="Unsupported assertion type"):
            YAMLLoader().load(str(golden))

    def test_construction_by_field_name_also_works(self):
        case = TestCase(
            id="c-1",
            query="q",
            expected_behavior="e",
            assertions=[_assertion("is_json")],
        )
        assert case.assertions[0].type == "is_json"


class TestEvaluationResultAssertions:
    def test_assertions_passed_none_when_no_assertions(self):
        assert _make_eval().assertions_passed() is None

    def test_assertions_passed_true_and_false(self):
        passing = _make_eval(
            assertion_results=[
                AssertionResult(type="is_json", passed=True, message="ok"),
            ]
        )
        assert passing.assertions_passed() is True
        failing = _make_eval(
            assertion_results=[
                AssertionResult(type="is_json", passed=True, message="ok"),
                AssertionResult(type="contains", value="x", passed=False, message="no"),
            ]
        )
        assert failing.assertions_passed() is False


class TestRunnerAssertionIntegration:
    @pytest.mark.asyncio
    async def test_failed_assertion_skips_judge(self, mocker):
        from promptlens.runners.runner import Runner

        runner = Runner.__new__(Runner)
        runner.config = mocker.MagicMock()
        runner.config.execution.retry_attempts = 0
        runner.config.execution.retry_delay_seconds = 0
        runner.config.execution.timeout_seconds = 30
        runner.semaphore = __import__("asyncio").Semaphore(1)
        runner.judge = mocker.MagicMock()
        runner.judge.evaluate = mocker.AsyncMock(return_value=_judge(5))

        provider = mocker.MagicMock()
        provider.supports_tools.return_value = True
        provider.generate = mocker.AsyncMock(return_value=_make_response(content="not json"))

        test_case = TestCase(
            id="t-1",
            query="Return JSON",
            expected_behavior="Valid JSON",
            assertions=[_assertion("is_json")],
        )

        progress = mocker.MagicMock()
        result = await runner._evaluate_single(test_case, provider, progress, 0)

        assert result.assertions_passed() is False
        assert result.judge_score is None
        runner.judge.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_passing_assertions_still_judge(self, mocker):
        from promptlens.runners.runner import Runner

        runner = Runner.__new__(Runner)
        runner.config = mocker.MagicMock()
        runner.config.execution.retry_attempts = 0
        runner.config.execution.retry_delay_seconds = 0
        runner.config.execution.timeout_seconds = 30
        runner.semaphore = __import__("asyncio").Semaphore(1)
        runner.judge = mocker.MagicMock()
        runner.judge.evaluate = mocker.AsyncMock(return_value=_judge(4))

        provider = mocker.MagicMock()
        provider.supports_tools.return_value = True
        provider.generate = mocker.AsyncMock(
            return_value=_make_response(content='{"name": "Jane"}')
        )

        test_case = TestCase(
            id="t-2",
            query="Return JSON",
            expected_behavior="Valid JSON",
            assertions=[_assertion("is_json")],
        )

        progress = mocker.MagicMock()
        result = await runner._evaluate_single(test_case, provider, progress, 0)

        assert result.assertions_passed() is True
        assert result.judge_score is not None
        assert result.judge_score.score == 4
        runner.judge.evaluate.assert_called_once()


class TestJUnitAssertionMapping:
    def _export(self, results, tmp_path):
        run = _make_run(results)
        out = tmp_path / "junit.xml"
        JUnitXMLExporter().export(run, str(out))
        return ET.parse(out).getroot()

    def test_failed_assertion_reported_as_failure(self, tmp_path):
        eval_result = _make_eval(
            assertion_results=[
                AssertionResult(type="is_json", passed=False, message="not valid JSON"),
            ]
        )
        root = self._export([eval_result], tmp_path)
        failure = root.find("./testsuite/testcase/failure")
        assert failure is not None
        assert failure.get("type") == "AssertionFailed"
        assert "is_json" in failure.get("message")
        assert root.get("failures") == "1"

    def test_passing_assertions_without_judge_is_pass(self, tmp_path):
        eval_result = _make_eval(
            assertion_results=[
                AssertionResult(type="is_json", passed=True, message="ok"),
            ]
        )
        root = self._export([eval_result], tmp_path)
        testcase = root.find("./testsuite/testcase")
        assert testcase.find("failure") is None
        assert testcase.find("skipped") is None
        assert root.get("failures") == "0"
        assert root.get("skipped") == "0"

    def test_no_assertions_no_judge_still_skipped(self, tmp_path):
        eval_result = _make_eval()
        root = self._export([eval_result], tmp_path)
        assert root.find("./testsuite/testcase/skipped") is not None

    def test_assertion_summary_in_system_out(self, tmp_path):
        eval_result = _make_eval(
            judge_score=_judge(5),
            assertion_results=[
                AssertionResult(type="is_json", passed=True, message="ok"),
                AssertionResult(type="contains", value="x", passed=True, message="ok"),
            ],
        )
        root = self._export([eval_result], tmp_path)
        system_out = root.find("./testsuite/testcase/system-out")
        assert "assertions: 2/2 passed" in system_out.text


class TestFailUnderGateWithAssertions:
    def test_assertion_failure_fails_gate_despite_good_scores(self):
        run = _make_run(
            [
                _make_eval(judge_score=_judge(5)),
                _make_eval(
                    test_case_id="case-2",
                    assertion_results=[
                        AssertionResult(type="is_json", passed=False, message="bad"),
                    ],
                ),
            ]
        )
        failing = _check_fail_under(run, 3.0)
        assert len(failing) == 1
        assert failing[0][0] == "model-a"

    def test_assertion_only_run_passes_gate(self):
        run = _make_run(
            [
                _make_eval(
                    assertion_results=[
                        AssertionResult(type="is_json", passed=True, message="ok"),
                    ]
                ),
            ]
        )
        assert _check_fail_under(run, 3.0) == []

    def test_no_scores_no_assertions_still_fails_gate(self):
        run = _make_run([_make_eval()])
        failing = _check_fail_under(run, 3.0)
        assert len(failing) == 1
