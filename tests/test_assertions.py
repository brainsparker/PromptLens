"""Tests for deterministic assertions: models, evaluator, runner wiring, and exporters."""

import asyncio
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from promptlens.assertions import evaluate_assertion, evaluate_assertions
from promptlens.cli import _collect_assertion_failures
from promptlens.exporters.csv_exporter import CSVExporter
from promptlens.exporters.html_exporter import HTMLExporter
from promptlens.exporters.junit_exporter import JUnitXMLExporter
from promptlens.exporters.markdown_exporter import MarkdownExporter
from promptlens.models.assertions import Assertion, AssertionResult
from promptlens.models.config import ExecutionConfig
from promptlens.models.result import (
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)
from promptlens.models.test_case import GoldenSet
from promptlens.models.test_case import TestCase as GoldenTestCase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assertion(type_, value=None, **kwargs):
    return Assertion(type=type_, value=value, **kwargs)


def _passes(type_, response, value=None, **kwargs):
    return evaluate_assertion(_assertion(type_, value, **kwargs), response).passed


def _make_response(content="The answer is 42.", model="model-a", error=None):
    return ModelResponse(
        content=content,
        model=model,
        provider="anthropic",
        latency_ms=100.0,
        cost_usd=0.001,
        error=error,
    )


def _make_score(score=4):
    return JudgeScore(
        score=score,
        explanation="fine",
        judge_model="judge",
        judge_provider="anthropic",
    )


def _make_eval(
    test_case_id,
    assertions=None,
    response="The answer is 42.",
    score=4,
    model="model-a",
    error=None,
):
    assertion_results = []
    if assertions and not error:
        assertion_results = evaluate_assertions(assertions, response)
    return EvaluationResult(
        test_case_id=test_case_id,
        query="What is the answer?",
        expected_behavior="Answers correctly",
        model_response=_make_response(content=response, model=model, error=error),
        judge_score=_make_score(score) if score is not None else None,
        assertion_results=assertion_results,
    )


def _make_run(results, models=None):
    return RunResult(
        run_id="run-assert",
        run_name="assertion-run",
        timestamp=datetime(2026, 9, 23, 9, 0, 0),
        golden_set_name="golden-set",
        models_tested=models or ["model-a"],
        results=results,
    )


# ---------------------------------------------------------------------------
# Assertion model validation
# ---------------------------------------------------------------------------


class TestAssertionModel:
    def test_unknown_type_rejected(self):
        with pytest.raises(ValidationError, match="unsupported assertion type"):
            Assertion(type="looks_good", value="x")

    def test_type_is_normalized(self):
        assertion = Assertion(type=" Contains ", value="x")
        assert assertion.type == "contains"

    @pytest.mark.parametrize(
        "type_", ["contains", "not_contains", "equals", "starts_with", "ends_with", "regex"]
    )
    def test_string_types_require_string_value(self, type_):
        with pytest.raises(ValidationError, match="requires a string value"):
            Assertion(type=type_, value=["a"])
        with pytest.raises(ValidationError, match="requires a string value"):
            Assertion(type=type_)

    def test_invalid_regex_rejected_at_load_time(self):
        with pytest.raises(ValidationError, match="invalid pattern"):
            Assertion(type="regex", value="(unclosed")

    @pytest.mark.parametrize("type_", ["contains_any", "contains_all"])
    def test_list_types_require_non_empty_string_list(self, type_):
        with pytest.raises(ValidationError, match="non-empty list of strings"):
            Assertion(type=type_, value="a")
        with pytest.raises(ValidationError, match="non-empty list of strings"):
            Assertion(type=type_, value=[])
        with pytest.raises(ValidationError, match="non-empty list of strings"):
            Assertion(type=type_, value=["a", 1])
        assert Assertion(type=type_, value=["a", "b"]).value == ["a", "b"]

    @pytest.mark.parametrize("type_", ["min_length", "max_length"])
    def test_length_types_require_non_negative_int(self, type_):
        with pytest.raises(ValidationError, match="integer character count"):
            Assertion(type=type_, value="10")
        with pytest.raises(ValidationError, match="integer character count"):
            Assertion(type=type_, value=True)
        with pytest.raises(ValidationError, match="must not be negative"):
            Assertion(type=type_, value=-1)

    def test_is_json_takes_no_value(self):
        assert Assertion(type="is_json").value is None
        with pytest.raises(ValidationError, match="does not take a value"):
            Assertion(type="is_json", value="x")

    def test_label_prefers_custom_name(self):
        assert Assertion(type="contains", value="x", name="mentions x").label == "mentions x"
        assert Assertion(type="contains", value="x").label == 'contains "x"'
        assert Assertion(type="is_json").label == "is_json"
        assert Assertion(type="contains_any", value=["a", "b"]).label == 'contains_any ["a", "b"]'


# ---------------------------------------------------------------------------
# Evaluator semantics
# ---------------------------------------------------------------------------


class TestEvaluator:
    def test_contains_and_not_contains(self):
        assert _passes("contains", "Reset your password here", "password")
        assert not _passes("contains", "Reset your PIN here", "password")
        assert _passes("not_contains", "Reset your PIN here", "password")
        assert not _passes("not_contains", "Reset your password here", "password")

    def test_case_insensitive_flag(self):
        assert not _passes("contains", "Reset your Password", "password")
        assert _passes("contains", "Reset your Password", "password", case_sensitive=False)
        assert _passes("regex", "Total: USD 5", r"usd \d+", case_sensitive=False)
        assert _passes("equals", "  YES ", "yes", case_sensitive=False)

    def test_contains_any_and_all(self):
        response = "Go to Settings, then Security, then Reset password."
        assert _passes("contains_any", response, ["Reset password", "Forgot password"])
        assert not _passes("contains_any", response, ["Forgot password", "Support ticket"])
        assert _passes("contains_all", response, ["Settings", "Security"])
        result = evaluate_assertion(_assertion("contains_all", ["Settings", "Billing"]), response)
        assert not result.passed
        assert "Billing" in result.message

    def test_equals_ignores_surrounding_whitespace(self):
        assert _passes("equals", "  Paris\n", "Paris")
        assert not _passes("equals", "Paris, France", "Paris")

    def test_starts_with_and_ends_with(self):
        assert _passes("starts_with", "\nSure! Here is", "Sure")
        assert not _passes("starts_with", "Here is", "Sure")
        assert _passes("ends_with", "Done.\n", "Done.")
        assert not _passes("ends_with", "Done. More", "Done.")

    def test_regex_reports_match(self):
        result = evaluate_assertion(_assertion("regex", r"\b\d{3}-\d{4}\b"), "Call 555-1234 now")
        assert result.passed
        assert "555-1234" in result.message
        assert not _passes("regex", "Call us now", r"\d{3}-\d{4}")

    def test_is_json_accepts_raw_and_fenced(self):
        assert _passes("is_json", '{"a": 1, "b": [1, 2]}')
        assert _passes("is_json", '```json\n{"a": 1}\n```')
        assert _passes("is_json", "```\n[1, 2, 3]\n```")
        assert not _passes("is_json", 'Here is the JSON: {"a": 1}')
        assert not _passes("is_json", "")

    def test_length_bounds(self):
        assert _passes("max_length", "short", 10)
        assert not _passes("max_length", "this is far too long", 10)
        assert _passes("min_length", "long enough", 5)
        assert not _passes("min_length", "no", 5)
        result = evaluate_assertion(_assertion("max_length", 3), "hello")
        assert "5 characters, maximum is 3" in result.message

    def test_evaluate_assertions_preserves_order_and_metadata(self):
        assertions = [
            _assertion("contains", "42", name="has answer"),
            _assertion("max_length", 5),
        ]
        results = evaluate_assertions(assertions, "The answer is 42.")
        assert [r.label for r in results] == ["has answer", "max_length 5"]
        assert [r.passed for r in results] == [True, False]
        assert results[0].expected == "42"
        assert results[1].expected == 5
        assert all(isinstance(r, AssertionResult) for r in results)

    def test_evaluation_is_deterministic(self):
        assertion = _assertion("regex", r"answer is \d+")
        first = evaluate_assertion(assertion, "The answer is 42.")
        for _ in range(20):
            assert evaluate_assertion(assertion, "The answer is 42.") == first


# ---------------------------------------------------------------------------
# Golden set loading
# ---------------------------------------------------------------------------


class TestGoldenSetIntegration:
    def test_test_case_without_assertions_still_loads(self):
        case = GoldenTestCase(id="t1", query="q", expected_behavior="e")
        assert case.assertions == []

    def test_golden_set_with_assertions_loads(self):
        golden = GoldenSet(
            name="gs",
            test_cases=[
                {
                    "id": "t1",
                    "query": "q",
                    "expected_behavior": "e",
                    "assertions": [
                        {"type": "contains", "value": "hello", "case_sensitive": False},
                        {"type": "is_json"},
                        {"type": "max_length", "value": 200},
                    ],
                }
            ],
        )
        assert len(golden.test_cases[0].assertions) == 3
        assert golden.test_cases[0].assertions[0].case_sensitive is False

    def test_bad_assertion_surfaces_as_validation_error(self):
        with pytest.raises(ValidationError, match="unsupported assertion type"):
            GoldenSet(
                name="gs",
                test_cases=[
                    {
                        "id": "t1",
                        "query": "q",
                        "expected_behavior": "e",
                        "assertions": [{"type": "vibes"}],
                    }
                ],
            )


# ---------------------------------------------------------------------------
# Result model helpers
# ---------------------------------------------------------------------------


class TestResultHelpers:
    def test_assertions_passed_tri_state(self):
        none_case = _make_eval("t0")
        assert none_case.assertions_passed is None
        assert none_case.has_assertions is False

        passing = _make_eval("t1", [_assertion("contains", "42")])
        assert passing.assertions_passed is True
        assert passing.failed_assertions == []

        failing = _make_eval("t2", [_assertion("contains", "42"), _assertion("max_length", 3)])
        assert failing.assertions_passed is False
        assert len(failing.failed_assertions) == 1

    def test_run_summary_and_pass_rate(self):
        run = _make_run(
            [
                _make_eval("t0"),
                _make_eval("t1", [_assertion("contains", "42")]),
                _make_eval("t2", [_assertion("contains", "42"), _assertion("max_length", 3)]),
                _make_eval("t3", [_assertion("is_json")], model="model-b"),
            ],
            models=["model-a", "model-b"],
        )
        summary = run.get_assertion_summary()
        assert summary == {"checks": 4, "failed": 2, "cases": 3, "cases_failed": 2}
        assert run.get_assertion_summary("model-a") == {
            "checks": 3,
            "failed": 1,
            "cases": 2,
            "cases_failed": 1,
        }
        assert run.get_assertion_pass_rate("model-a") == 0.5
        assert run.get_assertion_pass_rate("model-b") == 0.0
        assert _make_run([_make_eval("t0")]).get_assertion_pass_rate() is None

    def test_old_results_json_without_assertion_fields_still_loads(self):
        payload = {
            "test_case_id": "legacy",
            "query": "q",
            "expected_behavior": "e",
            "model_response": _make_response().model_dump(mode="json"),
            "judge_score": None,
        }
        loaded = EvaluationResult.model_validate(payload)
        assert loaded.assertion_results == []
        assert loaded.judge_skipped_reason is None


# ---------------------------------------------------------------------------
# Runner wiring
# ---------------------------------------------------------------------------


def _build_runner(skip_judge_on_assertion_failure, judge_mock, response_content):
    """Construct a Runner without touching providers or the network."""
    from promptlens.runners.runner import Runner

    runner = Runner.__new__(Runner)
    runner.config = MagicMock()
    runner.config.execution = ExecutionConfig(
        skip_judge_on_assertion_failure=skip_judge_on_assertion_failure
    )
    runner.judge = judge_mock

    provider = MagicMock()
    provider.supports_tools.return_value = True
    provider.provider_name = "mock"
    provider.generate = AsyncMock(return_value=_make_response(content=response_content))
    return runner, provider


class TestRunnerWiring:
    def _run_single(self, runner, provider, test_case):
        progress = MagicMock()

        async def go():
            # Create the semaphore inside the loop: on Python 3.9 primitives
            # bind to the loop that exists when they are constructed.
            runner.semaphore = asyncio.Semaphore(1)
            return await runner._evaluate_single(
                test_case=test_case, provider=provider, progress=progress, task_id=0
            )

        return asyncio.run(go())

    def test_assertions_are_evaluated_and_judge_still_runs_by_default(self):
        judge = MagicMock()
        judge.evaluate = AsyncMock(return_value=_make_score(2))
        runner, provider = _build_runner(False, judge, "The answer is 41.")
        case = GoldenTestCase(
            id="t1",
            query="q",
            expected_behavior="e",
            assertions=[_assertion("contains", "42")],
        )

        result = self._run_single(runner, provider, case)

        assert result.assertions_passed is False
        assert result.judge_score is not None
        assert result.judge_skipped_reason is None
        judge.evaluate.assert_awaited_once()

    def test_skip_judge_on_assertion_failure_saves_the_judge_call(self):
        judge = MagicMock()
        judge.evaluate = AsyncMock(return_value=_make_score(5))
        runner, provider = _build_runner(True, judge, "The answer is 41.")
        case = GoldenTestCase(
            id="t1",
            query="q",
            expected_behavior="e",
            assertions=[_assertion("contains", "42")],
        )

        result = self._run_single(runner, provider, case)

        assert result.assertions_passed is False
        assert result.judge_score is None
        assert result.judge_skipped_reason == "assertion failed"
        judge.evaluate.assert_not_awaited()

    def test_skip_option_does_not_skip_when_assertions_pass(self):
        judge = MagicMock()
        judge.evaluate = AsyncMock(return_value=_make_score(5))
        runner, provider = _build_runner(True, judge, "The answer is 42.")
        case = GoldenTestCase(
            id="t1",
            query="q",
            expected_behavior="e",
            assertions=[_assertion("contains", "42")],
        )

        result = self._run_single(runner, provider, case)

        assert result.assertions_passed is True
        assert result.judge_score is not None
        judge.evaluate.assert_awaited_once()

    def test_no_assertions_means_no_assertion_results(self):
        judge = MagicMock()
        judge.evaluate = AsyncMock(return_value=_make_score(5))
        runner, provider = _build_runner(True, judge, "anything")
        case = GoldenTestCase(id="t1", query="q", expected_behavior="e")

        result = self._run_single(runner, provider, case)

        assert result.assertion_results == []
        assert result.assertions_passed is None

    def test_errored_response_skips_assertions(self):
        judge = MagicMock()
        judge.evaluate = AsyncMock(return_value=_make_score(5))
        runner, provider = _build_runner(False, judge, "")
        provider.generate = AsyncMock(return_value=_make_response(content="", error="boom"))
        case = GoldenTestCase(
            id="t1",
            query="q",
            expected_behavior="e",
            assertions=[_assertion("contains", "42")],
        )

        result = self._run_single(runner, provider, case)

        assert result.assertion_results == []
        assert result.judge_score is None
        judge.evaluate.assert_not_awaited()


# ---------------------------------------------------------------------------
# CLI gate helper
# ---------------------------------------------------------------------------


def test_collect_assertion_failures_returns_only_failing_evaluations():
    run = _make_run(
        [
            _make_eval("t0"),
            _make_eval("t1", [_assertion("contains", "42")]),
            _make_eval("t2", [_assertion("max_length", 3)]),
        ]
    )
    failing = _collect_assertion_failures(run)
    assert [r.test_case_id for r in failing] == ["t2"]


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------


class TestJUnitAssertionMapping:
    def _export(self, run, tmp_path, fail_under=None):
        path = tmp_path / "junit.xml"
        JUnitXMLExporter(fail_under=fail_under).export(run, str(path))
        return ET.parse(path).getroot()

    def test_failed_assertion_is_a_failure_even_with_high_judge_score(self, tmp_path):
        run = _make_run([_make_eval("t1", [_assertion("max_length", 3)], score=5)])
        root = self._export(run, tmp_path)
        testcase = root.find("./testsuite/testcase")
        failure = testcase.find("failure")
        assert failure is not None
        assert failure.get("type") == "AssertionFailed"
        assert "1 of 1 assertion(s) failed" in failure.get("message")
        assert "max_length 3" in failure.text
        assert root.get("failures") == "1"

    def test_passing_assertions_without_judge_score_is_a_pass(self, tmp_path):
        run = _make_run([_make_eval("t1", [_assertion("contains", "42")], score=None)])
        root = self._export(run, tmp_path)
        testcase = root.find("./testsuite/testcase")
        assert testcase.find("failure") is None
        assert testcase.find("skipped") is None
        assert testcase.find("error") is None
        assert root.get("skipped") == "0"

    def test_no_assertions_and_no_judge_score_still_skipped(self, tmp_path):
        run = _make_run([_make_eval("t1", score=None)])
        root = self._export(run, tmp_path)
        assert root.find("./testsuite/testcase/skipped") is not None

    def test_system_out_and_properties_list_assertions(self, tmp_path):
        run = _make_run(
            [_make_eval("t1", [_assertion("contains", "42"), _assertion("max_length", 3)])]
        )
        root = self._export(run, tmp_path)
        system_out = root.find("./testsuite/testcase/system-out").text
        assert "assertions_passed: 1/2" in system_out
        assert 'assertion [contains "42"]: PASS' in system_out
        assert "assertion [max_length 3]: FAIL" in system_out
        props = {
            p.get("name"): p.get("value") for p in root.findall("./testsuite/properties/property")
        }
        assert props["assertion_checks_failed"] == "1/2"
        assert props["assertion_cases_failed"] == "1/1"


class TestOtherExporters:
    def test_csv_has_assertion_columns(self, tmp_path):
        run = _make_run(
            [
                _make_eval("t0"),
                _make_eval("t1", [_assertion("contains", "42"), _assertion("max_length", 3)]),
            ]
        )
        path = tmp_path / "results.csv"
        CSVExporter().export(run, str(path))
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["assertions_passed"] == ""
        assert rows[0]["assertions_failed"] == ""
        assert rows[1]["assertions_passed"] == "false"
        assert "max_length 3" in rows[1]["assertions_failed"]

    def test_markdown_lists_failed_assertions(self, tmp_path):
        run = _make_run([_make_eval("t1", [_assertion("max_length", 3)])])
        path = tmp_path / "results.md"
        MarkdownExporter().export(run, str(path))
        text = path.read_text(encoding="utf-8")
        assert "| Assertions |" in text
        assert "FAIL 0/1" in text
        assert "**Failed assertions:**" in text
        assert "maximum is 3" in text

    def test_html_renders_assertion_rows(self, tmp_path):
        run = _make_run(
            [_make_eval("t1", [_assertion("contains", "42"), _assertion("max_length", 3)])]
        )
        path = tmp_path / "report.html"
        HTMLExporter().export(run, str(path))
        html = path.read_text(encoding="utf-8")
        assert "assertions fail" in html
        assert 'class="assertion pass"' in html
        assert 'class="assertion fail"' in html
        assert "maximum is 3" in html

    def test_html_without_assertions_unchanged_shape(self, tmp_path):
        run = _make_run([_make_eval("t1")])
        path = tmp_path / "report.html"
        HTMLExporter().export(run, str(path))
        html = path.read_text(encoding="utf-8")
        assert "assertions fail" not in html
        assert "assertions pass" not in html
