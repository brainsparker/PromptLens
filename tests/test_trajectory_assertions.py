"""Tests for deterministic trajectory assertions.

Covers the assertion models (validation), the evaluator (matching semantics),
loader integration, the JUnit exporter mapping, and the CLI gate helper.
"""

import xml.etree.ElementTree as ET
from datetime import datetime

import pytest
from pydantic import ValidationError

from promptlens.assertions import evaluate_trajectory
from promptlens.cli import _collect_assertion_failures
from promptlens.exporters.junit_exporter import JUnitXMLExporter
from promptlens.models.result import (
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)
from promptlens.models.test_case import TestCase
from promptlens.models.tools import ToolCall
from promptlens.models.trajectory import (
    ToolCallMatcher,
    TrajectoryAssertions,
)


def _call(name, arguments=None, call_id="call-1"):
    return ToolCall(id=call_id, name=name, arguments=arguments or {})


def _calls(*specs):
    """Build a list of ToolCall from (name, args) tuples or bare names."""
    calls = []
    for i, spec in enumerate(specs):
        if isinstance(spec, str):
            calls.append(_call(spec, call_id=f"call-{i}"))
        else:
            name, args = spec
            calls.append(_call(name, args, call_id=f"call-{i}"))
    return calls


class TestModelValidation:
    def test_rejects_empty_assertions(self):
        with pytest.raises(ValidationError):
            TrajectoryAssertions()

    def test_rejects_invalid_args_match(self):
        with pytest.raises(ValidationError):
            ToolCallMatcher(name="get_weather", args_match="fuzzy")

    def test_rejects_max_times_below_min_times(self):
        with pytest.raises(ValidationError):
            ToolCallMatcher(name="get_weather", min_times=3, max_times=1)

    def test_rejects_min_times_zero_without_max_times(self):
        with pytest.raises(ValidationError):
            ToolCallMatcher(name="get_weather", min_times=0)

    def test_rejects_whitelist_without_allowed_tools(self):
        with pytest.raises(ValidationError):
            TrajectoryAssertions(must_not_call=["rm_rf"], allow_other_calls=False)

    def test_rejects_blank_tool_names(self):
        with pytest.raises(ValidationError):
            TrajectoryAssertions(must_not_call=["  "])

    def test_accepts_minimal_must_call(self):
        assertions = TrajectoryAssertions(must_call=[{"name": "get_weather"}])
        assert assertions.must_call[0].args_match == "partial"
        assert assertions.must_call[0].min_times == 1


class TestMustCall:
    def test_partial_match_passes_with_extra_actual_args(self):
        assertions = TrajectoryAssertions(
            must_call=[{"name": "get_weather", "args": {"location": "SF"}}]
        )
        result = evaluate_trajectory(
            assertions, _calls(("get_weather", {"location": "SF", "units": "c"}))
        )
        assert result.passed

    def test_partial_match_fails_on_wrong_value(self):
        assertions = TrajectoryAssertions(
            must_call=[{"name": "get_weather", "args": {"location": "SF"}}]
        )
        result = evaluate_trajectory(
            assertions, _calls(("get_weather", {"location": "NYC"}))
        )
        assert not result.passed
        assert "non-matching arguments" in result.failed_checks[0].detail

    def test_partial_match_fails_on_missing_key(self):
        assertions = TrajectoryAssertions(
            must_call=[{"name": "get_weather", "args": {"location": "SF"}}]
        )
        result = evaluate_trajectory(assertions, _calls(("get_weather", {})))
        assert not result.passed

    def test_exact_match_rejects_extra_actual_args(self):
        assertions = TrajectoryAssertions(
            must_call=[
                {
                    "name": "get_weather",
                    "args": {"location": "SF"},
                    "args_match": "exact",
                }
            ]
        )
        result = evaluate_trajectory(
            assertions, _calls(("get_weather", {"location": "SF", "units": "c"}))
        )
        assert not result.passed

    def test_ignore_mode_matches_on_name_only(self):
        assertions = TrajectoryAssertions(
            must_call=[
                {"name": "get_weather", "args": {"location": "SF"}, "args_match": "ignore"}
            ]
        )
        result = evaluate_trajectory(
            assertions, _calls(("get_weather", {"location": "anywhere"}))
        )
        assert result.passed

    def test_missing_tool_fails(self):
        assertions = TrajectoryAssertions(must_call=[{"name": "get_weather"}])
        result = evaluate_trajectory(assertions, _calls("search_web"))
        assert not result.passed
        assert result.failed_checks[0].kind == "must_call"

    def test_min_times_requires_repeated_calls(self):
        assertions = TrajectoryAssertions(
            must_call=[{"name": "fetch_page", "min_times": 2}]
        )
        assert not evaluate_trajectory(assertions, _calls("fetch_page")).passed
        assert evaluate_trajectory(
            assertions, _calls("fetch_page", "fetch_page")
        ).passed

    def test_max_times_caps_repeated_calls(self):
        assertions = TrajectoryAssertions(
            must_call=[{"name": "fetch_page", "min_times": 0, "max_times": 2}]
        )
        assert evaluate_trajectory(assertions, _calls()).passed
        assert evaluate_trajectory(
            assertions, _calls("fetch_page", "fetch_page")
        ).passed
        result = evaluate_trajectory(
            assertions, _calls("fetch_page", "fetch_page", "fetch_page")
        )
        assert not result.passed
        assert "at most 2" in result.failed_checks[0].detail


class TestMustNotCall:
    def test_passes_when_tool_never_called(self):
        assertions = TrajectoryAssertions(must_not_call=["delete_db"])
        result = evaluate_trajectory(assertions, _calls("get_weather"))
        assert result.passed

    def test_fails_when_forbidden_tool_called(self):
        assertions = TrajectoryAssertions(must_not_call=["delete_db"])
        result = evaluate_trajectory(assertions, _calls("get_weather", "delete_db"))
        assert not result.passed
        assert result.failed_checks[0].kind == "must_not_call"
        assert "called 1 time(s)" in result.failed_checks[0].detail


class TestCallOrder:
    def test_exact_order_passes(self):
        assertions = TrajectoryAssertions(call_order=["check", "book"])
        assert evaluate_trajectory(assertions, _calls("check", "book")).passed

    def test_subsequence_allows_interleaved_calls(self):
        assertions = TrajectoryAssertions(call_order=["check", "book"])
        result = evaluate_trajectory(
            assertions, _calls("login", "check", "get_prices", "book")
        )
        assert result.passed

    def test_wrong_order_fails(self):
        assertions = TrajectoryAssertions(call_order=["check", "book"])
        result = evaluate_trajectory(assertions, _calls("book", "check"))
        assert not result.passed
        assert result.failed_checks[0].kind == "call_order"

    def test_missing_step_fails_and_names_it(self):
        assertions = TrajectoryAssertions(call_order=["check", "book"])
        result = evaluate_trajectory(assertions, _calls("check"))
        assert not result.passed
        assert "'book'" in result.failed_checks[0].detail

    def test_no_calls_fails_order(self):
        assertions = TrajectoryAssertions(call_order=["check"])
        result = evaluate_trajectory(assertions, _calls())
        assert not result.passed
        assert "(no tool calls)" in result.failed_checks[0].detail


class TestMaxCallsAndWhitelist:
    def test_max_calls_budget(self):
        assertions = TrajectoryAssertions(max_calls=2)
        assert evaluate_trajectory(assertions, _calls("a", "b")).passed
        result = evaluate_trajectory(assertions, _calls("a", "b", "c"))
        assert not result.passed
        assert result.failed_checks[0].kind == "max_calls"

    def test_whitelist_passes_within_allowed_set(self):
        assertions = TrajectoryAssertions(
            must_call=[{"name": "get_order"}],
            call_order=["get_order"],
            allow_other_calls=False,
        )
        assert evaluate_trajectory(assertions, _calls("get_order")).passed

    def test_whitelist_fails_on_unexpected_tool(self):
        assertions = TrajectoryAssertions(
            must_call=[{"name": "get_order"}],
            allow_other_calls=False,
        )
        result = evaluate_trajectory(assertions, _calls("get_order", "refund_order"))
        assert not result.passed
        failed = [c for c in result.failed_checks if c.kind == "allowed_tools"]
        assert failed
        assert "refund_order" in failed[0].detail

    def test_observed_calls_recorded_in_order(self):
        assertions = TrajectoryAssertions(max_calls=10)
        result = evaluate_trajectory(assertions, _calls("b", "a", "b"))
        assert result.observed_calls == ["b", "a", "b"]


class TestTestCaseIntegration:
    def test_test_case_parses_trajectory_from_dict(self):
        test_case = TestCase(
            id="tc-1",
            query="Book a haircut",
            expected_behavior="Check availability first",
            trajectory={
                "call_order": ["check_availability", "create_booking"],
                "must_not_call": ["cancel_booking"],
                "max_calls": 4,
            },
        )
        assert test_case.trajectory is not None
        assert test_case.trajectory.call_order == [
            "check_availability",
            "create_booking",
        ]

    def test_test_case_without_trajectory_defaults_to_none(self):
        test_case = TestCase(
            id="tc-1", query="q", expected_behavior="e"
        )
        assert test_case.trajectory is None


def _make_response(model="model-a", tool_calls=None, error=None):
    return ModelResponse(
        content="Done.",
        model=model,
        provider="anthropic",
        latency_ms=100.0,
        cost_usd=0.001,
        error=error,
        tool_calls=tool_calls or [],
    )


def _make_eval(test_case_id, trajectory_result=None, score=None, error=None):
    judge_score = None
    if score is not None:
        judge_score = JudgeScore(
            score=score,
            explanation="ok",
            judge_model="judge",
            judge_provider="anthropic",
        )
    return EvaluationResult(
        test_case_id=test_case_id,
        query="q",
        expected_behavior="e",
        model_response=_make_response(error=error),
        judge_score=judge_score,
        trajectory_result=trajectory_result,
    )


def _make_run(results):
    return RunResult(
        run_id="run-1",
        run_name="run",
        timestamp=datetime(2026, 9, 1, 12, 0, 0),
        golden_set_name="golden-set",
        models_tested=["model-a"],
        results=results,
    )


def _failed_trajectory():
    assertions = TrajectoryAssertions(must_not_call=["delete_db"])
    return evaluate_trajectory(assertions, _calls("delete_db"))


def _passed_trajectory():
    assertions = TrajectoryAssertions(must_not_call=["delete_db"])
    return evaluate_trajectory(assertions, _calls("get_weather"))


class TestJUnitMapping:
    def _export(self, run, tmp_path):
        output = tmp_path / "junit.xml"
        JUnitXMLExporter().export(run, str(output))
        return ET.parse(str(output)).getroot()

    def test_failed_trajectory_is_junit_failure(self, tmp_path):
        run = _make_run([_make_eval("tc-1", trajectory_result=_failed_trajectory(), score=5)])
        root = self._export(run, tmp_path)
        failure = root.find(".//testcase/failure")
        assert failure is not None
        assert failure.get("type") == "TrajectoryAssertionFailure"
        assert root.get("failures") == "1"

    def test_passed_trajectory_without_judge_is_a_pass(self, tmp_path):
        run = _make_run([_make_eval("tc-1", trajectory_result=_passed_trajectory())])
        root = self._export(run, tmp_path)
        testcase = root.find(".//testcase")
        assert testcase.find("failure") is None
        assert testcase.find("skipped") is None
        assert root.get("skipped") == "0"

    def test_no_judge_and_no_trajectory_is_skipped(self, tmp_path):
        run = _make_run([_make_eval("tc-1")])
        root = self._export(run, tmp_path)
        assert root.find(".//testcase/skipped") is not None

    def test_response_error_takes_precedence(self, tmp_path):
        run = _make_run(
            [_make_eval("tc-1", trajectory_result=_failed_trajectory(), error="boom")]
        )
        root = self._export(run, tmp_path)
        assert root.find(".//testcase/error") is not None
        assert root.find(".//testcase/failure") is None

    def test_system_out_reports_trajectory_stats(self, tmp_path):
        run = _make_run([_make_eval("tc-1", trajectory_result=_passed_trajectory(), score=4)])
        root = self._export(run, tmp_path)
        system_out = root.find(".//testcase/system-out").text
        assert "trajectory_assertions: 1/1 passed" in system_out
        assert "observed_tool_calls: get_weather" in system_out


class TestCliGateHelper:
    def test_collects_only_failed_assertion_results(self):
        run = _make_run(
            [
                _make_eval("tc-pass", trajectory_result=_passed_trajectory()),
                _make_eval("tc-fail", trajectory_result=_failed_trajectory()),
                _make_eval("tc-none", score=5),
            ]
        )
        failures = _collect_assertion_failures(run)
        assert [r.test_case_id for r in failures] == ["tc-fail"]

    def test_empty_when_no_assertions_configured(self):
        run = _make_run([_make_eval("tc-1", score=5)])
        assert _collect_assertion_failures(run) == []
