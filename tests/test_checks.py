"""Tests for deterministic checks: models, engine, and exporter integration."""

import xml.etree.ElementTree as ET
from datetime import datetime

import pytest
from pydantic import ValidationError

from promptlens.checks import evaluate_checks
from promptlens.exporters.junit_exporter import JUnitXMLExporter
from promptlens.models.checks import CheckDefinition, CheckResult
from promptlens.models.result import (
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)
from promptlens.models.test_case import TestCase


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_response(content="The answer is 42.", **kwargs):
    defaults = {
        "content": content,
        "model": "model-a",
        "provider": "anthropic",
        "latency_ms": 1234.5,
        "cost_usd": 0.0021,
    }
    defaults.update(kwargs)
    return ModelResponse(**defaults)


def _make_case(checks, **kwargs):
    defaults = {
        "id": "tc-001",
        "query": "What is the answer?",
        "expected_behavior": "Answers correctly",
        "checks": checks,
    }
    defaults.update(kwargs)
    return TestCase(**defaults)


def _run_single(check_dict, response):
    case = _make_case([check_dict])
    results = evaluate_checks(case, response)
    assert len(results) == 1
    return results[0]


# ---------------------------------------------------------------------------
# CheckDefinition validation
# ---------------------------------------------------------------------------


class TestCheckDefinitionValidation:
    def test_valid_string_check(self):
        check = CheckDefinition(type="contains", value="hello")
        assert check.type == "contains"
        assert check.value == "hello"
        assert check.case_insensitive is False

    def test_valid_numeric_check(self):
        check = CheckDefinition(type="max_latency_ms", value=5000)
        assert check.value == 5000

    def test_valueless_check(self):
        check = CheckDefinition(type="is_valid_json")
        assert check.value is None

    def test_unknown_type_rejected(self):
        with pytest.raises(ValidationError):
            CheckDefinition(type="fuzzy_match", value="x")

    def test_string_check_requires_value(self):
        with pytest.raises(ValidationError, match="requires a value"):
            CheckDefinition(type="contains")

    def test_string_check_rejects_non_string(self):
        with pytest.raises(ValidationError, match="requires a string"):
            CheckDefinition(type="contains", value=42)

    def test_numeric_check_rejects_string(self):
        with pytest.raises(ValidationError, match="requires a numeric"):
            CheckDefinition(type="max_latency_ms", value="fast")

    def test_numeric_check_rejects_bool(self):
        with pytest.raises(ValidationError, match="requires a numeric"):
            CheckDefinition(type="max_cost_usd", value=True)

    def test_numeric_check_rejects_negative(self):
        with pytest.raises(ValidationError, match="non-negative"):
            CheckDefinition(type="min_length", value=-1)

    def test_valueless_check_rejects_value(self):
        with pytest.raises(ValidationError, match="does not take a value"):
            CheckDefinition(type="is_valid_json", value="{}")

    def test_invalid_regex_rejected(self):
        with pytest.raises(ValidationError, match="invalid regular expression"):
            CheckDefinition(type="regex", value="[unclosed")

    def test_test_case_accepts_checks(self):
        case = _make_case([{"type": "contains", "value": "42"}])
        assert len(case.checks) == 1
        assert case.checks[0].type == "contains"

    def test_test_case_defaults_to_no_checks(self):
        case = TestCase(
            id="tc-x", query="q", expected_behavior="e"
        )
        assert case.checks == []


# ---------------------------------------------------------------------------
# Engine: string checks
# ---------------------------------------------------------------------------


class TestStringChecks:
    def test_contains_pass(self):
        result = _run_single(
            {"type": "contains", "value": "42"}, _make_response()
        )
        assert result.passed is True

    def test_contains_fail(self):
        result = _run_single(
            {"type": "contains", "value": "43"}, _make_response()
        )
        assert result.passed is False
        assert "does not contain" in result.reason

    def test_contains_case_insensitive(self):
        result = _run_single(
            {"type": "contains", "value": "THE ANSWER", "case_insensitive": True},
            _make_response(),
        )
        assert result.passed is True

    def test_contains_case_sensitive_by_default(self):
        result = _run_single(
            {"type": "contains", "value": "THE ANSWER"}, _make_response()
        )
        assert result.passed is False

    def test_not_contains_pass(self):
        result = _run_single(
            {"type": "not_contains", "value": "I cannot help"}, _make_response()
        )
        assert result.passed is True

    def test_not_contains_fail(self):
        result = _run_single(
            {"type": "not_contains", "value": "42"}, _make_response()
        )
        assert result.passed is False
        assert "forbidden" in result.reason

    def test_regex_pass(self):
        result = _run_single(
            {"type": "regex", "value": r"answer is \d+"}, _make_response()
        )
        assert result.passed is True

    def test_regex_fail(self):
        result = _run_single(
            {"type": "regex", "value": r"^\d+$"}, _make_response()
        )
        assert result.passed is False

    def test_regex_case_insensitive(self):
        result = _run_single(
            {"type": "regex", "value": r"THE ANSWER", "case_insensitive": True},
            _make_response(),
        )
        assert result.passed is True

    def test_not_regex_pass(self):
        result = _run_single(
            {"type": "not_regex", "value": r"as an ai (language )?model"},
            _make_response(),
        )
        assert result.passed is True

    def test_not_regex_fail(self):
        result = _run_single(
            {"type": "not_regex", "value": r"\d+"}, _make_response()
        )
        assert result.passed is False

    def test_exact_match_pass_with_whitespace(self):
        result = _run_single(
            {"type": "exact_match", "value": "The answer is 42."},
            _make_response(content="  The answer is 42.\n"),
        )
        assert result.passed is True

    def test_exact_match_fail(self):
        result = _run_single(
            {"type": "exact_match", "value": "The answer is 43."},
            _make_response(),
        )
        assert result.passed is False

    def test_exact_match_case_insensitive(self):
        result = _run_single(
            {"type": "exact_match", "value": "the answer is 42.",
             "case_insensitive": True},
            _make_response(),
        )
        assert result.passed is True


# ---------------------------------------------------------------------------
# Engine: JSON checks
# ---------------------------------------------------------------------------


class TestJsonCheck:
    def test_raw_json_object(self):
        result = _run_single(
            {"type": "is_valid_json"},
            _make_response(content='{"summary": "ok", "items": [1, 2]}'),
        )
        assert result.passed is True

    def test_raw_json_array(self):
        result = _run_single(
            {"type": "is_valid_json"}, _make_response(content="[1, 2, 3]")
        )
        assert result.passed is True

    def test_fenced_json_block(self):
        content = '```json\n{"summary": "ok"}\n```'
        result = _run_single(
            {"type": "is_valid_json"}, _make_response(content=content)
        )
        assert result.passed is True

    def test_fenced_block_without_language(self):
        content = '```\n{"a": 1}\n```'
        result = _run_single(
            {"type": "is_valid_json"}, _make_response(content=content)
        )
        assert result.passed is True

    def test_invalid_json_fails(self):
        result = _run_single(
            {"type": "is_valid_json"},
            _make_response(content="The answer is: {broken"),
        )
        assert result.passed is False
        assert "not valid JSON" in result.reason

    def test_prose_around_json_fails(self):
        result = _run_single(
            {"type": "is_valid_json"},
            _make_response(content='Here you go: {"a": 1}'),
        )
        assert result.passed is False

    def test_empty_content_fails(self):
        result = _run_single(
            {"type": "is_valid_json"}, _make_response(content="")
        )
        assert result.passed is False


# ---------------------------------------------------------------------------
# Engine: budget and length checks
# ---------------------------------------------------------------------------


class TestBudgetChecks:
    def test_max_latency_pass(self):
        result = _run_single(
            {"type": "max_latency_ms", "value": 5000}, _make_response()
        )
        assert result.passed is True

    def test_max_latency_fail(self):
        result = _run_single(
            {"type": "max_latency_ms", "value": 1000}, _make_response()
        )
        assert result.passed is False
        assert "exceeds budget" in result.reason

    def test_max_cost_pass(self):
        result = _run_single(
            {"type": "max_cost_usd", "value": 0.01}, _make_response()
        )
        assert result.passed is True

    def test_max_cost_fail(self):
        result = _run_single(
            {"type": "max_cost_usd", "value": 0.001}, _make_response()
        )
        assert result.passed is False

    def test_max_cost_none_treated_as_zero(self):
        result = _run_single(
            {"type": "max_cost_usd", "value": 0.001},
            _make_response(cost_usd=None),
        )
        assert result.passed is True

    def test_min_length_pass(self):
        result = _run_single(
            {"type": "min_length", "value": 5}, _make_response()
        )
        assert result.passed is True

    def test_min_length_fail(self):
        result = _run_single(
            {"type": "min_length", "value": 500}, _make_response()
        )
        assert result.passed is False

    def test_max_length_pass(self):
        result = _run_single(
            {"type": "max_length", "value": 500}, _make_response()
        )
        assert result.passed is True

    def test_max_length_fail(self):
        result = _run_single(
            {"type": "max_length", "value": 5}, _make_response()
        )
        assert result.passed is False


# ---------------------------------------------------------------------------
# Engine: multiple checks and ordering
# ---------------------------------------------------------------------------


class TestMultipleChecks:
    def test_results_preserve_declaration_order(self):
        case = _make_case(
            [
                {"type": "contains", "value": "42"},
                {"type": "max_latency_ms", "value": 1},
                {"type": "min_length", "value": 1},
            ]
        )
        results = evaluate_checks(case, _make_response())
        assert [r.check_type for r in results] == [
            "contains",
            "max_latency_ms",
            "min_length",
        ]
        assert [r.passed for r in results] == [True, False, True]

    def test_no_checks_returns_empty(self):
        case = _make_case([])
        assert evaluate_checks(case, _make_response()) == []


# ---------------------------------------------------------------------------
# EvaluationResult helpers
# ---------------------------------------------------------------------------


class TestEvaluationResultHelpers:
    def _make_eval(self, check_results):
        return EvaluationResult(
            test_case_id="tc-001",
            query="q",
            expected_behavior="e",
            model_response=_make_response(),
            check_results=check_results,
        )

    def test_checks_passed_none_without_checks(self):
        assert self._make_eval([]).checks_passed is None

    def test_checks_passed_true(self):
        result = self._make_eval(
            [CheckResult(check_type="contains", passed=True, reason="ok")]
        )
        assert result.checks_passed is True

    def test_checks_passed_false(self):
        result = self._make_eval(
            [
                CheckResult(check_type="contains", passed=True, reason="ok"),
                CheckResult(check_type="regex", passed=False, reason="no"),
            ]
        )
        assert result.checks_passed is False
        assert [c.check_type for c in result.failed_checks] == ["regex"]

    def test_backward_compat_default(self):
        # Existing serialized results without check_results still load.
        result = EvaluationResult(
            test_case_id="tc-001",
            query="q",
            expected_behavior="e",
            model_response=_make_response(),
        )
        assert result.check_results == []

    def test_run_result_check_stats(self):
        run = RunResult(
            run_id="run-1",
            golden_set_name="gs",
            models_tested=["model-a"],
            results=[
                self._make_eval(
                    [
                        CheckResult(check_type="contains", passed=True, reason="ok"),
                        CheckResult(check_type="regex", passed=False, reason="no"),
                    ]
                ),
                self._make_eval(
                    [CheckResult(check_type="min_length", passed=True, reason="ok")]
                ),
            ],
        )
        assert run.get_check_stats() == {"passed": 2, "total": 3}
        assert run.get_check_stats("model-a") == {"passed": 2, "total": 3}
        assert run.get_check_stats("other-model") == {"passed": 0, "total": 0}


# ---------------------------------------------------------------------------
# JUnit exporter integration
# ---------------------------------------------------------------------------


class TestJUnitCheckIntegration:
    def _make_eval(self, check_results, score=None):
        return EvaluationResult(
            test_case_id="tc-001",
            query="q",
            expected_behavior="e",
            model_response=_make_response(),
            judge_score=(
                JudgeScore(
                    score=score,
                    explanation="fine",
                    judge_model="judge",
                    judge_provider="anthropic",
                )
                if score is not None
                else None
            ),
            check_results=check_results,
        )

    def _export(self, results, tmp_path):
        run = RunResult(
            run_id="run-1",
            run_name="ci",
            timestamp=datetime(2026, 8, 28, 12, 0, 0),
            golden_set_name="gs",
            models_tested=["model-a"],
            results=results,
        )
        out = tmp_path / "junit.xml"
        JUnitXMLExporter().export(run, str(out))
        return ET.parse(str(out)).getroot()

    def test_failed_check_reported_as_failure(self, tmp_path):
        root = self._export(
            [
                self._make_eval(
                    [CheckResult(check_type="contains", passed=False,
                                 reason="Response does not contain '42'")],
                    score=5,
                )
            ],
            tmp_path,
        )
        assert root.get("failures") == "1"
        failure = root.find("./testsuite/testcase/failure")
        assert failure is not None
        assert failure.get("type") == "DeterministicCheckFailed"
        assert "contains" in failure.get("message")
        assert "does not contain" in failure.text

    def test_passing_checks_high_score_is_pass(self, tmp_path):
        root = self._export(
            [
                self._make_eval(
                    [CheckResult(check_type="contains", passed=True, reason="ok")],
                    score=5,
                )
            ],
            tmp_path,
        )
        assert root.get("failures") == "0"
        assert root.get("errors") == "0"
        assert root.get("skipped") == "0"

    def test_passing_checks_without_judge_is_pass_not_skipped(self, tmp_path):
        root = self._export(
            [
                self._make_eval(
                    [CheckResult(check_type="contains", passed=True, reason="ok")]
                )
            ],
            tmp_path,
        )
        assert root.get("skipped") == "0"
        assert root.get("failures") == "0"

    def test_no_checks_no_judge_still_skipped(self, tmp_path):
        root = self._export([self._make_eval([])], tmp_path)
        assert root.get("skipped") == "1"

    def test_check_outcomes_in_system_out(self, tmp_path):
        root = self._export(
            [
                self._make_eval(
                    [
                        CheckResult(check_type="contains", passed=True, reason="ok"),
                        CheckResult(check_type="regex", passed=False, reason="no match"),
                    ],
                )
            ],
            tmp_path,
        )
        system_out = root.find("./testsuite/testcase/system-out")
        assert "checks: 1/2 passed" in system_out.text
        assert "check[contains]: pass" in system_out.text
        assert "check[regex]: FAIL" in system_out.text

    def test_model_error_takes_precedence_over_checks(self, tmp_path):
        result = EvaluationResult(
            test_case_id="tc-001",
            query="q",
            expected_behavior="e",
            model_response=_make_response(error="boom"),
            check_results=[
                CheckResult(check_type="contains", passed=False, reason="no")
            ],
        )
        root = self._export([result], tmp_path)
        assert root.get("errors") == "1"
        assert root.get("failures") == "0"
