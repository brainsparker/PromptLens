"""Tests for deterministic check specs and the check engine."""

import pytest
from pydantic import ValidationError

from promptlens.models.checks import CheckSpec, run_check, run_checks
from promptlens.models.test_case import TestCase as PLTestCase


# ---------------------------------------------------------------------------
# CheckSpec validation
# ---------------------------------------------------------------------------


def test_unknown_check_type_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown check type"):
        CheckSpec(type="fuzzy_match", value="hello")


@pytest.mark.parametrize("check_type", ["contains", "not_contains", "equals", "regex", "not_regex"])
def test_string_checks_require_string_value(check_type: str) -> None:
    with pytest.raises(ValidationError, match="non-empty string value"):
        CheckSpec(type=check_type)


def test_invalid_regex_rejected() -> None:
    with pytest.raises(ValidationError, match="invalid pattern"):
        CheckSpec(type="regex", value="[unclosed")


@pytest.mark.parametrize("check_type", ["min_length", "max_length"])
def test_length_checks_require_integer(check_type: str) -> None:
    with pytest.raises(ValidationError, match="integer value"):
        CheckSpec(type=check_type, value="ten")


def test_length_checks_reject_negative() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        CheckSpec(type="min_length", value=-1)


def test_length_checks_reject_bool() -> None:
    with pytest.raises(ValidationError, match="integer value"):
        CheckSpec(type="max_length", value=True)


def test_json_valid_rejects_value() -> None:
    with pytest.raises(ValidationError, match="does not take a value"):
        CheckSpec(type="json_valid", value="{}")


# ---------------------------------------------------------------------------
# Check engine behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec_kwargs,content,expected",
    [
        ({"type": "contains", "value": "password"}, "Reset your password now", True),
        ({"type": "contains", "value": "password"}, "Reset your login now", False),
        ({"type": "not_contains", "value": "sorry"}, "Here are the steps", True),
        ({"type": "not_contains", "value": "sorry"}, "I am sorry, I cannot", False),
        ({"type": "equals", "value": "yes"}, "  yes  ", True),
        ({"type": "equals", "value": "yes"}, "yes!", False),
        ({"type": "regex", "value": r"\d{3}-\d{4}"}, "Call 555-1234", True),
        ({"type": "regex", "value": r"\d{3}-\d{4}"}, "Call us", False),
        ({"type": "not_regex", "value": r"(?i)as an ai"}, "Here is the answer", True),
        ({"type": "not_regex", "value": r"(?i)as an ai"}, "As an AI, I cannot", False),
        ({"type": "json_valid"}, '{"ok": true}', True),
        ({"type": "json_valid"}, "not json {", False),
        ({"type": "min_length", "value": 5}, "hello", True),
        ({"type": "min_length", "value": 6}, "hello", False),
        ({"type": "max_length", "value": 5}, "hello", True),
        ({"type": "max_length", "value": 4}, "hello", False),
    ],
)
def test_run_check_outcomes(spec_kwargs: dict, content: str, expected: bool) -> None:
    result = run_check(CheckSpec(**spec_kwargs), content)

    assert result.passed is expected
    assert result.type == spec_kwargs["type"]
    assert result.detail


def test_contains_case_insensitive() -> None:
    spec = CheckSpec(type="contains", value="Password", case_sensitive=False)

    assert run_check(spec, "reset your PASSWORD").passed is True


def test_equals_case_sensitive_by_default() -> None:
    spec = CheckSpec(type="equals", value="Yes")

    assert run_check(spec, "yes").passed is False


def test_run_checks_preserves_order() -> None:
    specs = [
        CheckSpec(type="contains", value="alpha"),
        CheckSpec(type="max_length", value=100),
    ]

    results = run_checks(specs, "alpha beta")

    assert [r.type for r in results] == ["contains", "max_length"]
    assert all(r.passed for r in results)


# ---------------------------------------------------------------------------
# TestCase integration
# ---------------------------------------------------------------------------


def test_test_case_accepts_checks() -> None:
    case = PLTestCase(
        id="tc-1",
        query="How do I reset my password?",
        expected_behavior="Provide clear instructions",
        checks=[
            {"type": "contains", "value": "password", "case_sensitive": False},
            {"type": "max_length", "value": 2000},
        ],
    )

    assert len(case.checks) == 2
    assert case.checks[0].type == "contains"


def test_test_case_checks_default_empty() -> None:
    case = PLTestCase(id="tc-1", query="q", expected_behavior="b")

    assert case.checks == []


def test_test_case_rejects_invalid_check() -> None:
    with pytest.raises(ValidationError):
        PLTestCase(
            id="tc-1",
            query="q",
            expected_behavior="b",
            checks=[{"type": "regex", "value": "[bad"}],
        )
