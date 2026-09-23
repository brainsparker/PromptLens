"""Evaluate deterministic assertions against a response string.

Every check here is pure: no network, no model call, no randomness. The same
response and the same assertion always give the same result, which is what
makes assertions safe to gate a CI build on.
"""

import json
import re
from typing import List

from promptlens.models.assertions import Assertion, AssertionResult

_PREVIEW_LIMIT = 80


def evaluate_assertions(assertions: List[Assertion], response: str) -> List[AssertionResult]:
    """Evaluate every assertion against a response.

    Args:
        assertions: Assertions declared on the test case
        response: The model's response text

    Returns:
        One AssertionResult per assertion, in declaration order
    """
    return [evaluate_assertion(assertion, response) for assertion in assertions]


def evaluate_assertion(assertion: Assertion, response: str) -> AssertionResult:
    """Evaluate a single assertion against a response.

    Args:
        assertion: The assertion to check
        response: The model's response text

    Returns:
        AssertionResult describing whether the check passed and why
    """
    checker = _CHECKERS[assertion.type]
    passed, message = checker(assertion, response)
    return AssertionResult(
        type=assertion.type,
        label=assertion.label,
        passed=passed,
        message=message,
        expected=assertion.value,
    )


def _fold(assertion: Assertion, text: str) -> str:
    """Lower-case text when the assertion is case-insensitive."""
    return text if assertion.case_sensitive else text.lower()


def _preview(text: str) -> str:
    """Short single-line preview of a response for failure messages."""
    flat = " ".join(text.split())
    if len(flat) <= _PREVIEW_LIMIT:
        return flat
    return flat[: _PREVIEW_LIMIT - 3] + "..."


def _check_contains(assertion: Assertion, response: str):
    needle = _fold(assertion, str(assertion.value))
    if needle in _fold(assertion, response):
        return True, f"response contains {assertion.value!r}"
    return False, f"response does not contain {assertion.value!r}"


def _check_not_contains(assertion: Assertion, response: str):
    needle = _fold(assertion, str(assertion.value))
    if needle in _fold(assertion, response):
        return False, f"response contains forbidden text {assertion.value!r}"
    return True, f"response does not contain {assertion.value!r}"


def _check_contains_any(assertion: Assertion, response: str):
    haystack = _fold(assertion, response)
    found = [item for item in assertion.value if _fold(assertion, item) in haystack]
    if found:
        return True, f"response contains {found[0]!r}"
    return False, f"response contains none of {list(assertion.value)!r}"


def _check_contains_all(assertion: Assertion, response: str):
    haystack = _fold(assertion, response)
    missing = [item for item in assertion.value if _fold(assertion, item) not in haystack]
    if not missing:
        return True, f"response contains all of {list(assertion.value)!r}"
    return False, f"response is missing {missing!r}"


def _check_equals(assertion: Assertion, response: str):
    if _fold(assertion, response.strip()) == _fold(assertion, str(assertion.value).strip()):
        return True, "response matches expected text exactly"
    return False, f"response {_preview(response)!r} does not equal {assertion.value!r}"


def _check_starts_with(assertion: Assertion, response: str):
    if _fold(assertion, response.lstrip()).startswith(_fold(assertion, str(assertion.value))):
        return True, f"response starts with {assertion.value!r}"
    return False, f"response starts with {_preview(response)!r}, expected {assertion.value!r}"


def _check_ends_with(assertion: Assertion, response: str):
    if _fold(assertion, response.rstrip()).endswith(_fold(assertion, str(assertion.value))):
        return True, f"response ends with {assertion.value!r}"
    return False, f"response does not end with {assertion.value!r}"


def _check_regex(assertion: Assertion, response: str):
    flags = 0 if assertion.case_sensitive else re.IGNORECASE
    pattern = re.compile(str(assertion.value), flags)
    match = pattern.search(response)
    if match:
        return True, f"pattern matched {match.group(0)!r}"
    return False, f"pattern {assertion.value!r} did not match"


def _check_is_json(assertion: Assertion, response: str):
    text = response.strip()
    # Tolerate a fenced code block, which is how most models return JSON.
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        json.loads(text)
    except ValueError as exc:
        return False, f"response is not valid JSON: {exc}"
    return True, "response is valid JSON"


def _check_min_length(assertion: Assertion, response: str):
    length = len(response)
    if length >= int(assertion.value):
        return True, f"response has {length} characters (minimum {assertion.value})"
    return False, f"response has {length} characters, minimum is {assertion.value}"


def _check_max_length(assertion: Assertion, response: str):
    length = len(response)
    if length <= int(assertion.value):
        return True, f"response has {length} characters (maximum {assertion.value})"
    return False, f"response has {length} characters, maximum is {assertion.value}"


_CHECKERS = {
    "contains": _check_contains,
    "not_contains": _check_not_contains,
    "contains_any": _check_contains_any,
    "contains_all": _check_contains_all,
    "equals": _check_equals,
    "starts_with": _check_starts_with,
    "ends_with": _check_ends_with,
    "regex": _check_regex,
    "is_json": _check_is_json,
    "min_length": _check_min_length,
    "max_length": _check_max_length,
}
