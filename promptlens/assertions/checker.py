"""Evaluate deterministic assertions against a model response.

Everything here is pure and local: no network, no LLM, no randomness. A
failing check never raises; it returns an AssertionResult with passed=False
so the runner can record it and decide whether the LLM judge still runs.
"""

import json
import re
from typing import Any, List, Optional, Tuple

from promptlens.models.assertions import Assertion, AssertionResult
from promptlens.models.result import ModelResponse
from promptlens.models.test_case import TestCase

_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*\n(.*?)\n```\s*$", re.DOTALL)

_PREVIEW_CHARS = 80


def _preview(text: str) -> str:
    """Trim a response for inclusion in a failure message."""
    flat = " ".join(text.split())
    if len(flat) <= _PREVIEW_CHARS:
        return flat
    return flat[: _PREVIEW_CHARS - 3] + "..."


def extract_json(text: str) -> Tuple[Optional[Any], Optional[str]]:
    """Parse a response as JSON, tolerating a surrounding Markdown code fence.

    Args:
        text: Raw model response

    Returns:
        (parsed_value, None) on success or (None, error_message) on failure.
        Note that a valid JSON ``null`` parses to None with no error.
    """
    candidate = text.strip()
    fence = _FENCE_RE.match(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as e:
        return None, f"{e.msg} at line {e.lineno} column {e.colno}"


def _fold(text: str, case_sensitive: bool) -> str:
    return text if case_sensitive else text.casefold()


def check_assertion(assertion: Assertion, response: ModelResponse) -> AssertionResult:
    """Run one assertion against one response.

    Args:
        assertion: The check to run
        response: The model response (content and latency are inspected)

    Returns:
        AssertionResult describing the outcome
    """
    kind = assertion.type
    label = assertion.label()
    content = response.content or ""
    cs = assertion.case_sensitive

    def result(passed: bool, message: str) -> AssertionResult:
        return AssertionResult(type=kind, label=label, passed=passed, message=message)

    if kind == "contains":
        needle = str(assertion.value)
        passed = _fold(needle, cs) in _fold(content, cs)
        return result(passed, "found" if passed else f"'{needle}' not found in response")

    if kind == "not_contains":
        needle = str(assertion.value)
        passed = _fold(needle, cs) not in _fold(content, cs)
        return result(passed, "absent" if passed else f"'{needle}' unexpectedly present in response")

    if kind in ("regex", "not_regex"):
        pattern = re.compile(str(assertion.value), 0 if cs else re.IGNORECASE)
        matched = pattern.search(content) is not None
        if kind == "regex":
            return result(matched, "pattern matched" if matched else "pattern did not match response")
        return result(not matched, "pattern absent" if not matched else "pattern unexpectedly matched response")

    if kind == "equals":
        expected = str(assertion.value).strip()
        actual = content.strip()
        passed = _fold(expected, cs) == _fold(actual, cs)
        return result(
            passed,
            "exact match" if passed else f"expected '{_preview(expected)}', got '{_preview(actual)}'",
        )

    if kind == "starts_with":
        prefix = str(assertion.value)
        passed = _fold(content.lstrip(), cs).startswith(_fold(prefix, cs))
        return result(passed, "prefix matched" if passed else f"response does not start with '{prefix}'")

    if kind == "ends_with":
        suffix = str(assertion.value)
        passed = _fold(content.rstrip(), cs).endswith(_fold(suffix, cs))
        return result(passed, "suffix matched" if passed else f"response does not end with '{suffix}'")

    if kind == "json_valid":
        _, error = extract_json(content)
        return result(error is None, "valid JSON" if error is None else f"invalid JSON: {error}")

    if kind == "json_schema":
        parsed, error = extract_json(content)
        if error is not None:
            return result(False, f"invalid JSON: {error}")
        try:
            import jsonschema
        except ImportError:
            return result(
                False,
                "json_schema assertions need the optional 'jsonschema' package: "
                "pip install promptlens[schema]",
            )
        try:
            jsonschema.validate(instance=parsed, schema=assertion.value)
        except jsonschema.SchemaError as e:
            return result(False, f"invalid schema: {e.message}")
        except jsonschema.ValidationError as e:
            path = "/".join(str(p) for p in e.absolute_path) or "$"
            return result(False, f"schema violation at {path}: {e.message}")
        return result(True, "matches schema")

    if kind == "min_chars":
        limit = int(assertion.value)
        length = len(content)
        return result(length >= limit, f"{length} chars (min {limit})")

    if kind == "max_chars":
        limit = int(assertion.value)
        length = len(content)
        return result(length <= limit, f"{length} chars (max {limit})")

    if kind == "max_latency_ms":
        limit = float(assertion.value)
        latency = float(response.latency_ms)
        return result(latency <= limit, f"{latency:.0f} ms (max {limit:.0f} ms)")

    # Pydantic's Literal validation makes this unreachable, but keep the
    # failure explicit rather than silently passing an unknown check.
    return result(False, f"unknown assertion type '{kind}'")  # pragma: no cover


def run_assertions(test_case: TestCase, response: ModelResponse) -> List[AssertionResult]:
    """Run every assertion declared on a test case.

    Args:
        test_case: Test case whose ``assertions`` list is evaluated
        response: The model response to check

    Returns:
        One AssertionResult per assertion, in declaration order. Empty when
        the test case declares no assertions or the response errored.
    """
    if not test_case.assertions or response.error:
        return []
    return [check_assertion(a, response) for a in test_case.assertions]
