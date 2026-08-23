"""Deterministic assertion checks for test case responses.

Assertions run before the LLM judge and cost zero tokens. They cover the
checks that never need semantic judgment: does the response parse as JSON,
does it match a JSON Schema, does it contain (or avoid) a substring, does it
match a regex, does it start with a prefix.

When any assertion fails, the runner records the failure and skips the LLM
judge call entirely, so judge tokens are only spent on responses that pass
the cheap deterministic gate first.
"""

import json
import logging
import re
from typing import List

from promptlens.models.result import AssertionResult
from promptlens.models.test_case import Assertion

logger = logging.getLogger(__name__)

try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:  # pragma: no cover
    jsonschema = None  # type: ignore[assignment]
    HAS_JSONSCHEMA = False

# Maximum characters of the response shown in failure messages.
_PREVIEW_CHARS = 120


def _preview(content: str) -> str:
    """Return a short single-line preview of the response for messages."""
    flattened = " ".join(content.split())
    if len(flattened) <= _PREVIEW_CHARS:
        return flattened
    return flattened[:_PREVIEW_CHARS] + "..."


def _check_is_json(content: str, assertion: Assertion) -> AssertionResult:
    """Check that the response parses as JSON."""
    try:
        json.loads(content)
        return AssertionResult(
            type=assertion.type,
            passed=True,
            message="Response is valid JSON",
        )
    except (json.JSONDecodeError, ValueError) as e:
        return AssertionResult(
            type=assertion.type,
            passed=False,
            message=f"Response is not valid JSON: {e}",
        )


def _check_json_schema(content: str, assertion: Assertion) -> AssertionResult:
    """Check that the response parses as JSON and matches a JSON Schema."""
    if not HAS_JSONSCHEMA:
        return AssertionResult(
            type=assertion.type,
            passed=False,
            message=(
                "json_schema assertion requires the 'jsonschema' package. "
                "Install it with: pip install jsonschema"
            ),
        )

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, ValueError) as e:
        return AssertionResult(
            type=assertion.type,
            passed=False,
            message=f"Response is not valid JSON, cannot check schema: {e}",
        )

    try:
        jsonschema.validate(instance=parsed, schema=assertion.value)
        return AssertionResult(
            type=assertion.type,
            passed=True,
            message="Response matches JSON Schema",
        )
    except jsonschema.SchemaError as e:
        return AssertionResult(
            type=assertion.type,
            passed=False,
            message=f"Invalid JSON Schema in assertion: {e.message}",
        )
    except jsonschema.ValidationError as e:
        path = ".".join(str(p) for p in e.absolute_path) or "(root)"
        return AssertionResult(
            type=assertion.type,
            passed=False,
            message=f"Schema validation failed at {path}: {e.message}",
        )


def _check_contains(content: str, assertion: Assertion) -> AssertionResult:
    """Check that the response contains a substring (case-sensitive)."""
    if assertion.value in content:
        return AssertionResult(
            type=assertion.type,
            value=assertion.value,
            passed=True,
            message=f"Response contains '{assertion.value}'",
        )
    return AssertionResult(
        type=assertion.type,
        value=assertion.value,
        passed=False,
        message=(
            f"Response does not contain '{assertion.value}'. " f"Response: {_preview(content)}"
        ),
    )


def _check_not_contains(content: str, assertion: Assertion) -> AssertionResult:
    """Check that the response does not contain a substring (case-sensitive)."""
    if assertion.value not in content:
        return AssertionResult(
            type=assertion.type,
            value=assertion.value,
            passed=True,
            message=f"Response does not contain '{assertion.value}'",
        )
    return AssertionResult(
        type=assertion.type,
        value=assertion.value,
        passed=False,
        message=f"Response contains forbidden substring '{assertion.value}'",
    )


def _check_regex(content: str, assertion: Assertion) -> AssertionResult:
    """Check that the response matches a regex (re.search semantics)."""
    try:
        pattern = re.compile(assertion.value)
    except re.error as e:
        return AssertionResult(
            type=assertion.type,
            value=assertion.value,
            passed=False,
            message=f"Invalid regex pattern '{assertion.value}': {e}",
        )

    if pattern.search(content):
        return AssertionResult(
            type=assertion.type,
            value=assertion.value,
            passed=True,
            message=f"Response matches pattern '{assertion.value}'",
        )
    return AssertionResult(
        type=assertion.type,
        value=assertion.value,
        passed=False,
        message=(
            f"Response does not match pattern '{assertion.value}'. "
            f"Response: {_preview(content)}"
        ),
    )


def _check_starts_with(content: str, assertion: Assertion) -> AssertionResult:
    """Check that the response starts with a prefix (leading whitespace ignored)."""
    if content.lstrip().startswith(assertion.value):
        return AssertionResult(
            type=assertion.type,
            value=assertion.value,
            passed=True,
            message=f"Response starts with '{assertion.value}'",
        )
    return AssertionResult(
        type=assertion.type,
        value=assertion.value,
        passed=False,
        message=(
            f"Response does not start with '{assertion.value}'. " f"Response: {_preview(content)}"
        ),
    )


_CHECKS = {
    "is_json": _check_is_json,
    "json_schema": _check_json_schema,
    "contains": _check_contains,
    "not_contains": _check_not_contains,
    "regex": _check_regex,
    "starts_with": _check_starts_with,
}


def evaluate_assertions(content: str, assertions: List[Assertion]) -> List[AssertionResult]:
    """Evaluate all assertions against a model response.

    All assertions are always evaluated (no short-circuit) so the report
    shows every failure at once instead of one per run.

    Args:
        content: The model response text
        assertions: Assertions from the test case

    Returns:
        One AssertionResult per assertion, in order
    """
    results: List[AssertionResult] = []
    for assertion in assertions:
        check = _CHECKS.get(assertion.type)
        if check is None:  # pragma: no cover - blocked by model validation
            results.append(
                AssertionResult(
                    type=assertion.type,
                    passed=False,
                    message=f"Unknown assertion type '{assertion.type}'",
                )
            )
            continue
        results.append(check(content, assertion))
    return results
