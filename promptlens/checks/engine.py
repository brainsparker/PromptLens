"""Deterministic check evaluation engine.

Runs the ``checks`` declared on a test case against a model response.
All checks are pure Python: no API calls, no tokens, no latency. This
makes them suitable for tight CI loops where running the LLM judge on
every push would be too slow or too expensive.
"""

import json
import logging
import re
from typing import List

from promptlens.models.checks import CheckDefinition, CheckResult
from promptlens.models.result import ModelResponse
from promptlens.models.test_case import TestCase

logger = logging.getLogger(__name__)

# Matches a response that is entirely one fenced code block, e.g.
# ```json\n{...}\n``` (a very common shape for LLM JSON output).
_FENCE_PATTERN = re.compile(
    r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```$",
    re.DOTALL,
)


def evaluate_checks(
    test_case: TestCase,
    response: ModelResponse,
) -> List[CheckResult]:
    """Evaluate all deterministic checks for a test case.

    Args:
        test_case: The test case whose ``checks`` should run
        response: The model response to check

    Returns:
        One CheckResult per configured check, in declaration order.
        Empty list when the test case declares no checks.
    """
    results: List[CheckResult] = []
    for check in test_case.checks:
        try:
            results.append(_evaluate_one(check, response))
        except Exception as e:  # Defensive: a bad check must not kill the run
            logger.error(
                f"Check '{check.type}' raised unexpectedly for test case "
                f"'{test_case.id}': {e}"
            )
            results.append(
                CheckResult(
                    check_type=check.type,
                    passed=False,
                    reason=f"Check raised an unexpected error: {e}",
                    value=check.value,
                )
            )
    return results


def _evaluate_one(check: CheckDefinition, response: ModelResponse) -> CheckResult:
    """Evaluate a single check against a response."""
    content = response.content or ""

    if check.type == "contains":
        needle, haystack = _fold(check, str(check.value), content)
        passed = needle in haystack
        reason = (
            f"Response contains '{_short(check.value)}'"
            if passed
            else f"Response does not contain '{_short(check.value)}'"
        )

    elif check.type == "not_contains":
        needle, haystack = _fold(check, str(check.value), content)
        passed = needle not in haystack
        reason = (
            f"Response does not contain forbidden '{_short(check.value)}'"
            if passed
            else f"Response contains forbidden '{_short(check.value)}'"
        )

    elif check.type == "regex":
        flags = re.IGNORECASE if check.case_insensitive else 0
        passed = re.search(str(check.value), content, flags) is not None
        reason = (
            f"Response matches pattern '{_short(check.value)}'"
            if passed
            else f"Response does not match pattern '{_short(check.value)}'"
        )

    elif check.type == "not_regex":
        flags = re.IGNORECASE if check.case_insensitive else 0
        passed = re.search(str(check.value), content, flags) is None
        reason = (
            f"Response does not match forbidden pattern '{_short(check.value)}'"
            if passed
            else f"Response matches forbidden pattern '{_short(check.value)}'"
        )

    elif check.type == "exact_match":
        expected, actual = str(check.value).strip(), content.strip()
        if check.case_insensitive:
            expected, actual = expected.casefold(), actual.casefold()
        passed = expected == actual
        reason = (
            "Response exactly matches expected value"
            if passed
            else "Response does not exactly match expected value "
            f"(got '{_short(content.strip())}')"
        )

    elif check.type == "is_valid_json":
        passed, detail = _check_json(content)
        reason = (
            "Response is valid JSON"
            if passed
            else f"Response is not valid JSON: {detail}"
        )

    elif check.type == "max_latency_ms":
        latency = response.latency_ms or 0.0
        passed = latency <= float(check.value)
        reason = (
            f"Latency {latency:.0f}ms within budget {check.value}ms"
            if passed
            else f"Latency {latency:.0f}ms exceeds budget {check.value}ms"
        )

    elif check.type == "max_cost_usd":
        cost = response.cost_usd or 0.0
        passed = cost <= float(check.value)
        reason = (
            f"Cost ${cost:.6f} within budget ${float(check.value):.6f}"
            if passed
            else f"Cost ${cost:.6f} exceeds budget ${float(check.value):.6f}"
        )

    elif check.type == "min_length":
        passed = len(content) >= int(check.value)
        reason = (
            f"Response length {len(content)} meets minimum {int(check.value)}"
            if passed
            else f"Response length {len(content)} below minimum {int(check.value)}"
        )

    elif check.type == "max_length":
        passed = len(content) <= int(check.value)
        reason = (
            f"Response length {len(content)} within maximum {int(check.value)}"
            if passed
            else f"Response length {len(content)} exceeds maximum {int(check.value)}"
        )

    else:  # pragma: no cover - CheckDefinition validation prevents this
        passed = False
        reason = f"Unknown check type '{check.type}'"

    return CheckResult(
        check_type=check.type,
        passed=passed,
        reason=reason,
        value=check.value,
    )


def _fold(check: CheckDefinition, needle: str, haystack: str) -> tuple:
    """Apply case folding for case-insensitive string checks."""
    if check.case_insensitive:
        return needle.casefold(), haystack.casefold()
    return needle, haystack


def _check_json(content: str) -> tuple:
    """Check whether content parses as JSON.

    Accepts either raw JSON or a response that is entirely one fenced
    code block (e.g. ```json ... ```), which is a common LLM output shape.

    Returns:
        Tuple of (passed, detail) where detail describes the parse error.
    """
    candidate = content.strip()
    fence_match = _FENCE_PATTERN.match(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()
    if not candidate:
        return False, "response is empty"
    try:
        json.loads(candidate)
        return True, ""
    except json.JSONDecodeError as e:
        return False, str(e)


def _short(value: object, limit: int = 60) -> str:
    """Truncate a value for readable reasons."""
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
