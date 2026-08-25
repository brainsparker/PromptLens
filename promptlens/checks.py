"""Deterministic check evaluation engine.

Runs the checks declared on a test case against a model response, entirely
locally: no API calls, no cost, reproducible verdicts. See
promptlens.models.checks for the check schema.

The json_schema check implements a minimal, dependency-free subset of JSON
Schema: type, required, properties, items, and enum. That covers the common
"is the output well-formed structured JSON with the right fields" cases
without pulling in a new dependency. Schemas using unsupported keywords
still validate the supported ones and ignore the rest.
"""

import json
import re
from typing import Any, List, Optional, Tuple

from promptlens.models.checks import Check, CheckResult

_JSON_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def run_checks(checks: List[Check], response_text: str) -> List[CheckResult]:
    """Evaluate all checks against a response.

    Args:
        checks: Checks declared on the test case
        response_text: The model's response content

    Returns:
        One CheckResult per check, in declaration order
    """
    return [_run_check(check, response_text) for check in checks]


def _run_check(check: Check, text: str) -> CheckResult:
    """Evaluate a single check against a response."""
    if check.type == "contains":
        passed, detail = _check_contains(check, text, negate=False)
    elif check.type == "not_contains":
        passed, detail = _check_contains(check, text, negate=True)
    elif check.type == "regex":
        passed, detail = _check_regex(check, text)
    elif check.type == "exact_match":
        passed, detail = _check_exact_match(check, text)
    elif check.type == "json_valid":
        passed, detail = _check_json_valid(text)
    elif check.type == "json_schema":
        passed, detail = _check_json_schema(check, text)
    else:  # pragma: no cover - Check model validation prevents this
        passed, detail = False, f"Unknown check type: {check.type}"

    return CheckResult(
        check_type=check.type,
        description=check.describe(),
        passed=passed,
        detail=detail,
    )


def _normalize(text: str, case_sensitive: bool) -> str:
    return text if case_sensitive else text.lower()


def _check_contains(check: Check, text: str, negate: bool) -> Tuple[bool, str]:
    values = check.value if isinstance(check.value, list) else [check.value]
    haystack = _normalize(text, check.case_sensitive)
    found = [v for v in values if _normalize(v, check.case_sensitive) in haystack]
    missing = [v for v in values if v not in found]

    if negate:
        if found:
            return False, f"Response contains forbidden substring(s): {', '.join(repr(v) for v in found)}"
        return True, "No forbidden substrings found"

    if check.mode == "any":
        if found:
            return True, f"Found: {', '.join(repr(v) for v in found)}"
        return False, f"None of the expected substrings found: {', '.join(repr(v) for v in values)}"

    if missing:
        return False, f"Missing expected substring(s): {', '.join(repr(v) for v in missing)}"
    return True, "All expected substrings found"


def _check_regex(check: Check, text: str) -> Tuple[bool, str]:
    if re.search(check.pattern, text):
        return True, f"Pattern {check.pattern!r} matched"
    return False, f"Pattern {check.pattern!r} did not match the response"


def _check_exact_match(check: Check, text: str) -> Tuple[bool, str]:
    expected = check.value
    actual = text.strip() if check.strip else text
    if not check.case_sensitive:
        expected = expected.lower()
        actual = actual.lower()
    if actual == expected:
        return True, "Response matches expected value exactly"
    return False, f"Response does not exactly match {check.value!r}"


def _extract_json(text: str) -> Optional[Any]:
    """Parse JSON from a response, tolerating markdown code fences.

    Tries the raw text first, then the contents of the first fenced code
    block, since models frequently wrap JSON in ```json fences.
    """
    candidates = [text.strip()]
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _check_json_valid(text: str) -> Tuple[bool, str]:
    if _extract_json(text) is not None:
        return True, "Response is valid JSON"
    return False, "Response is not valid JSON (raw or fenced)"


def _check_json_schema(check: Check, text: str) -> Tuple[bool, str]:
    data = _extract_json(text)
    if data is None:
        return False, "Response is not valid JSON (raw or fenced)"

    errors: List[str] = []
    _validate_schema(data, check.json_schema, path="$", errors=errors)
    if errors:
        return False, "; ".join(errors[:5])
    return True, "Response JSON conforms to schema"


def _validate_schema(data: Any, schema: Any, path: str, errors: List[str]) -> None:
    """Validate data against a minimal JSON Schema subset.

    Supports: type, required, properties, items, enum. Unsupported
    keywords are ignored.
    """
    if not isinstance(schema, dict):
        return

    expected_type = schema.get("type")
    if expected_type is not None:
        expected_types = expected_type if isinstance(expected_type, list) else [expected_type]
        python_types: Tuple[type, ...] = ()
        for t in expected_types:
            mapped = _JSON_TYPE_MAP.get(t)
            if mapped is not None:
                python_types = python_types + (mapped if isinstance(mapped, tuple) else (mapped,))
        if python_types and not isinstance(data, python_types):
            # bool is a subclass of int in Python; keep them distinct for JSON
            errors.append(f"{path}: expected type {expected_type}, got {type(data).__name__}")
            return
        if python_types and isinstance(data, bool) and bool not in python_types:
            errors.append(f"{path}: expected type {expected_type}, got boolean")
            return

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and data not in enum_values:
        errors.append(f"{path}: value {data!r} not in enum {enum_values!r}")

    if isinstance(data, dict):
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if key not in data:
                    errors.append(f"{path}: missing required property {key!r}")

        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, subschema in properties.items():
                if key in data:
                    _validate_schema(data[key], subschema, f"{path}.{key}", errors)

    if isinstance(data, list):
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for index, item in enumerate(data):
                _validate_schema(item, items_schema, f"{path}[{index}]", errors)
