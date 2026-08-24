"""Deterministic check data models.

Deterministic checks are cheap, local assertions that run against a model
response before any LLM judge is invoked. When a check fails, the judge call
is skipped entirely, so obviously-broken outputs never spend judge tokens.
"""

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, model_validator

# Check types that require a string value
STRING_CHECK_TYPES = {"contains", "not_contains", "equals", "regex", "not_regex"}

# Check types that require a non-negative integer value
LENGTH_CHECK_TYPES = {"min_length", "max_length"}

# Check types that take no value
VALUELESS_CHECK_TYPES = {"json_valid"}

ALL_CHECK_TYPES = STRING_CHECK_TYPES | LENGTH_CHECK_TYPES | VALUELESS_CHECK_TYPES


class CheckSpec(BaseModel):
    """A single deterministic check declared on a test case.

    Attributes:
        type: Check type. One of: contains, not_contains, equals, regex,
            not_regex, json_valid, min_length, max_length
        value: Check argument. A string for text/regex checks, a
            non-negative integer for length checks, unused for json_valid
        case_sensitive: Whether text comparisons are case sensitive
            (applies to contains, not_contains, and equals)
    """

    type: str
    value: Optional[Any] = None
    case_sensitive: bool = True

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "type": "contains",
            "value": "reset your password",
            "case_sensitive": False,
        }
    })

    @model_validator(mode="after")
    def validate_spec(self) -> "CheckSpec":
        if self.type not in ALL_CHECK_TYPES:
            allowed = ", ".join(sorted(ALL_CHECK_TYPES))
            raise ValueError(
                f"unknown check type '{self.type}'. Allowed types: {allowed}"
            )

        if self.type in STRING_CHECK_TYPES:
            if not isinstance(self.value, str) or not self.value:
                raise ValueError(
                    f"check type '{self.type}' requires a non-empty string value"
                )
            if self.type in {"regex", "not_regex"}:
                try:
                    re.compile(self.value)
                except re.error as e:
                    raise ValueError(
                        f"check type '{self.type}' has an invalid pattern: {e}"
                    )

        if self.type in LENGTH_CHECK_TYPES:
            if isinstance(self.value, bool) or not isinstance(self.value, int):
                raise ValueError(
                    f"check type '{self.type}' requires an integer value"
                )
            if self.value < 0:
                raise ValueError(
                    f"check type '{self.type}' requires a non-negative integer value"
                )

        if self.type in VALUELESS_CHECK_TYPES and self.value is not None:
            raise ValueError(
                f"check type '{self.type}' does not take a value"
            )

        return self


class CheckResult(BaseModel):
    """Outcome of running one deterministic check against a response.

    Attributes:
        type: The check type that ran
        passed: Whether the check passed
        detail: Human-readable description of the outcome
    """

    type: str
    passed: bool
    detail: str


def run_check(spec: CheckSpec, content: str) -> CheckResult:
    """Run a single deterministic check against response content.

    Args:
        spec: The check specification
        content: The model response text

    Returns:
        CheckResult with pass/fail and a human-readable detail
    """
    if spec.type == "contains":
        haystack = content if spec.case_sensitive else content.lower()
        needle = spec.value if spec.case_sensitive else spec.value.lower()
        passed = needle in haystack
        detail = (
            f"response contains '{spec.value}'"
            if passed
            else f"response does not contain '{spec.value}'"
        )
    elif spec.type == "not_contains":
        haystack = content if spec.case_sensitive else content.lower()
        needle = spec.value if spec.case_sensitive else spec.value.lower()
        passed = needle not in haystack
        detail = (
            f"response does not contain forbidden text '{spec.value}'"
            if passed
            else f"response contains forbidden text '{spec.value}'"
        )
    elif spec.type == "equals":
        left = content if spec.case_sensitive else content.lower()
        right = spec.value if spec.case_sensitive else spec.value.lower()
        passed = left.strip() == right.strip()
        detail = (
            "response matches expected text exactly"
            if passed
            else "response does not match expected text exactly"
        )
    elif spec.type == "regex":
        passed = re.search(spec.value, content) is not None
        detail = (
            f"response matches pattern /{spec.value}/"
            if passed
            else f"response does not match pattern /{spec.value}/"
        )
    elif spec.type == "not_regex":
        passed = re.search(spec.value, content) is None
        detail = (
            f"response does not match forbidden pattern /{spec.value}/"
            if passed
            else f"response matches forbidden pattern /{spec.value}/"
        )
    elif spec.type == "json_valid":
        try:
            json.loads(content)
            passed = True
            detail = "response is valid JSON"
        except (ValueError, TypeError):
            passed = False
            detail = "response is not valid JSON"
    elif spec.type == "min_length":
        passed = len(content) >= spec.value
        detail = (
            f"response length {len(content)} >= minimum {spec.value}"
            if passed
            else f"response length {len(content)} is below minimum {spec.value}"
        )
    elif spec.type == "max_length":
        passed = len(content) <= spec.value
        detail = (
            f"response length {len(content)} <= maximum {spec.value}"
            if passed
            else f"response length {len(content)} exceeds maximum {spec.value}"
        )
    else:  # pragma: no cover - CheckSpec validation prevents this
        raise ValueError(f"unknown check type '{spec.type}'")

    return CheckResult(type=spec.type, passed=passed, detail=detail)


def run_checks(specs, content: str):
    """Run a list of deterministic checks against response content.

    Args:
        specs: List of CheckSpec objects
        content: The model response text

    Returns:
        List of CheckResult objects, one per spec, in order
    """
    return [run_check(spec, content) for spec in specs]
