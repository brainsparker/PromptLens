"""Deterministic assertion data models.

Assertions are cheap, deterministic checks that run on a model response
before (and independently of) the LLM judge. They cover the questions that
have one right answer: does the output contain a required phrase, match a
regex, parse as JSON, satisfy a schema, stay under a length or latency cap.

They are declared per test case in the golden set:

    test_cases:
      - id: "refund-001"
        query: "What's your refund policy?"
        expected_behavior: "Explain the 30-day refund policy"
        assertions:
          - type: contains
            value: "30 days"
          - type: not_contains
            value: ["As an AI", "I cannot"]
          - type: max_chars
            value: 1200
"""

import re
from typing import Any, List, Optional

from pydantic import BaseModel, field_validator, model_validator

# Assertion types that read the response text.
TEXT_ASSERTION_TYPES = (
    "equals",
    "contains",
    "not_contains",
    "contains_any",
    "starts_with",
    "ends_with",
    "regex",
    "not_regex",
    "is_json",
    "json_schema",
    "max_chars",
    "min_chars",
)

# Assertion types that read response metadata rather than text.
METADATA_ASSERTION_TYPES = (
    "max_latency_ms",
    "max_cost_usd",
)

ASSERTION_TYPES = TEXT_ASSERTION_TYPES + METADATA_ASSERTION_TYPES

# Types whose value is one string or a list of strings.
_STRING_OR_LIST_TYPES = {"contains", "not_contains", "contains_any"}
# Types whose value is exactly one string.
_STRING_TYPES = {"equals", "starts_with", "ends_with", "regex", "not_regex"}
# Types whose value is a positive number.
_NUMBER_TYPES = {"max_chars", "min_chars", "max_latency_ms", "max_cost_usd"}
# Types that take no value.
_NO_VALUE_TYPES = {"is_json"}
# Types whose value is a JSON Schema mapping.
_SCHEMA_TYPES = {"json_schema"}


class Assertion(BaseModel):
    """One deterministic check declared on a test case.

    Attributes:
        type: Assertion type (see ASSERTION_TYPES)
        value: Expected value. Shape depends on type: a string for equals,
            starts_with, ends_with, regex and not_regex; a string or list of
            strings for contains, not_contains and contains_any; a positive
            number for max_chars, min_chars, max_latency_ms and max_cost_usd;
            a JSON Schema mapping for json_schema; omitted for is_json.
        ignore_case: Case-insensitive matching for text assertions
        name: Optional human-readable label shown in reports
        message: Optional custom message shown when the assertion fails
    """

    type: str
    value: Optional[Any] = None
    ignore_case: bool = False
    name: Optional[str] = None
    message: Optional[str] = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_")
        if normalized not in ASSERTION_TYPES:
            supported = ", ".join(ASSERTION_TYPES)
            raise ValueError(f"Unknown assertion type '{value}'. Supported types: {supported}")
        return normalized

    @model_validator(mode="after")
    def validate_value_for_type(self) -> "Assertion":
        kind = self.type
        value = self.value

        if kind in _NO_VALUE_TYPES:
            if value is not None:
                raise ValueError(f"Assertion '{kind}' does not take a value")
            return self

        if value is None:
            raise ValueError(f"Assertion '{kind}' requires a value")

        if kind in _STRING_TYPES:
            if not isinstance(value, str) or not value:
                raise ValueError(f"Assertion '{kind}' requires a non-empty string value")
            if kind in ("regex", "not_regex"):
                try:
                    re.compile(value)
                except re.error as exc:
                    raise ValueError(f"Assertion '{kind}' has an invalid pattern: {exc}")
            return self

        if kind in _STRING_OR_LIST_TYPES:
            if isinstance(value, str):
                if not value:
                    raise ValueError(f"Assertion '{kind}' requires a non-empty string value")
                return self
            if isinstance(value, list):
                if not value or not all(isinstance(item, str) and item for item in value):
                    raise ValueError(
                        f"Assertion '{kind}' requires a non-empty list of non-empty strings"
                    )
                return self
            raise ValueError(f"Assertion '{kind}' requires a string or a list of strings")

        if kind in _NUMBER_TYPES:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Assertion '{kind}' requires a numeric value")
            if value <= 0:
                raise ValueError(f"Assertion '{kind}' requires a value greater than 0")
            return self

        if kind in _SCHEMA_TYPES:
            if not isinstance(value, dict):
                raise ValueError(f"Assertion '{kind}' requires a JSON Schema mapping as its value")
            return self

        return self  # pragma: no cover - every type is handled above

    @property
    def values(self) -> List[str]:
        """The expected strings for string-or-list assertions."""
        if isinstance(self.value, list):
            return list(self.value)
        if isinstance(self.value, str):
            return [self.value]
        return []

    @property
    def label(self) -> str:
        """A short display label for reports."""
        if self.name:
            return self.name
        if self.type in _NO_VALUE_TYPES:
            return self.type
        if isinstance(self.value, dict):
            return f"{self.type}"
        if isinstance(self.value, list):
            shown = ", ".join(repr(v) for v in self.value[:3])
            if len(self.value) > 3:
                shown += ", ..."
            return f"{self.type} [{shown}]"
        return f"{self.type} {self.value!r}"


class AssertionResult(BaseModel):
    """Outcome of one assertion against one model response.

    Attributes:
        type: Assertion type that was evaluated
        name: Display label for the assertion
        passed: Whether the check passed
        message: Human-readable explanation of the outcome
    """

    type: str
    name: str
    passed: bool
    message: str = ""


class AssertionSummary(BaseModel):
    """Aggregate assertion counts for a slice of results.

    Attributes:
        cases_with_assertions: Evaluations that declared at least one assertion
        cases_passed: Evaluations whose assertions all passed
        cases_failed: Evaluations with at least one failed assertion
        assertions_total: Individual assertions evaluated
        assertions_failed: Individual assertions that failed
    """

    cases_with_assertions: int = 0
    cases_passed: int = 0
    cases_failed: int = 0
    assertions_total: int = 0
    assertions_failed: int = 0

    @property
    def pass_rate(self) -> Optional[float]:
        """Fraction of evaluations whose assertions all passed, or None."""
        if not self.cases_with_assertions:
            return None
        return self.cases_passed / self.cases_with_assertions
