"""Deterministic check data models.

Checks are deterministic assertions attached to a test case. They run
locally against the model's response text, cost nothing, and produce
reproducible pass/fail verdicts, which makes them safe to gate CI on.
They complement (and can fully replace) LLM-as-judge scoring for test
cases where the expectation is cheaply and mechanically verifiable.
"""

import re
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# Check types that require a `value`
_VALUE_TYPES = {"contains", "not_contains", "exact_match"}

SUPPORTED_CHECK_TYPES = [
    "contains",
    "not_contains",
    "regex",
    "exact_match",
    "json_valid",
    "json_schema",
]


class Check(BaseModel):
    """A single deterministic check on a model response.

    Attributes:
        type: Check type. One of: contains, not_contains, regex,
            exact_match, json_valid, json_schema
        value: Expected value(s). A string or list of strings for
            contains/not_contains, a string for exact_match
        pattern: Regular expression pattern (regex type only)
        json_schema: Minimal JSON schema object (json_schema type only).
            Supported keywords: type, required, properties, items, enum
        mode: For contains with a list of values: "all" requires every
            substring, "any" requires at least one. Default "all"
        case_sensitive: Whether string comparison is case sensitive.
            Applies to contains, not_contains, and exact_match.
            Default False
        strip: Strip surrounding whitespace before exact_match comparison.
            Default True
    """

    type: str
    value: Optional[Union[str, List[str]]] = None
    pattern: Optional[str] = None
    json_schema: Optional[Dict[str, Any]] = None
    mode: str = "all"
    case_sensitive: bool = False
    strip: bool = True

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_CHECK_TYPES:
            supported = ", ".join(SUPPORTED_CHECK_TYPES)
            raise ValueError(
                f"Unsupported check type '{value}'. Supported types: {supported}"
            )
        return normalized

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"all", "any"}:
            raise ValueError("mode must be 'all' or 'any'")
        return normalized

    @model_validator(mode="after")
    def validate_fields_for_type(self) -> "Check":
        if self.type in _VALUE_TYPES:
            if self.value is None:
                raise ValueError(f"Check type '{self.type}' requires a 'value'")
            if isinstance(self.value, list):
                if self.type == "exact_match":
                    raise ValueError("exact_match takes a single string value, not a list")
                if not self.value:
                    raise ValueError(f"Check type '{self.type}' requires a non-empty value list")
                for item in self.value:
                    if not isinstance(item, str):
                        raise ValueError(f"Check type '{self.type}' values must be strings")

        if self.type == "regex":
            if not self.pattern:
                raise ValueError("Check type 'regex' requires a 'pattern'")
            try:
                re.compile(self.pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")

        if self.type == "json_schema" and self.json_schema is None:
            raise ValueError("Check type 'json_schema' requires a 'json_schema' object")

        return self

    def describe(self) -> str:
        """Return a short human-readable description of the check."""
        if self.type in {"contains", "not_contains"}:
            values = self.value if isinstance(self.value, list) else [self.value]
            joined = ", ".join(repr(v) for v in values)
            qualifier = f" ({self.mode})" if isinstance(self.value, list) else ""
            return f"{self.type}{qualifier}: {joined}"
        if self.type == "exact_match":
            return f"exact_match: {self.value!r}"
        if self.type == "regex":
            return f"regex: {self.pattern!r}"
        if self.type == "json_schema":
            return "json_schema"
        return self.type


class CheckResult(BaseModel):
    """Outcome of evaluating a single check against a response.

    Attributes:
        check_type: The check type that was evaluated
        description: Short human-readable description of the check
        passed: Whether the check passed
        detail: Explanation of the outcome, most useful on failure
    """

    check_type: str
    description: str
    passed: bool
    detail: str = ""
