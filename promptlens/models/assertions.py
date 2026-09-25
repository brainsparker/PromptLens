"""Deterministic assertion models.

Assertions are cheap, judge-free checks that run against a model response
before (or instead of) the LLM judge. They are declared per test case in the
golden set and evaluated locally, so they cost no tokens and work with any
provider, including keyless local models.

Supported assertion types:

    contains         Response contains the given substring
    not_contains     Response does not contain the given substring
    regex            Response matches the given regular expression (re.search)
    not_regex        Response does not match the given regular expression
    equals           Response equals the given string (whitespace-trimmed)
    starts_with      Response starts with the given string
    ends_with        Response ends with the given string
    json_valid       Response parses as JSON (a leading Markdown code fence is tolerated)
    json_schema      Response parses as JSON and validates against the given
                     JSON Schema (requires the optional ``jsonschema`` package)
    min_chars        Response has at least this many characters
    max_chars        Response has at most this many characters
    max_latency_ms   Model latency was at most this many milliseconds

Text checks honour ``case_sensitive`` (default true). ``regex`` and
``not_regex`` compile with ``re.IGNORECASE`` when ``case_sensitive`` is false.
"""

import re
from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, field_validator, model_validator

AssertionType = Literal[
    "contains",
    "not_contains",
    "regex",
    "not_regex",
    "equals",
    "starts_with",
    "ends_with",
    "json_valid",
    "json_schema",
    "min_chars",
    "max_chars",
    "max_latency_ms",
]

TEXT_ASSERTIONS = {
    "contains",
    "not_contains",
    "regex",
    "not_regex",
    "equals",
    "starts_with",
    "ends_with",
}
NUMERIC_ASSERTIONS = {"min_chars", "max_chars", "max_latency_ms"}
VALUELESS_ASSERTIONS = {"json_valid"}
SCHEMA_ASSERTIONS = {"json_schema"}


class Assertion(BaseModel):
    """A single deterministic check declared on a test case.

    Attributes:
        type: Which check to run (see module docstring)
        value: Check operand: a string for text checks, a number for
            numeric checks, a JSON Schema mapping for json_schema, and
            omitted for json_valid
        case_sensitive: Whether text checks are case sensitive
        description: Optional human-readable label shown in reports
    """

    type: AssertionType
    value: Optional[Union[str, int, float, Dict[str, Any]]] = None
    case_sensitive: bool = True
    description: Optional[str] = None

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower().replace("-", "_")
        return value

    @field_validator("value", mode="before")
    @classmethod
    def reject_booleans(cls, value: Any) -> Any:
        # YAML `value: true` would otherwise coerce to 1 for numeric checks.
        if isinstance(value, bool):
            raise ValueError("assertion value cannot be a boolean")
        return value

    @model_validator(mode="after")
    def validate_value_for_type(self) -> "Assertion":
        kind = self.type
        value = self.value

        if kind in TEXT_ASSERTIONS:
            if not isinstance(value, str):
                raise ValueError(f"assertion '{kind}' requires a string value")
            if kind in ("contains", "not_contains", "starts_with", "ends_with") and value == "":
                raise ValueError(f"assertion '{kind}' requires a non-empty string value")
            if kind in ("regex", "not_regex"):
                try:
                    re.compile(value)
                except re.error as e:
                    raise ValueError(f"assertion '{kind}' has an invalid pattern: {e}")

        elif kind in NUMERIC_ASSERTIONS:
            if not isinstance(value, (int, float)):
                raise ValueError(f"assertion '{kind}' requires a numeric value")
            if value < 0:
                raise ValueError(f"assertion '{kind}' requires a non-negative value")

        elif kind in VALUELESS_ASSERTIONS:
            if value is not None:
                raise ValueError(f"assertion '{kind}' does not take a value")

        elif kind in SCHEMA_ASSERTIONS:
            if not isinstance(value, dict):
                raise ValueError("assertion 'json_schema' requires a JSON Schema mapping as its value")

        return self

    def label(self) -> str:
        """Short human-readable label for reports."""
        if self.description:
            return self.description
        if self.type in VALUELESS_ASSERTIONS:
            return self.type
        if self.type in SCHEMA_ASSERTIONS:
            return "json_schema"
        shown = str(self.value)
        if len(shown) > 60:
            shown = shown[:57] + "..."
        return f"{self.type}: {shown}"


class AssertionResult(BaseModel):
    """Outcome of one assertion against one model response.

    Attributes:
        type: The assertion type that ran
        label: Human-readable label (description or type plus operand)
        passed: Whether the check passed
        message: Short explanation, most useful on failure
    """

    type: str
    label: str
    passed: bool
    message: str = ""
