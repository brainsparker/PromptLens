"""Deterministic assertion models.

Assertions are checks on a model response that need no LLM to evaluate:
string containment, regular expressions, JSON validity, length bounds, and
token-level F1 against a reference answer. They return the same result every
time they run and cost nothing, which makes them suitable as CI merge gates
where LLM-as-judge scores are too noisy or too expensive to run per commit.
"""

import re
from typing import Any, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

# Assertion types that compare the response against a string value.
STRING_ASSERTIONS = frozenset(
    {"equals", "contains", "not_contains", "starts_with", "ends_with"}
)

# Assertion types that compile the value as a regular expression.
REGEX_ASSERTIONS = frozenset({"regex", "not_regex"})

# Assertion types that compare response length (in characters) to an integer.
LENGTH_ASSERTIONS = frozenset({"min_length", "max_length"})

# Assertion types that need no value at all.
VALUELESS_ASSERTIONS = frozenset({"is_json"})

# Assertion types that produce a 0.0 to 1.0 score compared to a threshold.
SCORED_ASSERTIONS = frozenset({"token_f1"})

ASSERTION_TYPES = frozenset(
    STRING_ASSERTIONS
    | REGEX_ASSERTIONS
    | LENGTH_ASSERTIONS
    | VALUELESS_ASSERTIONS
    | SCORED_ASSERTIONS
)

DEFAULT_F1_THRESHOLD = 0.5


class Assertion(BaseModel):
    """A single deterministic check declared on a golden-set test case.

    Attributes:
        type: Assertion type. One of: equals, contains, not_contains,
            starts_with, ends_with, regex, not_regex, is_json, min_length,
            max_length, token_f1.
        value: Comparison value. A string for text and regex assertions, an
            integer for length assertions. For token_f1 it is the reference
            text; when omitted the test case's reference_answer is used.
        case_sensitive: Whether text assertions compare case-sensitively.
            Applies to equals, contains, not_contains, starts_with, ends_with.
            Regex assertions use the re.IGNORECASE flag when this is False.
        threshold: Minimum score for scored assertions (token_f1). Defaults
            to 0.5.
        name: Optional human-readable label shown in reports.
    """

    type: str
    # Strict types so YAML "10" stays a string and true is not silently an int.
    value: Optional[Union[StrictStr, StrictInt, StrictFloat]] = None
    case_sensitive: bool = True
    threshold: Optional[float] = None
    name: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "contains",
                "value": "30 days",
                "case_sensitive": False,
            }
        }
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ASSERTION_TYPES:
            allowed = ", ".join(sorted(ASSERTION_TYPES))
            raise ValueError(
                f"unsupported assertion type '{value}'. Supported types: {allowed}"
            )
        return normalized

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        if not 0.0 <= value <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        return value

    @model_validator(mode="after")
    def validate_value_for_type(self) -> "Assertion":
        kind = self.type

        if kind in STRING_ASSERTIONS or kind in REGEX_ASSERTIONS:
            if self.value is None or not isinstance(self.value, str):
                raise ValueError(f"assertion '{kind}' requires a string value")
            if kind in REGEX_ASSERTIONS:
                try:
                    re.compile(self.value)
                except re.error as exc:
                    raise ValueError(f"assertion '{kind}' has an invalid regex: {exc}")

        elif kind in LENGTH_ASSERTIONS:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise ValueError(f"assertion '{kind}' requires an integer value")
            if self.value < 0 or int(self.value) != self.value:
                raise ValueError(f"assertion '{kind}' requires a non-negative integer value")
            self.value = int(self.value)

        elif kind in VALUELESS_ASSERTIONS:
            if self.value is not None:
                raise ValueError(f"assertion '{kind}' does not take a value")

        elif kind in SCORED_ASSERTIONS:
            if self.value is not None and not isinstance(self.value, str):
                raise ValueError(
                    f"assertion '{kind}' value must be a string when provided"
                )

        if self.threshold is not None and kind not in SCORED_ASSERTIONS:
            scored = ", ".join(sorted(SCORED_ASSERTIONS))
            raise ValueError(f"threshold is only valid for scored assertions ({scored})")

        return self

    def label(self) -> str:
        """Short label used in explanations and reports."""
        if self.name:
            return self.name
        if self.type in VALUELESS_ASSERTIONS:
            return self.type
        if self.type in SCORED_ASSERTIONS:
            threshold = self.threshold if self.threshold is not None else DEFAULT_F1_THRESHOLD
            return f"{self.type} >= {threshold:g}"
        return f"{self.type} {_short(self.value)}"


def _short(value: Any, limit: int = 60) -> str:
    text = repr(value) if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
