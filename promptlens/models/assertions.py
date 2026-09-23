"""Deterministic assertion data models.

Assertions are cheap, local checks that run against a model response before
(or instead of) the LLM judge. They never call a model, so they are free,
instant, and give the same answer every time. Use them for the parts of a
response that have a right answer (a required phrase, a forbidden phrase,
a length budget, valid JSON) and leave the judgment calls to the judge.
"""

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, model_validator

# Assertion types that compare the response against a single string value.
STRING_ASSERTIONS = frozenset(
    {"contains", "not_contains", "equals", "starts_with", "ends_with", "regex"}
)
# Assertion types that compare the response against a list of strings.
LIST_ASSERTIONS = frozenset({"contains_any", "contains_all"})
# Assertion types that compare the response length against an integer.
LENGTH_ASSERTIONS = frozenset({"min_length", "max_length"})
# Assertion types that take no value at all.
VALUELESS_ASSERTIONS = frozenset({"is_json"})

SUPPORTED_ASSERTION_TYPES = frozenset(
    STRING_ASSERTIONS | LIST_ASSERTIONS | LENGTH_ASSERTIONS | VALUELESS_ASSERTIONS
)


class Assertion(BaseModel):
    """A single deterministic check against a model response.

    Attributes:
        type: The check to perform. One of: contains, not_contains,
            contains_any, contains_all, equals, starts_with, ends_with,
            regex, is_json, min_length, max_length.
        value: The expected value. A string for text checks, a list of
            strings for contains_any/contains_all, an integer (characters)
            for min_length/max_length, and omitted for is_json.
        case_sensitive: Whether text comparisons respect case. Applies to
            every text check including regex. Defaults to True.
        name: Optional label shown in reports instead of the generated one.
    """

    type: str
    # Typed as Any so the validator below owns every error message. A Union
    # would let pydantic coerce values (True -> 1, 5 -> "5") or emit three
    # confusing branch errors for one bad value.
    value: Any = None
    case_sensitive: bool = True
    name: Optional[str] = None

    @model_validator(mode="after")
    def validate_value_for_type(self) -> "Assertion":
        """Reject assertions whose value does not fit their type.

        Validation happens at load time so `promptlens validate` catches a
        bad regex or a missing value before any paid model call is made.
        """
        assertion_type = self.type.strip().lower()
        if assertion_type not in SUPPORTED_ASSERTION_TYPES:
            supported = ", ".join(sorted(SUPPORTED_ASSERTION_TYPES))
            raise ValueError(
                f"unsupported assertion type '{self.type}'. Supported types: {supported}"
            )
        self.type = assertion_type

        if assertion_type in STRING_ASSERTIONS:
            if not isinstance(self.value, str):
                raise ValueError(f"assertion '{assertion_type}' requires a string value")
            if assertion_type == "regex":
                try:
                    re.compile(self.value)
                except re.error as exc:
                    raise ValueError(f"assertion 'regex' has an invalid pattern: {exc}")
        elif assertion_type in LIST_ASSERTIONS:
            if (
                not isinstance(self.value, list)
                or not self.value
                or not all(isinstance(item, str) for item in self.value)
            ):
                raise ValueError(
                    f"assertion '{assertion_type}' requires a non-empty list of strings"
                )
        elif assertion_type in LENGTH_ASSERTIONS:
            # bool is a subclass of int; reject it explicitly.
            if not isinstance(self.value, int) or isinstance(self.value, bool):
                raise ValueError(
                    f"assertion '{assertion_type}' requires an integer character count"
                )
            if self.value < 0:
                raise ValueError(f"assertion '{assertion_type}' must not be negative")
        elif assertion_type in VALUELESS_ASSERTIONS and self.value is not None:
            raise ValueError(f"assertion '{assertion_type}' does not take a value")

        return self

    @property
    def label(self) -> str:
        """Human-readable label for reports."""
        if self.name:
            return self.name
        if self.value is None:
            return self.type
        return f"{self.type} {json.dumps(self.value, ensure_ascii=False)}"


class AssertionResult(BaseModel):
    """Outcome of one assertion against one model response.

    Attributes:
        type: The assertion type that was evaluated
        label: Human-readable label (custom name or generated description)
        passed: Whether the response satisfied the assertion
        message: Short explanation, most useful when the assertion failed
        expected: The expected value, serialized for reports
    """

    type: str
    label: str
    passed: bool
    message: str
    expected: Any = None
