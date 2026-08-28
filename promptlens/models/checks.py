"""Deterministic check data models.

Checks are zero-cost, zero-latency assertions evaluated in plain Python
against a model response. They complement the LLM judge: checks catch
hard rule violations (missing required strings, invalid JSON, latency or
cost budgets) without spending judge tokens, while the judge handles
nuanced quality grading.
"""

import re
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

try:  # Python 3.9 compatibility
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal


CheckType = Literal[
    "contains",
    "not_contains",
    "regex",
    "not_regex",
    "exact_match",
    "is_valid_json",
    "max_latency_ms",
    "max_cost_usd",
    "min_length",
    "max_length",
]

# Checks that compare against a string value
_STRING_CHECKS = {"contains", "not_contains", "regex", "not_regex", "exact_match"}

# Checks that compare against a non-negative numeric value
_NUMERIC_CHECKS = {"max_latency_ms", "max_cost_usd", "min_length", "max_length"}

# Checks that take no value at all
_VALUELESS_CHECKS = {"is_valid_json"}


class CheckDefinition(BaseModel):
    """A single deterministic check attached to a test case.

    Attributes:
        type: The kind of check to run (e.g., "contains", "regex",
            "max_latency_ms")
        value: The comparison value. Required for all check types except
            "is_valid_json". String checks take a string; budget and
            length checks take a non-negative number.
        case_insensitive: For string checks, compare case-insensitively
            (default: False). Ignored by other check types.
    """

    type: CheckType
    value: Optional[Any] = None
    case_insensitive: bool = Field(
        default=False,
        description="Case-insensitive comparison for string checks",
    )

    @model_validator(mode="after")
    def _validate_value_for_type(self) -> "CheckDefinition":
        """Validate that the value matches what the check type expects."""
        if self.type in _VALUELESS_CHECKS:
            if self.value is not None:
                raise ValueError(
                    f"Check '{self.type}' does not take a value"
                )
            return self

        if self.value is None:
            raise ValueError(f"Check '{self.type}' requires a value")

        if self.type in _STRING_CHECKS:
            if not isinstance(self.value, str):
                raise ValueError(
                    f"Check '{self.type}' requires a string value, "
                    f"got {type(self.value).__name__}"
                )
            if self.type in ("regex", "not_regex"):
                try:
                    re.compile(self.value)
                except re.error as e:
                    raise ValueError(
                        f"Check '{self.type}' has an invalid regular "
                        f"expression: {e}"
                    )

        if self.type in _NUMERIC_CHECKS:
            if isinstance(self.value, bool) or not isinstance(
                self.value, (int, float)
            ):
                raise ValueError(
                    f"Check '{self.type}' requires a numeric value, "
                    f"got {type(self.value).__name__}"
                )
            if self.value < 0:
                raise ValueError(
                    f"Check '{self.type}' requires a non-negative value"
                )

        return self


class CheckResult(BaseModel):
    """Outcome of one deterministic check against one model response.

    Attributes:
        check_type: The type of check that was run
        passed: Whether the check passed
        reason: Human-readable explanation of the outcome
        value: The configured comparison value (for reporting)
    """

    check_type: str
    passed: bool
    reason: str
    value: Optional[Any] = None
