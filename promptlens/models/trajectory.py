"""Trajectory assertion data models for deterministic agent evaluation.

Trajectory assertions verify agent BEHAVIOR (which tools were called, in what
order, with which arguments) rather than output QUALITY. They are evaluated
locally against the tool calls captured on a model response, with zero LLM
calls: no judge cost, no judge variance, fully reproducible pass/fail.

They complement (not replace) LLM-as-judge scoring:
    - Trajectory assertions catch logic bugs: the model called delete_booking
      instead of update_booking, skipped a required availability check, or
      looped on the same tool.
    - Judge scores catch quality issues: unhelpful answers, hallucinations.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Argument matching modes for ToolCallMatcher.
ARGS_MATCH_MODES = ("partial", "exact", "ignore")


class ToolCallMatcher(BaseModel):
    """Matcher for a single expected tool call in a trajectory assertion.

    Attributes:
        name: Name of the tool that must be called
        args: Expected arguments (interpretation depends on args_match)
        args_match: How to compare arguments:
            - "partial" (default): every key in args must be present in the
              actual call arguments with an equal value; extra actual
              arguments are allowed
            - "exact": actual arguments must equal args exactly
            - "ignore": match on tool name only, arguments are not checked
        min_times: Minimum number of matching calls required (0 allows
            expressing "at most max_times" constraints)
        max_times: Optional maximum number of matching calls allowed
    """

    name: str = Field(..., min_length=1, description="Name of the tool that must be called")
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Expected arguments; comparison controlled by args_match",
    )
    args_match: str = Field(
        default="partial",
        description="Argument comparison mode: partial, exact, or ignore",
    )
    min_times: int = Field(
        default=1,
        ge=0,
        description="Minimum number of matching calls required",
    )
    max_times: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum number of matching calls allowed",
    )

    @field_validator("args_match")
    @classmethod
    def _validate_args_match(cls, value: str) -> str:
        if value not in ARGS_MATCH_MODES:
            raise ValueError(
                f"args_match must be one of {ARGS_MATCH_MODES}, got '{value}'"
            )
        return value

    @model_validator(mode="after")
    def _validate_times(self) -> "ToolCallMatcher":
        if self.max_times is not None and self.max_times < self.min_times:
            raise ValueError(
                f"max_times ({self.max_times}) must be >= min_times ({self.min_times})"
            )
        if self.min_times == 0 and self.max_times is None:
            raise ValueError(
                "min_times=0 requires max_times, otherwise the matcher asserts nothing"
            )
        return self


class TrajectoryAssertions(BaseModel):
    """Deterministic assertions over the sequence of tool calls in a response.

    All configured checks are evaluated; a trajectory passes only if every
    check passes. At least one check must be configured.

    Attributes:
        must_call: Tools that must be called (with optional argument matching
            and call-count bounds)
        must_not_call: Tool names that must never be called
        call_order: Tool names that must appear in this relative order
            (as a subsequence; other calls may occur in between)
        max_calls: Maximum total number of tool calls allowed
        allow_other_calls: When False, every observed tool call must be named
            in must_call or call_order (whitelist mode, useful for safety
            testing)
    """

    must_call: List[ToolCallMatcher] = Field(
        default_factory=list,
        description="Tools that must be called",
    )
    must_not_call: List[str] = Field(
        default_factory=list,
        description="Tool names that must never be called",
    )
    call_order: List[str] = Field(
        default_factory=list,
        description="Tool names that must appear in this relative order",
    )
    max_calls: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum total number of tool calls allowed",
    )
    allow_other_calls: bool = Field(
        default=True,
        description="Whether tools outside must_call/call_order may be called",
    )

    @field_validator("must_not_call", "call_order")
    @classmethod
    def _validate_names(cls, value: List[str]) -> List[str]:
        for name in value:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("tool names must be non-empty strings")
        return value

    @model_validator(mode="after")
    def _validate_has_assertions(self) -> "TrajectoryAssertions":
        has_any = (
            self.must_call
            or self.must_not_call
            or self.call_order
            or self.max_calls is not None
            or not self.allow_other_calls
        )
        if not has_any:
            raise ValueError(
                "trajectory must configure at least one assertion "
                "(must_call, must_not_call, call_order, max_calls, "
                "or allow_other_calls: false)"
            )
        if not self.allow_other_calls and not (self.must_call or self.call_order):
            raise ValueError(
                "allow_other_calls: false requires must_call or call_order "
                "to define the set of allowed tools"
            )
        return self


class TrajectoryCheck(BaseModel):
    """Result of a single trajectory assertion check.

    Attributes:
        kind: Check type (must_call, must_not_call, call_order, max_calls,
            allowed_tools)
        passed: Whether the check passed
        detail: Human-readable description of what was checked and the outcome
    """

    kind: str
    passed: bool
    detail: str


class TrajectoryResult(BaseModel):
    """Aggregate result of evaluating trajectory assertions for one response.

    Attributes:
        passed: True only if every check passed
        checks: Individual check results, in evaluation order
        observed_calls: Names of the tool calls the model actually made,
            in order
    """

    passed: bool
    checks: List[TrajectoryCheck] = Field(default_factory=list)
    observed_calls: List[str] = Field(default_factory=list)

    @property
    def failed_checks(self) -> List[TrajectoryCheck]:
        """Checks that failed, in evaluation order."""
        return [check for check in self.checks if not check.passed]
