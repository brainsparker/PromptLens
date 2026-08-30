"""Trajectory assertion data models for deterministic agent evaluation.

Trajectory assertions check the sequence of tool calls a model makes against
declarative constraints, with zero LLM cost. They complement (and can run
without) LLM-as-judge scoring: the checks are pure functions of the recorded
tool calls, so results are stable across runs and free to compute.

A trajectory spec supports five kinds of checks:
    - require: tools (optionally with argument subsets) that must be called
    - forbid: tools that must never be called
    - order: sequences of tool names that must appear in order (as a
      subsequence of the actual call sequence, other calls may interleave)
    - min_calls / max_calls: bounds on the total number of tool calls
    - no_repeat_calls: the same tool must not be called twice with
      identical arguments (catches agent loops)
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class TrajectoryToolRef(BaseModel):
    """Reference to a tool call in a trajectory constraint.

    Used in `require` entries. Matches a tool call by name and, when
    `args` is provided, by argument subset: every key in `args` must be
    present in the actual call's arguments with a deeply equal value.
    Extra arguments in the actual call are ignored.

    Attributes:
        name: Tool name to match
        args: Optional argument subset that must match exactly
    """

    name: str = Field(..., min_length=1, description="Tool name to match")
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Argument subset that must be present with equal values",
    )


class TrajectorySpec(BaseModel):
    """Deterministic constraints on the tool-call sequence of a response.

    All checks are evaluated without any LLM. An empty spec is rejected so
    that a typo'd or accidentally empty `trajectory:` block in a golden set
    fails loudly at load time instead of silently passing every response.

    Attributes:
        require: Tool calls that must appear at least once. Entries are
            either a tool name string or a mapping with `name` and `args`.
        forbid: Tool names that must never be called.
        order: Ordered constraints. Each entry is a list of tool names that
            must appear in that relative order (subsequence match).
        min_calls: Minimum number of tool calls (inclusive).
        max_calls: Maximum number of tool calls (inclusive). A step budget.
        no_repeat_calls: If True, fail when the same tool is called more
            than once with identical arguments.
    """

    require: List[Union[str, TrajectoryToolRef]] = Field(
        default_factory=list,
        description="Tool calls that must appear at least once",
    )
    forbid: List[str] = Field(
        default_factory=list,
        description="Tool names that must never be called",
    )
    order: List[List[str]] = Field(
        default_factory=list,
        description="Sequences of tool names that must appear in order",
    )
    min_calls: Optional[int] = Field(
        None, ge=0, description="Minimum number of tool calls (inclusive)"
    )
    max_calls: Optional[int] = Field(
        None, ge=0, description="Maximum number of tool calls (inclusive)"
    )
    no_repeat_calls: bool = Field(
        False,
        description="Fail when the same tool is called twice with identical arguments",
    )

    @field_validator("forbid")
    @classmethod
    def _forbid_entries_non_empty(cls, value: List[str]) -> List[str]:
        for entry in value:
            if not entry or not entry.strip():
                raise ValueError("forbid entries must be non-empty tool names")
        return value

    @field_validator("order")
    @classmethod
    def _order_sequences_valid(cls, value: List[List[str]]) -> List[List[str]]:
        for sequence in value:
            if len(sequence) < 2:
                raise ValueError(
                    "each order constraint needs at least two tool names; "
                    "use require for single-tool presence checks"
                )
            for name in sequence:
                if not name or not name.strip():
                    raise ValueError("order entries must be non-empty tool names")
        return value

    @model_validator(mode="after")
    def _validate_spec(self) -> "TrajectorySpec":
        if (
            not self.require
            and not self.forbid
            and not self.order
            and self.min_calls is None
            and self.max_calls is None
            and not self.no_repeat_calls
        ):
            raise ValueError(
                "trajectory spec must define at least one check "
                "(require, forbid, order, min_calls, max_calls, or no_repeat_calls)"
            )

        if (
            self.min_calls is not None
            and self.max_calls is not None
            and self.min_calls > self.max_calls
        ):
            raise ValueError(
                f"min_calls ({self.min_calls}) cannot exceed max_calls ({self.max_calls})"
            )

        return self

    def normalized_require(self) -> List[TrajectoryToolRef]:
        """Return require entries as TrajectoryToolRef objects.

        String entries become name-only refs (any arguments match).

        Returns:
            List of TrajectoryToolRef entries
        """
        normalized = []
        for entry in self.require:
            if isinstance(entry, str):
                normalized.append(TrajectoryToolRef(name=entry))
            else:
                normalized.append(entry)
        return normalized

    @field_validator("require")
    @classmethod
    def _require_entries_non_empty(
        cls, value: List[Union[str, TrajectoryToolRef]]
    ) -> List[Union[str, TrajectoryToolRef]]:
        for entry in value:
            if isinstance(entry, str) and (not entry or not entry.strip()):
                raise ValueError("require entries must be non-empty tool names")
        return value


class TrajectoryCheckResult(BaseModel):
    """Outcome of a single trajectory check.

    Attributes:
        check: Kind of check (require, forbid, order, min_calls, max_calls,
            no_repeat_calls)
        constraint: Human-readable rendering of the constraint checked
        passed: Whether the check passed
        detail: Human-readable explanation of the outcome
    """

    check: str = Field(..., description="Kind of check")
    constraint: str = Field(..., description="The constraint that was checked")
    passed: bool = Field(..., description="Whether the check passed")
    detail: str = Field(default="", description="Explanation of the outcome")


class TrajectoryEvaluation(BaseModel):
    """Complete deterministic evaluation of a response's tool-call trajectory.

    Attributes:
        passed: True only when every individual check passed
        checks: Outcome of each individual check
        calls_observed: Tool names in the order they were called
        call_count: Total number of tool calls observed
    """

    passed: bool = Field(..., description="Whether all checks passed")
    checks: List[TrajectoryCheckResult] = Field(
        default_factory=list, description="Individual check outcomes"
    )
    calls_observed: List[str] = Field(
        default_factory=list, description="Tool names in call order"
    )
    call_count: int = Field(0, ge=0, description="Total number of tool calls")

    @property
    def failed_checks(self) -> List[TrajectoryCheckResult]:
        """Return only the checks that failed."""
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        """One-line human-readable summary of the evaluation."""
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        status = "passed" if self.passed else "FAILED"
        return f"trajectory {status}: {passed}/{total} checks passed over {self.call_count} tool call(s)"
