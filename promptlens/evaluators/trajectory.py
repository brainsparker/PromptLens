"""Deterministic trajectory evaluator for agent tool-call sequences.

Evaluates a model response's recorded tool calls against a TrajectorySpec.
Pure function of its inputs: no LLM, no network, no cost, stable results.
"""

import json
import logging
from typing import Any, Dict, List

from promptlens.models.tools import ToolCall
from promptlens.models.trajectory import (
    TrajectoryCheckResult,
    TrajectoryEvaluation,
    TrajectorySpec,
    TrajectoryToolRef,
)

logger = logging.getLogger(__name__)


def evaluate_trajectory(
    spec: TrajectorySpec,
    tool_calls: List[ToolCall],
) -> TrajectoryEvaluation:
    """Evaluate a sequence of tool calls against a trajectory spec.

    Args:
        spec: The trajectory constraints to check
        tool_calls: Tool calls made by the model, in call order

    Returns:
        TrajectoryEvaluation with per-check outcomes and an overall pass flag
    """
    call_names = [call.name for call in tool_calls]
    checks: List[TrajectoryCheckResult] = []

    for ref in spec.normalized_require():
        checks.append(_check_require(ref, tool_calls))

    for forbidden in spec.forbid:
        checks.append(_check_forbid(forbidden, call_names))

    for sequence in spec.order:
        checks.append(_check_order(sequence, call_names))

    if spec.min_calls is not None:
        checks.append(_check_min_calls(spec.min_calls, len(tool_calls)))

    if spec.max_calls is not None:
        checks.append(_check_max_calls(spec.max_calls, len(tool_calls)))

    if spec.no_repeat_calls:
        checks.append(_check_no_repeats(tool_calls))

    return TrajectoryEvaluation(
        passed=all(c.passed for c in checks),
        checks=checks,
        calls_observed=call_names,
        call_count=len(tool_calls),
    )


def _args_subset_match(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    """Check that every expected key exists in actual with a deeply equal value."""
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        if actual[key] != expected_value:
            return False
    return True


def _render_ref(ref: TrajectoryToolRef) -> str:
    """Render a tool ref for check output."""
    if ref.args:
        try:
            args_repr = json.dumps(ref.args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            args_repr = str(ref.args)
        return f"{ref.name}({args_repr})"
    return ref.name


def _check_require(
    ref: TrajectoryToolRef,
    tool_calls: List[ToolCall],
) -> TrajectoryCheckResult:
    """Check that a required tool call is present."""
    constraint = _render_ref(ref)
    name_matches = [call for call in tool_calls if call.name == ref.name]

    if not name_matches:
        return TrajectoryCheckResult(
            check="require",
            constraint=constraint,
            passed=False,
            detail=f"tool '{ref.name}' was never called",
        )

    if not ref.args:
        return TrajectoryCheckResult(
            check="require",
            constraint=constraint,
            passed=True,
            detail=f"tool '{ref.name}' called {len(name_matches)} time(s)",
        )

    for call in name_matches:
        if _args_subset_match(ref.args, call.arguments):
            return TrajectoryCheckResult(
                check="require",
                constraint=constraint,
                passed=True,
                detail=f"tool '{ref.name}' called with matching arguments",
            )

    return TrajectoryCheckResult(
        check="require",
        constraint=constraint,
        passed=False,
        detail=(
            f"tool '{ref.name}' called {len(name_matches)} time(s) but never "
            f"with the required arguments"
        ),
    )


def _check_forbid(forbidden: str, call_names: List[str]) -> TrajectoryCheckResult:
    """Check that a forbidden tool was never called."""
    count = call_names.count(forbidden)
    if count:
        return TrajectoryCheckResult(
            check="forbid",
            constraint=forbidden,
            passed=False,
            detail=f"forbidden tool '{forbidden}' was called {count} time(s)",
        )
    return TrajectoryCheckResult(
        check="forbid",
        constraint=forbidden,
        passed=True,
        detail=f"forbidden tool '{forbidden}' was not called",
    )


def _check_order(sequence: List[str], call_names: List[str]) -> TrajectoryCheckResult:
    """Check that a sequence of tool names appears in order.

    Subsequence match: other calls may interleave, but the named tools must
    appear in the given relative order. Repeated tool names in the sequence
    require distinct calls in order (e.g. ["a", "a"] needs two calls to a).
    """
    constraint = " -> ".join(sequence)
    position = 0
    for name in call_names:
        if position < len(sequence) and name == sequence[position]:
            position += 1

    if position == len(sequence):
        return TrajectoryCheckResult(
            check="order",
            constraint=constraint,
            passed=True,
            detail="tools called in the required order",
        )

    return TrajectoryCheckResult(
        check="order",
        constraint=constraint,
        passed=False,
        detail=(
            f"order broken at step {position + 1} ('{sequence[position]}'): "
            f"observed sequence was [{', '.join(call_names) or 'no tool calls'}]"
        ),
    )


def _check_min_calls(minimum: int, count: int) -> TrajectoryCheckResult:
    """Check the minimum call count."""
    passed = count >= minimum
    return TrajectoryCheckResult(
        check="min_calls",
        constraint=f">= {minimum}",
        passed=passed,
        detail=f"observed {count} tool call(s), minimum is {minimum}",
    )


def _check_max_calls(maximum: int, count: int) -> TrajectoryCheckResult:
    """Check the maximum call count (step budget)."""
    passed = count <= maximum
    return TrajectoryCheckResult(
        check="max_calls",
        constraint=f"<= {maximum}",
        passed=passed,
        detail=f"observed {count} tool call(s), budget is {maximum}",
    )


def _stable_args_key(arguments: Dict[str, Any]) -> str:
    """Produce a stable string key for a call's arguments."""
    try:
        return json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(sorted(arguments.items(), key=lambda item: item[0]))


def _check_no_repeats(tool_calls: List[ToolCall]) -> TrajectoryCheckResult:
    """Check that no tool was called twice with identical arguments."""
    seen: Dict[str, int] = {}
    repeats: List[str] = []

    for call in tool_calls:
        key = f"{call.name}::{_stable_args_key(call.arguments)}"
        seen[key] = seen.get(key, 0) + 1

    for key, count in seen.items():
        if count > 1:
            tool_name = key.split("::", 1)[0]
            repeats.append(f"'{tool_name}' x{count}")

    if repeats:
        return TrajectoryCheckResult(
            check="no_repeat_calls",
            constraint="no identical repeated calls",
            passed=False,
            detail=f"repeated identical call(s): {', '.join(sorted(repeats))}",
        )

    return TrajectoryCheckResult(
        check="no_repeat_calls",
        constraint="no identical repeated calls",
        passed=True,
        detail="no identical repeated calls observed",
    )
