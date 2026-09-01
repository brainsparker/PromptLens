"""Deterministic trajectory assertion evaluation.

Evaluates TrajectoryAssertions against the tool calls captured on a model
response. Pure Python, no LLM calls, no network: results are free to compute
and fully reproducible, which makes them safe to gate CI on.
"""

from typing import Any, Dict, List

from promptlens.models.tools import ToolCall
from promptlens.models.trajectory import (
    ToolCallMatcher,
    TrajectoryAssertions,
    TrajectoryCheck,
    TrajectoryResult,
)


def _args_match(matcher: ToolCallMatcher, arguments: Dict[str, Any]) -> bool:
    """Check whether a tool call's arguments satisfy a matcher.

    Args:
        matcher: The matcher with expected args and comparison mode
        arguments: Actual arguments from the tool call

    Returns:
        True if the arguments satisfy the matcher's args_match mode
    """
    if matcher.args_match == "ignore":
        return True
    if matcher.args_match == "exact":
        return arguments == matcher.args
    # partial: every expected key present with an equal value
    for key, expected_value in matcher.args.items():
        if key not in arguments or arguments[key] != expected_value:
            return False
    return True


def _matching_calls(matcher: ToolCallMatcher, tool_calls: List[ToolCall]) -> int:
    """Count tool calls that match a matcher (name and arguments).

    Args:
        matcher: The matcher to test against
        tool_calls: Observed tool calls

    Returns:
        Number of matching calls
    """
    return sum(
        1
        for call in tool_calls
        if call.name == matcher.name and _args_match(matcher, call.arguments)
    )


def _describe_matcher(matcher: ToolCallMatcher) -> str:
    """Build a short human-readable description of a matcher."""
    parts = [matcher.name]
    if matcher.args and matcher.args_match != "ignore":
        parts.append(f"args {matcher.args} ({matcher.args_match})")
    if matcher.min_times != 1 or matcher.max_times is not None:
        bounds = f"min {matcher.min_times}"
        if matcher.max_times is not None:
            bounds += f", max {matcher.max_times}"
        parts.append(bounds)
    return " ".join(parts)


def _check_must_call(
    assertions: TrajectoryAssertions, tool_calls: List[ToolCall]
) -> List[TrajectoryCheck]:
    """Evaluate every must_call matcher."""
    checks = []
    for matcher in assertions.must_call:
        count = _matching_calls(matcher, tool_calls)
        within_max = matcher.max_times is None or count <= matcher.max_times
        passed = count >= matcher.min_times and within_max
        if passed:
            detail = f"must_call {_describe_matcher(matcher)}: matched {count} call(s)"
        elif count < matcher.min_times:
            name_only = sum(1 for c in tool_calls if c.name == matcher.name)
            if name_only > count:
                detail = (
                    f"must_call {_describe_matcher(matcher)}: {count} matching call(s) "
                    f"(expected at least {matcher.min_times}); {name_only} call(s) to "
                    f"'{matcher.name}' had non-matching arguments"
                )
            else:
                detail = (
                    f"must_call {_describe_matcher(matcher)}: {count} matching call(s) "
                    f"(expected at least {matcher.min_times})"
                )
        else:
            detail = (
                f"must_call {_describe_matcher(matcher)}: {count} matching call(s) "
                f"(expected at most {matcher.max_times})"
            )
        checks.append(TrajectoryCheck(kind="must_call", passed=passed, detail=detail))
    return checks


def _check_must_not_call(
    assertions: TrajectoryAssertions, observed_names: List[str]
) -> List[TrajectoryCheck]:
    """Evaluate every must_not_call name."""
    checks = []
    for name in assertions.must_not_call:
        count = observed_names.count(name)
        passed = count == 0
        detail = (
            f"must_not_call {name}: not called"
            if passed
            else f"must_not_call {name}: called {count} time(s)"
        )
        checks.append(TrajectoryCheck(kind="must_not_call", passed=passed, detail=detail))
    return checks


def _check_call_order(
    assertions: TrajectoryAssertions, observed_names: List[str]
) -> List[TrajectoryCheck]:
    """Evaluate call_order as a subsequence of the observed call names."""
    if not assertions.call_order:
        return []

    position = 0
    for name in observed_names:
        if position < len(assertions.call_order) and name == assertions.call_order[position]:
            position += 1

    expected = " -> ".join(assertions.call_order)
    if position == len(assertions.call_order):
        detail = f"call_order {expected}: observed in order"
        passed = True
    else:
        missing = assertions.call_order[position]
        observed = " -> ".join(observed_names) if observed_names else "(no tool calls)"
        detail = (
            f"call_order {expected}: '{missing}' not observed in order "
            f"(observed: {observed})"
        )
        passed = False
    return [TrajectoryCheck(kind="call_order", passed=passed, detail=detail)]


def _check_max_calls(
    assertions: TrajectoryAssertions, observed_names: List[str]
) -> List[TrajectoryCheck]:
    """Evaluate the max_calls budget."""
    if assertions.max_calls is None:
        return []
    total = len(observed_names)
    passed = total <= assertions.max_calls
    detail = (
        f"max_calls {assertions.max_calls}: {total} call(s) made"
        if passed
        else f"max_calls {assertions.max_calls}: exceeded with {total} call(s)"
    )
    return [TrajectoryCheck(kind="max_calls", passed=passed, detail=detail)]


def _check_allowed_tools(
    assertions: TrajectoryAssertions, observed_names: List[str]
) -> List[TrajectoryCheck]:
    """Evaluate the whitelist when allow_other_calls is False."""
    if assertions.allow_other_calls:
        return []
    allowed = {matcher.name for matcher in assertions.must_call}
    allowed.update(assertions.call_order)
    unexpected = sorted({name for name in observed_names if name not in allowed})
    passed = not unexpected
    detail = (
        f"allowed_tools {sorted(allowed)}: all calls within the allowed set"
        if passed
        else f"allowed_tools {sorted(allowed)}: unexpected tool(s) called: {unexpected}"
    )
    return [TrajectoryCheck(kind="allowed_tools", passed=passed, detail=detail)]


def evaluate_trajectory(
    assertions: TrajectoryAssertions,
    tool_calls: List[ToolCall],
) -> TrajectoryResult:
    """Evaluate trajectory assertions against observed tool calls.

    Args:
        assertions: The assertions configured on the test case
        tool_calls: Tool calls captured from the model response, in order

    Returns:
        TrajectoryResult with per-check outcomes; passed is True only when
        every check passed
    """
    observed_names = [call.name for call in tool_calls]

    checks: List[TrajectoryCheck] = []
    checks.extend(_check_must_call(assertions, tool_calls))
    checks.extend(_check_must_not_call(assertions, observed_names))
    checks.extend(_check_call_order(assertions, observed_names))
    checks.extend(_check_max_calls(assertions, observed_names))
    checks.extend(_check_allowed_tools(assertions, observed_names))

    return TrajectoryResult(
        passed=all(check.passed for check in checks),
        checks=checks,
        observed_calls=observed_names,
    )
