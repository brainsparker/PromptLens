"""Cost and latency budget checks.

An eval that only scores quality can pass while a prompt change quietly
doubles the bill or pushes p95 latency past what the product tolerates.
Budgets make cost and latency first-class pass/fail criteria next to the
judge score.

Two levels of budget exist:

- Case budgets: `max_cost_usd` and `max_latency_ms`. Defaults come from the
  config's `budgets` section (`max_case_cost_usd`, `max_case_latency_ms`); a
  test case can override them with its own fields.
- Run budgets: `max_total_cost_usd`, the ceiling for the whole run.

Rules:

- Budgets never stop a run early. Every response is generated and judged,
  then checked, so a report always shows the full picture.
- Responses that errored are not budget-checked; they are already errors.
- A cost budget is skipped when the provider reported no cost estimate
  (local models, unknown pricing). Latency is always known, so latency
  budgets always apply.
"""

from typing import List, Optional, Tuple

from promptlens.models.config import BudgetConfig
from promptlens.models.result import BudgetViolation, EvaluationResult, ModelResponse, RunResult
from promptlens.models.test_case import TestCase

KIND_COST = "cost"
KIND_LATENCY = "latency"
KIND_TOTAL_COST = "total_cost"

SCOPE_CASE = "case"
SCOPE_RUN = "run"


def resolve_case_limits(
    test_case: TestCase, budgets: BudgetConfig
) -> Tuple[Optional[float], Optional[float]]:
    """Return the effective (cost_limit, latency_limit) for a test case.

    A limit set on the test case wins over the run-level default. Either
    value is None when no budget applies.
    """
    cost_limit = (
        test_case.max_cost_usd if test_case.max_cost_usd is not None else budgets.max_case_cost_usd
    )
    latency_limit = (
        test_case.max_latency_ms
        if test_case.max_latency_ms is not None
        else budgets.max_case_latency_ms
    )
    return cost_limit, latency_limit


def check_case_budget(
    test_case: TestCase,
    model_response: ModelResponse,
    budgets: BudgetConfig,
) -> List[BudgetViolation]:
    """Check one response against the effective budgets for its test case.

    Args:
        test_case: The test case that produced the response
        model_response: The response to check
        budgets: Run-level budget defaults

    Returns:
        Violations found, in the order cost then latency. Empty when the
        response is within budget, errored, or no budget applies.
    """
    if model_response.error:
        return []

    cost_limit, latency_limit = resolve_case_limits(test_case, budgets)
    violations: List[BudgetViolation] = []

    cost = model_response.cost_usd
    if cost_limit is not None and cost is not None and cost > cost_limit:
        violations.append(
            BudgetViolation(
                kind=KIND_COST,
                scope=SCOPE_CASE,
                limit=cost_limit,
                actual=cost,
                message=f"cost ${cost:.4f} exceeds budget ${cost_limit:.4f}",
            )
        )

    latency = model_response.latency_ms
    if latency_limit is not None and latency > latency_limit:
        violations.append(
            BudgetViolation(
                kind=KIND_LATENCY,
                scope=SCOPE_CASE,
                limit=latency_limit,
                actual=latency,
                message=f"latency {latency:.0f}ms exceeds budget {latency_limit:.0f}ms",
            )
        )

    return violations


def check_run_budget(run_result: RunResult, budgets: BudgetConfig) -> List[BudgetViolation]:
    """Check run-level totals against the run budgets.

    Args:
        run_result: The completed run
        budgets: Run-level budgets

    Returns:
        Violations found. Currently only total cost is checked.
    """
    violations: List[BudgetViolation] = []

    limit = budgets.max_total_cost_usd
    if limit is not None and run_result.total_cost_usd > limit:
        violations.append(
            BudgetViolation(
                kind=KIND_TOTAL_COST,
                scope=SCOPE_RUN,
                limit=limit,
                actual=run_result.total_cost_usd,
                message=(f"run cost ${run_result.total_cost_usd:.4f} exceeds budget ${limit:.4f}"),
            )
        )

    return violations


def describe_violations(run_result: RunResult) -> List[str]:
    """Render every violation in a run as one line each, case lines first.

    Case lines read "<test_case_id> (<model>): <message>"; run lines carry
    the message alone. Used by the CLI summary and the markdown report so
    both describe a failure the same way.
    """
    lines: List[str] = []
    for eval_result in run_result.results:
        for violation in eval_result.budget_violations:
            lines.append(
                f"{eval_result.test_case_id} ({eval_result.model_response.model}): "
                f"{violation.message}"
            )
    for violation in run_result.budget_violations:
        lines.append(violation.message)
    return lines


def count_case_violations(results: List[EvaluationResult]) -> int:
    """Return the number of results that exceeded at least one case budget."""
    return sum(1 for r in results if r.budget_violations)
