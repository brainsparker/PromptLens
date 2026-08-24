"""Judge spend tracking and budget enforcement.

Tracks the cumulative cost of LLM judge calls during a run and stops
authorizing new judge calls once a configured budget is exhausted.
Model-under-test calls are never blocked; only judge spend is governed.
"""

from typing import Optional


class JudgeSpendTracker:
    """Tracks judge spend for one run and enforces an optional budget.

    Note on concurrency: evaluations run concurrently, so a burst of
    in-flight judge calls can overshoot the budget by up to
    (parallel_requests - 1) calls. The budget is a circuit breaker
    against runaway spend, not an exact accounting limit.
    """

    def __init__(self, budget_usd: Optional[float] = None) -> None:
        """Initialize the tracker.

        Args:
            budget_usd: Maximum judge spend for this run in USD, or None
                for unlimited
        """
        self.budget_usd = budget_usd
        self.spent_usd = 0.0
        self.skipped_count = 0

    def allows_call(self) -> bool:
        """Whether a new judge call is currently within budget."""
        if self.budget_usd is None:
            return True
        return self.spent_usd < self.budget_usd

    def record_cost(self, cost_usd: Optional[float]) -> None:
        """Record the actual cost of a completed judge call.

        Args:
            cost_usd: Cost in USD (None is treated as zero)
        """
        self.spent_usd += cost_usd or 0.0

    def record_skip(self) -> None:
        """Record a judge call that was skipped because the budget ran out."""
        self.skipped_count += 1

    @property
    def exhausted(self) -> bool:
        """Whether the budget has been reached."""
        return self.budget_usd is not None and self.spent_usd >= self.budget_usd
