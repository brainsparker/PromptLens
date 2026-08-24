"""Tests for the judge spend guard: check gating, caching, and budgets."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from promptlens.judges.cache import JudgeCache
from promptlens.judges.spend import JudgeSpendTracker
from promptlens.models.config import JudgeConfig, RunConfig
from promptlens.models.result import JudgeScore, ModelResponse
from promptlens.models.test_case import TestCase as PLTestCase
from promptlens.runners.runner import Runner


class FakeJudge:
    """Judge stub that counts calls and returns a fixed verdict."""

    def __init__(self, cost_usd: float = 0.01) -> None:
        self.calls = 0
        self.cost_usd = cost_usd

    async def evaluate(self, test_case, model_response) -> JudgeScore:
        self.calls += 1
        return JudgeScore(
            score=4,
            explanation="Looks good",
            judge_model="fake-judge",
            judge_provider="fake",
            timestamp=datetime.utcnow(),
            cost_usd=self.cost_usd,
        )


def make_runner(
    judge: FakeJudge,
    judge_cache=None,
    budget_usd=None,
    judge_config=None,
) -> Runner:
    """Build a Runner with only the pieces _judge_with_guard needs."""
    runner = Runner.__new__(Runner)
    runner.judge = judge
    runner.judge_cache = judge_cache
    runner.spend = JudgeSpendTracker(budget_usd)
    runner.gated_count = 0

    config = RunConfig(
        golden_set="golden.yaml",
        models=[
            {"name": "Test Model", "provider": "http", "model": "test-model"}
        ],
        judge=(judge_config or JudgeConfig()),
    )
    runner.config = config
    return runner


def make_test_case(**overrides) -> PLTestCase:
    data = {
        "id": "tc-1",
        "query": "How do I reset my password?",
        "expected_behavior": "Provide clear instructions",
    }
    data.update(overrides)
    return PLTestCase(**data)


def make_response(content: str = "Step 1: open settings and reset your password") -> ModelResponse:
    return ModelResponse(
        content=content,
        model="test-model",
        provider="test",
        latency_ms=10.0,
        cost_usd=0.001,
    )


# ---------------------------------------------------------------------------
# Deterministic check gating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failing_checks_skip_judge_and_score_one() -> None:
    judge = FakeJudge()
    runner = make_runner(judge)
    case = make_test_case(checks=[{"type": "contains", "value": "refund"}])

    score, check_results, skipped_reason = await runner._judge_with_guard(
        case, make_response()
    )

    assert judge.calls == 0
    assert score is not None
    assert score.score == 1
    assert score.cost_usd == 0.0
    assert score.judge_model == "deterministic-checks"
    assert "deterministic checks" in score.explanation.lower()
    assert skipped_reason is None
    assert runner.gated_count == 1
    assert len(check_results) == 1
    assert check_results[0].passed is False


@pytest.mark.asyncio
async def test_passing_checks_proceed_to_judge() -> None:
    judge = FakeJudge()
    runner = make_runner(judge)
    case = make_test_case(
        checks=[{"type": "contains", "value": "password", "case_sensitive": False}]
    )

    score, check_results, skipped_reason = await runner._judge_with_guard(
        case, make_response()
    )

    assert judge.calls == 1
    assert score is not None
    assert score.score == 4
    assert runner.gated_count == 0
    assert len(check_results) == 1
    assert check_results[0].passed is True
    assert skipped_reason is None


@pytest.mark.asyncio
async def test_no_checks_proceed_to_judge() -> None:
    judge = FakeJudge()
    runner = make_runner(judge)

    score, check_results, _ = await runner._judge_with_guard(
        make_test_case(), make_response()
    )

    assert judge.calls == 1
    assert score is not None
    assert check_results == []


# ---------------------------------------------------------------------------
# Judge cache integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_skips_judge_call(tmp_path) -> None:
    judge = FakeJudge()
    cache = JudgeCache(tmp_path / "cache.json")
    runner = make_runner(judge, judge_cache=cache)
    case = make_test_case()
    response = make_response()

    first, _, _ = await runner._judge_with_guard(case, response)
    second, _, _ = await runner._judge_with_guard(case, response)

    assert judge.calls == 1
    assert first is not None and first.cached is False
    assert second is not None and second.cached is True
    assert second.score == first.score
    assert second.cost_usd == 0.0


@pytest.mark.asyncio
async def test_different_responses_miss_cache(tmp_path) -> None:
    judge = FakeJudge()
    cache = JudgeCache(tmp_path / "cache.json")
    runner = make_runner(judge, judge_cache=cache)
    case = make_test_case()

    await runner._judge_with_guard(case, make_response("answer A"))
    await runner._judge_with_guard(case, make_response("answer B"))

    assert judge.calls == 2


@pytest.mark.asyncio
async def test_cache_disabled_always_calls_judge() -> None:
    judge = FakeJudge()
    runner = make_runner(judge, judge_cache=None)
    case = make_test_case()
    response = make_response()

    await runner._judge_with_guard(case, response)
    await runner._judge_with_guard(case, response)

    assert judge.calls == 2


# ---------------------------------------------------------------------------
# Judge budget enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_exhaustion_skips_judge_calls() -> None:
    judge = FakeJudge(cost_usd=0.01)
    runner = make_runner(judge, budget_usd=0.01)

    first, _, first_reason = await runner._judge_with_guard(
        make_test_case(id="tc-1"), make_response("answer 1")
    )
    second, _, second_reason = await runner._judge_with_guard(
        make_test_case(id="tc-2"), make_response("answer 2")
    )

    assert judge.calls == 1
    assert first is not None and first_reason is None
    assert second is None
    assert second_reason is not None
    assert "budget" in second_reason
    assert runner.spend.skipped_count == 1
    assert runner.spend.spent_usd == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_no_budget_never_skips() -> None:
    judge = FakeJudge(cost_usd=100.0)
    runner = make_runner(judge, budget_usd=None)

    for i in range(3):
        score, _, reason = await runner._judge_with_guard(
            make_test_case(id=f"tc-{i}"), make_response(f"answer {i}")
        )
        assert score is not None
        assert reason is None

    assert judge.calls == 3


@pytest.mark.asyncio
async def test_gated_cases_do_not_consume_budget() -> None:
    judge = FakeJudge(cost_usd=0.01)
    runner = make_runner(judge, budget_usd=0.05)
    case = make_test_case(checks=[{"type": "contains", "value": "refund"}])

    await runner._judge_with_guard(case, make_response())

    assert runner.spend.spent_usd == 0.0
    assert judge.calls == 0


# ---------------------------------------------------------------------------
# Spend tracker unit behavior
# ---------------------------------------------------------------------------


def test_spend_tracker_allows_until_budget_reached() -> None:
    tracker = JudgeSpendTracker(budget_usd=0.05)

    assert tracker.allows_call() is True
    tracker.record_cost(0.03)
    assert tracker.allows_call() is True
    tracker.record_cost(0.02)
    assert tracker.allows_call() is False
    assert tracker.exhausted is True


def test_spend_tracker_treats_none_cost_as_zero() -> None:
    tracker = JudgeSpendTracker(budget_usd=1.0)

    tracker.record_cost(None)

    assert tracker.spent_usd == 0.0


def test_spend_tracker_unlimited_by_default() -> None:
    tracker = JudgeSpendTracker()

    tracker.record_cost(10_000.0)

    assert tracker.allows_call() is True
    assert tracker.exhausted is False


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_judge_config_rejects_zero_budget() -> None:
    with pytest.raises(ValidationError, match="budget_usd must be greater than 0"):
        JudgeConfig(budget_usd=0)


def test_judge_config_rejects_negative_budget() -> None:
    with pytest.raises(ValidationError, match="budget_usd must be greater than 0"):
        JudgeConfig(budget_usd=-5.0)


def test_judge_config_defaults() -> None:
    config = JudgeConfig()

    assert config.cache is True
    assert config.budget_usd is None
