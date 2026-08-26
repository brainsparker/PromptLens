"""Tests for multi-sample judging, score aggregation, and the noise-aware gate."""

from datetime import datetime

import pytest

from promptlens.cli import _check_fail_under, _gate_failures
from promptlens.judges.base import BaseJudge
from promptlens.judges.stability import (
    UNSTABLE_STDEV_THRESHOLD,
    aggregate_judge_scores,
    sample_judge_scores,
)
from promptlens.models.config import JudgeConfig
from promptlens.models.result import (
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)
from promptlens.models.test_case import TestCase


def _make_score(score, explanation="Looks correct.", **kwargs):
    return JudgeScore(
        score=score,
        explanation=explanation,
        judge_model="judge-model",
        judge_provider="anthropic",
        **kwargs,
    )


def _make_response(model="model-a"):
    return ModelResponse(
        content="The answer is 42.",
        model=model,
        provider="anthropic",
        latency_ms=1234.5,
    )


def _make_test_case():
    return TestCase(
        id="tc-1",
        query="What is the answer?",
        expected_behavior="Answers correctly",
    )


def _make_eval(test_case_id, model="model-a", score=None, stdev=None):
    judge_score = None
    if score is not None:
        judge_score = _make_score(score, score_stdev=stdev)
    return EvaluationResult(
        test_case_id=test_case_id,
        query="What is the answer?",
        expected_behavior="Answers correctly",
        model_response=_make_response(model=model),
        judge_score=judge_score,
    )


def _make_run(results, models=None):
    models = models or ["model-a"]
    return RunResult(
        run_id="run-123",
        timestamp=datetime(2026, 8, 26, 12, 0, 0),
        golden_set_name="golden-set",
        models_tested=models,
        results=results,
    )


class StubJudge(BaseJudge):
    """Judge that returns a scripted sequence of scores or exceptions."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def evaluate(self, test_case, model_response):
        outcome = self.outcomes[self.calls % len(self.outcomes)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return _make_score(outcome, explanation=f"sample score {outcome}")

    @property
    def judge_model(self):
        return "stub-judge"

    @property
    def judge_provider(self):
        return "stub"


class TestAggregateJudgeScores:
    def test_empty_samples_raises(self):
        with pytest.raises(ValueError):
            aggregate_judge_scores([])

    def test_single_sample_passthrough_with_stability_fields(self):
        aggregated = aggregate_judge_scores([_make_score(4)])
        assert aggregated.score == 4
        assert aggregated.sample_scores == [4]
        assert aggregated.score_mean == 4.0
        assert aggregated.score_stdev == 0.0
        assert aggregated.explanation == "Looks correct."

    def test_median_is_robust_to_outlier(self):
        aggregated = aggregate_judge_scores(
            [_make_score(4), _make_score(4), _make_score(1)]
        )
        assert aggregated.score == 4
        assert aggregated.score_mean == pytest.approx(3.0)
        assert sorted(aggregated.sample_scores) == [1, 4, 4]
        assert aggregated.score_stdev == pytest.approx(1.7320508, rel=1e-4)

    def test_even_sample_count_rounds_median(self):
        aggregated = aggregate_judge_scores([_make_score(3), _make_score(4)])
        # Median of [3, 4] is 3.5, which rounds to 4 (banker's rounding
        # applies to .5 values: round(3.5) == 4).
        assert aggregated.score == 4

    def test_identical_samples_have_zero_stdev(self):
        aggregated = aggregate_judge_scores([_make_score(5), _make_score(5)])
        assert aggregated.score == 5
        assert aggregated.score_stdev == 0.0
        # No stability note appended when samples fully agree
        assert "[Judge stability" not in aggregated.explanation

    def test_disagreeing_samples_append_stability_note(self):
        aggregated = aggregate_judge_scores(
            [_make_score(2), _make_score(4), _make_score(4)]
        )
        assert "[Judge stability: 3 samples" in aggregated.explanation

    def test_representative_explanation_comes_from_median_sample(self):
        aggregated = aggregate_judge_scores(
            [
                _make_score(1, explanation="harsh outlier"),
                _make_score(4, explanation="representative"),
                _make_score(5, explanation="generous"),
            ]
        )
        assert aggregated.explanation.startswith("representative")

    def test_aggregate_score_is_clamped_to_valid_range(self):
        aggregated = aggregate_judge_scores([_make_score(1), _make_score(1)])
        assert aggregated.score == 1
        aggregated = aggregate_judge_scores([_make_score(5), _make_score(5)])
        assert aggregated.score == 5


class TestSampleJudgeScores:
    @pytest.mark.asyncio
    async def test_invalid_sample_count_raises(self):
        judge = StubJudge([4])
        with pytest.raises(ValueError):
            await sample_judge_scores(
                judge, _make_test_case(), _make_response(), samples=0
            )

    @pytest.mark.asyncio
    async def test_single_sample_calls_judge_once(self):
        judge = StubJudge([4])
        result = await sample_judge_scores(
            judge, _make_test_case(), _make_response(), samples=1
        )
        assert judge.calls == 1
        assert result.score == 4
        assert result.sample_scores == [4]

    @pytest.mark.asyncio
    async def test_multi_sample_aggregates_all_samples(self):
        judge = StubJudge([3, 4, 5])
        result = await sample_judge_scores(
            judge, _make_test_case(), _make_response(), samples=3
        )
        assert judge.calls == 3
        assert result.score == 4
        assert sorted(result.sample_scores) == [3, 4, 5]
        assert result.score_mean == pytest.approx(4.0)

    @pytest.mark.asyncio
    async def test_partial_sample_failure_still_aggregates(self):
        judge = StubJudge([4, RuntimeError("judge API down"), 4])
        result = await sample_judge_scores(
            judge, _make_test_case(), _make_response(), samples=3
        )
        assert result.score == 4
        assert sorted(result.sample_scores) == [4, 4]

    @pytest.mark.asyncio
    async def test_all_samples_failing_raises_first_error(self):
        judge = StubJudge([RuntimeError("judge API down")])
        with pytest.raises(RuntimeError, match="judge API down"):
            await sample_judge_scores(
                judge, _make_test_case(), _make_response(), samples=3
            )


class TestJudgeConfigSamples:
    def test_default_is_one(self):
        assert JudgeConfig().samples == 1

    def test_valid_samples_accepted(self):
        assert JudgeConfig(samples=3).samples == 3
        assert JudgeConfig(samples=10).samples == 10

    @pytest.mark.parametrize("value", [0, -1, 11])
    def test_out_of_range_samples_rejected(self, value):
        with pytest.raises(ValueError):
            JudgeConfig(samples=value)


class TestAverageJudgeStdev:
    def test_no_stability_data_returns_none(self):
        run = _make_run([_make_eval("tc-1", score=4)])
        assert run.get_average_judge_stdev("model-a") is None

    def test_average_of_per_case_stdevs(self):
        run = _make_run(
            [
                _make_eval("tc-1", score=4, stdev=0.5),
                _make_eval("tc-2", score=3, stdev=1.5),
            ]
        )
        assert run.get_average_judge_stdev("model-a") == pytest.approx(1.0)

    def test_filters_by_model(self):
        run = _make_run(
            [
                _make_eval("tc-1", model="model-a", score=4, stdev=0.5),
                _make_eval("tc-1", model="model-b", score=4, stdev=1.5),
            ],
            models=["model-a", "model-b"],
        )
        assert run.get_average_judge_stdev("model-a") == pytest.approx(0.5)
        assert run.get_average_judge_stdev("model-b") == pytest.approx(1.5)


class TestNoiseAwareGate:
    def test_noise_margin_rescues_model_within_noise(self):
        # Average 2.5 fails a 3.0 gate, but with an average judge stdev of
        # 0.7 the effective threshold drops to 2.3 and the model passes.
        run = _make_run(
            [
                _make_eval("tc-1", score=2, stdev=0.7),
                _make_eval("tc-2", score=3, stdev=0.7),
            ]
        )
        assert _gate_failures(run, 3.0, noise_aware=False) != []
        assert _gate_failures(run, 3.0, noise_aware=True) == []

    def test_drop_beyond_noise_still_fails(self):
        run = _make_run(
            [
                _make_eval("tc-1", score=1, stdev=0.3),
                _make_eval("tc-2", score=2, stdev=0.3),
            ]
        )
        failing = _gate_failures(run, 3.0, noise_aware=True)
        assert len(failing) == 1
        model, avg, margin = failing[0]
        assert model == "model-a"
        assert avg == pytest.approx(1.5)
        assert margin == pytest.approx(0.3)

    def test_no_stability_data_falls_back_to_absolute_gate(self):
        run = _make_run([_make_eval("tc-1", score=2)])
        failing = _gate_failures(run, 3.0, noise_aware=True)
        assert len(failing) == 1
        assert failing[0][2] == 0.0

    def test_model_with_no_scores_always_fails(self):
        run = _make_run([_make_eval("tc-1")])
        failing = _gate_failures(run, 3.0, noise_aware=True)
        assert failing == [("model-a", None, 0.0)]

    def test_check_fail_under_keeps_two_tuple_shape(self):
        run = _make_run([_make_eval("tc-1", score=2)])
        assert _check_fail_under(run, 3.0) == [("model-a", pytest.approx(2.0))]


class TestUnstableThreshold:
    def test_threshold_marks_full_point_disagreement(self):
        # Sanity-check the constant used by the run summary: three samples
        # spanning two full points should exceed the instability threshold.
        aggregated = aggregate_judge_scores(
            [_make_score(2), _make_score(3), _make_score(4)]
        )
        assert aggregated.score_stdev >= UNSTABLE_STDEV_THRESHOLD
