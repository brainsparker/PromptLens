"""Tests for the deterministic judge, judge factory, and judge config."""

import asyncio

import pytest
from pydantic import ValidationError

from promptlens.judges.deterministic_judge import (
    DETERMINISTIC_JUDGE_MODEL,
    DETERMINISTIC_JUDGE_PROVIDER,
    DeterministicJudge,
    score_from_assertions,
)
from promptlens.judges.factory import get_judge
from promptlens.models.config import JudgeConfig, RunConfig
from promptlens.models.result import AssertionResult, ModelResponse
from promptlens.models.test_case import TestCase


def _response(content: str) -> ModelResponse:
    return ModelResponse(content=content, model="m", provider="p", latency_ms=1.0)


def _case(assertions, reference_answer=None) -> TestCase:
    return TestCase(
        id="tc",
        query="What is the refund window?",
        expected_behavior="States the 30 day refund window",
        reference_answer=reference_answer,
        assertions=assertions,
    )


def _result(passed: bool) -> AssertionResult:
    return AssertionResult(type="contains", label="x", passed=passed)


class TestJudgeConfig:
    def test_default_type_is_llm(self):
        assert JudgeConfig().type == "llm"

    def test_type_normalized_and_validated(self):
        assert JudgeConfig(type=" Deterministic ").type == "deterministic"
        with pytest.raises(ValidationError, match="unsupported judge type"):
            JudgeConfig(type="vibes")

    def test_run_config_accepts_deterministic_judge(self):
        config = RunConfig(
            golden_set="tests.yaml",
            models=[{"name": "m", "provider": "http", "model": "local"}],
            judge={"type": "deterministic"},
        )
        assert config.judge.type == "deterministic"


class TestScoreMapping:
    def test_all_pass_scores_five(self):
        assert score_from_assertions([_result(True), _result(True)]) == 5

    def test_none_pass_scores_one(self):
        assert score_from_assertions([_result(False), _result(False)]) == 1

    def test_partial_scales_linearly(self):
        assert score_from_assertions([_result(True), _result(False)]) == 3
        three_of_four = [_result(True), _result(True), _result(True), _result(False)]
        assert score_from_assertions(three_of_four) == 4
        one_of_four = [_result(True), _result(False), _result(False), _result(False)]
        assert score_from_assertions(one_of_four) == 2

    def test_empty_results_rejected(self):
        with pytest.raises(ValueError):
            score_from_assertions([])


class TestDeterministicJudge:
    def test_factory_returns_deterministic_judge(self):
        judge = get_judge(JudgeConfig(type="deterministic"))
        assert isinstance(judge, DeterministicJudge)
        assert judge.judge_model == DETERMINISTIC_JUDGE_MODEL
        assert judge.judge_provider == DETERMINISTIC_JUDGE_PROVIDER

    def test_factory_rejects_unknown_type(self):
        config = JudgeConfig()
        config.type = "mystery"  # bypass validation to exercise the factory guard
        with pytest.raises(ValueError, match="Unsupported judge type"):
            get_judge(config)

    def test_can_evaluate_requires_assertions(self):
        judge = DeterministicJudge(JudgeConfig(type="deterministic"))
        assert judge.can_evaluate(_case([{"type": "is_json"}]))
        assert not judge.can_evaluate(_case([]))

    def test_evaluate_all_pass(self):
        judge = DeterministicJudge(JudgeConfig(type="deterministic"))
        case = _case(
            [
                {"type": "contains", "value": "30 days"},
                {"type": "not_contains", "value": "90 days"},
                {"type": "max_length", "value": 200},
            ]
        )
        score = asyncio.run(judge.evaluate(case, _response("Refunds are accepted within 30 days.")))
        assert score.score == 5
        assert score.assertions_passed
        assert len(score.assertion_results) == 3
        assert score.judge_model == DETERMINISTIC_JUDGE_MODEL
        assert "3/3 passed" in score.explanation

    def test_evaluate_partial_failure_lists_failed_assertions(self):
        judge = DeterministicJudge(JudgeConfig(type="deterministic"))
        case = _case(
            [
                {"type": "contains", "value": "30 days"},
                {"type": "is_json"},
            ]
        )
        score = asyncio.run(judge.evaluate(case, _response("Refunds are accepted within 30 days.")))
        assert score.score == 3
        assert not score.assertions_passed
        assert [a.type for a in score.failed_assertions] == ["is_json"]
        assert "[FAIL] is_json" in score.explanation
        assert "[PASS] contains '30 days'" in score.explanation

    def test_evaluate_uses_reference_answer_for_token_f1(self):
        judge = DeterministicJudge(JudgeConfig(type="deterministic"))
        case = _case(
            [{"type": "token_f1", "threshold": 0.5}], reference_answer="30 day refund window"
        )
        score = asyncio.run(judge.evaluate(case, _response("There is a 30 day refund window.")))
        assert score.score == 5
        assert score.assertion_results[0].score is not None

    def test_evaluate_without_assertions_raises(self):
        judge = DeterministicJudge(JudgeConfig(type="deterministic"))
        with pytest.raises(ValueError, match="no assertions"):
            asyncio.run(judge.evaluate(_case([]), _response("x")))

    def test_same_input_same_output(self):
        judge = DeterministicJudge(JudgeConfig(type="deterministic"))
        case = _case(
            [{"type": "regex", "value": r"\d+ days"}, {"type": "token_f1", "value": "30 days"}]
        )
        first = asyncio.run(judge.evaluate(case, _response("30 days")))
        second = asyncio.run(judge.evaluate(case, _response("30 days")))
        assert first.score == second.score
        assert [a.model_dump(exclude={"detail"}) for a in first.assertion_results] == [
            a.model_dump(exclude={"detail"}) for a in second.assertion_results
        ]
