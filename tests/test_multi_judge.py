"""Tests for multi-judge consensus scoring."""

import asyncio
from typing import Dict, List

import pytest
from pydantic import ValidationError

from promptlens.judges.factory import create_judge
from promptlens.judges.llm_judge import LLMJudge
from promptlens.judges.multi_judge import MultiJudge
from promptlens.models.config import JudgeConfig, JudgeInstanceConfig
from promptlens.models.result import JudgeScore, ModelResponse


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeProvider:
    """Provider stub that returns a canned judge response per model."""

    responses: Dict[str, str] = {}

    def __init__(self, model_config) -> None:
        self.model = model_config.model

    async def generate(self, prompt: str) -> _FakeResponse:
        return _FakeResponse(self.responses[self.model])


@pytest.fixture
def fake_providers(monkeypatch):
    """Patch the provider factory used by LLMJudge with canned responses."""
    _FakeProvider.responses = {}

    def _get_provider(model_config):
        return _FakeProvider(model_config)

    monkeypatch.setattr("promptlens.judges.llm_judge.get_provider", _get_provider)
    return _FakeProvider.responses


def _panel_config(models: List[str], threshold: int = 1) -> JudgeConfig:
    return JudgeConfig(
        judges=[
            JudgeInstanceConfig(provider="anthropic", model=m) for m in models
        ],
        agreement_threshold=threshold,
    )


def _test_case():
    from promptlens.models.test_case import TestCase

    return TestCase(
        id="tc-001",
        query="How do I reset my password?",
        expected_behavior="Provide clear reset steps",
    )


def _model_response() -> ModelResponse:
    return ModelResponse(
        content="Click 'Forgot password' and follow the emailed link.",
        model="test-model",
        provider="anthropic",
        latency_ms=100.0,
    )


def _judge_reply(score: int, explanation: str = "Reasonable answer.") -> str:
    return f"SCORE: {score}\nEXPLANATION: {explanation}"


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestJudgeConfigValidation:
    def test_single_judge_panel_rejected(self):
        with pytest.raises(ValidationError, match="at least two judges"):
            JudgeConfig(
                judges=[JudgeInstanceConfig(provider="anthropic", model="only-one")]
            )

    def test_panel_capped_at_five(self):
        with pytest.raises(ValidationError, match="at most 5"):
            JudgeConfig(
                judges=[
                    JudgeInstanceConfig(provider="anthropic", model=f"judge-{i}")
                    for i in range(6)
                ]
            )

    def test_agreement_threshold_range(self):
        with pytest.raises(ValidationError, match="agreement_threshold"):
            JudgeConfig(agreement_threshold=5)
        with pytest.raises(ValidationError, match="agreement_threshold"):
            JudgeConfig(agreement_threshold=-1)

    def test_empty_judge_model_rejected(self):
        with pytest.raises(ValidationError):
            JudgeInstanceConfig(provider="anthropic", model="  ")

    def test_default_config_has_no_panel(self):
        config = JudgeConfig()
        assert config.judges == []
        assert config.agreement_threshold == 1


class TestJudgeFactory:
    def test_single_judge_by_default(self, fake_providers):
        judge = create_judge(JudgeConfig())
        assert isinstance(judge, LLMJudge)

    def test_multi_judge_when_panel_configured(self, fake_providers):
        judge = create_judge(_panel_config(["judge-a", "judge-b"]))
        assert isinstance(judge, MultiJudge)

    def test_multi_judge_requires_panel(self, fake_providers):
        with pytest.raises(ValueError, match="at least two judges"):
            MultiJudge(JudgeConfig())


class TestConsensusScoring:
    def test_median_consensus_and_agreement_gap(self, fake_providers):
        fake_providers["judge-a"] = _judge_reply(4, "Accurate.")
        fake_providers["judge-b"] = _judge_reply(4, "Clear.")
        fake_providers["judge-c"] = _judge_reply(2, "Missing edge cases.")

        judge = MultiJudge(_panel_config(["judge-a", "judge-b", "judge-c"]))
        score = _run(judge.evaluate(_test_case(), _model_response()))

        assert score.score == 4  # median of [4, 4, 2]
        assert score.agreement_gap == 2
        assert score.low_confidence is True
        assert len(score.individual_scores) == 3
        assert {s.judge_model for s in score.individual_scores} == {
            "judge-a",
            "judge-b",
            "judge-c",
        }
        assert "low confidence" in score.explanation

    def test_agreeing_panel_not_flagged(self, fake_providers):
        for model in ["judge-a", "judge-b", "judge-c"]:
            fake_providers[model] = _judge_reply(4)

        judge = MultiJudge(_panel_config(["judge-a", "judge-b", "judge-c"]))
        score = _run(judge.evaluate(_test_case(), _model_response()))

        assert score.score == 4
        assert score.agreement_gap == 0
        assert score.low_confidence is False
        assert "low confidence" not in score.explanation

    def test_even_panel_median_rounds_half_up(self, fake_providers):
        fake_providers["judge-a"] = _judge_reply(3)
        fake_providers["judge-b"] = _judge_reply(4)

        judge = MultiJudge(_panel_config(["judge-a", "judge-b"]))
        score = _run(judge.evaluate(_test_case(), _model_response()))

        assert score.score == 4  # median 3.5 rounds half up
        assert score.agreement_gap == 1
        assert score.low_confidence is False

    def test_gap_within_custom_threshold_not_flagged(self, fake_providers):
        fake_providers["judge-a"] = _judge_reply(5)
        fake_providers["judge-b"] = _judge_reply(3)

        judge = MultiJudge(_panel_config(["judge-a", "judge-b"], threshold=2))
        score = _run(judge.evaluate(_test_case(), _model_response()))

        assert score.agreement_gap == 2
        assert score.low_confidence is False

    def test_zero_threshold_flags_any_disagreement(self, fake_providers):
        fake_providers["judge-a"] = _judge_reply(4)
        fake_providers["judge-b"] = _judge_reply(5)

        judge = MultiJudge(_panel_config(["judge-a", "judge-b"], threshold=0))
        score = _run(judge.evaluate(_test_case(), _model_response()))

        assert score.agreement_gap == 1
        assert score.low_confidence is True

    def test_panel_identity_fields(self, fake_providers):
        fake_providers["judge-a"] = _judge_reply(4)
        fake_providers["judge-b"] = _judge_reply(4)

        judge = MultiJudge(_panel_config(["judge-a", "judge-b"]))
        score = _run(judge.evaluate(_test_case(), _model_response()))

        assert score.judge_model == "judge-a+judge-b"
        assert score.judge_provider == "consensus:anthropic+anthropic"

    def test_failed_judge_falls_back_to_default_score(self, fake_providers):
        """A judge whose provider errors contributes LLMJudge's fallback score."""
        fake_providers["judge-a"] = _judge_reply(5)
        fake_providers["judge-b"] = _judge_reply(5)
        # judge-c has no canned response, so its provider raises KeyError
        # and LLMJudge returns its documented fallback score of 3.

        judge = MultiJudge(_panel_config(["judge-a", "judge-b", "judge-c"]))
        score = _run(judge.evaluate(_test_case(), _model_response()))

        assert score.score == 5  # median of [5, 5, 3]
        assert score.agreement_gap == 2
        assert score.low_confidence is True
        assert len(score.individual_scores) == 3


class TestMedianHelper:
    def test_median_values(self):
        assert MultiJudge._median_score([1, 5]) == 3
        assert MultiJudge._median_score([2, 3]) == 3  # 2.5 rounds half up
        assert MultiJudge._median_score([4, 4, 2]) == 4
        assert MultiJudge._median_score([1, 1, 1]) == 1
        assert MultiJudge._median_score([5, 5, 5, 5, 5]) == 5


class TestBackwardCompatibility:
    def test_judge_score_defaults(self):
        score = JudgeScore(
            score=4,
            explanation="Fine.",
            judge_model="judge-model",
            judge_provider="anthropic",
        )
        assert score.individual_scores == []
        assert score.agreement_gap is None
        assert score.low_confidence is False

    def test_single_judge_config_yaml_shape_still_valid(self):
        config = JudgeConfig(provider="anthropic", model="some-judge-model")
        assert config.judges == []
