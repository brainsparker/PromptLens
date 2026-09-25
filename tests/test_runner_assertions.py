"""Runner behaviour with deterministic assertions: judge gating and result plumbing."""

import asyncio
import textwrap
from typing import ClassVar, List

import pytest

from promptlens.models.config import RunConfig
from promptlens.models.result import JudgeScore, ModelResponse
from promptlens.runners import runner as runner_module
from promptlens.runners.runner import Runner


class FakeProvider:
    """Minimal provider returning a canned response per test-case id."""

    provider_name = "fake"

    def __init__(self, config, responses):
        self.config = config
        self._responses = responses

    def supports_tools(self):
        return False

    async def generate(self, query, tools=None, **kwargs):
        content = self._responses.get(query, "")
        return ModelResponse(
            content=content,
            model=self.config.model,
            provider=self.provider_name,
            latency_ms=42.0,
            cost_usd=0.0,
        )


class FakeJudge:
    """Judge that records every call and always returns a 4."""

    instances: ClassVar[List["FakeJudge"]] = []

    def __init__(self, config):
        self.config = config
        self.calls = []
        FakeJudge.instances.append(self)

    async def evaluate(self, test_case, model_response):
        self.calls.append(test_case.id)
        return JudgeScore(
            score=4,
            explanation="fine",
            judge_model=self.config.model,
            judge_provider=self.config.provider,
        )


GOLDEN = textwrap.dedent(
    """
    name: "Runner assertions"
    test_cases:
      - id: "judged-pass"
        query: "q-pass"
        expected_behavior: "mentions Paris"
        assertions:
          - type: contains
            value: "Paris"
      - id: "judged-fail"
        query: "q-fail"
        expected_behavior: "mentions Paris"
        assertions:
          - type: contains
            value: "Paris"
          - type: max_chars
            value: 5
      - id: "assertions-only"
        query: "q-only"
        expected_behavior: "says OK"
        evaluation_mode: assertions_only
        assertions:
          - type: equals
            value: "OK"
      - id: "plain"
        query: "q-plain"
        expected_behavior: "anything"
    """
)

RESPONSES = {
    "q-pass": "Paris is the capital of France.",
    "q-fail": "Lyon, obviously.",
    "q-only": "OK",
    "q-plain": "Whatever you say.",
}


@pytest.fixture
def make_config(tmp_path, monkeypatch):
    golden = tmp_path / "golden.yaml"
    golden.write_text(GOLDEN)
    FakeJudge.instances = []
    monkeypatch.setattr(runner_module, "get_provider", lambda cfg: FakeProvider(cfg, RESPONSES))
    monkeypatch.setattr(runner_module, "LLMJudge", FakeJudge)

    def _make(**judge_overrides):
        return RunConfig(
            golden_set=str(golden),
            models=[{"name": "Fake", "provider": "fake", "model": "fake-1"}],
            judge={"provider": "fake", "model": "fake-judge", **judge_overrides},
            output={"directory": str(tmp_path / "out"), "formats": ["json"]},
        )

    return _make


def _run(config):
    # Runner owns an asyncio.Semaphore, which on Python 3.9 binds to the
    # current loop at construction, so build and run inside one loop.
    async def go():
        return await Runner(config).run()

    return asyncio.run(go())


def _by_id(result):
    return {r.test_case_id: r for r in result.results}


class TestRunnerAssertions:
    def test_assertion_results_attached_and_judge_still_runs_by_default(self, make_config):
        result = _run(make_config())
        by_id = _by_id(result)

        assert by_id["judged-pass"].assertions_passed is True
        assert by_id["judged-pass"].judge_score is not None

        failing = by_id["judged-fail"]
        assert failing.assertions_passed is False
        assert [a.passed for a in failing.assertion_results] == [False, False]
        # Default: the judge still scores responses that failed a check
        assert failing.judge_score is not None
        assert failing.judge_skipped_reason is None

        assert by_id["plain"].assertions_passed is None
        assert by_id["plain"].assertion_results == []

    def test_assertions_only_skips_judge(self, make_config):
        result = _run(make_config())
        only = _by_id(result)["assertions-only"]

        assert only.assertions_passed is True
        assert only.judge_score is None
        assert only.judge_skipped_reason == "assertions_only"
        judge = FakeJudge.instances[0]
        assert "assertions-only" not in judge.calls
        assert sorted(judge.calls) == ["judged-fail", "judged-pass", "plain"]

    def test_skip_on_assertion_failure_saves_judge_calls(self, make_config):
        result = _run(make_config(skip_on_assertion_failure=True))
        by_id = _by_id(result)

        failing = by_id["judged-fail"]
        assert failing.judge_score is None
        assert failing.judge_skipped_reason == "assertion_failure"
        judge = FakeJudge.instances[0]
        assert sorted(judge.calls) == ["judged-pass", "plain"]

    def test_run_result_aggregates(self, make_config):
        result = _run(make_config())
        # Three asserted results, two passed
        assert result.get_assertion_pass_rate() == pytest.approx(2 / 3)
        assert result.get_assertion_pass_rate("fake-1") == pytest.approx(2 / 3)
        assert [r.test_case_id for r in result.get_assertion_failures()] == ["judged-fail"]

    def test_judge_is_lazy(self, make_config):
        config = make_config()

        async def go():
            runner = Runner(config)
            assert FakeJudge.instances == []
            _ = runner.judge
            assert len(FakeJudge.instances) == 1
            _ = runner.judge
            assert len(FakeJudge.instances) == 1

        asyncio.run(go())


class TestAssertionsOnlyGoldenSetNeedsNoJudge:
    def test_judge_never_constructed(self, tmp_path, monkeypatch):
        golden = tmp_path / "golden.yaml"
        golden.write_text(textwrap.dedent(
            """
            name: "Only"
            test_cases:
              - id: "a"
                query: "q-only"
                expected_behavior: "OK"
                evaluation_mode: assertions_only
                assertions:
                  - type: equals
                    value: "OK"
            """
        ))

        def exploding_judge(config):
            raise AssertionError("judge must not be constructed for assertions_only sets")

        monkeypatch.setattr(runner_module, "get_provider", lambda cfg: FakeProvider(cfg, RESPONSES))
        monkeypatch.setattr(runner_module, "LLMJudge", exploding_judge)

        config = RunConfig(
            golden_set=str(golden),
            models=[{"name": "Fake", "provider": "fake", "model": "fake-1"}],
            output={"directory": str(tmp_path / "out"), "formats": ["json"]},
        )
        result = _run(config)
        assert result.results[0].assertions_passed is True
        assert result.get_average_score() is None
