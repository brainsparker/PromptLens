"""Tests for how the runner wires deterministic assertions around the judge."""

import asyncio
from typing import List, Optional

from promptlens.models.config import RunConfig
from promptlens.models.result import JudgeScore, ModelResponse, RunResult
from promptlens.models.test_case import TestCase
from promptlens.providers.base import BaseProvider
from promptlens.runners import runner as runner_module


class FakeProvider(BaseProvider):
    """Provider that returns canned content and never touches the network."""

    def __init__(self, config, content: str = "Refunds are accepted within 30 days.", error=None):
        self.config = config
        self.content = content
        self.error = error
        self.calls = 0

    def _initialize_client(self) -> None:  # pragma: no cover - not used
        pass

    async def generate(self, prompt: str, tools=None, **kwargs) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            content=self.content if not self.error else "",
            model=self.config.model,
            provider=self.config.name,
            latency_ms=120.0,
            cost_usd=0.002,
            error=self.error,
        )

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return 0.002

    @property
    def provider_name(self) -> str:
        return self.config.name

    def supports_tools(self) -> bool:
        return False


class FakeJudge:
    """Judge that records calls and returns a fixed score."""

    def __init__(self, *_args, **_kwargs):
        self.calls: List[str] = []

    async def evaluate(self, test_case: TestCase, model_response: ModelResponse) -> JudgeScore:
        self.calls.append(test_case.id)
        return JudgeScore(
            score=4,
            explanation="fine",
            judge_model="fake-judge",
            judge_provider="fake",
        )


def _config(tmp_path, golden_set_path, judge_on_assertion_failure: Optional[bool] = None):
    execution = {"parallel_requests": 2}
    if judge_on_assertion_failure is not None:
        execution["judge_on_assertion_failure"] = judge_on_assertion_failure
    return RunConfig(
        golden_set=str(golden_set_path),
        models=[{"name": "Fake", "provider": "fake", "model": "fake-1"}],
        judge={"provider": "fake", "model": "fake-judge"},
        execution=execution,
        output={"directory": str(tmp_path / "out"), "formats": ["json"]},
    )


def _write_golden_set(tmp_path, cases):
    import yaml

    path = tmp_path / "golden.yaml"
    path.write_text(yaml.safe_dump({"name": "assertions-suite", "test_cases": cases}))
    return path


def _execute(
    monkeypatch,
    config,
    provider_content="Refunds are accepted within 30 days.",
    provider_error=None,
):
    """Build a Runner with fake provider and judge, run it, and return (result, fakes).

    The runner is constructed inside the event loop because asyncio.Semaphore
    binds to the current loop on Python 3.9.
    """
    created = {}

    def fake_get_provider(model_config):
        from promptlens.models.config import ProviderConfig

        provider = FakeProvider(
            ProviderConfig(name=model_config.provider, model=model_config.model),
            content=provider_content,
            error=provider_error,
        )
        created["provider"] = provider
        return provider

    def fake_judge_factory(_judge_config):
        judge = FakeJudge()
        created["judge"] = judge
        return judge

    monkeypatch.setattr(runner_module, "get_provider", fake_get_provider)
    monkeypatch.setattr(runner_module, "LLMJudge", fake_judge_factory)

    async def go() -> RunResult:
        runner = runner_module.Runner(config)
        return await runner.run()

    return asyncio.run(go()), created


CASES = [
    {
        "id": "passes",
        "query": "Refund policy?",
        "expected_behavior": "Mentions 30 days",
        "assertions": [{"type": "contains", "value": "30 days"}],
    },
    {
        "id": "fails",
        "query": "Refund policy?",
        "expected_behavior": "Mentions warranty",
        "assertions": [
            {"type": "contains", "value": "warranty"},
            {"type": "max_chars", "value": 5},
        ],
    },
    {
        "id": "no-assertions",
        "query": "Refund policy?",
        "expected_behavior": "Anything",
    },
]


def test_assertions_run_and_failed_cases_skip_the_judge_by_default(tmp_path, monkeypatch):
    golden = _write_golden_set(tmp_path, CASES)
    result, created = _execute(monkeypatch, _config(tmp_path, golden))
    by_id = {r.test_case_id: r for r in result.results}

    passing = by_id["passes"]
    assert passing.assertions_passed is True
    assert passing.judge_score is not None
    assert passing.judge_skipped_reason is None

    failing = by_id["fails"]
    assert failing.assertions_passed is False
    assert [a.passed for a in failing.assertion_results] == [False, False]
    assert failing.judge_score is None
    assert failing.judge_skipped_reason is not None
    assert "contains 'warranty'" in failing.judge_skipped_reason

    plain = by_id["no-assertions"]
    assert plain.assertions_passed is None
    assert plain.assertion_results == []
    assert plain.judge_score is not None

    # The judge was called exactly for the two cases that did not fail assertions.
    assert sorted(created["judge"].calls) == ["no-assertions", "passes"]

    summary = result.get_assertion_summary()
    assert summary.cases_with_assertions == 2
    assert summary.cases_passed == 1
    assert summary.cases_failed == 1
    assert summary.assertions_total == 3
    assert summary.assertions_failed == 2
    assert [r.test_case_id for r in result.get_assertion_failures()] == ["fails"]


def test_judge_on_assertion_failure_opt_in_still_judges(tmp_path, monkeypatch):
    golden = _write_golden_set(tmp_path, CASES)
    result, created = _execute(
        monkeypatch, _config(tmp_path, golden, judge_on_assertion_failure=True)
    )
    failing = next(r for r in result.results if r.test_case_id == "fails")

    assert failing.assertions_passed is False
    assert failing.judge_score is not None
    assert failing.judge_skipped_reason is None
    assert sorted(created["judge"].calls) == ["fails", "no-assertions", "passes"]


def test_assertions_are_skipped_when_generation_errors(tmp_path, monkeypatch):
    golden = _write_golden_set(tmp_path, CASES[:2])
    result, created = _execute(
        monkeypatch, _config(tmp_path, golden), provider_error="503 upstream"
    )
    for eval_result in result.results:
        assert eval_result.model_response.error == "503 upstream"
        assert eval_result.assertion_results == []
        assert eval_result.assertions_passed is None
        assert eval_result.judge_score is None
    assert created["judge"].calls == []


def test_summary_prints_assertion_lines(tmp_path, monkeypatch, capsys):
    golden = _write_golden_set(tmp_path, CASES)
    _execute(monkeypatch, _config(tmp_path, golden))
    out = capsys.readouterr().out

    assert "Assertions:" in out
    assert "1/2 cases passed" in out
    assert "Failed assertions" in out
    assert "contains 'warranty'" in out


def test_evaluation_result_serializes_assertions_to_json(tmp_path, monkeypatch):
    golden = _write_golden_set(tmp_path, CASES[:2])
    result, _ = _execute(monkeypatch, _config(tmp_path, golden))
    data = result.model_dump(mode="json")
    failing = next(r for r in data["results"] if r["test_case_id"] == "fails")

    assert failing["assertion_results"][0]["type"] == "contains"
    assert failing["assertion_results"][0]["passed"] is False
    assert failing["judge_skipped_reason"].startswith("skipped:")

    # And it round-trips, so `promptlens compare` can read runs with assertions.
    restored = RunResult(**data)
    assert restored.get_assertion_summary().cases_failed == 1
