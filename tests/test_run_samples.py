"""Tests for --samples in the runner and the --fail-on-flaky CLI gate."""

import asyncio
import json
from typing import Optional

import pytest
import yaml
from click.testing import CliRunner

from promptlens.cli import _check_fail_on_flaky, cli
from promptlens.models.config import ExecutionConfig, RunConfig
from promptlens.models.result import JudgeScore, ModelResponse
from promptlens.providers.base import BaseProvider
from promptlens.runners import runner as runner_module
from promptlens.runners.runner import Runner


class _ScriptedProvider(BaseProvider):
    """Provider that returns a fixed response and counts calls."""

    def __init__(self, config):
        super().__init__(config)
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "scripted"

    def supports_tools(self) -> bool:
        return False

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return 0.001

    async def generate(self, prompt: str, tools=None, **kwargs) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            content=f"response {self.calls}",
            model=self.config.model,
            provider="scripted",
            latency_ms=10.0,
            cost_usd=0.001,
        )


class _ScriptedJudge:
    """Judge that hands out scores per test case in call order."""

    def __init__(self, config, scores_by_case):
        self.scores_by_case = scores_by_case
        self.calls = {}

    async def evaluate(self, test_case, model_response) -> JudgeScore:
        index = self.calls.get(test_case.id, 0)
        self.calls[test_case.id] = index + 1
        scores = self.scores_by_case[test_case.id]
        return JudgeScore(
            score=scores[index % len(scores)],
            explanation="scripted",
            judge_model="judge",
            judge_provider="scripted",
        )


def _write_golden_set(tmp_path, case_ids):
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        yaml.safe_dump(
            {
                "name": "Scripted Set",
                "description": "Deterministic cases",
                "version": "1.0",
                "test_cases": [
                    {"id": case_id, "query": f"Q {case_id}", "expected_behavior": "Correct"}
                    for case_id in case_ids
                ],
            }
        )
    )
    return golden


def _write_config(tmp_path, golden, output_dir, samples: Optional[int] = None, formats=None):
    execution = {"parallel_requests": 4}
    if samples is not None:
        execution["samples"] = samples
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "golden_set": str(golden),
                "models": [{"name": "Scripted", "provider": "scripted", "model": "scripted-1"}],
                "judge": {"provider": "scripted", "model": "judge"},
                "execution": execution,
                "output": {"directory": str(output_dir), "formats": formats or ["json", "junit"]},
            }
        )
    )
    return config


@pytest.fixture
def scripted(monkeypatch):
    """Patch provider and judge construction so runs are offline and deterministic."""
    state = {"providers": [], "scores": {}}

    def fake_get_provider(model_config):
        provider = _ScriptedProvider(model_config)
        state["providers"].append(provider)
        return provider

    def fake_judge(config):
        return _ScriptedJudge(config, state["scores"])

    monkeypatch.setattr(runner_module, "get_provider", fake_get_provider)
    monkeypatch.setattr(runner_module, "LLMJudge", fake_judge)
    return state


class TestExecutionConfigSamples:
    def test_defaults_to_one(self):
        assert ExecutionConfig().samples == 1

    def test_rejects_zero(self):
        with pytest.raises(ValueError):
            ExecutionConfig(samples=0)


class TestRunnerSamples:
    def test_each_case_runs_samples_times(self, tmp_path, scripted):
        golden = _write_golden_set(tmp_path, ["tc-1", "tc-2"])
        scripted["scores"] = {"tc-1": [5], "tc-2": [4]}
        config = RunConfig(
            golden_set=str(golden),
            models=[{"name": "Scripted", "provider": "scripted", "model": "scripted-1"}],
            judge={"provider": "scripted", "model": "judge"},
            execution={"samples": 3},
            output={"directory": str(tmp_path / "out"), "formats": ["json"]},
        )

        result = asyncio.run(Runner(config).run())

        assert len(result.results) == 6
        assert scripted["providers"][0].calls == 6
        assert result.metadata["samples"] == 3
        assert result.sample_count == 3
        assert result.test_case_count == 2
        indices = sorted(r.sample_index for r in result.results if r.test_case_id == "tc-1")
        assert indices == [0, 1, 2]

    def test_single_sample_default_is_unchanged(self, tmp_path, scripted):
        golden = _write_golden_set(tmp_path, ["tc-1"])
        scripted["scores"] = {"tc-1": [5]}
        config = RunConfig(
            golden_set=str(golden),
            models=[{"name": "Scripted", "provider": "scripted", "model": "scripted-1"}],
            judge={"provider": "scripted", "model": "judge"},
            output={"directory": str(tmp_path / "out"), "formats": ["json"]},
        )

        result = asyncio.run(Runner(config).run())

        assert len(result.results) == 1
        assert result.results[0].sample_index == 0
        assert result.metadata["samples"] == 1


class TestFailOnFlakyGate:
    def _invoke(self, tmp_path, scripted, scores, args, config_samples=None):
        golden = _write_golden_set(tmp_path, list(scores))
        scripted["scores"] = scores
        output_dir = tmp_path / "results"
        config = _write_config(tmp_path, golden, output_dir, samples=config_samples)
        result = CliRunner().invoke(cli, ["run", str(config), *args])
        return result, output_dir

    def test_flaky_case_fails_the_gate(self, tmp_path, scripted):
        result, _ = self._invoke(
            tmp_path, scripted, {"tc-1": [5, 2, 5]}, ["--samples", "3", "--fail-on-flaky"]
        )
        assert result.exit_code == 2, result.output
        assert "Stability gate failed" in result.output
        assert "tc-1" in result.output

    def test_stable_run_passes_the_gate(self, tmp_path, scripted):
        result, output_dir = self._invoke(
            tmp_path, scripted, {"tc-1": [5], "tc-2": [1]}, ["--samples", "3", "--fail-on-flaky"]
        )
        assert result.exit_code == 0, result.output
        assert "Stability gate passed" in result.output

        run_dirs = [p for p in output_dir.iterdir() if p.is_dir()]
        data = json.loads((run_dirs[0] / "results.json").read_text())
        assert data["metadata"]["samples"] == 3
        assert data["metadata"]["pass_threshold"] == 3.0
        assert data["stability"]["models"][0]["pass_hat_k"] == 0.5

    def test_gate_uses_fail_under_as_threshold(self, tmp_path, scripted):
        # Scores 4,3,4 are stable at the default threshold (3.0) but flip at 4.0.
        result, _ = self._invoke(
            tmp_path,
            scripted,
            {"tc-1": [4, 3, 4]},
            ["--samples", "3", "--fail-on-flaky", "--fail-under", "4"],
        )
        assert result.exit_code == 2, result.output
        assert "Stability gate failed" in result.output
        # Average 3.67 is also below --fail-under 4, so both gates report.
        assert "Quality gate failed" in result.output

    def test_gate_requires_multiple_samples(self, tmp_path, scripted):
        result, _ = self._invoke(tmp_path, scripted, {"tc-1": [5]}, ["--fail-on-flaky"])
        assert result.exit_code == 1
        assert "at least 2 samples" in result.output

    def test_samples_from_config_file(self, tmp_path, scripted):
        result, _ = self._invoke(
            tmp_path, scripted, {"tc-1": [5, 2]}, ["--fail-on-flaky"], config_samples=2
        )
        assert result.exit_code == 2, result.output

    def test_cli_samples_override_config(self, tmp_path, scripted):
        result, _ = self._invoke(
            tmp_path, scripted, {"tc-1": [5]}, ["--samples", "4"], config_samples=2
        )
        assert result.exit_code == 0, result.output
        assert scripted["providers"][0].calls == 4

    def test_dry_run_reports_samples(self, tmp_path, scripted):
        result, _ = self._invoke(tmp_path, scripted, {"tc-1": [5]}, ["--samples", "3", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "Samples per test case: 3" in result.output
        assert scripted["providers"] == []

    def test_check_fail_on_flaky_helper(self, tmp_path, scripted):
        from promptlens.models.result import RunResult
        from tests.test_stability import _make_run, _sampled

        run = _make_run(_sampled("tc-1", [5, 1]) + _sampled("tc-2", [5, 5]), samples=2)
        flaky = _check_fail_on_flaky(run, 3.0)
        assert [c.test_case_id for c in flaky] == ["tc-1"]
