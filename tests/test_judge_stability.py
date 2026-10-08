"""Tests for multi-run judge sampling, score aggregation, and the stability gate."""

import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional

import pytest
from click.testing import CliRunner

from promptlens.cli import _check_max_flip_rate, cli
from promptlens.exporters.csv_exporter import CSVExporter
from promptlens.exporters.html_exporter import HTMLExporter
from promptlens.exporters.junit_exporter import JUnitXMLExporter
from promptlens.exporters.markdown_exporter import MarkdownExporter
from promptlens.judges.llm_judge import LLMJudge
from promptlens.judges.stability import (
    aggregate_judge_scores,
    aggregate_scores,
    compute_agreement,
)
from promptlens.models.config import JudgeConfig
from promptlens.models.result import (
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)
from promptlens.models.test_case import TestCase as GoldenTestCase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score(value: int, explanation: str = "Looks fine.", **kwargs) -> JudgeScore:
    return JudgeScore(
        score=value,
        explanation=explanation,
        judge_model="judge-model",
        judge_provider="anthropic",
        **kwargs,
    )


def _response(model: str = "model-a", error: Optional[str] = None) -> ModelResponse:
    return ModelResponse(
        content="The answer is 42." if error is None else "",
        model=model,
        provider="anthropic",
        latency_ms=1000.0,
        cost_usd=0.001,
        error=error,
    )


def _eval(
    test_case_id: str,
    model: str = "model-a",
    samples: Optional[List[int]] = None,
    score: Optional[int] = None,
) -> EvaluationResult:
    judge_score = None
    if samples is not None:
        judge_score = aggregate_judge_scores([_score(s) for s in samples])
    elif score is not None:
        judge_score = _score(score)
    return EvaluationResult(
        test_case_id=test_case_id,
        query="What is the answer?",
        expected_behavior="Answers correctly",
        model_response=_response(model=model),
        judge_score=judge_score,
    )


def _run(results: List[EvaluationResult], models: Optional[List[str]] = None) -> RunResult:
    return RunResult(
        run_id="run-1",
        run_name="stability-run",
        timestamp=datetime(2026, 10, 8, 12, 0, 0),
        golden_set_name="golden-set",
        models_tested=models or ["model-a"],
        results=results,
    )


class _ScriptedProvider:
    """Stand-in judge provider that returns scripted judge responses in order."""

    def __init__(self, scores: List[int]) -> None:
        self.scores = list(scores)
        self.calls = 0

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        value = self.scores[self.calls % len(self.scores)]
        self.calls += 1
        return ModelResponse(
            content=f"SCORE: {value}\nEXPLANATION: Run {self.calls} scored {value}.",
            model="judge-model",
            provider="anthropic",
            latency_ms=10.0,
        )


def _judge_with_scripted_provider(runs: int, scores: List[int], aggregation: str = "median"):
    judge = LLMJudge.__new__(LLMJudge)
    judge.config = JudgeConfig(
        provider="anthropic", model="judge-model", runs=runs, aggregation=aggregation
    )
    judge.provider = _ScriptedProvider(scores)
    return judge


def _test_case() -> GoldenTestCase:
    return GoldenTestCase(
        id="case-1",
        query="What is the answer?",
        expected_behavior="Answers correctly",
    )


# ---------------------------------------------------------------------------
# aggregate_scores / compute_agreement
# ---------------------------------------------------------------------------


def test_median_aggregation_returns_a_sampled_value_for_even_counts():
    assert aggregate_scores([3, 4], "median") == 4
    assert aggregate_scores([2, 4, 4, 5], "median") == 4
    assert aggregate_scores([5, 1, 5], "median") == 5


def test_mean_aggregation_rounds_half_up():
    assert aggregate_scores([3, 4], "mean") == 4
    assert aggregate_scores([3, 3, 4], "mean") == 3
    assert aggregate_scores([1, 5], "mean") == 3


def test_min_aggregation_is_strictest():
    assert aggregate_scores([5, 5, 2], "min") == 2


def test_aggregate_scores_rejects_empty_and_unknown_methods():
    with pytest.raises(ValueError):
        aggregate_scores([], "median")
    with pytest.raises(ValueError):
        aggregate_scores([3], "mode")


def test_compute_agreement_is_share_of_samples_matching_final():
    assert compute_agreement([4, 4, 4], 4) == 1.0
    assert compute_agreement([4, 4, 5], 4) == pytest.approx(2 / 3)
    assert compute_agreement([1, 5], 3) == 0.0


# ---------------------------------------------------------------------------
# aggregate_judge_scores
# ---------------------------------------------------------------------------


def test_single_score_is_marked_unanimous_without_changing_explanation():
    aggregated = aggregate_judge_scores([_score(4, "Only run.")])

    assert aggregated.score == 4
    assert aggregated.runs == 1
    assert aggregated.samples == [4]
    assert aggregated.agreement == 1.0
    assert aggregated.explanation == "Only run."
    assert aggregated.is_unanimous


def test_disagreeing_runs_keep_the_explanation_that_matches_the_final_score():
    aggregated = aggregate_judge_scores(
        [_score(5, "Generous run."), _score(3, "Harsh run."), _score(5, "Second generous run.")]
    )

    assert aggregated.score == 5
    assert aggregated.runs == 3
    assert aggregated.samples == [5, 3, 5]
    assert aggregated.agreement == pytest.approx(2 / 3)
    assert not aggregated.is_unanimous
    assert aggregated.explanation.startswith(
        "[3 judge runs scored 5, 3, 5; median = 5, agreement 67%] "
    )
    assert aggregated.explanation.endswith("Generous run.")


def test_unanimous_runs_get_a_short_prefix():
    aggregated = aggregate_judge_scores([_score(4), _score(4)])

    assert aggregated.explanation.startswith("[2 judge runs, all scored 4] ")
    assert aggregated.agreement == 1.0
    assert aggregated.is_unanimous


def test_criteria_scores_aggregate_per_key():
    aggregated = aggregate_judge_scores(
        [
            _score(4, criteria_scores={"accuracy": 4, "tool_selection": 5}),
            _score(2, criteria_scores={"accuracy": 2, "tool_selection": 5}),
            _score(4, criteria_scores={"accuracy": 5}),
        ]
    )

    assert aggregated.criteria_scores == {"accuracy": 4, "tool_selection": 5}


def test_tool_scores_average_across_runs():
    aggregated = aggregate_judge_scores(
        [
            _score(4, tool_usage_score=1.0, tool_efficiency_score=0.8),
            _score(4, tool_usage_score=0.5, tool_efficiency_score=0.6),
        ]
    )

    assert aggregated.tool_usage_score == pytest.approx(0.75)
    assert aggregated.tool_efficiency_score == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# LLMJudge sampling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_calls_provider_once_per_run_and_aggregates():
    judge = _judge_with_scripted_provider(runs=5, scores=[4, 4, 5, 4, 3])

    result = await judge.evaluate(_test_case(), _response())

    assert judge.provider.calls == 5
    assert result.runs == 5
    assert sorted(result.samples) == [3, 4, 4, 4, 5]
    assert result.score == 4
    assert result.agreement == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_judge_with_one_run_makes_one_call_and_stays_backward_compatible():
    judge = _judge_with_scripted_provider(runs=1, scores=[2])

    result = await judge.evaluate(_test_case(), _response())

    assert judge.provider.calls == 1
    assert result.score == 2
    assert result.runs == 1
    assert result.samples == [2]
    assert result.agreement == 1.0
    assert result.explanation == "Run 1 scored 2."


@pytest.mark.asyncio
async def test_judge_honors_min_aggregation():
    judge = _judge_with_scripted_provider(runs=3, scores=[5, 5, 2], aggregation="min")

    result = await judge.evaluate(_test_case(), _response())

    assert result.score == 2


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_judge_config_defaults_to_single_median_run():
    config = JudgeConfig()
    assert config.runs == 1
    assert config.aggregation == "median"


def test_judge_config_rejects_bad_runs_and_aggregation():
    with pytest.raises(ValueError):
        JudgeConfig(runs=0)
    with pytest.raises(ValueError):
        JudgeConfig(runs=26)
    with pytest.raises(ValueError):
        JudgeConfig(aggregation="mode")
    assert JudgeConfig(aggregation=" Mean ").aggregation == "mean"


def test_legacy_judge_score_json_without_stability_fields_still_loads():
    legacy = {
        "score": 4,
        "explanation": "Old result.",
        "judge_model": "judge-model",
        "judge_provider": "anthropic",
    }
    loaded = JudgeScore(**legacy)

    assert loaded.runs == 1
    assert loaded.samples == []
    assert loaded.agreement is None
    assert loaded.is_unanimous


# ---------------------------------------------------------------------------
# RunResult flip rate
# ---------------------------------------------------------------------------


def test_flip_rate_counts_only_multi_run_cases_that_disagreed():
    run = _run(
        [
            _eval("c1", samples=[4, 4, 4]),
            _eval("c2", samples=[4, 5, 4]),
            _eval("c3", samples=[2, 2, 2]),
            _eval("c4", samples=[1, 5, 3]),
            _eval("c5", score=3),  # single-run legacy score, not counted
        ]
    )

    assert run.get_flip_rate("model-a") == pytest.approx(0.5)
    assert run.get_unstable_case_ids("model-a") == ["c2", "c4"]
    assert run.get_judge_runs() == 3


def test_flip_rate_is_none_when_nothing_was_judged_more_than_once():
    run = _run([_eval("c1", score=4), _eval("c2")])

    assert run.get_flip_rate("model-a") is None
    assert run.get_unstable_case_ids("model-a") == []
    assert run.get_judge_runs() == 1


def test_flip_rate_is_per_model():
    run = _run(
        [
            _eval("c1", model="model-a", samples=[4, 4]),
            _eval("c1", model="model-b", samples=[4, 2]),
        ],
        models=["model-a", "model-b"],
    )

    assert run.get_flip_rate("model-a") == 0.0
    assert run.get_flip_rate("model-b") == 1.0
    assert run.get_flip_rate() == 0.5


# ---------------------------------------------------------------------------
# Stability gate
# ---------------------------------------------------------------------------


def test_check_max_flip_rate_flags_models_over_the_gate():
    run = _run(
        [
            _eval("c1", model="model-a", samples=[4, 4]),
            _eval("c2", model="model-a", samples=[4, 5]),
            _eval("c1", model="model-b", samples=[4, 4]),
            _eval("c2", model="model-b", samples=[4, 4]),
        ],
        models=["model-a", "model-b"],
    )

    failing = _check_max_flip_rate(run, max_flip_rate=0.25)

    assert failing == [("model-a", 0.5, ["c2"])]


def test_check_max_flip_rate_passes_when_flip_rate_equals_gate():
    run = _run([_eval("c1", samples=[4, 4]), _eval("c2", samples=[4, 5])])

    assert _check_max_flip_rate(run, max_flip_rate=0.5) == []


def test_check_max_flip_rate_fails_models_with_no_multi_run_scores():
    run = _run([_eval("c1", score=4)])

    failing = _check_max_flip_rate(run, max_flip_rate=0.5)

    assert failing == [("model-a", None, [])]


def test_cli_rejects_max_flip_rate_without_multiple_judge_runs(tmp_path):
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        "name: g\nversion: '1.0'\ntest_cases:\n"
        "  - id: t1\n    query: q\n    expected_behavior: e\n"
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"golden_set: {golden}\n" "models:\n  - name: m\n    provider: anthropic\n    model: x\n"
    )

    result = CliRunner().invoke(cli, ["run", str(config), "--max-flip-rate", "0.2"])

    assert result.exit_code == 1
    assert "--max-flip-rate needs judge runs greater than 1" in result.output


def test_cli_judge_runs_override_is_visible_in_dry_run(tmp_path):
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        "name: g\nversion: '1.0'\ntest_cases:\n"
        "  - id: t1\n    query: q\n    expected_behavior: e\n"
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"golden_set: {golden}\n"
        "models:\n  - name: m\n    provider: anthropic\n    model: x\n"
        "judge:\n  runs: 2\n  aggregation: min\n"
    )

    result = CliRunner().invoke(
        cli, ["run", str(config), "--dry-run", "--judge-runs", "7", "--max-flip-rate", "0.1"]
    )

    assert result.exit_code == 0, result.output
    assert "Judge runs: 7 (min)" in result.output


# ---------------------------------------------------------------------------
# Exporters surface stability data
# ---------------------------------------------------------------------------


def _mixed_run() -> RunResult:
    return _run([_eval("c1", samples=[4, 4, 4]), _eval("c2", samples=[3, 5, 4])])


def test_csv_export_includes_samples_and_agreement(tmp_path):
    out = tmp_path / "results.csv"
    CSVExporter().export(_mixed_run(), str(out))

    rows = list(csv.DictReader(out.open()))

    assert rows[0]["judge_runs"] == "3"
    assert rows[0]["score_samples"] == "4 4 4"
    assert rows[0]["judge_agreement"] == "1.0"
    assert rows[1]["score_samples"] == "3 5 4"


def test_markdown_export_reports_stability(tmp_path):
    out = tmp_path / "results.md"
    MarkdownExporter().export(_mixed_run(), str(out))
    text = out.read_text()

    assert "| Judge Stability | 50% unanimous over 3 runs (1 unstable) |" in text
    assert "4/5 (runs: 3,5,4)" in text


def test_html_export_reports_stability(tmp_path):
    out = tmp_path / "report.html"
    HTMLExporter().export(_mixed_run(), str(out))
    html = out.read_text()

    assert "Judge stability (3 runs)" in html
    assert "50% unanimous" in html
    assert "runs: 3, 5, 4" in html
    assert 'class="samples flipped"' in html


def test_html_export_hides_stability_column_for_single_run(tmp_path):
    out = tmp_path / "report.html"
    HTMLExporter().export(_run([_eval("c1", score=4)]), str(out))
    html = out.read_text()

    assert "Judge stability" not in html
    assert "runs:" not in html


def test_junit_export_records_samples_and_flip_rate(tmp_path):
    out = tmp_path / "junit.xml"
    JUnitXMLExporter(fail_under=3.0).export(_mixed_run(), str(out))
    root = ET.parse(out).getroot()

    props = {p.get("name"): p.get("value") for p in root.iter("property")}
    assert props["judge_runs"] == "3"
    assert props["judge_flip_rate"] == "0.50"

    outs = [tc.find("system-out").text for tc in root.iter("testcase")]
    assert "judge_samples: 3 5 4" in outs[1]
    assert "judge_agreement: 0.33" in outs[1]
