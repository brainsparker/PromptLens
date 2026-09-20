"""Tests for judge sampling, stability aggregation, and the disagreement gate."""

import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List

import pytest
from click.testing import CliRunner

from promptlens.cli import _check_judge_disagreement, cli
from promptlens.exporters.csv_exporter import CSVExporter
from promptlens.exporters.html_exporter import HTMLExporter
from promptlens.exporters.json_exporter import JSONExporter
from promptlens.exporters.junit_exporter import JUnitXMLExporter
from promptlens.exporters.markdown_exporter import MarkdownExporter
from promptlens.judges.base import BaseJudge
from promptlens.judges.stability import (
    aggregate_judge_scores,
    straddles_threshold,
    summarize_stability,
)
from promptlens.models.config import MAX_JUDGE_SAMPLES, JudgeConfig, OutputConfig, RunConfig
from promptlens.models.result import (
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)
from promptlens.models.test_case import TestCase as GoldenTestCase


def _score(value: int, explanation: str = None) -> JudgeScore:
    return JudgeScore(
        score=value,
        explanation=explanation or f"explanation for {value}",
        judge_model="judge-model",
        judge_provider="anthropic",
    )


def _response(model: str = "model-a", error: str = None) -> ModelResponse:
    return ModelResponse(
        content="The answer is 42.",
        model=model,
        provider="anthropic",
        latency_ms=100.0,
        cost_usd=0.001,
        tokens_used=20,
        error=error,
    )


def _eval(
    test_case_id: str, judge_score: JudgeScore = None, model: str = "model-a"
) -> EvaluationResult:
    return EvaluationResult(
        test_case_id=test_case_id,
        query="What is the answer?",
        expected_behavior="Answers correctly",
        model_response=_response(model=model),
        judge_score=judge_score,
    )


def _run(results: List[EvaluationResult], judge_samples: int = 1, models=None) -> RunResult:
    return RunResult(
        run_id="run-abc",
        run_name="stability-run",
        timestamp=datetime(2026, 9, 20, 12, 0, 0),
        golden_set_name="golden-set",
        models_tested=models or ["model-a"],
        results=results,
        metadata={"judge_samples": judge_samples},
    )


def _test_case() -> GoldenTestCase:
    return GoldenTestCase(id="tc-1", query="What is the answer?", expected_behavior="Says 42")


class ScriptedJudge(BaseJudge):
    """Judge that returns a scripted sequence of scores and counts calls."""

    def __init__(self, scores: List[int]) -> None:
        self._scores = list(scores)
        self.calls = 0

    async def evaluate(self, test_case, model_response) -> JudgeScore:
        value = self._scores[self.calls % len(self._scores)]
        self.calls += 1
        return _score(value, explanation=f"call {self.calls} scored {value}")

    @property
    def judge_model(self) -> str:
        return "scripted"

    @property
    def judge_provider(self) -> str:
        return "test"


# ---------------------------------------------------------------------------
# aggregate_judge_scores
# ---------------------------------------------------------------------------


def test_aggregate_uses_median_and_records_spread():
    aggregate = aggregate_judge_scores([_score(3), _score(5), _score(4)])

    assert aggregate.score == 4
    assert aggregate.sample_scores == [3, 5, 4]
    assert aggregate.score_mean == pytest.approx(4.0)
    assert aggregate.score_std == pytest.approx(0.8165, abs=1e-3)
    assert aggregate.score_min == 3
    assert aggregate.score_max == 5
    assert aggregate.is_sampled
    assert aggregate.sample_count == 3
    assert aggregate.score_range == 2


def test_aggregate_flags_disagreement_at_configured_range():
    spread_two = aggregate_judge_scores([_score(3), _score(5)], disagreement_range=2)
    spread_one = aggregate_judge_scores([_score(4), _score(5)], disagreement_range=2)
    strict = aggregate_judge_scores([_score(4), _score(5)], disagreement_range=1)

    assert spread_two.disagreement is True
    assert spread_one.disagreement is False
    assert strict.disagreement is True


def test_aggregate_rounds_even_median_half_up_and_picks_closest_explanation():
    aggregate = aggregate_judge_scores([_score(2, "low"), _score(5, "high")])

    assert aggregate.score == 4  # median 3.5 rounds up
    assert aggregate.explanation == "high"  # closest sample to the aggregate
    assert aggregate.sample_explanations == ["low", "high"]


def test_aggregate_explanation_comes_from_a_sample_matching_the_median():
    aggregate = aggregate_judge_scores(
        [_score(5, "five"), _score(3, "three"), _score(3, "three-b")]
    )

    assert aggregate.score == 3
    assert aggregate.explanation == "three"


def test_aggregate_single_sample_is_returned_unchanged():
    single = _score(4)
    assert aggregate_judge_scores([single]) is single
    assert not single.is_sampled
    assert single.effective_score == 4.0


def test_aggregate_rejects_empty_samples():
    with pytest.raises(ValueError):
        aggregate_judge_scores([])


def test_aggregate_keeps_tool_fields_from_representative_sample():
    with_tools = _score(4)
    with_tools.tool_usage_score = 0.9
    with_tools.criteria_scores = {"overall_score": 4, "tool_efficiency": 5}
    aggregate = aggregate_judge_scores([with_tools, _score(4), _score(2)])

    assert aggregate.score == 4
    assert aggregate.tool_usage_score == 0.9
    assert aggregate.criteria_scores == {"overall_score": 4, "tool_efficiency": 5}


# ---------------------------------------------------------------------------
# straddles_threshold
# ---------------------------------------------------------------------------


def test_straddle_requires_samples_on_both_sides_of_threshold():
    mixed = aggregate_judge_scores([_score(3), _score(4)])
    all_pass = aggregate_judge_scores([_score(4), _score(5)])
    all_fail = aggregate_judge_scores([_score(1), _score(2)])

    assert straddles_threshold(mixed, 3.5) is True
    assert straddles_threshold(mixed, 3.0) is False  # 3 passes a 3.0 gate
    assert straddles_threshold(all_pass, 3.5) is False
    assert straddles_threshold(all_fail, 3.5) is False
    assert straddles_threshold(mixed, None) is False
    assert straddles_threshold(_score(3), 3.5) is False


# ---------------------------------------------------------------------------
# BaseJudge.evaluate_sampled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_sampled_calls_judge_n_times_and_aggregates():
    judge = ScriptedJudge([5, 3, 4])

    result = await judge.evaluate_sampled(_test_case(), _response(), samples=3)

    assert judge.calls == 3
    assert result.score == 4
    assert sorted(result.sample_scores) == [3, 4, 5]
    assert result.disagreement is True


@pytest.mark.asyncio
async def test_evaluate_sampled_with_one_sample_is_plain_evaluate():
    judge = ScriptedJudge([2])

    result = await judge.evaluate_sampled(_test_case(), _response(), samples=1)

    assert judge.calls == 1
    assert result.score == 2
    assert result.sample_scores == []
    assert result.score_mean is None


@pytest.mark.asyncio
async def test_evaluate_sampled_rejects_zero_samples():
    judge = ScriptedJudge([3])
    with pytest.raises(ValueError):
        await judge.evaluate_sampled(_test_case(), _response(), samples=0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_judge_config_defaults_to_one_sample():
    config = JudgeConfig()
    assert config.samples == 1
    assert config.disagreement_range == 2


@pytest.mark.parametrize("value", [0, -1, MAX_JUDGE_SAMPLES + 1])
def test_judge_config_rejects_out_of_range_samples(value):
    with pytest.raises(ValueError):
        JudgeConfig(samples=value)


@pytest.mark.parametrize("value", [0, 5])
def test_judge_config_rejects_out_of_range_disagreement_range(value):
    with pytest.raises(ValueError):
        JudgeConfig(disagreement_range=value)


def test_output_config_accepts_junit_format():
    config = OutputConfig(formats=["html", "junit"])
    assert config.formats == ["html", "junit"]


# ---------------------------------------------------------------------------
# RunResult averages and summarize_stability
# ---------------------------------------------------------------------------


def test_average_score_uses_sample_mean_for_sampled_results():
    sampled = aggregate_judge_scores([_score(3), _score(4)])  # mean 3.5, median rounds to 4
    run = _run([_eval("tc-1", sampled), _eval("tc-2", _score(2))], judge_samples=2)

    assert run.get_average_score("model-a") == pytest.approx((3.5 + 2) / 2)
    assert run.judge_samples == 2


def test_summarize_stability_counts_disagreements_and_gate_straddles():
    disagreeing = aggregate_judge_scores(
        [_score(1), _score(3), _score(3)]
    )  # spread 2, all fail 3.5
    straddling = aggregate_judge_scores(
        [_score(3), _score(4), _score(4)]
    )  # spread 1, straddles 3.5
    stable = aggregate_judge_scores([_score(5), _score(5), _score(5)])
    run = _run(
        [
            _eval("tc-disagree", disagreeing),
            _eval("tc-straddle", straddling),
            _eval("tc-stable", stable),
            _eval("tc-unjudged", None),
        ],
        judge_samples=3,
    )

    summary = summarize_stability(run, threshold=3.5)

    assert summary.sampled
    assert summary.judged_results == 3
    assert summary.disagreements == 1
    assert summary.gate_straddles == 1
    assert summary.disagreement_rate == pytest.approx(1 / 3)
    assert summary.has_gate_risk
    ids = {c.test_case_id for c in summary.unstable_cases}
    assert ids == {"tc-disagree", "tc-straddle"}
    straddle_case = next(c for c in summary.unstable_cases if c.test_case_id == "tc-straddle")
    assert straddle_case.straddles_gate and not straddle_case.disagreement
    assert straddle_case.spread_label == "3-4"


def test_summarize_stability_is_empty_for_unsampled_run():
    run = _run([_eval("tc-1", _score(4)), _eval("tc-2", _score(2))])

    summary = summarize_stability(run, threshold=3.0)

    assert not summary.sampled
    assert summary.disagreements == 0
    assert summary.gate_straddles == 0
    assert summary.unstable_cases == []
    assert summary.mean_std is None


def test_check_judge_disagreement_reports_unstable_cases():
    run = _run([_eval("tc-1", aggregate_judge_scores([_score(1), _score(5)]))], judge_samples=2)

    summary = _check_judge_disagreement(run, fail_under=None)

    assert len(summary.unstable_cases) == 1


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------


def _sampled_run() -> RunResult:
    return _run(
        [
            _eval("tc-agree", aggregate_judge_scores([_score(4), _score(4), _score(4)])),
            _eval("tc-disagree", aggregate_judge_scores([_score(2), _score(4), _score(5)])),
        ],
        judge_samples=3,
    )


def test_csv_exporter_adds_stability_columns(tmp_path):
    output = tmp_path / "results.csv"
    CSVExporter().export(_sampled_run(), str(output))

    with open(output, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_id = {row["test_case_id"]: row for row in rows}
    assert by_id["tc-disagree"]["judge_samples"] == "3"
    assert by_id["tc-disagree"]["score_samples"] == "2|4|5"
    assert by_id["tc-disagree"]["score_min"] == "2"
    assert by_id["tc-disagree"]["score_max"] == "5"
    assert by_id["tc-disagree"]["judge_disagreement"] == "True"
    assert by_id["tc-agree"]["judge_disagreement"] == "False"


def test_csv_exporter_leaves_stability_columns_blank_for_single_sample(tmp_path):
    output = tmp_path / "results.csv"
    CSVExporter().export(_run([_eval("tc-1", _score(4))]), str(output))

    with open(output, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["score"] == "4"
    assert rows[0]["judge_samples"] == "1"
    assert rows[0]["score_samples"] == ""


def test_json_exporter_includes_sample_scores(tmp_path):
    import json

    output = tmp_path / "results.json"
    JSONExporter().export(_sampled_run(), str(output))

    data = json.loads(output.read_text(encoding="utf-8"))
    disagree = next(r for r in data["results"] if r["test_case_id"] == "tc-disagree")
    assert disagree["judge_score"]["sample_scores"] == [2, 4, 5]
    assert disagree["judge_score"]["disagreement"] is True
    assert data["metadata"]["judge_samples"] == 3


def test_markdown_exporter_renders_stability_section(tmp_path):
    output = tmp_path / "results.md"
    MarkdownExporter().export(_sampled_run(), str(output))

    text = output.read_text(encoding="utf-8")
    assert "## Judge Stability" in text
    assert "Disagreements | 1/2" in text
    assert "`tc-disagree`" in text
    assert "4/5 (samples 2-5) (disagreement)" in text
    assert "4/5 (samples 4-4)" in text


def test_markdown_exporter_omits_stability_section_for_single_sample(tmp_path):
    output = tmp_path / "results.md"
    MarkdownExporter().export(_run([_eval("tc-1", _score(4))]), str(output))

    text = output.read_text(encoding="utf-8")
    assert "Judge Stability" not in text
    assert "| 4/5 |" in text


def test_html_exporter_renders_stability_summary(tmp_path):
    output = tmp_path / "report.html"
    HTMLExporter().export(_sampled_run(), str(output))

    html = output.read_text(encoding="utf-8")
    assert "Judge stability" in html
    assert "Judge disagreements" in html
    assert "samples 2-5, disagreement" in html
    assert "tc-disagree" in html


def test_html_exporter_has_no_stability_section_for_single_sample(tmp_path):
    output = tmp_path / "report.html"
    HTMLExporter().export(_run([_eval("tc-1", _score(4))]), str(output))

    html = output.read_text(encoding="utf-8")
    assert "Judge stability" not in html


def test_junit_exporter_records_stability_properties_and_output(tmp_path):
    output = tmp_path / "junit.xml"
    JUnitXMLExporter(fail_under=3.5).export(_sampled_run(), str(output))

    root = ET.parse(output).getroot()
    suite = root.find("testsuite")
    props = {p.get("name"): p.get("value") for p in suite.find("properties")}
    assert props["judge_samples"] == "3"
    assert props["judge_disagreements"] == "1"
    assert props["judge_gate_straddles"] == "1"  # 2,4,5 straddles 3.5

    cases = {tc.get("name"): tc for tc in suite.findall("testcase")}
    out = cases["tc-disagree"].find("system-out").text
    assert "judge_score_samples: 2,4,5" in out
    assert "judge_disagreement: true" in out
    assert "judge_gate_straddle: true" in out
    # Median 4 passes the 3.5 gate even though one sample failed it
    assert cases["tc-disagree"].find("failure") is None


def test_junit_exporter_unchanged_for_single_sample(tmp_path):
    output = tmp_path / "junit.xml"
    JUnitXMLExporter().export(_run([_eval("tc-1", _score(4))]), str(output))

    root = ET.parse(output).getroot()
    suite = root.find("testsuite")
    props = {p.get("name") for p in suite.find("properties")}
    assert "judge_samples" not in props
    assert "judge_score_samples" not in suite.find("testcase").find("system-out").text


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def _write_config(tmp_path, judge_block: str = "") -> str:
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        "name: tiny\n"
        "test_cases:\n"
        "  - id: tc-1\n"
        "    query: hello\n"
        "    expected_behavior: greets\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"golden_set: {golden}\n"
        "models:\n"
        "  - name: Local\n"
        "    provider: http\n"
        "    model: local\n"
        "    endpoint: http://localhost:9\n"
        f"{judge_block}",
        encoding="utf-8",
    )
    return str(config)


def test_cli_judge_samples_override_shows_in_dry_run(tmp_path):
    config = _write_config(tmp_path)

    result = CliRunner().invoke(cli, ["run", config, "--dry-run", "--judge-samples", "3"])

    assert result.exit_code == 0, result.output
    assert "Judge samples: 3" in result.output


def test_cli_judge_samples_from_config_shows_in_dry_run(tmp_path):
    config = _write_config(tmp_path, judge_block="judge:\n  samples: 5\n")

    result = CliRunner().invoke(cli, ["run", config, "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Judge samples: 5" in result.output


def test_cli_disagreement_gate_requires_multiple_samples(tmp_path):
    config = _write_config(tmp_path)

    result = CliRunner().invoke(cli, ["run", config, "--dry-run", "--fail-on-judge-disagreement"])

    assert result.exit_code == 1
    assert "at least 2 judge samples" in result.output


def test_cli_rejects_judge_samples_above_maximum(tmp_path):
    config = _write_config(tmp_path)

    result = CliRunner().invoke(
        cli, ["run", config, "--dry-run", "--judge-samples", str(MAX_JUDGE_SAMPLES + 1)]
    )

    assert result.exit_code != 0


def test_run_config_accepts_judge_samples_block():
    config = RunConfig(
        golden_set="golden.yaml",
        models=[{"name": "m", "provider": "http", "model": "x", "endpoint": "http://localhost:9"}],
        judge={"samples": 3, "disagreement_range": 1},
    )
    assert config.judge.samples == 3
    assert config.judge.disagreement_range == 1
