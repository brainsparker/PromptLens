"""Tests for cost and latency budgets and the --fail-on-budget CI gate."""

import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import ClassVar, List, Optional

import pytest
from click.testing import CliRunner

from promptlens.budgets import (
    KIND_COST,
    KIND_LATENCY,
    KIND_TOTAL_COST,
    SCOPE_RUN,
    check_case_budget,
    check_run_budget,
    count_case_violations,
    describe_violations,
    resolve_case_limits,
)
from promptlens.cli import cli
from promptlens.exporters.html_exporter import HTMLExporter
from promptlens.exporters.junit_exporter import JUnitXMLExporter
from promptlens.exporters.markdown_exporter import MarkdownExporter
from promptlens.models.config import BudgetConfig, OutputConfig, RunConfig
from promptlens.models.result import (
    BudgetViolation,
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)
from promptlens.models.test_case import TestCase


def _case(**kwargs):
    defaults = {
        "id": "tc-1",
        "query": "What is the answer?",
        "expected_behavior": "Answers correctly",
    }
    defaults.update(kwargs)
    return TestCase(**defaults)


def _response(cost=0.002, latency=1000.0, error=None, model="model-a"):
    return ModelResponse(
        content="The answer is 42." if error is None else "",
        model=model,
        provider="anthropic",
        latency_ms=latency,
        cost_usd=cost,
        error=error,
    )


def _score(score):
    return JudgeScore(
        score=score,
        explanation="Judged.",
        judge_model="judge-model",
        judge_provider="anthropic",
    )


def _eval(test_case_id="tc-1", model="model-a", score=None, violations=None, **resp):
    return EvaluationResult(
        test_case_id=test_case_id,
        query="What is the answer?",
        expected_behavior="Answers correctly",
        model_response=_response(model=model, **resp),
        judge_score=_score(score) if score is not None else None,
        budget_violations=violations or [],
    )


def _violation(kind=KIND_COST, limit=0.001, actual=0.002, message="cost over"):
    return BudgetViolation(kind=kind, limit=limit, actual=actual, message=message)


def _run(results, models=None, run_violations=None, total_cost=None):
    return RunResult(
        run_id="run-1",
        run_name="budget-run",
        timestamp=datetime(2026, 9, 19, 12, 0, 0),
        golden_set_name="golden-set",
        models_tested=models or ["model-a"],
        results=results,
        total_cost_usd=(
            total_cost
            if total_cost is not None
            else sum(r.model_response.cost_usd or 0.0 for r in results)
        ),
        budget_violations=run_violations or [],
    )


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


class TestBudgetModels:
    def test_test_case_accepts_positive_budgets(self):
        case = _case(max_cost_usd=0.01, max_latency_ms=2500)
        assert case.max_cost_usd == 0.01
        assert case.max_latency_ms == 2500

    def test_test_case_budgets_default_to_none(self):
        case = _case()
        assert case.max_cost_usd is None
        assert case.max_latency_ms is None

    @pytest.mark.parametrize("field", ["max_cost_usd", "max_latency_ms"])
    @pytest.mark.parametrize("value", [0, -1, -0.5])
    def test_test_case_rejects_non_positive_budget(self, field, value):
        with pytest.raises(ValueError, match="greater than 0"):
            _case(**{field: value})

    @pytest.mark.parametrize(
        "field", ["max_case_cost_usd", "max_case_latency_ms", "max_total_cost_usd"]
    )
    def test_budget_config_rejects_non_positive(self, field):
        with pytest.raises(ValueError, match="greater than 0"):
            BudgetConfig(**{field: 0})

    def test_budget_config_is_configured(self):
        assert not BudgetConfig().is_configured()
        assert BudgetConfig(max_case_cost_usd=0.01).is_configured()
        assert BudgetConfig(max_total_cost_usd=1.0).is_configured()

    def test_run_config_defaults_budgets(self):
        config = RunConfig(
            golden_set="./tests.yaml",
            models=[{"name": "m", "provider": "anthropic", "model": "x"}],
        )
        assert config.budgets == BudgetConfig()

    def test_run_config_parses_budgets_section(self):
        config = RunConfig(
            golden_set="./tests.yaml",
            models=[{"name": "m", "provider": "anthropic", "model": "x"}],
            budgets={"max_case_cost_usd": 0.02, "max_total_cost_usd": 0.5},
        )
        assert config.budgets.max_case_cost_usd == 0.02
        assert config.budgets.max_case_latency_ms is None
        assert config.budgets.max_total_cost_usd == 0.5

    def test_output_formats_accept_junit(self):
        # README documents junit as an output format; the validator must allow it
        assert OutputConfig(formats=["json", "junit"]).formats == ["json", "junit"]

    def test_run_result_loads_without_budget_fields(self):
        # results.json files written before budgets existed must still load
        data = {
            "run_id": "old",
            "golden_set_name": "gs",
            "models_tested": ["model-a"],
            "results": [
                {
                    "test_case_id": "tc-1",
                    "query": "q",
                    "expected_behavior": "e",
                    "model_response": {
                        "content": "c",
                        "model": "model-a",
                        "provider": "anthropic",
                        "latency_ms": 10.0,
                    },
                }
            ],
        }
        run = RunResult(**data)
        assert run.budget_violations == []
        assert run.results[0].budget_violations == []
        assert not run.has_budget_violations


# ---------------------------------------------------------------------------
# Budget checks
# ---------------------------------------------------------------------------


class TestCaseBudgetChecks:
    def test_no_budgets_means_no_violations(self):
        assert check_case_budget(_case(), _response(), BudgetConfig()) == []

    def test_within_budget(self):
        budgets = BudgetConfig(max_case_cost_usd=0.01, max_case_latency_ms=5000)
        assert check_case_budget(_case(), _response(cost=0.005, latency=1000), budgets) == []

    def test_cost_at_limit_passes(self):
        budgets = BudgetConfig(max_case_cost_usd=0.005)
        assert check_case_budget(_case(), _response(cost=0.005), budgets) == []

    def test_cost_over_config_default(self):
        budgets = BudgetConfig(max_case_cost_usd=0.001)
        violations = check_case_budget(_case(), _response(cost=0.002), budgets)
        assert len(violations) == 1
        v = violations[0]
        assert v.kind == KIND_COST
        assert v.scope == "case"
        assert v.limit == 0.001
        assert v.actual == 0.002
        assert "cost $0.0020 exceeds budget $0.0010" == v.message

    def test_latency_over_config_default(self):
        budgets = BudgetConfig(max_case_latency_ms=500)
        violations = check_case_budget(_case(), _response(latency=1234.6), budgets)
        assert [v.kind for v in violations] == [KIND_LATENCY]
        assert violations[0].message == "latency 1235ms exceeds budget 500ms"

    def test_both_budgets_exceeded_reports_cost_then_latency(self):
        budgets = BudgetConfig(max_case_cost_usd=0.001, max_case_latency_ms=500)
        violations = check_case_budget(_case(), _response(cost=0.002, latency=1000), budgets)
        assert [v.kind for v in violations] == [KIND_COST, KIND_LATENCY]

    def test_case_budget_overrides_config_default(self):
        # Config says 1 cent, case allows 5 cents: a 2 cent response is fine
        budgets = BudgetConfig(max_case_cost_usd=0.01)
        case = _case(max_cost_usd=0.05)
        assert check_case_budget(case, _response(cost=0.02), budgets) == []

        # Config says 5 cents, case tightens to 1 cent: a 2 cent response fails
        budgets = BudgetConfig(max_case_cost_usd=0.05)
        case = _case(max_cost_usd=0.01)
        violations = check_case_budget(case, _response(cost=0.02), budgets)
        assert len(violations) == 1
        assert violations[0].limit == 0.01

    def test_case_budget_applies_without_config_default(self):
        case = _case(max_latency_ms=100)
        violations = check_case_budget(case, _response(latency=250), BudgetConfig())
        assert [v.kind for v in violations] == [KIND_LATENCY]

    def test_resolve_case_limits(self):
        budgets = BudgetConfig(max_case_cost_usd=0.01, max_case_latency_ms=1000)
        assert resolve_case_limits(_case(), budgets) == (0.01, 1000)
        assert resolve_case_limits(_case(max_cost_usd=0.5), budgets) == (0.5, 1000)
        assert resolve_case_limits(_case(max_latency_ms=5), BudgetConfig()) == (None, 5)

    def test_errored_response_is_not_checked(self):
        budgets = BudgetConfig(max_case_cost_usd=0.0001, max_case_latency_ms=1)
        response = _response(cost=1.0, latency=99999, error="API timeout")
        assert check_case_budget(_case(), response, budgets) == []

    def test_unknown_cost_skips_cost_budget_but_keeps_latency(self):
        budgets = BudgetConfig(max_case_cost_usd=0.0001, max_case_latency_ms=100)
        response = _response(cost=None, latency=500)
        violations = check_case_budget(_case(), response, budgets)
        assert [v.kind for v in violations] == [KIND_LATENCY]


class TestRunBudgetChecks:
    def test_no_total_budget(self):
        run = _run([_eval(cost=0.5)])
        assert check_run_budget(run, BudgetConfig()) == []

    def test_total_within_budget(self):
        run = _run([_eval(cost=0.1), _eval(test_case_id="tc-2", cost=0.1)])
        assert check_run_budget(run, BudgetConfig(max_total_cost_usd=0.25)) == []

    def test_total_over_budget(self):
        run = _run([_eval(cost=0.2), _eval(test_case_id="tc-2", cost=0.2)])
        violations = check_run_budget(run, BudgetConfig(max_total_cost_usd=0.25))
        assert len(violations) == 1
        v = violations[0]
        assert v.kind == KIND_TOTAL_COST
        assert v.scope == SCOPE_RUN
        assert v.actual == pytest.approx(0.4)
        assert v.message == "run cost $0.4000 exceeds budget $0.2500"


class TestViolationHelpers:
    def test_describe_lists_case_lines_then_run_lines(self):
        run = _run(
            [
                _eval("tc-1", violations=[_violation(message="cost over")]),
                _eval("tc-2", violations=[]),
                _eval(
                    "tc-3",
                    model="model-b",
                    violations=[_violation(KIND_LATENCY, message="slow")],
                ),
            ],
            models=["model-a", "model-b"],
            run_violations=[
                BudgetViolation(
                    kind=KIND_TOTAL_COST, scope=SCOPE_RUN, limit=1, actual=2, message="run over"
                )
            ],
        )
        assert describe_violations(run) == [
            "tc-1 (model-a): cost over",
            "tc-3 (model-b): slow",
            "run over",
        ]

    def test_count_and_has_violations(self):
        clean = _run([_eval("tc-1"), _eval("tc-2")])
        assert count_case_violations(clean.results) == 0
        assert not clean.has_budget_violations
        assert clean.get_over_budget_results() == []

        dirty = _run([_eval("tc-1", violations=[_violation()]), _eval("tc-2")])
        assert count_case_violations(dirty.results) == 1
        assert dirty.has_budget_violations
        assert [r.test_case_id for r in dirty.get_over_budget_results()] == ["tc-1"]
        assert dirty.results[0].over_budget

    def test_run_level_violation_alone_counts(self):
        run = _run(
            [_eval("tc-1")],
            run_violations=[
                BudgetViolation(
                    kind=KIND_TOTAL_COST, scope=SCOPE_RUN, limit=1, actual=2, message="m"
                )
            ],
        )
        assert run.has_budget_violations


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------


def _junit(run, tmp_path, fail_under=None):
    output = tmp_path / "junit.xml"
    JUnitXMLExporter(fail_under=fail_under).export(run, str(output))
    return ET.parse(str(output)).getroot()


class TestJUnitBudgetMapping:
    def test_budget_violation_is_a_failure(self, tmp_path):
        run = _run([_eval("tc-1", score=5, violations=[_violation(message="cost over")])])
        root = _junit(run, tmp_path)

        assert root.get("failures") == "1"
        failure = root.find("./testsuite/testcase/failure")
        assert failure.get("type") == "BudgetExceeded"
        assert failure.get("message") == "Budget exceeded: cost over"
        assert "Budget: cost over" in failure.text
        assert "Score: 5" in failure.text

    def test_low_score_and_budget_share_one_failure(self, tmp_path):
        run = _run([_eval("tc-1", score=1, violations=[_violation(message="cost over")])])
        root = _junit(run, tmp_path)

        failures = root.findall("./testsuite/testcase/failure")
        assert len(failures) == 1
        assert failures[0].get("type") == "JudgeScoreBelowThreshold,BudgetExceeded"
        assert "below threshold" in failures[0].get("message")
        assert "Budget exceeded" in failures[0].get("message")

    def test_unjudged_but_over_budget_is_failure_not_skipped(self, tmp_path):
        run = _run([_eval("tc-1", violations=[_violation(message="slow")])])
        root = _junit(run, tmp_path)

        assert root.get("skipped") == "0"
        assert root.get("failures") == "1"
        testcase = root.find("./testsuite/testcase")
        assert testcase.find("skipped") is None
        assert testcase.find("failure").get("type") == "BudgetExceeded"

    def test_errored_response_stays_an_error(self, tmp_path):
        run = _run([_eval("tc-1", error="boom", violations=[_violation()])])
        root = _junit(run, tmp_path)
        assert root.get("errors") == "1"
        assert root.get("failures") == "0"

    def test_within_budget_case_is_unchanged(self, tmp_path):
        run = _run([_eval("tc-1", score=5)])
        root = _junit(run, tmp_path)
        assert root.get("failures") == "0"
        props = {
            p.get("name"): p.get("value") for p in root.findall("./testsuite/properties/property")
        }
        assert props["budget_violations"] == "0"

    def test_suite_properties_report_budget_counts_and_run_violations(self, tmp_path):
        run = _run(
            [_eval("tc-1", score=5, violations=[_violation()]), _eval("tc-2", score=5)],
            run_violations=[
                BudgetViolation(
                    kind=KIND_TOTAL_COST, scope=SCOPE_RUN, limit=1, actual=2, message="run over"
                )
            ],
        )
        root = _junit(run, tmp_path)
        props = {
            p.get("name"): p.get("value") for p in root.findall("./testsuite/properties/property")
        }
        assert props["budget_violations"] == "1"
        assert props["run_budget_total_cost"] == "run over"

    def test_system_out_lists_violations(self, tmp_path):
        run = _run([_eval("tc-1", score=5, violations=[_violation(message="cost over")])])
        root = _junit(run, tmp_path)
        system_out = root.find("./testsuite/testcase/system-out")
        assert "budget_violation: cost over" in system_out.text


class TestMarkdownBudgetSection:
    def test_no_section_without_violations(self, tmp_path):
        output = tmp_path / "r.md"
        MarkdownExporter().export(_run([_eval("tc-1", score=5)]), str(output))
        text = output.read_text()
        assert "Budget Violations" not in text
        assert "over budget" not in text

    def test_section_lists_violations(self, tmp_path):
        run = _run(
            [_eval("tc-1", score=5, violations=[_violation(message="cost over")])],
            run_violations=[
                BudgetViolation(
                    kind=KIND_TOTAL_COST, scope=SCOPE_RUN, limit=1, actual=2, message="run over"
                )
            ],
        )
        output = tmp_path / "r.md"
        MarkdownExporter().export(run, str(output))
        text = output.read_text()
        assert "| Budget Violations | 2 |" in text
        assert "## Budget Violations" in text
        assert "- tc-1 (model-a): cost over" in text
        assert "- run over" in text
        assert "(over budget)" in text


class TestHTMLBudgetDisplay:
    def test_html_marks_over_budget_responses(self, tmp_path):
        run = _run(
            [_eval("tc-1", score=5, violations=[_violation(message="cost over")])],
            run_violations=[
                BudgetViolation(
                    kind=KIND_TOTAL_COST, scope=SCOPE_RUN, limit=1, actual=2, message="run over"
                )
            ],
        )
        output = tmp_path / "report.html"
        HTMLExporter().export(run, str(output))
        html = output.read_text()
        assert "over budget" in html
        assert "Budget: cost over" in html
        assert "Run budget exceeded: run over" in html

    def test_html_omits_budget_ui_when_clean(self, tmp_path):
        output = tmp_path / "report.html"
        HTMLExporter().export(_run([_eval("tc-1", score=5)]), str(output))
        html = output.read_text()
        assert "Over budget" not in html
        assert "Run budget exceeded" not in html


# ---------------------------------------------------------------------------
# CLI gate
# ---------------------------------------------------------------------------


class _FakeRunner:
    """Stands in for Runner so CLI tests need no provider keys.

    Records the RunConfig it was built with and returns a canned RunResult,
    running the real run-level budget check so --max-total-cost is exercised.
    """

    captured_config: ClassVar[Optional[RunConfig]] = None
    canned_results: ClassVar[List[EvaluationResult]] = []

    def __init__(self, config):
        _FakeRunner.captured_config = config
        self.config = config
        self.run_id = "fake-run"

    async def run(self):
        from promptlens.budgets import check_run_budget

        run = _run(list(_FakeRunner.canned_results))
        run.budget_violations = check_run_budget(run, self.config.budgets)
        return run


def _write_config(tmp_path: Path, extra: str = "") -> Path:
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        "name: gs\ntest_cases:\n  - id: tc-1\n    query: q\n    expected_behavior: e\n"
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"golden_set: {golden}\n"
        "models:\n  - name: m\n    provider: anthropic\n    model: x\n"
        f"output:\n  directory: {tmp_path / 'results'}\n  formats: [json]\n"
        f"{extra}"
    )
    return config


@pytest.fixture
def fake_runner(monkeypatch):
    monkeypatch.setattr("promptlens.cli.Runner", _FakeRunner)
    _FakeRunner.captured_config = None
    _FakeRunner.canned_results = []
    return _FakeRunner


class TestFailOnBudgetGate:
    def test_violations_without_flag_do_not_change_exit_code(self, tmp_path, fake_runner):
        fake_runner.canned_results = [_eval("tc-1", score=5, violations=[_violation()])]
        result = CliRunner().invoke(cli, ["run", str(_write_config(tmp_path))])
        assert result.exit_code == 0, result.output

    def test_flag_exits_2_on_case_violation(self, tmp_path, fake_runner):
        fake_runner.canned_results = [
            _eval("tc-1", score=5, violations=[_violation(message="cost over")])
        ]
        result = CliRunner().invoke(cli, ["run", str(_write_config(tmp_path)), "--fail-on-budget"])
        assert result.exit_code == 2, result.output
        assert "Budget gate failed" in result.output
        assert "tc-1 (model-a): cost over" in result.output

    def test_flag_passes_when_within_configured_budget(self, tmp_path, fake_runner):
        fake_runner.canned_results = [_eval("tc-1", score=5, cost=0.001)]
        config = _write_config(tmp_path, "budgets:\n  max_case_cost_usd: 0.01\n")
        result = CliRunner().invoke(cli, ["run", str(config), "--fail-on-budget"])
        assert result.exit_code == 0, result.output
        assert "Budget gate passed" in result.output

    def test_flag_warns_when_nothing_is_budgeted(self, tmp_path, fake_runner):
        fake_runner.canned_results = [_eval("tc-1", score=5)]
        result = CliRunner().invoke(cli, ["run", str(_write_config(tmp_path)), "--fail-on-budget"])
        assert result.exit_code == 0, result.output
        assert "no budgets are configured" in result.output

    def test_max_total_cost_overrides_config_and_implies_gate(self, tmp_path, fake_runner):
        fake_runner.canned_results = [_eval("tc-1", score=5, cost=0.3)]
        config = _write_config(tmp_path, "budgets:\n  max_total_cost_usd: 5.0\n")
        result = CliRunner().invoke(cli, ["run", str(config), "--max-total-cost", "0.25"])
        assert result.exit_code == 2, result.output
        assert fake_runner.captured_config.budgets.max_total_cost_usd == 0.25
        assert "run cost $0.3000 exceeds budget $0.2500" in result.output

    def test_max_total_cost_rejects_zero(self, tmp_path, fake_runner):
        result = CliRunner().invoke(
            cli, ["run", str(_write_config(tmp_path)), "--max-total-cost", "0"]
        )
        assert result.exit_code == 2
        assert "Invalid value" in result.output

    def test_both_gates_are_reported_before_exit(self, tmp_path, fake_runner):
        fake_runner.canned_results = [
            _eval("tc-1", score=1, violations=[_violation(message="cost over")])
        ]
        result = CliRunner().invoke(
            cli,
            ["run", str(_write_config(tmp_path)), "--fail-under", "3", "--fail-on-budget"],
        )
        assert result.exit_code == 2, result.output
        assert "Quality gate failed" in result.output
        assert "Budget gate failed" in result.output

    def test_violations_are_persisted_in_results_json(self, tmp_path, fake_runner):
        fake_runner.canned_results = [
            _eval("tc-1", score=5, violations=[_violation(message="cost over")])
        ]
        result = CliRunner().invoke(cli, ["run", str(_write_config(tmp_path))])
        assert result.exit_code == 0, result.output
        # The output directory is named after the RunResult's run_id
        data = json.loads((tmp_path / "results" / "run-1" / "results.json").read_text())
        assert data["results"][0]["budget_violations"][0]["message"] == "cost over"
        reloaded = RunResult(**data)
        assert reloaded.has_budget_violations
