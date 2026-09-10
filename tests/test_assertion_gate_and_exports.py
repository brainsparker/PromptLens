"""Tests for the --fail-on-assertion gate, runner wiring, and exporter output
of deterministic assertion results."""

import asyncio
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional

from promptlens.cli import _check_assertion_failures
from promptlens.exporters.csv_exporter import CSVExporter
from promptlens.exporters.html_exporter import HTMLExporter
from promptlens.exporters.junit_exporter import JUnitXMLExporter
from promptlens.exporters.markdown_exporter import MarkdownExporter
from promptlens.models.config import RunConfig
from promptlens.models.result import (
    AssertionResult,
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)
from promptlens.providers.base import BaseProvider
from promptlens.runners import runner as runner_module


def _response(content="Refunds within 30 days.", model="model-a", error=None):
    return ModelResponse(
        content=content, model=model, provider="fake", latency_ms=10.0, cost_usd=0.0, error=error
    )


def _score(score, assertions: Optional[List[AssertionResult]] = None):
    return JudgeScore(
        score=score,
        explanation="explained",
        judge_model="judge",
        judge_provider="fake",
        assertion_results=assertions or [],
    )


def _eval(test_case_id, judge_score=None, model="model-a", error=None):
    return EvaluationResult(
        test_case_id=test_case_id,
        query="What is the refund window?",
        expected_behavior="Mentions 30 days",
        model_response=_response(model=model, error=error),
        judge_score=judge_score,
    )


def _run(results, models=None):
    return RunResult(
        run_id="run-1",
        run_name="assertions",
        timestamp=datetime(2026, 9, 10, 12, 0, 0),
        golden_set_name="golden",
        models_tested=models or ["model-a"],
        results=results,
    )


PASS = AssertionResult(type="contains", label="contains '30 days'", passed=True, detail="matched")
FAIL = AssertionResult(type="is_json", label="is_json", passed=False, detail="invalid JSON: x")


class TestAssertionGate:
    def test_no_failures_when_all_pass(self):
        run = _run([_eval("tc-1", _score(5, [PASS])), _eval("tc-2", _score(4))])
        assert _check_assertion_failures(run) == []

    def test_collects_failed_assertions_per_evaluation(self):
        run = _run(
            [
                _eval("tc-1", _score(5, [PASS])),
                _eval("tc-2", _score(3, [PASS, FAIL])),
                _eval("tc-3", None),
                _eval("tc-4", _score(1, [FAIL]), model="model-b"),
            ],
            models=["model-a", "model-b"],
        )
        failures = _check_assertion_failures(run)
        assert [(m, tc) for m, tc, _ in failures] == [("model-a", "tc-2"), ("model-b", "tc-4")]
        assert [a.type for a in failures[0][2]] == ["is_json"]

    def test_run_result_assertion_summary(self):
        run = _run(
            [
                _eval("tc-1", _score(5, [PASS, PASS])),
                _eval("tc-2", _score(3, [PASS, FAIL])),
                _eval("tc-3", _score(4)),
            ]
        )
        summary = run.get_assertion_summary("model-a")
        assert summary == {"passed": 3, "failed": 1, "total": 4, "cases_failed": 1}
        assert run.get_assertion_summary("missing")["total"] == 0


class TestJUnitAssertionFailures:
    def _export(self, run, tmp_path, fail_under=None):
        output = tmp_path / "junit.xml"
        JUnitXMLExporter(fail_under=fail_under).export(run, str(output))
        return ET.parse(str(output)).getroot()

    def test_failed_assertion_is_failure_even_with_high_score(self, tmp_path):
        run = _run([_eval("tc-1", _score(5, [PASS, FAIL]))])
        root = self._export(run, tmp_path)
        failure = root.find("./testsuite/testcase/failure")
        assert failure is not None
        assert failure.get("type") == "AssertionFailed"
        assert "1 of 2" in failure.get("message")
        assert "FAIL is_json" in failure.text
        assert root.get("failures") == "1"

    def test_passing_assertions_do_not_fail(self, tmp_path):
        run = _run([_eval("tc-1", _score(5, [PASS]))])
        root = self._export(run, tmp_path)
        assert root.find("./testsuite/testcase/failure") is None
        system_out = root.find("./testsuite/testcase/system-out").text
        assert "assertions: 1/1 passed" in system_out

    def test_suite_properties_include_assertion_counts(self, tmp_path):
        run = _run([_eval("tc-1", _score(5, [PASS, FAIL]))])
        root = self._export(run, tmp_path)
        props = {
            p.get("name"): p.get("value")
            for p in root.findall("./testsuite/properties/property")
        }
        assert props["assertions_passed"] == "1"
        assert props["assertions_total"] == "2"

    def test_low_score_still_fails_without_assertions(self, tmp_path):
        run = _run([_eval("tc-1", _score(1))])
        root = self._export(run, tmp_path)
        failure = root.find("./testsuite/testcase/failure")
        assert failure.get("type") == "JudgeScoreBelowThreshold"


class TestOtherExporters:
    def test_csv_has_assertion_columns(self, tmp_path):
        run = _run([_eval("tc-1", _score(3, [PASS, FAIL])), _eval("tc-2", _score(4))])
        output = tmp_path / "results.csv"
        CSVExporter().export(run, str(output))
        with open(output, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["assertions_passed"] == "1"
        assert rows[0]["assertions_total"] == "2"
        assert rows[0]["failed_assertions"] == "is_json"
        assert rows[1]["assertions_total"] == "0"
        assert rows[1]["failed_assertions"] == ""

    def test_markdown_shows_assertion_column(self, tmp_path):
        run = _run([_eval("tc-1", _score(3, [PASS, FAIL])), _eval("tc-2", _score(4))])
        output = tmp_path / "results.md"
        MarkdownExporter().export(run, str(output))
        text = output.read_text(encoding="utf-8")
        assert "| Assertions |" in text
        assert "| 1/2 |" in text
        assert "| - |" in text

    def test_html_renders_assertion_list(self, tmp_path):
        run = _run([_eval("tc-1", _score(3, [PASS, FAIL]))])
        output = tmp_path / "report.html"
        HTMLExporter().export(run, str(output))
        html = output.read_text(encoding="utf-8")
        assert 'class="assertions"' in html
        assert "FAIL" in html and "is_json" in html
        assert "PASS" in html and "contains" in html


class _FakeProvider(BaseProvider):
    """Provider stub that returns canned content without any network calls."""

    def __init__(self, config, content):
        self.config = config
        self._content = content

    async def generate(self, prompt, tools=None, **kwargs):
        return ModelResponse(
            content=self._content,
            model=self.config.model,
            provider="fake",
            latency_ms=1.0,
            cost_usd=0.0,
        )

    def estimate_cost(self, *args, **kwargs):
        return 0.0

    @property
    def provider_name(self):
        return "fake"


def _write_golden_set(path, cases):
    import yaml

    data = {"name": "Assertions", "version": "1.0", "test_cases": cases}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


class TestRunnerWiring:
    def _config(self, golden_path, judge):
        return RunConfig(
            golden_set=str(golden_path),
            models=[{"name": "Fake", "provider": "http", "model": "fake-model", "endpoint": "http://localhost:1"}],
            judge=judge,
            output={"formats": ["json"]},
        )

    def _patch_provider(self, monkeypatch, content):
        from promptlens.models.config import ProviderConfig

        def fake_get_provider(model_config):
            return _FakeProvider(ProviderConfig(name="fake", model=model_config.model), content)

        monkeypatch.setattr(runner_module, "get_provider", fake_get_provider)

    @staticmethod
    def _run(config):
        # Same shape as the CLI: construct the Runner outside any event loop,
        # then asyncio.run() its coroutine.
        runner = runner_module.Runner(config)
        return asyncio.run(runner.run())

    def test_deterministic_judge_scores_without_api_key(self, tmp_path, monkeypatch):
        golden = tmp_path / "golden.yaml"
        _write_golden_set(
            golden,
            [
                {
                    "id": "tc-1",
                    "query": "Refund window?",
                    "expected_behavior": "Mentions 30 days",
                    "assertions": [
                        {"type": "contains", "value": "30 days"},
                        {"type": "not_contains", "value": "I don't know"},
                    ],
                },
                {
                    "id": "tc-2",
                    "query": "No assertions here",
                    "expected_behavior": "Anything",
                },
            ],
        )
        self._patch_provider(monkeypatch, "Refunds are accepted within 30 days.")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        result = self._run(self._config(golden, {"type": "deterministic"}))

        by_id = {r.test_case_id: r for r in result.results}
        assert by_id["tc-1"].judge_score.score == 5
        assert by_id["tc-1"].judge_score.judge_model == "deterministic"
        assert len(by_id["tc-1"].judge_score.assertion_results) == 2
        # Cases without assertions are left unscored, not given a placeholder
        assert by_id["tc-2"].judge_score is None
        assert result.total_cost_usd == 0.0

    def test_llm_judge_records_assertions_alongside_score(self, tmp_path, monkeypatch):
        golden = tmp_path / "golden.yaml"
        _write_golden_set(
            golden,
            [
                {
                    "id": "tc-1",
                    "query": "Refund window?",
                    "expected_behavior": "Mentions 30 days",
                    "assertions": [{"type": "is_json"}],
                }
            ],
        )
        self._patch_provider(monkeypatch, "Refunds are accepted within 30 days.")

        class _FakeLLMJudge:
            def can_evaluate(self, test_case):
                return True

            async def evaluate(self, test_case, model_response):
                return _score(5)

        monkeypatch.setattr(runner_module, "get_judge", lambda config: _FakeLLMJudge())

        result = self._run(self._config(golden, {"type": "llm"}))

        score = result.results[0].judge_score
        assert score.score == 5
        assert [a.type for a in score.assertion_results] == ["is_json"]
        assert not score.assertions_passed
        assert _check_assertion_failures(result)[0][1] == "tc-1"
