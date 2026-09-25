"""Assertion results in every exporter plus the --fail-on-assertions gate."""

import csv
import json
import xml.etree.ElementTree as ET
from datetime import datetime

from promptlens.cli import _check_assertion_failures
from promptlens.exporters.csv_exporter import CSVExporter
from promptlens.exporters.html_exporter import HTMLExporter
from promptlens.exporters.json_exporter import JSONExporter
from promptlens.exporters.junit_exporter import JUnitXMLExporter
from promptlens.exporters.markdown_exporter import MarkdownExporter
from promptlens.models.assertions import AssertionResult
from promptlens.models.result import (
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)


def _response(content="Paris.", model="model-a"):
    return ModelResponse(
        content=content, model=model, provider="anthropic", latency_ms=100.0, cost_usd=0.001
    )


def _score(score=4):
    return JudgeScore(
        score=score, explanation="ok", judge_model="judge", judge_provider="anthropic"
    )


def _check(passed, label="contains: Paris", message=None):
    return AssertionResult(
        type="contains",
        label=label,
        passed=passed,
        message=message or ("found" if passed else "'Paris' not found in response"),
    )


def _eval(test_case_id, checks=(), score=None, skipped=None, model="model-a"):
    return EvaluationResult(
        test_case_id=test_case_id,
        query="Capital of France?",
        expected_behavior="Paris",
        model_response=_response(model=model),
        judge_score=_score(score) if score is not None else None,
        assertion_results=list(checks),
        judge_skipped_reason=skipped,
    )


def _run(results, models=None):
    return RunResult(
        run_id="run-1",
        run_name="assertions",
        timestamp=datetime(2026, 9, 25, 9, 0, 0),
        golden_set_name="golden",
        models_tested=models or ["model-a"],
        results=results,
    )


MIXED = [
    _eval("pass-judged", [_check(True)], score=5),
    _eval("fail-judged", [_check(True), _check(False, "max_chars: 5", "12 chars (max 5)")], score=4),
    _eval("pass-only", [_check(True)], skipped="assertions_only"),
    _eval("skipped-fail", [_check(False)], skipped="assertion_failure"),
    _eval("plain", [], score=3),
    _eval("unjudged-plain", []),
]


class TestJUnit:
    def _root(self, tmp_path, results=MIXED):
        out = tmp_path / "junit.xml"
        JUnitXMLExporter().export(_run(results), str(out))
        return ET.parse(str(out)).getroot()

    def _case(self, root, name):
        return next(tc for tc in root.iter("testcase") if tc.get("name") == name)

    def test_assertion_failure_is_failure_even_with_good_score(self, tmp_path):
        root = self._root(tmp_path)
        failure = self._case(root, "fail-judged").find("failure")
        assert failure is not None
        assert failure.get("type") == "AssertionFailed"
        assert "1 of 2 assertion(s) failed" in failure.get("message")
        assert "FAILED max_chars: 5: 12 chars (max 5)" in failure.text
        assert "Judge score: 4" in failure.text

    def test_assertions_only_pass_is_a_pass_not_skipped(self, tmp_path):
        root = self._root(tmp_path)
        case = self._case(root, "pass-only")
        assert case.find("skipped") is None
        assert case.find("failure") is None
        assert "judge_skipped: assertions_only" in case.find("system-out").text
        assert "assertion PASS contains: Paris" in case.find("system-out").text

    def test_skipped_judge_after_failure_is_failure(self, tmp_path):
        root = self._root(tmp_path)
        assert self._case(root, "skipped-fail").find("failure").get("type") == "AssertionFailed"

    def test_unjudged_without_assertions_still_skipped(self, tmp_path):
        root = self._root(tmp_path)
        assert self._case(root, "unjudged-plain").find("skipped") is not None

    def test_suite_counts_and_pass_rate_property(self, tmp_path):
        root = self._root(tmp_path)
        suite = root.find("testsuite")
        assert suite.get("tests") == "6"
        assert suite.get("failures") == "2"
        assert suite.get("skipped") == "1"
        props = {p.get("name"): p.get("value") for p in suite.iter("property")}
        # 4 asserted results, 2 passed
        assert props["assertion_pass_rate"] == "0.5000"


class TestTabularExporters:
    def test_csv_columns(self, tmp_path):
        out = tmp_path / "r.csv"
        CSVExporter().export(_run(MIXED), str(out))
        with open(out, newline="", encoding="utf-8") as f:
            rows = {row["test_case_id"]: row for row in csv.DictReader(f)}
        assert rows["pass-judged"]["assertions_passed"] == "true"
        assert rows["fail-judged"]["assertions_passed"] == "false"
        assert "max_chars: 5: 12 chars (max 5)" in rows["fail-judged"]["assertions_failed"]
        assert rows["plain"]["assertions_passed"] == ""
        assert rows["plain"]["assertions_failed"] == ""

    def test_markdown_checks_column_and_failure_details(self, tmp_path):
        out = tmp_path / "r.md"
        MarkdownExporter().export(_run(MIXED), str(out))
        text = out.read_text(encoding="utf-8")
        assert "| Assertions Passed | 50% |" in text
        assert "| Checks |" in text
        assert "1/2 passed" in text
        assert "Failed checks for `model-a`:" in text
        assert "- max_chars: 5: 12 chars (max 5)" in text

    def test_markdown_without_assertions_keeps_old_table(self, tmp_path):
        out = tmp_path / "r.md"
        MarkdownExporter().export(_run([_eval("plain", [], score=3)]), str(out))
        text = out.read_text(encoding="utf-8")
        assert "| Checks |" not in text
        assert "Assertions Passed" not in text

    def test_json_includes_assertion_fields(self, tmp_path):
        out = tmp_path / "r.json"
        JSONExporter().export(_run(MIXED), str(out))
        data = json.loads(out.read_text(encoding="utf-8"))
        by_id = {r["test_case_id"]: r for r in data["results"]}
        assert by_id["fail-judged"]["assertions_passed"] is False
        assert by_id["fail-judged"]["assertion_results"][1]["passed"] is False
        assert by_id["pass-only"]["judge_skipped_reason"] == "assertions_only"
        assert by_id["plain"]["assertions_passed"] is None


class TestHTML:
    def test_report_renders_checks(self, tmp_path):
        out = tmp_path / "report.html"
        HTMLExporter().export(_run(MIXED), str(out))
        html = out.read_text(encoding="utf-8")
        assert "<th>Assertions</th>" in html
        assert "checks fail" in html and "checks pass" in html
        assert "12 chars (max 5)" in html
        assert "Judge not run: test case is assertions_only" in html
        assert "Judge skipped: a deterministic assertion failed" in html

    def test_report_without_assertions_has_no_column(self, tmp_path):
        out = tmp_path / "report.html"
        HTMLExporter().export(_run([_eval("plain", [], score=3)]), str(out))
        html = out.read_text(encoding="utf-8")
        assert "<th>Assertions</th>" not in html


class TestOutputFormats:
    def test_junit_is_an_accepted_output_format(self):
        # The README documents `junit` in output.formats and the CLI exports it,
        # but the config validator used to reject it.
        from promptlens.models.config import OutputConfig

        assert OutputConfig(formats=["html", "JUnit"]).formats == ["html", "junit"]


class TestGate:
    def test_check_assertion_failures(self):
        failing = _check_assertion_failures(_run(MIXED))
        assert sorted(r.test_case_id for r in failing) == ["fail-judged", "skipped-fail"]

    def test_no_failures(self):
        assert _check_assertion_failures(_run([_eval("plain", [], score=3)])) == []
