"""Tests for how assertions surface in exporters, the CI gate, and compare."""

import csv
import xml.etree.ElementTree as ET
from datetime import datetime

from promptlens.cli import _check_assertion_failures
from promptlens.comparison import compare_runs
from promptlens.exporters.csv_exporter import CSVExporter
from promptlens.exporters.html_exporter import HTMLExporter
from promptlens.exporters.junit_exporter import JUnitXMLExporter
from promptlens.exporters.markdown_exporter import MarkdownExporter
from promptlens.models.assertions import AssertionResult
from promptlens.models.comparison import (
    REASON_ASSERTIONS_FIXED,
    REASON_ASSERTIONS_NEWLY_FAILING,
    REASON_ASSERTIONS_STILL_FAILING,
    REASON_SCORE_MISSING,
    REASON_WITHIN_THRESHOLD,
    STATUS_IMPROVED,
    STATUS_REGRESSED,
    STATUS_UNCHANGED,
)
from promptlens.models.result import (
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)

PASS = AssertionResult(type="contains", name="contains '30 days'", passed=True, message="ok")
FAIL = AssertionResult(
    type="not_contains",
    name="not_contains 'As an AI'",
    passed=False,
    message="found forbidden text: 'As an AI'",
)


def _response(model="model-a", error=None):
    return ModelResponse(
        content="As an AI, refunds are accepted within 30 days.",
        model=model,
        provider="anthropic",
        latency_ms=900.0,
        cost_usd=0.003,
        error=error,
    )


def _score(score):
    return JudgeScore(
        score=score, explanation="judged", judge_model="j", judge_provider="anthropic"
    )


def _eval(case_id, model="model-a", score=None, assertions=None, skipped=None, error=None):
    return EvaluationResult(
        test_case_id=case_id,
        query="Refund policy?",
        expected_behavior="Mentions 30 days without AI disclaimers",
        model_response=_response(model=model, error=error),
        judge_score=_score(score) if score is not None else None,
        assertion_results=assertions or [],
        judge_skipped_reason=skipped,
    )


def _run(results, run_id="run-1", models=None):
    return RunResult(
        run_id=run_id,
        run_name="assertions",
        timestamp=datetime(2026, 9, 30, 9, 0, 0),
        golden_set_name="golden",
        models_tested=models or ["model-a"],
        results=results,
    )


class TestJUnitAssertions:
    def _export(self, run, tmp_path, fail_under=None):
        out = tmp_path / "junit.xml"
        JUnitXMLExporter(fail_under=fail_under).export(run, str(out))
        return ET.parse(str(out)).getroot()

    def test_failed_assertion_is_a_failure_even_without_judge(self, tmp_path):
        run = _run([_eval("tc-1", assertions=[PASS, FAIL], skipped="skipped: failed")])
        root = self._export(run, tmp_path)
        case = root.find("./testsuite/testcase")
        failure = case.find("failure")

        assert root.get("failures") == "1"
        assert root.get("skipped") == "0"
        assert failure is not None
        assert failure.get("type") == "AssertionFailed"
        assert "not_contains 'As an AI'" in failure.get("message")
        assert "FAILED not_contains" in failure.text
        assert "assertion FAIL" in case.find("system-out").text
        assert "judge: skipped: failed" in case.find("system-out").text

    def test_failed_assertion_beats_a_high_judge_score(self, tmp_path):
        run = _run([_eval("tc-1", score=5, assertions=[FAIL])])
        root = self._export(run, tmp_path, fail_under=3.0)
        failure = root.find("./testsuite/testcase/failure")
        assert failure is not None
        assert failure.get("type") == "AssertionFailed"
        assert "Judge score: 5" in failure.text

    def test_passing_assertions_with_skipped_judge_count_as_pass(self, tmp_path):
        run = _run([_eval("tc-1", assertions=[PASS])])
        root = self._export(run, tmp_path)
        case = root.find("./testsuite/testcase")
        assert case.find("failure") is None
        assert case.find("skipped") is None
        assert root.get("skipped") == "0"

    def test_unjudged_case_without_assertions_is_still_skipped(self, tmp_path):
        run = _run([_eval("tc-1")])
        root = self._export(run, tmp_path)
        assert root.get("skipped") == "1"

    def test_suite_properties_include_assertion_counts(self, tmp_path):
        run = _run([_eval("tc-1", assertions=[PASS]), _eval("tc-2", assertions=[PASS, FAIL])])
        root = self._export(run, tmp_path)
        props = {
            p.get("name"): p.get("value") for p in root.findall("./testsuite/properties/property")
        }
        assert props["assertions_passed_cases"] == "1/2"
        assert props["assertions_failed"] == "1/3"


class TestReportExporters:
    def test_html_shows_assertions(self, tmp_path):
        run = _run([_eval("tc-1", score=4, assertions=[PASS, FAIL])])
        out = tmp_path / "report.html"
        HTMLExporter().export(run, str(out))
        html = out.read_text(encoding="utf-8")

        assert "assertions fail" in html
        assert "found forbidden text" in html
        assert "contains &#39;30 days&#39;" in html or "contains '30 days'" in html
        # Summary card and comparison column render when assertions exist
        assert ">Assertions<" in html
        assert "0/1" in html

    def test_html_hides_assertion_chrome_when_none_declared(self, tmp_path):
        run = _run([_eval("tc-1", score=4)])
        out = tmp_path / "report.html"
        HTMLExporter().export(run, str(out))
        html = out.read_text(encoding="utf-8")
        assert ">Assertions<" not in html
        assert "assertions pass" not in html

    def test_csv_has_assertion_columns(self, tmp_path):
        run = _run([_eval("tc-1", score=4, assertions=[FAIL]), _eval("tc-2", score=4)])
        out = tmp_path / "results.csv"
        CSVExporter().export(run, str(out))
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["assertions_passed"] == "false"
        assert "As an AI" in rows[0]["assertions_failed"]
        assert rows[1]["assertions_passed"] == ""
        assert rows[1]["assertions_failed"] == ""

    def test_markdown_lists_failed_assertions(self, tmp_path):
        run = _run([_eval("tc-1", score=4, assertions=[PASS, FAIL])])
        out = tmp_path / "results.md"
        MarkdownExporter().export(run, str(out))
        md = out.read_text(encoding="utf-8")

        assert "| Assertions |" in md
        assert "FAIL (1/2)" in md
        assert "failed `not_contains 'As an AI'`" in md
        assert "0/1 cases passed" in md


class TestGateHelper:
    def test_collects_only_failed_cases(self):
        run = _run(
            [
                _eval("ok", assertions=[PASS]),
                _eval("bad", assertions=[PASS, FAIL]),
                _eval("plain", score=3),
            ]
        )
        failures = _check_assertion_failures(run)
        assert len(failures) == 1
        case_id, model, failed = failures[0]
        assert (case_id, model) == ("bad", "model-a")
        assert [a.name for a in failed] == ["not_contains 'As an AI'"]

    def test_empty_when_everything_passes(self):
        run = _run([_eval("ok", assertions=[PASS]), _eval("plain", score=4)])
        assert _check_assertion_failures(run) == []


class TestCompareWithAssertions:
    def _compare(self, baseline_eval, candidate_eval):
        baseline = _run([baseline_eval], run_id="base")
        candidate = _run([candidate_eval], run_id="cand")
        comparison = compare_runs(baseline, candidate)
        assert len(comparison.cases) == 1
        return comparison.cases[0]

    def test_newly_failing_assertions_regress_even_with_higher_score(self):
        case = self._compare(
            _eval("tc", score=3, assertions=[PASS]),
            _eval("tc", score=5, assertions=[FAIL]),
        )
        assert case.status == STATUS_REGRESSED
        assert case.reason == REASON_ASSERTIONS_NEWLY_FAILING
        assert case.baseline_assertions_passed is True
        assert case.candidate_assertions_passed is False

    def test_newly_added_assertions_that_fail_regress(self):
        case = self._compare(_eval("tc", score=4), _eval("tc", score=4, assertions=[FAIL]))
        assert case.status == STATUS_REGRESSED
        assert case.reason == REASON_ASSERTIONS_NEWLY_FAILING

    def test_fixed_assertions_improve(self):
        case = self._compare(
            _eval("tc", assertions=[FAIL], skipped="skipped"),
            _eval("tc", score=4, assertions=[PASS]),
        )
        assert case.status == STATUS_IMPROVED
        assert case.reason == REASON_ASSERTIONS_FIXED

    def test_still_failing_assertions_are_unchanged(self):
        case = self._compare(
            _eval("tc", assertions=[FAIL], skipped="skipped"),
            _eval("tc", assertions=[FAIL], skipped="skipped"),
        )
        assert case.status == STATUS_UNCHANGED
        assert case.reason == REASON_ASSERTIONS_STILL_FAILING

    def test_score_rules_still_apply_when_assertions_pass_on_both_sides(self):
        case = self._compare(
            _eval("tc", score=5, assertions=[PASS]),
            _eval("tc", score=3, assertions=[PASS]),
        )
        assert case.status == STATUS_REGRESSED
        assert case.score_delta == -2

    def test_deliberately_skipped_judge_with_passing_assertions_is_not_a_regression(self):
        case = self._compare(
            _eval("tc", score=4, assertions=[PASS]),
            _eval("tc", assertions=[PASS], skipped="skipped: judge disabled"),
        )
        assert case.status == STATUS_UNCHANGED
        assert case.reason == REASON_WITHIN_THRESHOLD

    def test_lost_score_without_a_deliberate_skip_is_still_a_regression(self):
        case = self._compare(
            _eval("tc", score=4, assertions=[PASS]),
            _eval("tc", assertions=[PASS]),
        )
        assert case.status == STATUS_REGRESSED
        assert case.reason == REASON_SCORE_MISSING

    def test_assertion_only_runs_compare_as_unchanged(self):
        case = self._compare(_eval("tc", assertions=[PASS]), _eval("tc", assertions=[PASS]))
        assert case.status == STATUS_UNCHANGED
        assert case.reason == REASON_WITHIN_THRESHOLD
