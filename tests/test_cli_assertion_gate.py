"""End-to-end tests for the run command's assertion-aware quality gate."""

from datetime import datetime

import yaml
from click.testing import CliRunner

from promptlens import cli as cli_module
from promptlens.models.assertions import AssertionResult
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


def _eval(case_id, score=None, assertions=None, skipped=None):
    return EvaluationResult(
        test_case_id=case_id,
        query="Refund policy?",
        expected_behavior="Mentions 30 days",
        model_response=ModelResponse(
            content="As an AI, refunds are accepted within 30 days.",
            model="model-a",
            provider="anthropic",
            latency_ms=500.0,
            cost_usd=0.001,
        ),
        judge_score=(
            JudgeScore(score=score, explanation="judged", judge_model="j", judge_provider="a")
            if score is not None
            else None
        ),
        assertion_results=assertions or [],
        judge_skipped_reason=skipped,
    )


def _write_config(tmp_path):
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        yaml.safe_dump(
            {
                "name": "gate",
                "test_cases": [
                    {"id": "tc-1", "query": "q", "expected_behavior": "e"},
                ],
            }
        )
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "golden_set": str(golden),
                "models": [{"name": "A", "provider": "anthropic", "model": "model-a"}],
                "output": {"directory": str(tmp_path / "out"), "formats": ["json", "junit"]},
            }
        )
    )
    return config


def _install_fake_runner(monkeypatch, results):
    run_result = RunResult(
        run_id="run-gate",
        run_name="gate",
        timestamp=datetime(2026, 9, 30, 9, 0, 0),
        golden_set_name="gate",
        models_tested=["model-a"],
        results=results,
    )

    class FakeRunner:
        def __init__(self, _config):
            pass

        async def run(self):
            return run_result

    monkeypatch.setattr(cli_module, "Runner", FakeRunner)
    return run_result


def _invoke(config, *args):
    return CliRunner().invoke(cli_module.cli, ["run", str(config), *args])


def test_fail_on_assertions_exits_2_when_an_assertion_fails(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    _install_fake_runner(monkeypatch, [_eval("tc-1", score=5, assertions=[FAIL])])

    result = _invoke(config, "--fail-on-assertions")

    assert result.exit_code == 2, result.output
    assert "Quality gate failed" in result.output
    assert "not_contains 'As an AI'" in result.output


def test_fail_on_assertions_passes_when_assertions_hold(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    _install_fake_runner(
        monkeypatch,
        [_eval("tc-1", assertions=[PASS], skipped="skipped"), _eval("tc-2", score=2)],
    )

    result = _invoke(config, "--fail-on-assertions")

    # A low judge score does not trip the assertion-only gate.
    assert result.exit_code == 0, result.output
    assert "Quality gate passed (--fail-on-assertions)" in result.output


def test_fail_under_also_fails_on_assertion_failures(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    # Average score is a healthy 5.0, but one hard check failed.
    _install_fake_runner(monkeypatch, [_eval("tc-1", score=5, assertions=[FAIL])])

    result = _invoke(config, "--fail-under", "3.0")

    assert result.exit_code == 2, result.output
    assert "failed assertions" in result.output


def test_fail_under_reports_both_failure_kinds(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    _install_fake_runner(
        monkeypatch,
        [_eval("tc-1", score=1, assertions=[FAIL]), _eval("tc-2", score=2)],
    )

    result = _invoke(config, "--fail-under", "3.0")

    assert result.exit_code == 2, result.output
    assert "evaluation(s) failed assertions" in result.output
    assert "average judge score" in result.output


def test_no_gate_flags_never_changes_exit_code(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    _install_fake_runner(monkeypatch, [_eval("tc-1", score=1, assertions=[FAIL])])

    result = _invoke(config)

    assert result.exit_code == 0, result.output
    assert "Quality gate" not in result.output


def test_junit_export_from_run_marks_assertion_failure(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    _install_fake_runner(monkeypatch, [_eval("tc-1", score=5, assertions=[FAIL])])

    result = _invoke(config, "--fail-on-assertions")
    assert result.exit_code == 2

    junit = (tmp_path / "out" / "run-gate" / "junit.xml").read_text(encoding="utf-8")
    assert 'type="AssertionFailed"' in junit


def test_validate_reports_assertion_counts(tmp_path):
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        yaml.safe_dump(
            {
                "name": "with-assertions",
                "test_cases": [
                    {
                        "id": "a",
                        "query": "q",
                        "expected_behavior": "e",
                        "assertions": [{"type": "is_json"}, {"type": "max_chars", "value": 10}],
                    },
                    {"id": "b", "query": "q", "expected_behavior": "e"},
                ],
            }
        )
    )

    result = CliRunner().invoke(cli_module.cli, ["validate", str(golden)])

    assert result.exit_code == 0, result.output
    assert "Assertions: 2 across 1 test case(s)" in result.output
    assert "- a (2 assertions)" in result.output


def test_validate_rejects_bad_assertion(tmp_path):
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        yaml.safe_dump(
            {
                "name": "broken",
                "test_cases": [
                    {
                        "id": "a",
                        "query": "q",
                        "expected_behavior": "e",
                        "assertions": [{"type": "regex", "value": "("}],
                    }
                ],
            }
        )
    )

    result = CliRunner().invoke(cli_module.cli, ["validate", str(golden)])

    assert result.exit_code == 1
    assert "invalid pattern" in result.output
