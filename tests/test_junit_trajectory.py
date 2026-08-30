"""Tests for trajectory assertion reporting in the JUnit exporter."""

import xml.etree.ElementTree as ET
from datetime import datetime


from promptlens.evaluators.trajectory import evaluate_trajectory
from promptlens.exporters.junit_exporter import JUnitXMLExporter
from promptlens.models.result import (
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)
from promptlens.models.tools import ToolCall
from promptlens.models.trajectory import TrajectorySpec


def _make_response(model="model-a", tool_calls=None, error=None):
    return ModelResponse(
        content="Done.",
        model=model,
        provider="anthropic",
        latency_ms=100.0,
        cost_usd=0.001,
        tokens_used=50,
        error=error,
        tool_calls=tool_calls or [],
    )


def _make_score(score):
    return JudgeScore(
        score=score,
        explanation="Judged.",
        judge_model="judge-model",
        judge_provider="anthropic",
    )


def _make_eval(test_case_id, spec=None, tool_calls=None, score=None, model="model-a"):
    response = _make_response(model=model, tool_calls=tool_calls)
    trajectory_evaluation = (
        evaluate_trajectory(spec, response.tool_calls) if spec is not None else None
    )
    return EvaluationResult(
        test_case_id=test_case_id,
        query="Do the thing",
        expected_behavior="Does the thing safely",
        model_response=response,
        judge_score=_make_score(score) if score is not None else None,
        trajectory_evaluation=trajectory_evaluation,
    )


def _make_run(results, model="model-a"):
    return RunResult(
        run_id="run-1",
        golden_set_name="Trajectory Set",
        timestamp=datetime(2026, 8, 30, 12, 0, 0),
        models_tested=[model],
        results=results,
    )


def _export(run, tmp_path):
    path = tmp_path / "junit.xml"
    JUnitXMLExporter().export(run, str(path))
    return ET.parse(str(path)).getroot()


class TestJUnitTrajectoryMapping:
    def test_trajectory_failure_reported_as_failure(self, tmp_path):
        spec = TrajectorySpec(require=["verify_identity"])
        result = _make_eval("case-1", spec=spec, tool_calls=[], score=5)
        root = _export(_make_run([result]), tmp_path)

        failure = root.find("./testsuite/testcase/failure")
        assert failure is not None
        assert failure.get("type") == "TrajectoryAssertionFailed"
        assert "verify_identity" in failure.text
        assert root.get("failures") == "1"

    def test_trajectory_failure_wins_over_passing_judge_score(self, tmp_path):
        spec = TrajectorySpec(forbid=["delete_account"])
        calls = [ToolCall(id="c1", name="delete_account", arguments={})]
        result = _make_eval("case-1", spec=spec, tool_calls=calls, score=5)
        root = _export(_make_run([result]), tmp_path)

        assert root.find("./testsuite/testcase/failure") is not None
        assert root.get("failures") == "1"

    def test_passing_trajectory_without_judge_counts_as_pass(self, tmp_path):
        spec = TrajectorySpec(max_calls=1)
        calls = [ToolCall(id="c1", name="search", arguments={})]
        result = _make_eval("case-1", spec=spec, tool_calls=calls, score=None)
        root = _export(_make_run([result]), tmp_path)

        testcase = root.find("./testsuite/testcase")
        assert testcase.find("failure") is None
        assert testcase.find("skipped") is None
        assert root.get("skipped") == "0"
        assert root.get("failures") == "0"

    def test_no_judge_and_no_trajectory_still_skipped(self, tmp_path):
        result = _make_eval("case-1", spec=None, score=None)
        root = _export(_make_run([result]), tmp_path)

        assert root.find("./testsuite/testcase/skipped") is not None
        assert root.get("skipped") == "1"

    def test_trajectory_summary_in_system_out(self, tmp_path):
        spec = TrajectorySpec(max_calls=3)
        calls = [ToolCall(id="c1", name="search", arguments={})]
        result = _make_eval("case-1", spec=spec, tool_calls=calls, score=4)
        root = _export(_make_run([result]), tmp_path)

        system_out = root.find("./testsuite/testcase/system-out")
        assert "trajectory passed" in system_out.text

    def test_low_judge_score_still_fails_when_trajectory_passes(self, tmp_path):
        spec = TrajectorySpec(max_calls=5)
        calls = [ToolCall(id="c1", name="search", arguments={})]
        result = _make_eval("case-1", spec=spec, tool_calls=calls, score=1)
        root = _export(_make_run([result]), tmp_path)

        failure = root.find("./testsuite/testcase/failure")
        assert failure is not None
        assert failure.get("type") == "JudgeScoreBelowThreshold"


class TestRunResultTrajectoryHelpers:
    def test_get_trajectory_failures_filters_by_model(self):
        spec = TrajectorySpec(require=["a"])
        failing = _make_eval("case-1", spec=spec, tool_calls=[], model="model-a")
        passing = _make_eval(
            "case-2",
            spec=spec,
            tool_calls=[ToolCall(id="c1", name="a", arguments={})],
            model="model-b",
        )
        run = RunResult(
            run_id="run-1",
            golden_set_name="Set",
            models_tested=["model-a", "model-b"],
            results=[failing, passing],
        )

        assert len(run.get_trajectory_failures()) == 1
        assert len(run.get_trajectory_failures(model="model-a")) == 1
        assert len(run.get_trajectory_failures(model="model-b")) == 0

    def test_get_trajectory_stats(self):
        spec = TrajectorySpec(require=["a"])
        failing = _make_eval("case-1", spec=spec, tool_calls=[])
        passing = _make_eval(
            "case-2", spec=spec, tool_calls=[ToolCall(id="c1", name="a", arguments={})]
        )
        no_spec = _make_eval("case-3", spec=None)
        run = RunResult(
            run_id="run-1",
            golden_set_name="Set",
            models_tested=["model-a"],
            results=[failing, passing, no_spec],
        )

        stats = run.get_trajectory_stats()
        assert stats == {"evaluated": 2, "passed": 1, "failed": 1}
