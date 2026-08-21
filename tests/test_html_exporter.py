"""Tests for the HTML exporter and its bundled report template."""

from pathlib import Path

from promptlens.exporters.html_exporter import HTMLExporter
from promptlens.models.result import (
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)


def _sample_run() -> RunResult:
    results = [
        EvaluationResult(
            test_case_id="tc-001",
            query="How do I reset my password?",
            expected_behavior="Provide clear reset steps",
            model_response=ModelResponse(
                content="Click 'Forgot password' and follow the emailed link.",
                model="Test Model",
                provider="anthropic",
                latency_ms=1234.5,
                cost_usd=0.0042,
            ),
            judge_score=JudgeScore(
                score=4,
                explanation="Accurate and clear.",
                judge_model="judge-model",
                judge_provider="anthropic",
            ),
        ),
        EvaluationResult(
            test_case_id="tc-002",
            query="What is your refund policy?",
            expected_behavior="Explain the refund policy",
            model_response=ModelResponse(
                content="",
                model="Test Model",
                provider="anthropic",
                latency_ms=800.0,
                error="Provider timeout",
            ),
            judge_score=None,
        ),
    ]
    return RunResult(
        run_id="test-run",
        run_name="Exporter Test Run",
        golden_set_name="Exporter Golden Set",
        models_tested=["Test Model"],
        results=results,
        total_cost_usd=0.0042,
        total_time_ms=2034.5,
    )


def test_html_export_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "report.html"

    HTMLExporter().export(_sample_run(), str(output))

    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert "Exporter Test Run" in html
    assert "Exporter Golden Set" in html
    assert "How do I reset my password?" in html
    # Scored response renders its judge score badge.
    assert "4/5" in html
    # Errored response surfaces the error instead of content.
    assert "Provider timeout" in html


def test_html_export_template_ships_with_package() -> None:
    template = (
        Path(HTMLExporter.__module__.replace(".", "/")).parent.parent
        / "templates"
        / "report.html"
    )
    assert template.exists(), (
        "promptlens/templates/report.html must ship with the package; "
        "HTML export crashes without it"
    )
