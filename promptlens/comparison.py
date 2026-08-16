"""Run-to-run comparison for regression detection.

Compares two exported run results (baseline vs. current) and reports
per-case score deltas, per-model summaries, and regressions. Built for
CI workflows: pair with ``promptlens compare --fail-on-regression`` to
block merges when a prompt or model change scores worse than the
baseline run.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from promptlens.models.result import EvaluationResult, RunResult

#: Minimum per-case score drop (baseline - current) that counts as a
#: regression. Judge scores are integers on a 1-5 scale, so the default
#: of 0.5 means any drop of a full point or more is flagged.
DEFAULT_REGRESSION_THRESHOLD = 0.5


class CaseComparison(BaseModel):
    """Comparison of a single (test case, model) pair across two runs.

    Attributes:
        test_case_id: ID of the test case
        model: Model identifier
        baseline_score: Judge score in the baseline run (None if unscored)
        current_score: Judge score in the current run (None if unscored)
        score_delta: current - baseline (None if either side is unscored)
        latency_delta_ms: Current latency minus baseline latency
        cost_delta_usd: Current cost minus baseline cost (None if unknown)
        baseline_error: Error message from the baseline run, if any
        current_error: Error message from the current run, if any
        status: One of "regressed", "improved", "unchanged", "new_error",
            "fixed_error", or "unscored"
    """

    test_case_id: str
    model: str
    baseline_score: Optional[int] = None
    current_score: Optional[int] = None
    score_delta: Optional[float] = None
    latency_delta_ms: float = 0.0
    cost_delta_usd: Optional[float] = None
    baseline_error: Optional[str] = None
    current_error: Optional[str] = None
    status: str


class ModelComparisonSummary(BaseModel):
    """Aggregate comparison for one model across two runs.

    Attributes:
        model: Model identifier
        baseline_avg_score: Average judge score in the baseline run
        current_avg_score: Average judge score in the current run
        avg_score_delta: current - baseline (None if either side has no scores)
        regressed: Number of cases flagged as regressed
        improved: Number of cases that improved past the threshold
        unchanged: Number of cases within the threshold
        new_errors: Number of cases that went from success to error
        fixed_errors: Number of cases that went from error to success
        cost_delta_usd: Total current cost minus total baseline cost
        latency_delta_ms: Total current latency minus total baseline latency
    """

    model: str
    baseline_avg_score: Optional[float] = None
    current_avg_score: Optional[float] = None
    avg_score_delta: Optional[float] = None
    regressed: int = 0
    improved: int = 0
    unchanged: int = 0
    new_errors: int = 0
    fixed_errors: int = 0
    cost_delta_usd: float = 0.0
    latency_delta_ms: float = 0.0


class RunComparison(BaseModel):
    """Full comparison of two runs.

    Attributes:
        baseline_run_id: run_id of the baseline run
        current_run_id: run_id of the current run
        threshold: Score-drop threshold used to flag regressions
        cases: Per-case comparisons for pairs present in both runs
        model_summaries: Aggregate summaries per model
        only_in_baseline: (test_case_id, model) pairs missing from current
        only_in_current: (test_case_id, model) pairs missing from baseline
    """

    baseline_run_id: str
    current_run_id: str
    threshold: float = DEFAULT_REGRESSION_THRESHOLD
    cases: List[CaseComparison] = Field(default_factory=list)
    model_summaries: List[ModelComparisonSummary] = Field(default_factory=list)
    only_in_baseline: List[Tuple[str, str]] = Field(default_factory=list)
    only_in_current: List[Tuple[str, str]] = Field(default_factory=list)

    @property
    def total_regressions(self) -> int:
        """Total regressed cases plus new errors across all models."""
        return sum(s.regressed + s.new_errors for s in self.model_summaries)

    @property
    def has_regressions(self) -> bool:
        """Whether any case regressed or newly errored."""
        return self.total_regressions > 0


def load_run_result(path: str) -> RunResult:
    """Load a RunResult from a JSON file produced by the JSON exporter.

    Args:
        path: Path to a results.json file

    Returns:
        Parsed RunResult

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is not valid run-result JSON
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Run result file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} is not valid JSON: {e}") from e

    try:
        return RunResult.model_validate(data)
    except Exception as e:
        raise ValueError(
            f"{path} is not a valid PromptLens run result. "
            f"Expected the JSON produced by the json export format. ({e})"
        ) from e


def _index_results(result: RunResult) -> Dict[Tuple[str, str], EvaluationResult]:
    """Index evaluation results by (test_case_id, model).

    If duplicates exist for a pair, the last one wins, matching the
    order-dependent behavior of the exporters.
    """
    return {
        (r.test_case_id, r.model_response.model): r
        for r in result.results
    }


def _classify(
    baseline: EvaluationResult,
    current: EvaluationResult,
    threshold: float,
) -> CaseComparison:
    """Build the CaseComparison for one paired result."""
    baseline_error = baseline.model_response.error
    current_error = current.model_response.error

    baseline_score = baseline.judge_score.score if baseline.judge_score else None
    current_score = current.judge_score.score if current.judge_score else None

    score_delta: Optional[float] = None
    if baseline_score is not None and current_score is not None:
        score_delta = float(current_score - baseline_score)

    baseline_cost = baseline.model_response.cost_usd
    current_cost = current.model_response.cost_usd
    cost_delta = None
    if baseline_cost is not None and current_cost is not None:
        cost_delta = current_cost - baseline_cost

    latency_delta = (
        current.model_response.latency_ms - baseline.model_response.latency_ms
    )

    if current_error and not baseline_error:
        status = "new_error"
    elif baseline_error and not current_error:
        status = "fixed_error"
    elif score_delta is None:
        status = "unscored"
    elif score_delta <= -threshold:
        status = "regressed"
    elif score_delta >= threshold:
        status = "improved"
    else:
        status = "unchanged"

    return CaseComparison(
        test_case_id=baseline.test_case_id,
        model=baseline.model_response.model,
        baseline_score=baseline_score,
        current_score=current_score,
        score_delta=score_delta,
        latency_delta_ms=latency_delta,
        cost_delta_usd=cost_delta,
        baseline_error=baseline_error,
        current_error=current_error,
        status=status,
    )


def compare_runs(
    baseline: RunResult,
    current: RunResult,
    threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> RunComparison:
    """Compare two runs and detect regressions.

    Results are paired by (test_case_id, model). Pairs present in only
    one run are reported separately and never counted as regressions,
    so adding or removing test cases does not break the gate.

    Args:
        baseline: The baseline (older / known-good) run
        current: The current (candidate) run
        threshold: Minimum score drop that counts as a regression

    Returns:
        RunComparison with per-case and per-model breakdowns
    """
    if threshold < 0:
        raise ValueError("threshold must be >= 0")

    baseline_index = _index_results(baseline)
    current_index = _index_results(current)

    shared_keys = [k for k in baseline_index if k in current_index]
    only_in_baseline = sorted(k for k in baseline_index if k not in current_index)
    only_in_current = sorted(k for k in current_index if k not in baseline_index)

    cases = [
        _classify(baseline_index[key], current_index[key], threshold)
        for key in sorted(shared_keys)
    ]

    models = sorted({c.model for c in cases})
    summaries = []
    for model in models:
        model_cases = [c for c in cases if c.model == model]

        baseline_scores = [
            c.baseline_score for c in model_cases if c.baseline_score is not None
        ]
        current_scores = [
            c.current_score for c in model_cases if c.current_score is not None
        ]
        baseline_avg = (
            sum(baseline_scores) / len(baseline_scores) if baseline_scores else None
        )
        current_avg = (
            sum(current_scores) / len(current_scores) if current_scores else None
        )
        avg_delta = None
        if baseline_avg is not None and current_avg is not None:
            avg_delta = current_avg - baseline_avg

        summaries.append(
            ModelComparisonSummary(
                model=model,
                baseline_avg_score=baseline_avg,
                current_avg_score=current_avg,
                avg_score_delta=avg_delta,
                regressed=sum(1 for c in model_cases if c.status == "regressed"),
                improved=sum(1 for c in model_cases if c.status == "improved"),
                unchanged=sum(
                    1 for c in model_cases if c.status in ("unchanged", "unscored")
                ),
                new_errors=sum(1 for c in model_cases if c.status == "new_error"),
                fixed_errors=sum(1 for c in model_cases if c.status == "fixed_error"),
                cost_delta_usd=sum(c.cost_delta_usd or 0.0 for c in model_cases),
                latency_delta_ms=sum(c.latency_delta_ms for c in model_cases),
            )
        )

    return RunComparison(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        threshold=threshold,
        cases=cases,
        model_summaries=summaries,
        only_in_baseline=only_in_baseline,
        only_in_current=only_in_current,
    )


def _format_delta(value: Optional[float], suffix: str = "", digits: int = 2) -> str:
    """Format a signed delta for display, or 'n/a' when unknown."""
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}{suffix}"


def render_markdown(comparison: RunComparison) -> str:
    """Render a comparison as a markdown report.

    The output is designed to be posted as a PR comment from CI, in the
    style of the diff summaries hosted eval platforms attach to pull
    requests.

    Args:
        comparison: The comparison to render

    Returns:
        Markdown text
    """
    lines: List[str] = []
    verdict = (
        f"🔴 {comparison.total_regressions} regression(s) detected"
        if comparison.has_regressions
        else "🟢 No regressions detected"
    )

    lines.append("# PromptLens Run Comparison")
    lines.append("")
    lines.append(f"**Baseline:** `{comparison.baseline_run_id}`  ")
    lines.append(f"**Current:** `{comparison.current_run_id}`  ")
    lines.append(f"**Regression threshold:** {comparison.threshold:g} points  ")
    lines.append(f"**Verdict:** {verdict}")
    lines.append("")

    lines.append("## Model Summary")
    lines.append("")
    lines.append(
        "| Model | Baseline Avg | Current Avg | Delta | Regressed | Improved "
        "| New Errors | Fixed Errors | Cost Delta | Latency Delta |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in comparison.model_summaries:
        baseline_avg = (
            f"{s.baseline_avg_score:.2f}" if s.baseline_avg_score is not None else "n/a"
        )
        current_avg = (
            f"{s.current_avg_score:.2f}" if s.current_avg_score is not None else "n/a"
        )
        lines.append(
            f"| {s.model} | {baseline_avg} | {current_avg} "
            f"| {_format_delta(s.avg_score_delta)} | {s.regressed} | {s.improved} "
            f"| {s.new_errors} | {s.fixed_errors} "
            f"| {_format_delta(s.cost_delta_usd, ' USD', 4)} "
            f"| {_format_delta(s.latency_delta_ms, ' ms', 0)} |"
        )
    lines.append("")

    flagged = [c for c in comparison.cases if c.status in ("regressed", "new_error")]
    if flagged:
        lines.append("## Regressions")
        lines.append("")
        lines.append("| Test Case | Model | Baseline | Current | Delta | Status |")
        lines.append("|---|---|---|---|---|---|")
        for c in flagged:
            baseline_display = (
                str(c.baseline_score) if c.baseline_score is not None else "error"
                if c.baseline_error else "n/a"
            )
            current_display = (
                "error" if c.current_error
                else str(c.current_score) if c.current_score is not None else "n/a"
            )
            lines.append(
                f"| {c.test_case_id} | {c.model} | {baseline_display} "
                f"| {current_display} | {_format_delta(c.score_delta)} | {c.status} |"
            )
        lines.append("")

    if comparison.only_in_baseline or comparison.only_in_current:
        lines.append("## Coverage Changes")
        lines.append("")
        for test_case_id, model in comparison.only_in_baseline:
            lines.append(f"- Removed: `{test_case_id}` on {model} (baseline only)")
        for test_case_id, model in comparison.only_in_current:
            lines.append(f"- Added: `{test_case_id}` on {model} (current only)")
        lines.append("")

    return "\n".join(lines)
