"""Run-to-run comparison and regression detection.

Compares two evaluation runs (a baseline and a candidate) and classifies
each (test case, model) pair as regressed, improved, unchanged, fixed,
errored, added, or removed. Used by the ``promptlens compare`` CLI command
to answer the one question that matters after a prompt or model change:
is this better or worse than what we had?

Pairing is by (test_case_id, model), so comparisons survive reordered,
added, and removed test cases.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from promptlens.models.result import EvaluationResult, RunResult


class CaseStatus(str, Enum):
    """Classification of a paired (test case, model) result."""

    REGRESSED = "regressed"  # score dropped beyond threshold, or ok -> error
    IMPROVED = "improved"    # score rose beyond threshold, or error -> ok
    UNCHANGED = "unchanged"  # within threshold, same error state
    ADDED = "added"          # present only in the candidate run
    REMOVED = "removed"      # present only in the baseline run
    UNSCORED = "unscored"    # present in both, but a judge score is missing


class CaseComparison(BaseModel):
    """Comparison of one (test case, model) pair across two runs.

    Attributes:
        test_case_id: ID of the test case
        model: Model identifier
        status: Classification of the change
        baseline_score: Judge score in the baseline run (1-5), if any
        candidate_score: Judge score in the candidate run (1-5), if any
        score_delta: candidate_score - baseline_score, if both exist
        baseline_error: Error message in the baseline run, if any
        candidate_error: Error message in the candidate run, if any
        cost_delta_usd: Candidate cost minus baseline cost, if both known
        latency_delta_ms: Candidate latency minus baseline latency
    """

    test_case_id: str
    model: str
    status: CaseStatus
    baseline_score: Optional[int] = None
    candidate_score: Optional[int] = None
    score_delta: Optional[int] = None
    baseline_error: Optional[str] = None
    candidate_error: Optional[str] = None
    cost_delta_usd: Optional[float] = None
    latency_delta_ms: Optional[float] = None


class ModelSummary(BaseModel):
    """Aggregate comparison for one model across all paired test cases."""

    model: str
    baseline_avg_score: Optional[float] = None
    candidate_avg_score: Optional[float] = None
    avg_score_delta: Optional[float] = None
    baseline_cost_usd: float = 0.0
    candidate_cost_usd: float = 0.0
    regressed: int = 0
    improved: int = 0
    unchanged: int = 0
    unscored: int = 0


class ComparisonResult(BaseModel):
    """Full result of comparing a candidate run against a baseline run.

    Attributes:
        baseline_run_id: Run ID of the baseline
        candidate_run_id: Run ID of the candidate
        score_threshold: Minimum absolute score delta to count as a change
        cases: Per-(test case, model) comparisons, regressions first
        model_summaries: Per-model aggregates
    """

    baseline_run_id: str
    candidate_run_id: str
    baseline_run_name: Optional[str] = None
    candidate_run_name: Optional[str] = None
    score_threshold: int = 0
    cases: List[CaseComparison] = Field(default_factory=list)
    model_summaries: List[ModelSummary] = Field(default_factory=list)

    @property
    def regressions(self) -> List[CaseComparison]:
        """All cases classified as regressed."""
        return [c for c in self.cases if c.status == CaseStatus.REGRESSED]

    @property
    def improvements(self) -> List[CaseComparison]:
        """All cases classified as improved."""
        return [c for c in self.cases if c.status == CaseStatus.IMPROVED]

    def has_regressions(self) -> bool:
        """Whether any paired case regressed."""
        return any(c.status == CaseStatus.REGRESSED for c in self.cases)


def _index_results(run: RunResult) -> Dict[Tuple[str, str], EvaluationResult]:
    """Index a run's results by (test_case_id, model).

    If duplicate (test_case_id, model) pairs exist, the last one wins,
    matching the intuition that a later result supersedes an earlier one.
    """
    index: Dict[Tuple[str, str], EvaluationResult] = {}
    for result in run.results:
        index[(result.test_case_id, result.model_response.model)] = result
    return index


def _classify_pair(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    score_threshold: int,
) -> CaseComparison:
    """Classify one paired (test case, model) result."""
    baseline_error = baseline.model_response.error
    candidate_error = candidate.model_response.error
    baseline_score = baseline.judge_score.score if baseline.judge_score else None
    candidate_score = candidate.judge_score.score if candidate.judge_score else None

    score_delta: Optional[int] = None
    if baseline_score is not None and candidate_score is not None:
        score_delta = candidate_score - baseline_score

    cost_delta: Optional[float] = None
    if (
        baseline.model_response.cost_usd is not None
        and candidate.model_response.cost_usd is not None
    ):
        cost_delta = candidate.model_response.cost_usd - baseline.model_response.cost_usd

    latency_delta = (
        candidate.model_response.latency_ms - baseline.model_response.latency_ms
    )

    # Error-state transitions dominate score comparisons: a case that used to
    # succeed and now errors is a regression no matter what the judge said.
    if candidate_error and not baseline_error:
        status = CaseStatus.REGRESSED
    elif baseline_error and not candidate_error:
        status = CaseStatus.IMPROVED
    elif baseline_error and candidate_error:
        status = CaseStatus.UNCHANGED
    elif baseline_score is None or candidate_score is None:
        status = CaseStatus.UNSCORED
    elif score_delta is not None and score_delta < -score_threshold:
        status = CaseStatus.REGRESSED
    elif score_delta is not None and score_delta > score_threshold:
        status = CaseStatus.IMPROVED
    else:
        status = CaseStatus.UNCHANGED

    return CaseComparison(
        test_case_id=baseline.test_case_id,
        model=baseline.model_response.model,
        status=status,
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        score_delta=score_delta,
        baseline_error=baseline_error,
        candidate_error=candidate_error,
        cost_delta_usd=cost_delta,
        latency_delta_ms=latency_delta,
    )


_STATUS_ORDER = {
    CaseStatus.REGRESSED: 0,
    CaseStatus.IMPROVED: 1,
    CaseStatus.UNCHANGED: 2,
    CaseStatus.UNSCORED: 3,
    CaseStatus.ADDED: 4,
    CaseStatus.REMOVED: 5,
}


def compare_runs(
    baseline: RunResult,
    candidate: RunResult,
    score_threshold: int = 0,
) -> ComparisonResult:
    """Compare a candidate run against a baseline run.

    Args:
        baseline: The known-good run to compare against
        candidate: The new run being evaluated
        score_threshold: Minimum absolute judge-score delta (on the 1-5
            scale) required to count a paired case as regressed/improved.
            0 means any decrease is a regression.

    Returns:
        A ComparisonResult with per-case classifications (regressions
        first) and per-model aggregates.
    """
    if score_threshold < 0:
        raise ValueError("score_threshold must be >= 0")

    baseline_index = _index_results(baseline)
    candidate_index = _index_results(candidate)

    cases: List[CaseComparison] = []

    for key, baseline_result in baseline_index.items():
        candidate_result = candidate_index.get(key)
        if candidate_result is None:
            cases.append(
                CaseComparison(
                    test_case_id=baseline_result.test_case_id,
                    model=baseline_result.model_response.model,
                    status=CaseStatus.REMOVED,
                    baseline_score=(
                        baseline_result.judge_score.score
                        if baseline_result.judge_score
                        else None
                    ),
                    baseline_error=baseline_result.model_response.error,
                )
            )
        else:
            cases.append(
                _classify_pair(baseline_result, candidate_result, score_threshold)
            )

    for key, candidate_result in candidate_index.items():
        if key not in baseline_index:
            cases.append(
                CaseComparison(
                    test_case_id=candidate_result.test_case_id,
                    model=candidate_result.model_response.model,
                    status=CaseStatus.ADDED,
                    candidate_score=(
                        candidate_result.judge_score.score
                        if candidate_result.judge_score
                        else None
                    ),
                    candidate_error=candidate_result.model_response.error,
                )
            )

    cases.sort(
        key=lambda c: (_STATUS_ORDER[c.status], c.model, c.test_case_id)
    )

    models = sorted(
        set(baseline.models_tested) | set(candidate.models_tested)
    )
    summaries: List[ModelSummary] = []
    for model in models:
        model_cases = [c for c in cases if c.model == model]
        baseline_avg = baseline.get_average_score(model)
        candidate_avg = candidate.get_average_score(model)
        avg_delta: Optional[float] = None
        if baseline_avg is not None and candidate_avg is not None:
            avg_delta = candidate_avg - baseline_avg
        summaries.append(
            ModelSummary(
                model=model,
                baseline_avg_score=baseline_avg,
                candidate_avg_score=candidate_avg,
                avg_score_delta=avg_delta,
                baseline_cost_usd=baseline.get_total_cost(model),
                candidate_cost_usd=candidate.get_total_cost(model),
                regressed=sum(
                    1 for c in model_cases if c.status == CaseStatus.REGRESSED
                ),
                improved=sum(
                    1 for c in model_cases if c.status == CaseStatus.IMPROVED
                ),
                unchanged=sum(
                    1 for c in model_cases if c.status == CaseStatus.UNCHANGED
                ),
                unscored=sum(
                    1 for c in model_cases if c.status == CaseStatus.UNSCORED
                ),
            )
        )

    return ComparisonResult(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline_run_name=baseline.run_name,
        candidate_run_name=candidate.run_name,
        score_threshold=score_threshold,
        cases=cases,
        model_summaries=summaries,
    )


def _format_delta(value: Optional[float], suffix: str = "", digits: int = 2) -> str:
    """Format a signed delta for display, or a dash when unknown."""
    if value is None:
        return "-"
    return "{:+.{digits}f}{}".format(value, suffix, digits=digits)


def render_markdown(comparison: ComparisonResult) -> str:
    """Render a comparison as a Markdown report.

    The report is suitable for pasting into a pull request comment or
    committing as a CI artifact.
    """
    lines: List[str] = []
    baseline_label = comparison.baseline_run_name or comparison.baseline_run_id
    candidate_label = comparison.candidate_run_name or comparison.candidate_run_id

    lines.append("# PromptLens Comparison Report")
    lines.append("")
    lines.append("| | Run |")
    lines.append("|---|---|")
    lines.append("| Baseline | `{}` ({}) |".format(comparison.baseline_run_id, baseline_label))
    lines.append("| Candidate | `{}` ({}) |".format(comparison.candidate_run_id, candidate_label))
    lines.append("")

    regressed = len(comparison.regressions)
    improved = len(comparison.improvements)
    verdict = (
        "REGRESSIONS DETECTED" if regressed else "No regressions detected"
    )
    lines.append(
        "**{}** ({} regressed, {} improved, score threshold {})".format(
            verdict, regressed, improved, comparison.score_threshold
        )
    )
    lines.append("")

    lines.append("## Per-model summary")
    lines.append("")
    lines.append(
        "| Model | Baseline avg | Candidate avg | Delta | Regressed | Improved | Cost delta |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for summary in comparison.model_summaries:
        baseline_avg = (
            "{:.2f}".format(summary.baseline_avg_score)
            if summary.baseline_avg_score is not None
            else "-"
        )
        candidate_avg = (
            "{:.2f}".format(summary.candidate_avg_score)
            if summary.candidate_avg_score is not None
            else "-"
        )
        cost_delta = _format_delta(
            summary.candidate_cost_usd - summary.baseline_cost_usd, "", 4
        )
        lines.append(
            "| {} | {} | {} | {} | {} | {} | ${} |".format(
                summary.model,
                baseline_avg,
                candidate_avg,
                _format_delta(summary.avg_score_delta),
                summary.regressed,
                summary.improved,
                cost_delta,
            )
        )
    lines.append("")

    changed = [
        c
        for c in comparison.cases
        if c.status
        in (
            CaseStatus.REGRESSED,
            CaseStatus.IMPROVED,
            CaseStatus.ADDED,
            CaseStatus.REMOVED,
            CaseStatus.UNSCORED,
        )
    ]
    lines.append("## Changed cases")
    lines.append("")
    if not changed:
        lines.append("No changed cases. All paired results are within threshold.")
    else:
        lines.append(
            "| Status | Test case | Model | Baseline | Candidate | Score delta | Latency delta |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for case in changed:
            baseline_cell = (
                "error" if case.baseline_error else (
                    str(case.baseline_score) if case.baseline_score is not None else "-"
                )
            )
            candidate_cell = (
                "error" if case.candidate_error else (
                    str(case.candidate_score) if case.candidate_score is not None else "-"
                )
            )
            score_delta = (
                "{:+d}".format(case.score_delta)
                if case.score_delta is not None
                else "-"
            )
            latency_delta = (
                _format_delta(case.latency_delta_ms, " ms", 0)
                if case.latency_delta_ms is not None
                else "-"
            )
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    case.status.value,
                    case.test_case_id,
                    case.model,
                    baseline_cell,
                    candidate_cell,
                    score_delta,
                    latency_delta,
                )
            )
    lines.append("")

    return "\n".join(lines)
