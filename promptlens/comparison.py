"""Run-over-run comparison for regression detection.

Compares two saved evaluation runs (a baseline and a candidate) case by
case, classifies each shared test case as improved, regressed, or
unchanged, and aggregates score, cost, and latency deltas. This is the
local equivalent of the experiment-diff view in hosted eval platforms:
it turns "I think the new prompt is better" into a per-case diff you can
read in a terminal, post on a pull request, or gate CI on.
"""

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from promptlens.models.result import EvaluationResult, RunResult

# Case status values
STATUS_IMPROVED = "improved"
STATUS_REGRESSED = "regressed"
STATUS_UNCHANGED = "unchanged"
STATUS_ADDED = "added"
STATUS_REMOVED = "removed"
STATUS_UNSCORED = "unscored"


class CaseComparison(BaseModel):
    """Comparison of a single test case between two runs.

    Attributes:
        test_case_id: ID of the test case
        baseline_model: Model that produced the baseline result (if present)
        candidate_model: Model that produced the candidate result (if present)
        query: The test case query
        baseline_score: Judge score in the baseline run (1-5), if scored
        candidate_score: Judge score in the candidate run (1-5), if scored
        score_delta: candidate_score - baseline_score, if both scored
        baseline_error: Error message from the baseline run, if any
        candidate_error: Error message from the candidate run, if any
        latency_delta_ms: Candidate latency minus baseline latency
        cost_delta_usd: Candidate cost minus baseline cost
        status: One of improved / regressed / unchanged / added / removed / unscored
        detail: Human-readable note (e.g. judge explanation for a regression)
    """

    test_case_id: str
    baseline_model: Optional[str] = None
    candidate_model: Optional[str] = None
    query: str = ""
    baseline_score: Optional[int] = None
    candidate_score: Optional[int] = None
    score_delta: Optional[int] = None
    baseline_error: Optional[str] = None
    candidate_error: Optional[str] = None
    latency_delta_ms: Optional[float] = None
    cost_delta_usd: Optional[float] = None
    status: str
    detail: str = ""


class RunComparison(BaseModel):
    """Full comparison between a baseline run and a candidate run.

    Attributes:
        baseline_run_id: Run ID of the baseline
        candidate_run_id: Run ID of the candidate
        baseline_run_name: Optional run name of the baseline
        candidate_run_name: Optional run name of the candidate
        regression_threshold: Minimum score drop counted as a regression
        cases: Per-case comparisons, regressions first
        improved: Number of improved cases
        regressed: Number of regressed cases
        unchanged: Number of unchanged cases
        added: Cases only present in the candidate run
        removed: Cases only present in the baseline run
        unscored: Shared cases with no judge score on one or both sides
        baseline_avg_score: Average judge score across compared baseline cases
        candidate_avg_score: Average judge score across compared candidate cases
        avg_score_delta: candidate_avg_score - baseline_avg_score
        total_cost_delta_usd: Candidate total cost minus baseline total cost
        avg_latency_delta_ms: Average per-case latency delta
    """

    baseline_run_id: str
    candidate_run_id: str
    baseline_run_name: Optional[str] = None
    candidate_run_name: Optional[str] = None
    regression_threshold: float = 0.0
    cases: List[CaseComparison] = Field(default_factory=list)
    improved: int = 0
    regressed: int = 0
    unchanged: int = 0
    added: int = 0
    removed: int = 0
    unscored: int = 0
    baseline_avg_score: Optional[float] = None
    candidate_avg_score: Optional[float] = None
    avg_score_delta: Optional[float] = None
    total_cost_delta_usd: float = 0.0
    avg_latency_delta_ms: Optional[float] = None

    @property
    def has_regressions(self) -> bool:
        """Whether any compared case regressed."""
        return self.regressed > 0


def _index_results(
    run: RunResult, model_filter: Optional[str], by_model: bool
) -> "Dict[Tuple[str, str], EvaluationResult]":
    """Index a run's results for matching.

    Args:
        run: The run to index
        model_filter: If set, only include results from this model
        by_model: If True, key by (test_case_id, model); otherwise by
            (test_case_id, "") so cases match across different models

    Returns:
        Mapping from match key to the first result seen for that key
    """
    index: Dict[Tuple[str, str], EvaluationResult] = {}
    for result in run.results:
        model = result.model_response.model
        if model_filter is not None and model != model_filter:
            continue
        key = (result.test_case_id, model if by_model else "")
        if key not in index:
            index[key] = result
    return index


def _classify_pair(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    regression_threshold: float,
) -> CaseComparison:
    """Build the comparison entry for a test case present in both runs."""
    baseline_error = baseline.model_response.error
    candidate_error = candidate.model_response.error
    baseline_score = baseline.judge_score.score if baseline.judge_score else None
    candidate_score = candidate.judge_score.score if candidate.judge_score else None

    comparison = CaseComparison(
        test_case_id=baseline.test_case_id,
        baseline_model=baseline.model_response.model,
        candidate_model=candidate.model_response.model,
        query=baseline.query,
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        baseline_error=baseline_error,
        candidate_error=candidate_error,
        latency_delta_ms=(
            candidate.model_response.latency_ms - baseline.model_response.latency_ms
        ),
        cost_delta_usd=(
            (candidate.model_response.cost_usd or 0.0)
            - (baseline.model_response.cost_usd or 0.0)
        ),
        status=STATUS_UNCHANGED,
    )

    # Error transitions dominate score comparison: a case that used to
    # succeed and now errors is a regression even without judge scores.
    if candidate_error and not baseline_error:
        comparison.status = STATUS_REGRESSED
        comparison.detail = f"Candidate errored: {candidate_error}"
        return comparison
    if baseline_error and not candidate_error:
        comparison.status = STATUS_IMPROVED
        comparison.detail = "Baseline errored; candidate succeeded"
        return comparison
    if baseline_error and candidate_error:
        comparison.status = STATUS_UNCHANGED
        comparison.detail = "Errored in both runs"
        return comparison

    if baseline_score is None or candidate_score is None:
        comparison.status = STATUS_UNSCORED
        comparison.detail = "Missing judge score in one or both runs"
        return comparison

    delta = candidate_score - baseline_score
    comparison.score_delta = delta
    if delta < 0 and abs(delta) >= regression_threshold:
        comparison.status = STATUS_REGRESSED
        if candidate.judge_score and candidate.judge_score.explanation:
            comparison.detail = candidate.judge_score.explanation
    elif delta > 0:
        comparison.status = STATUS_IMPROVED
    else:
        comparison.status = STATUS_UNCHANGED
    return comparison


def compare_runs(
    baseline: RunResult,
    candidate: RunResult,
    regression_threshold: float = 0.0,
    baseline_model: Optional[str] = None,
    candidate_model: Optional[str] = None,
) -> RunComparison:
    """Compare two runs case by case.

    By default, results are matched on (test_case_id, model), so a run
    that tested several models compares each model against itself. Pass
    baseline_model and candidate_model to compare across models (e.g.
    the same golden set on claude vs. gpt): results are then matched on
    test_case_id alone within the selected models.

    Args:
        baseline: The reference run (e.g. main branch, previous prompt)
        candidate: The new run to evaluate against the baseline
        regression_threshold: Minimum score drop (in judge points) for a
            case to count as regressed. 0.0 means any drop regresses.
            Drops smaller than the threshold are classified unchanged.
        baseline_model: Restrict the baseline to one model and match on
            test_case_id only (requires candidate_model)
        candidate_model: Restrict the candidate to one model and match
            on test_case_id only (requires baseline_model)

    Returns:
        A RunComparison with per-case entries and aggregate deltas

    Raises:
        ValueError: If only one of baseline_model / candidate_model is set
    """
    if (baseline_model is None) != (candidate_model is None):
        raise ValueError(
            "baseline_model and candidate_model must be provided together"
        )
    cross_model = baseline_model is not None

    baseline_index = _index_results(baseline, baseline_model, by_model=not cross_model)
    candidate_index = _index_results(candidate, candidate_model, by_model=not cross_model)

    cases: List[CaseComparison] = []
    shared_keys = [key for key in baseline_index if key in candidate_index]

    for key in shared_keys:
        cases.append(
            _classify_pair(baseline_index[key], candidate_index[key], regression_threshold)
        )

    for key, result in baseline_index.items():
        if key in candidate_index:
            continue
        cases.append(
            CaseComparison(
                test_case_id=result.test_case_id,
                baseline_model=result.model_response.model,
                query=result.query,
                baseline_score=result.judge_score.score if result.judge_score else None,
                baseline_error=result.model_response.error,
                status=STATUS_REMOVED,
                detail="Only present in the baseline run",
            )
        )

    for key, result in candidate_index.items():
        if key in baseline_index:
            continue
        cases.append(
            CaseComparison(
                test_case_id=result.test_case_id,
                candidate_model=result.model_response.model,
                query=result.query,
                candidate_score=result.judge_score.score if result.judge_score else None,
                candidate_error=result.model_response.error,
                status=STATUS_ADDED,
                detail="Only present in the candidate run",
            )
        )

    # Regressions first, then improvements, then everything else.
    status_order = {
        STATUS_REGRESSED: 0,
        STATUS_IMPROVED: 1,
        STATUS_UNCHANGED: 2,
        STATUS_UNSCORED: 3,
        STATUS_ADDED: 4,
        STATUS_REMOVED: 5,
    }
    cases.sort(key=lambda c: (status_order.get(c.status, 9), c.test_case_id))

    comparison = RunComparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline_run_name=baseline.run_name,
        candidate_run_name=candidate.run_name,
        regression_threshold=regression_threshold,
        cases=cases,
    )

    scored_pairs = [
        c
        for c in cases
        if c.baseline_score is not None and c.candidate_score is not None
    ]
    comparison.improved = sum(1 for c in cases if c.status == STATUS_IMPROVED)
    comparison.regressed = sum(1 for c in cases if c.status == STATUS_REGRESSED)
    comparison.unchanged = sum(1 for c in cases if c.status == STATUS_UNCHANGED)
    comparison.added = sum(1 for c in cases if c.status == STATUS_ADDED)
    comparison.removed = sum(1 for c in cases if c.status == STATUS_REMOVED)
    comparison.unscored = sum(1 for c in cases if c.status == STATUS_UNSCORED)

    if scored_pairs:
        baseline_avg = sum(c.baseline_score for c in scored_pairs) / len(scored_pairs)
        candidate_avg = sum(c.candidate_score for c in scored_pairs) / len(scored_pairs)
        comparison.baseline_avg_score = baseline_avg
        comparison.candidate_avg_score = candidate_avg
        comparison.avg_score_delta = candidate_avg - baseline_avg

    shared_cases = [c for c in cases if c.status not in (STATUS_ADDED, STATUS_REMOVED)]
    if shared_cases:
        comparison.total_cost_delta_usd = sum(c.cost_delta_usd or 0.0 for c in shared_cases)
        latency_deltas = [
            c.latency_delta_ms for c in shared_cases if c.latency_delta_ms is not None
        ]
        if latency_deltas:
            comparison.avg_latency_delta_ms = sum(latency_deltas) / len(latency_deltas)

    return comparison


def _format_signed(value: float, suffix: str = "", decimals: int = 2) -> str:
    """Format a numeric delta with an explicit sign."""
    return f"{value:+.{decimals}f}{suffix}"


def _case_label(case: CaseComparison) -> str:
    """Label a case with its model(s) when they differ."""
    if case.baseline_model and case.candidate_model:
        if case.baseline_model == case.candidate_model:
            return f"{case.test_case_id} ({case.candidate_model})"
        return f"{case.test_case_id} ({case.baseline_model} vs {case.candidate_model})"
    model = case.baseline_model or case.candidate_model
    return f"{case.test_case_id} ({model})" if model else case.test_case_id


def comparison_to_markdown(comparison: RunComparison) -> str:
    """Render a comparison as Markdown suitable for a PR comment.

    Args:
        comparison: The comparison to render

    Returns:
        A Markdown document with a summary table and per-case sections
    """
    baseline_label = comparison.baseline_run_name or comparison.baseline_run_id
    candidate_label = comparison.candidate_run_name or comparison.candidate_run_id

    lines: List[str] = []
    lines.append("# PromptLens Run Comparison")
    lines.append("")
    lines.append(f"Baseline: `{baseline_label}` ({comparison.baseline_run_id})")
    lines.append(f"Candidate: `{candidate_label}` ({comparison.candidate_run_id})")
    lines.append("")
    lines.append("| Metric | Baseline | Candidate | Delta |")
    lines.append("| --- | --- | --- | --- |")

    if comparison.baseline_avg_score is not None:
        lines.append(
            "| Average judge score | "
            f"{comparison.baseline_avg_score:.2f} | "
            f"{comparison.candidate_avg_score:.2f} | "
            f"{_format_signed(comparison.avg_score_delta)} |"
        )
    lines.append(
        f"| Total cost (shared cases) | | | {_format_signed(comparison.total_cost_delta_usd, ' USD', 4)} |"
    )
    if comparison.avg_latency_delta_ms is not None:
        lines.append(
            f"| Average latency | | | {_format_signed(comparison.avg_latency_delta_ms, ' ms', 1)} |"
        )
    lines.append("")
    lines.append(
        f"**{comparison.regressed} regressed** / {comparison.improved} improved / "
        f"{comparison.unchanged} unchanged"
        + (f" / {comparison.unscored} unscored" if comparison.unscored else "")
        + (f" / {comparison.added} added" if comparison.added else "")
        + (f" / {comparison.removed} removed" if comparison.removed else "")
    )
    lines.append("")

    regressions = [c for c in comparison.cases if c.status == STATUS_REGRESSED]
    improvements = [c for c in comparison.cases if c.status == STATUS_IMPROVED]

    if regressions:
        lines.append("## Regressions")
        lines.append("")
        lines.append("| Test case | Baseline | Candidate | Delta | Detail |")
        lines.append("| --- | --- | --- | --- | --- |")
        for case in regressions:
            baseline_cell = (
                str(case.baseline_score) if case.baseline_score is not None else "error"
                if case.baseline_error
                else "-"
            )
            candidate_cell = (
                str(case.candidate_score)
                if case.candidate_score is not None
                else "error"
                if case.candidate_error
                else "-"
            )
            delta_cell = (
                _format_signed(case.score_delta, decimals=0)
                if case.score_delta is not None
                else "-"
            )
            detail = case.detail.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {_case_label(case)} | {baseline_cell} | {candidate_cell} | "
                f"{delta_cell} | {detail} |"
            )
        lines.append("")

    if improvements:
        lines.append("## Improvements")
        lines.append("")
        lines.append("| Test case | Baseline | Candidate | Delta |")
        lines.append("| --- | --- | --- | --- |")
        for case in improvements:
            baseline_cell = (
                str(case.baseline_score)
                if case.baseline_score is not None
                else "error"
                if case.baseline_error
                else "-"
            )
            candidate_cell = (
                str(case.candidate_score) if case.candidate_score is not None else "-"
            )
            delta_cell = (
                _format_signed(case.score_delta, decimals=0)
                if case.score_delta is not None
                else "-"
            )
            lines.append(
                f"| {_case_label(case)} | {baseline_cell} | {candidate_cell} | {delta_cell} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"
