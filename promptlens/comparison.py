"""Cross-run comparison: diff two evaluation runs and detect regressions.

Pairs results by (test_case_id, model), computes score, cost, and latency
deltas, and classifies each pair as regressed, improved, unchanged, or
incomparable. Built for the regression-testing workflow: run your golden set
before and after a prompt or model change, then compare the two runs.
"""

import logging
from typing import Dict, List, Optional, Tuple

from promptlens.models.comparison import (
    REASON_BOTH_ERRORED,
    REASON_ERROR_RESOLVED,
    REASON_NEW_ERROR,
    REASON_NO_BASELINE_SCORE,
    REASON_NO_SCORES,
    REASON_SCORE_DROP,
    REASON_SCORE_GAIN,
    REASON_SCORE_MISSING,
    REASON_WITHIN_THRESHOLD,
    STATUS_IMPROVED,
    STATUS_INCOMPARABLE,
    STATUS_REGRESSED,
    STATUS_UNCHANGED,
    CaseComparison,
    ComparisonResult,
    ModelComparison,
)
from promptlens.models.result import EvaluationResult, RunResult

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 1.0


def _index_results(run: RunResult) -> Dict[Tuple[str, str], EvaluationResult]:
    """Index a run's results by (test_case_id, model).

    If a pair appears more than once, the first occurrence wins.
    """
    index: Dict[Tuple[str, str], EvaluationResult] = {}
    for result in run.results:
        key = (result.test_case_id, result.model_response.model)
        if key not in index:
            index[key] = result
    return index


def _classify(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    threshold: float,
) -> Tuple[str, str]:
    """Classify a paired case as regressed, improved, unchanged, or incomparable.

    Rules, in order:
    1. A case that newly errors in the candidate run is a regression.
    2. A case whose baseline error is resolved in the candidate run is an
       improvement.
    3. A case that errors in both runs is unchanged.
    4. With scores on both sides, the score delta decides: a drop of at least
       ``threshold`` is a regression, a gain of at least ``threshold`` is an
       improvement, anything smaller is unchanged.
    5. A case that was scored in the baseline but has no candidate score is a
       regression: the gate cannot be evaluated without a score, and a silent
       pass would be misleading.
    6. A case with no baseline score cannot be compared.
    """
    baseline_error = baseline.model_response.error
    candidate_error = candidate.model_response.error

    if candidate_error and not baseline_error:
        return STATUS_REGRESSED, REASON_NEW_ERROR
    if baseline_error and not candidate_error:
        return STATUS_IMPROVED, REASON_ERROR_RESOLVED
    if baseline_error and candidate_error:
        return STATUS_UNCHANGED, REASON_BOTH_ERRORED

    baseline_score = baseline.judge_score.score if baseline.judge_score else None
    candidate_score = candidate.judge_score.score if candidate.judge_score else None

    if baseline_score is not None and candidate_score is not None:
        delta = candidate_score - baseline_score
        if delta <= -threshold:
            return STATUS_REGRESSED, REASON_SCORE_DROP
        if delta >= threshold:
            return STATUS_IMPROVED, REASON_SCORE_GAIN
        return STATUS_UNCHANGED, REASON_WITHIN_THRESHOLD

    if baseline_score is not None and candidate_score is None:
        return STATUS_REGRESSED, REASON_SCORE_MISSING
    if baseline_score is None and candidate_score is not None:
        return STATUS_INCOMPARABLE, REASON_NO_BASELINE_SCORE
    return STATUS_INCOMPARABLE, REASON_NO_SCORES


def _build_case_comparison(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    threshold: float,
) -> CaseComparison:
    """Build a CaseComparison for one paired (test case, model)."""
    status, reason = _classify(baseline, candidate, threshold)

    baseline_score = baseline.judge_score.score if baseline.judge_score else None
    candidate_score = candidate.judge_score.score if candidate.judge_score else None
    score_delta = None
    if baseline_score is not None and candidate_score is not None:
        score_delta = candidate_score - baseline_score

    return CaseComparison(
        test_case_id=candidate.test_case_id,
        model=candidate.model_response.model,
        query=candidate.query,
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        score_delta=score_delta,
        baseline_cost_usd=baseline.model_response.cost_usd,
        candidate_cost_usd=candidate.model_response.cost_usd,
        baseline_latency_ms=baseline.model_response.latency_ms,
        candidate_latency_ms=candidate.model_response.latency_ms,
        baseline_error=baseline.model_response.error,
        candidate_error=candidate.model_response.error,
        status=status,
        reason=reason,
    )


def _average(values: List[float]) -> Optional[float]:
    """Average of a list, or None when empty."""
    return sum(values) / len(values) if values else None


def _build_model_summary(
    model: str,
    cases: List[CaseComparison],
) -> ModelComparison:
    """Aggregate case comparisons for one model."""
    baseline_scores = [float(c.baseline_score) for c in cases if c.baseline_score is not None]
    candidate_scores = [float(c.candidate_score) for c in cases if c.candidate_score is not None]

    baseline_avg = _average(baseline_scores)
    candidate_avg = _average(candidate_scores)
    avg_delta = None
    if baseline_avg is not None and candidate_avg is not None:
        avg_delta = candidate_avg - baseline_avg

    baseline_cost = sum(c.baseline_cost_usd or 0.0 for c in cases)
    candidate_cost = sum(c.candidate_cost_usd or 0.0 for c in cases)

    baseline_latency = _average(
        [c.baseline_latency_ms for c in cases if c.baseline_latency_ms is not None]
    )
    candidate_latency = _average(
        [c.candidate_latency_ms for c in cases if c.candidate_latency_ms is not None]
    )
    latency_delta = None
    if baseline_latency is not None and candidate_latency is not None:
        latency_delta = candidate_latency - baseline_latency

    return ModelComparison(
        model=model,
        baseline_avg_score=baseline_avg,
        candidate_avg_score=candidate_avg,
        avg_score_delta=avg_delta,
        regressed=sum(1 for c in cases if c.status == STATUS_REGRESSED),
        improved=sum(1 for c in cases if c.status == STATUS_IMPROVED),
        unchanged=sum(1 for c in cases if c.status == STATUS_UNCHANGED),
        incomparable=sum(1 for c in cases if c.status == STATUS_INCOMPARABLE),
        baseline_cost_usd=baseline_cost,
        candidate_cost_usd=candidate_cost,
        cost_delta_usd=candidate_cost - baseline_cost,
        baseline_avg_latency_ms=baseline_latency,
        candidate_avg_latency_ms=candidate_latency,
        latency_delta_ms=latency_delta,
    )


def compare_runs(
    baseline: RunResult,
    candidate: RunResult,
    threshold: float = DEFAULT_THRESHOLD,
) -> ComparisonResult:
    """Compare two evaluation runs and classify every shared case.

    Args:
        baseline: The reference run (for example, the last known-good run)
        candidate: The new run being checked against the baseline
        threshold: Minimum absolute judge-score delta counted as a change.
            Judge scores are integers on a 1-5 scale, so the default of 1.0
            flags every score movement.

    Returns:
        A ComparisonResult with per-case and per-model deltas
    """
    if threshold <= 0:
        raise ValueError("threshold must be greater than 0")

    baseline_index = _index_results(baseline)
    candidate_index = _index_results(candidate)

    baseline_models = list(dict.fromkeys(baseline.models_tested))
    candidate_models = list(dict.fromkeys(candidate.models_tested))
    models_compared = [m for m in candidate_models if m in baseline_models]
    models_added = [m for m in candidate_models if m not in baseline_models]
    models_removed = [m for m in baseline_models if m not in candidate_models]

    baseline_case_ids = list(dict.fromkeys(r.test_case_id for r in baseline.results))
    candidate_case_ids = list(dict.fromkeys(r.test_case_id for r in candidate.results))
    cases_added = [c for c in candidate_case_ids if c not in baseline_case_ids]
    cases_removed = [c for c in baseline_case_ids if c not in candidate_case_ids]

    cases: List[CaseComparison] = []
    for model in models_compared:
        for case_id in candidate_case_ids:
            key = (case_id, model)
            if key in baseline_index and key in candidate_index:
                cases.append(
                    _build_case_comparison(
                        baseline_index[key], candidate_index[key], threshold
                    )
                )

    model_summaries = [
        _build_model_summary(model, [c for c in cases if c.model == model])
        for model in models_compared
    ]

    return ComparisonResult(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline_run_name=baseline.run_name,
        candidate_run_name=candidate.run_name,
        baseline_timestamp=baseline.timestamp,
        candidate_timestamp=candidate.timestamp,
        golden_set_name=candidate.golden_set_name,
        threshold=threshold,
        models_compared=models_compared,
        models_added=models_added,
        models_removed=models_removed,
        cases_added=cases_added,
        cases_removed=cases_removed,
        cases=cases,
        model_summaries=model_summaries,
    )


def _format_optional(value: Optional[float], fmt: str, missing: str = "-") -> str:
    """Format an optional number, or a placeholder when missing."""
    return fmt.format(value) if value is not None else missing


def render_markdown(comparison: ComparisonResult) -> str:
    """Render a comparison as a markdown report.

    Suitable for posting as a pull request comment or CI artifact.
    """
    lines: List[str] = []
    verdict = "REGRESSED" if comparison.has_regressions else "PASS"
    emoji = "🔴" if comparison.has_regressions else "🟢"

    lines.append("# PromptLens Run Comparison")
    lines.append("")
    lines.append(f"**Verdict:** {emoji} {verdict}  ")
    lines.append(
        f"**Baseline:** `{comparison.baseline_run_id}`"
        + (f" ({comparison.baseline_run_name})" if comparison.baseline_run_name else "")
        + "  "
    )
    lines.append(
        f"**Candidate:** `{comparison.candidate_run_id}`"
        + (f" ({comparison.candidate_run_name})" if comparison.candidate_run_name else "")
        + "  "
    )
    lines.append(f"**Golden Set:** {comparison.golden_set_name}  ")
    lines.append(f"**Score Threshold:** {comparison.threshold:g}  ")
    lines.append("")

    lines.append("## Model Summary")
    lines.append("")
    lines.append(
        "| Model | Baseline Avg | Candidate Avg | Delta | Regressed | Improved "
        "| Unchanged | Cost Delta | Latency Delta |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for summary in comparison.model_summaries:
        lines.append(
            "| {model} | {b_avg} | {c_avg} | {delta} | {reg} | {imp} | {unc} "
            "| {cost} | {lat} |".format(
                model=summary.model,
                b_avg=_format_optional(summary.baseline_avg_score, "{:.2f}"),
                c_avg=_format_optional(summary.candidate_avg_score, "{:.2f}"),
                delta=_format_optional(summary.avg_score_delta, "{:+.2f}"),
                reg=summary.regressed,
                imp=summary.improved,
                unc=summary.unchanged,
                cost=f"${summary.cost_delta_usd:+.4f}",
                lat=_format_optional(summary.latency_delta_ms, "{:+.0f}ms"),
            )
        )
    lines.append("")

    regressed = comparison.regressed_cases
    if regressed:
        lines.append(f"## Regressions ({len(regressed)})")
        lines.append("")
        lines.append("| Test Case | Model | Before | After | Delta | Reason |")
        lines.append("|---|---|---|---|---|---|")
        for case in regressed:
            lines.append(
                "| {tc} | {model} | {before} | {after} | {delta} | {reason} |".format(
                    tc=case.test_case_id,
                    model=case.model,
                    before=case.baseline_score if case.baseline_score is not None else "-",
                    after=case.candidate_score if case.candidate_score is not None else "-",
                    delta=(
                        f"{case.score_delta:+d}" if case.score_delta is not None else "-"
                    ),
                    reason=case.reason,
                )
            )
        lines.append("")

    improved = comparison.improved_cases
    if improved:
        lines.append(f"## Improvements ({len(improved)})")
        lines.append("")
        lines.append("| Test Case | Model | Before | After | Delta | Reason |")
        lines.append("|---|---|---|---|---|---|")
        for case in improved:
            lines.append(
                "| {tc} | {model} | {before} | {after} | {delta} | {reason} |".format(
                    tc=case.test_case_id,
                    model=case.model,
                    before=case.baseline_score if case.baseline_score is not None else "-",
                    after=case.candidate_score if case.candidate_score is not None else "-",
                    delta=(
                        f"{case.score_delta:+d}" if case.score_delta is not None else "-"
                    ),
                    reason=case.reason,
                )
            )
        lines.append("")

    drift_notes: List[str] = []
    if comparison.models_added:
        drift_notes.append(f"Models added: {', '.join(comparison.models_added)}")
    if comparison.models_removed:
        drift_notes.append(f"Models removed: {', '.join(comparison.models_removed)}")
    if comparison.cases_added:
        drift_notes.append(f"Test cases added: {', '.join(comparison.cases_added)}")
    if comparison.cases_removed:
        drift_notes.append(f"Test cases removed: {', '.join(comparison.cases_removed)}")
    if drift_notes:
        lines.append("## Suite Drift")
        lines.append("")
        lines.append(
            "The following models or test cases exist in only one run and were "
            "excluded from the comparison:"
        )
        lines.append("")
        for note in drift_notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by PromptLens*")
    lines.append("")

    return "\n".join(lines)
