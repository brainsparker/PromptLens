"""Run-to-run comparison for regression detection.

Compares a current evaluation run against a baseline run, pairing results
by (test_case_id, model). Produces per-case score deltas, per-model
aggregate deltas, and CI gate checks so a pull request can fail when it
makes quality worse relative to main, not just when it drops below an
absolute threshold.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from promptlens.models.result import EvaluationResult, RunResult


class CaseStatus(str, Enum):
    """Classification of a single (test case, model) pair across two runs."""

    REGRESSED = "regressed"
    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    ADDED = "added"  # present in current run only
    REMOVED = "removed"  # present in baseline run only
    UNSCORED = "unscored"  # present in both but missing a judge score


class CaseComparison(BaseModel):
    """Comparison of one test case for one model across two runs.

    Attributes:
        test_case_id: ID of the test case
        model: Model identifier
        baseline_score: Judge score in the baseline run (1-5), if scored
        current_score: Judge score in the current run (1-5), if scored
        delta: current_score minus baseline_score, when both are scored
        status: Classification of the change
        baseline_error: Error message from the baseline response, if any
        current_error: Error message from the current response, if any
    """

    test_case_id: str
    model: str
    baseline_score: Optional[int] = None
    current_score: Optional[int] = None
    delta: Optional[int] = None
    status: CaseStatus
    baseline_error: Optional[str] = None
    current_error: Optional[str] = None


class ModelComparison(BaseModel):
    """Aggregate comparison for one model across two runs.

    Attributes:
        model: Model identifier
        baseline_average: Average judge score in the baseline run
        current_average: Average judge score in the current run
        average_delta: current_average minus baseline_average, when both exist
        baseline_cost_usd: Total cost for this model in the baseline run
        current_cost_usd: Total cost for this model in the current run
        baseline_latency_ms: Total latency for this model in the baseline run
        current_latency_ms: Total latency for this model in the current run
        regressed: Number of cases whose score dropped
        improved: Number of cases whose score rose
        unchanged: Number of cases whose score is identical
        added: Number of cases only present in the current run
        removed: Number of cases only present in the baseline run
        unscored: Number of shared cases missing a judge score on either side
    """

    model: str
    baseline_average: Optional[float] = None
    current_average: Optional[float] = None
    average_delta: Optional[float] = None
    baseline_cost_usd: float = 0.0
    current_cost_usd: float = 0.0
    baseline_latency_ms: float = 0.0
    current_latency_ms: float = 0.0
    regressed: int = 0
    improved: int = 0
    unchanged: int = 0
    added: int = 0
    removed: int = 0
    unscored: int = 0


class GateFailure(BaseModel):
    """A single quality gate violation.

    Attributes:
        model: Model that violated the gate
        kind: "average" for a model average drop, "case" for a per-case drop
        test_case_id: The offending test case, for per-case violations
        drop: Magnitude of the score drop (positive number)
        allowed: The configured tolerance that was exceeded
    """

    model: str
    kind: str
    test_case_id: Optional[str] = None
    drop: float
    allowed: float


class RunComparison(BaseModel):
    """Full comparison of a current run against a baseline run.

    Attributes:
        baseline_run_id: Run ID of the baseline
        current_run_id: Run ID of the current run
        baseline_run_name: Human-readable baseline name, if set
        current_run_name: Human-readable current name, if set
        golden_set_name: Golden set of the current run
        golden_set_mismatch: True when the two runs used different golden sets
        model_comparisons: Aggregate comparison per model
        case_comparisons: Per (test case, model) comparisons
    """

    baseline_run_id: str
    current_run_id: str
    baseline_run_name: Optional[str] = None
    current_run_name: Optional[str] = None
    golden_set_name: str
    golden_set_mismatch: bool = False
    model_comparisons: List[ModelComparison] = Field(default_factory=list)
    case_comparisons: List[CaseComparison] = Field(default_factory=list)

    @property
    def regressions(self) -> List[CaseComparison]:
        """All per-case regressions, worst first."""
        regressed = [c for c in self.case_comparisons if c.status == CaseStatus.REGRESSED]
        return sorted(regressed, key=lambda c: c.delta if c.delta is not None else 0)

    @property
    def has_regressions(self) -> bool:
        """True when any case regressed or any model average dropped."""
        if self.regressions:
            return True
        return any(
            m.average_delta is not None and m.average_delta < 0
            for m in self.model_comparisons
        )

    def check_gates(
        self,
        max_regression: Optional[float] = None,
        max_case_regression: Optional[float] = None,
    ) -> List[GateFailure]:
        """Evaluate quality gates against this comparison.

        Args:
            max_regression: Maximum tolerated drop in a model's average judge
                score. 0 means any average drop fails.
            max_case_regression: Maximum tolerated drop in any single test
                case's judge score. 0 means any per-case drop fails.

        Returns:
            List of gate failures. Empty when all gates pass.
        """
        failures: List[GateFailure] = []

        if max_regression is not None:
            for mc in self.model_comparisons:
                if mc.average_delta is not None and -mc.average_delta > max_regression:
                    failures.append(
                        GateFailure(
                            model=mc.model,
                            kind="average",
                            drop=round(-mc.average_delta, 4),
                            allowed=max_regression,
                        )
                    )

        if max_case_regression is not None:
            for cc in self.case_comparisons:
                if cc.delta is not None and -cc.delta > max_case_regression:
                    failures.append(
                        GateFailure(
                            model=cc.model,
                            kind="case",
                            test_case_id=cc.test_case_id,
                            drop=float(-cc.delta),
                            allowed=max_case_regression,
                        )
                    )

        return failures


def _index_results(run: RunResult) -> Dict[Tuple[str, str], EvaluationResult]:
    """Index a run's results by (test_case_id, model).

    When duplicates exist for the same key, the last result wins, matching
    the order in which the runner appended them.
    """
    indexed: Dict[Tuple[str, str], EvaluationResult] = {}
    for result in run.results:
        indexed[(result.test_case_id, result.model_response.model)] = result
    return indexed


def _score_of(result: Optional[EvaluationResult]) -> Optional[int]:
    if result is None or result.judge_score is None:
        return None
    return result.judge_score.score


def compare_runs(
    baseline: RunResult,
    current: RunResult,
    model: Optional[str] = None,
) -> RunComparison:
    """Compare a current run against a baseline run.

    Pairs results by (test_case_id, model). Cases present on only one side
    are reported as added or removed rather than silently dropped, so a
    shrinking golden set cannot mask a regression.

    Args:
        baseline: The baseline run (for example, from the main branch)
        current: The current run (for example, from a pull request)
        model: Optional model identifier to restrict the comparison to

    Returns:
        A RunComparison with per-case and per-model deltas.
    """
    baseline_index = _index_results(baseline)
    current_index = _index_results(current)

    all_keys = sorted(set(baseline_index) | set(current_index))
    if model is not None:
        all_keys = [k for k in all_keys if k[1] == model]

    case_comparisons: List[CaseComparison] = []
    for test_case_id, case_model in all_keys:
        b = baseline_index.get((test_case_id, case_model))
        c = current_index.get((test_case_id, case_model))
        b_score = _score_of(b)
        c_score = _score_of(c)

        if b is None:
            status = CaseStatus.ADDED
        elif c is None:
            status = CaseStatus.REMOVED
        elif b_score is None or c_score is None:
            status = CaseStatus.UNSCORED
        elif c_score < b_score:
            status = CaseStatus.REGRESSED
        elif c_score > b_score:
            status = CaseStatus.IMPROVED
        else:
            status = CaseStatus.UNCHANGED

        delta = None
        if b_score is not None and c_score is not None:
            delta = c_score - b_score

        case_comparisons.append(
            CaseComparison(
                test_case_id=test_case_id,
                model=case_model,
                baseline_score=b_score,
                current_score=c_score,
                delta=delta,
                status=status,
                baseline_error=b.model_response.error if b else None,
                current_error=c.model_response.error if c else None,
            )
        )

    models = sorted(
        {m for m in baseline.models_tested} | {m for m in current.models_tested}
    )
    if model is not None:
        models = [m for m in models if m == model]

    model_comparisons: List[ModelComparison] = []
    for m in models:
        cases = [c for c in case_comparisons if c.model == m]
        baseline_avg = baseline.get_average_score(m)
        current_avg = current.get_average_score(m)
        average_delta = None
        if baseline_avg is not None and current_avg is not None:
            average_delta = current_avg - baseline_avg

        model_comparisons.append(
            ModelComparison(
                model=m,
                baseline_average=baseline_avg,
                current_average=current_avg,
                average_delta=average_delta,
                baseline_cost_usd=baseline.get_total_cost(m),
                current_cost_usd=current.get_total_cost(m),
                baseline_latency_ms=baseline.get_total_latency(m),
                current_latency_ms=current.get_total_latency(m),
                regressed=sum(1 for c in cases if c.status == CaseStatus.REGRESSED),
                improved=sum(1 for c in cases if c.status == CaseStatus.IMPROVED),
                unchanged=sum(1 for c in cases if c.status == CaseStatus.UNCHANGED),
                added=sum(1 for c in cases if c.status == CaseStatus.ADDED),
                removed=sum(1 for c in cases if c.status == CaseStatus.REMOVED),
                unscored=sum(1 for c in cases if c.status == CaseStatus.UNSCORED),
            )
        )

    return RunComparison(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        baseline_run_name=baseline.run_name,
        current_run_name=current.run_name,
        golden_set_name=current.golden_set_name,
        golden_set_mismatch=baseline.golden_set_name != current.golden_set_name,
        model_comparisons=model_comparisons,
        case_comparisons=case_comparisons,
    )


def _fmt_avg(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def _fmt_delta(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}"


def render_markdown(comparison: RunComparison) -> str:
    """Render a comparison as a markdown report.

    The output is sized for a pull request comment: a per-model summary
    table, then only the cases that changed.

    Args:
        comparison: The run comparison to render

    Returns:
        Markdown string
    """
    lines: List[str] = []
    lines.append("# PromptLens Comparison")
    lines.append("")
    lines.append(f"**Baseline:** `{comparison.baseline_run_id}`"
                 + (f" ({comparison.baseline_run_name})" if comparison.baseline_run_name else ""))
    lines.append("")
    lines.append(f"**Current:** `{comparison.current_run_id}`"
                 + (f" ({comparison.current_run_name})" if comparison.current_run_name else ""))
    lines.append("")

    if comparison.golden_set_mismatch:
        lines.append(
            "> **Warning:** the two runs used different golden sets. "
            "Score deltas may not be meaningful."
        )
        lines.append("")

    lines.append("## Models")
    lines.append("")
    lines.append("| Model | Baseline Avg | Current Avg | Delta | Regressed | Improved | Unchanged |")
    lines.append("|-------|--------------|-------------|-------|-----------|----------|-----------|")
    for mc in comparison.model_comparisons:
        lines.append(
            f"| {mc.model} | {_fmt_avg(mc.baseline_average)} | {_fmt_avg(mc.current_average)} "
            f"| {_fmt_delta(mc.average_delta)} | {mc.regressed} | {mc.improved} | {mc.unchanged} |"
        )
    lines.append("")

    changed = [
        c
        for c in comparison.case_comparisons
        if c.status in (CaseStatus.REGRESSED, CaseStatus.IMPROVED, CaseStatus.ADDED, CaseStatus.REMOVED)
    ]
    if changed:
        lines.append("## Changed Cases")
        lines.append("")
        lines.append("| Test Case | Model | Baseline | Current | Delta | Status |")
        lines.append("|-----------|-------|----------|---------|-------|--------|")
        for c in changed:
            baseline_display = f"{c.baseline_score}/5" if c.baseline_score is not None else "n/a"
            current_display = f"{c.current_score}/5" if c.current_score is not None else "n/a"
            delta_display = f"{c.delta:+d}" if c.delta is not None else "n/a"
            lines.append(
                f"| `{c.test_case_id}` | {c.model} | {baseline_display} "
                f"| {current_display} | {delta_display} | {c.status.value} |"
            )
        lines.append("")
    else:
        lines.append("No score changes between runs.")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Generated by [PromptLens](https://github.com/brainsparker/PromptLens)*")
    return "\n".join(lines)
