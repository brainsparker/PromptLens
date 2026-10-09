"""Multi-sample stability analysis: pass@k, pass^k, and flaky-case detection.

A single run per test case is a single draw from a distribution. Model
sampling, provider-side drift, inference nondeterminism (even at
temperature 0) and the LLM judge itself all add noise, so one score cannot
tell a regression from a bad roll. Running each test case several times and
looking at the spread does.

Terminology (k is the number of samples per test case):
    - pass@k: the share of test cases that passed in at least one sample.
      Optimistic bound: what the model can do.
    - pass^k: the share of test cases that passed in every sample.
      Pessimistic bound: what the model reliably does. This is the number to
      gate releases on.
    - flaky case: a test case that passed in some samples and failed in
      others. The gap between pass@k and pass^k is made of flaky cases.

A sample "passes" when its judge score is at or above ``pass_threshold``,
matching the per-test threshold used by the JUnit exporter and the
``--fail-under`` gate. Samples that errored or were never judged are left
out of pass/fail counts, so an infrastructure failure is never reported as a
quality failure.
"""

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from promptlens.models.result import EvaluationResult, RunResult
from promptlens.models.stability import (
    CaseStability,
    ModelStability,
    StabilityReport,
)

# Judge scores are on a 1-5 scale. Scores below this value count as failures
# unless a different threshold is provided. Kept in sync with the JUnit
# exporter's DEFAULT_FAIL_UNDER (which imports this constant).
DEFAULT_PASS_THRESHOLD = 3.0

# Two-sided 95% confidence, normal approximation.
_Z_95 = 1.96


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _sample_std_dev(values: List[float]) -> Optional[float]:
    """Sample standard deviation (n - 1 denominator); 0.0 for a single value."""
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def confidence_interval(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """95% confidence interval for the mean, normal approximation.

    Returns (None, None) when there are no values, and a zero-width interval
    for a single value. With small sample counts this is a rough guide, not a
    rigorous bound; it is here so two averages that differ by less than
    their own uncertainty are not read as a real change.
    """
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, mean
    std_dev = _sample_std_dev(values) or 0.0
    half_width = _Z_95 * std_dev / math.sqrt(len(values))
    return mean - half_width, mean + half_width


def group_samples(
    run: RunResult,
) -> "OrderedDict[Tuple[str, str], List[EvaluationResult]]":
    """Group a run's results by (test_case_id, model), preserving first-seen order.

    Within a group, results are ordered by sample_index.
    """
    groups: "OrderedDict[Tuple[str, str], List[EvaluationResult]]" = OrderedDict()
    for result in run.results:
        key = (result.test_case_id, result.model_response.model)
        groups.setdefault(key, []).append(result)
    for samples in groups.values():
        samples.sort(key=lambda r: r.sample_index)
    return groups


def analyze_case(
    test_case_id: str,
    model: str,
    samples: List[EvaluationResult],
    pass_threshold: float,
) -> CaseStability:
    """Compute stability statistics for one (test case, model) group."""
    scores: List[int] = []
    scored_indices: List[int] = []
    errors = 0
    for sample in samples:
        if sample.model_response.error:
            errors += 1
            continue
        if sample.judge_score is None:
            continue
        scores.append(sample.judge_score.score)
        scored_indices.append(sample.sample_index)

    float_scores = [float(s) for s in scores]
    passed = sum(1 for s in scores if s >= pass_threshold)
    scored = len(scores)

    pass_rate = (passed / scored) if scored else None
    pass_at_k = passed > 0
    pass_hat_k = scored > 0 and passed == scored
    flaky = pass_at_k and not pass_hat_k

    return CaseStability(
        test_case_id=test_case_id,
        model=model,
        query=samples[0].query if samples else "",
        samples=len(samples),
        scored_samples=scored,
        errored_samples=errors,
        scores=scores,
        mean_score=_mean(float_scores),
        min_score=min(scores) if scores else None,
        max_score=max(scores) if scores else None,
        std_dev=_sample_std_dev(float_scores),
        passed_samples=passed,
        pass_rate=pass_rate,
        pass_at_k=pass_at_k,
        pass_hat_k=pass_hat_k,
        flaky=flaky,
    )


def _analyze_model(
    model: str,
    cases: List[CaseStability],
    samples: int,
    pass_threshold: float,
) -> ModelStability:
    """Aggregate per-case stability for one model."""
    all_scores: List[float] = [float(s) for c in cases for s in c.scores]
    comparable = [c for c in cases if c.scored_samples > 0]
    ci_low, ci_high = confidence_interval(all_scores)

    case_std_devs = [c.std_dev for c in comparable if c.std_dev is not None]

    pass_at_k_count = sum(1 for c in comparable if c.pass_at_k)
    pass_hat_k_count = sum(1 for c in comparable if c.pass_hat_k)
    flaky_count = sum(1 for c in comparable if c.flaky)
    n = len(comparable)

    return ModelStability(
        model=model,
        samples=samples,
        pass_threshold=pass_threshold,
        cases=len(cases),
        scored_cases=n,
        mean_score=_mean(all_scores),
        score_ci_low=ci_low,
        score_ci_high=ci_high,
        mean_case_std_dev=_mean(case_std_devs),
        pass_at_k_cases=pass_at_k_count,
        pass_hat_k_cases=pass_hat_k_count,
        flaky_cases=flaky_count,
        pass_at_k=(pass_at_k_count / n) if n else None,
        pass_hat_k=(pass_hat_k_count / n) if n else None,
        flaky_rate=(flaky_count / n) if n else None,
    )


def compute_stability(
    run: RunResult,
    pass_threshold: Optional[float] = None,
) -> StabilityReport:
    """Analyze a run's per-case score stability across samples.

    Works on single-sample runs too (every scored case is then trivially
    stable), so callers do not have to special-case ``--samples 1``.

    Args:
        run: The run to analyze
        pass_threshold: Judge score (1-5) at or above which a sample passes.
            Defaults to the run's recorded ``pass_threshold`` metadata, then
            to DEFAULT_PASS_THRESHOLD.

    Returns:
        A StabilityReport with per-model and per-case statistics
    """
    if pass_threshold is None:
        recorded = run.metadata.get("pass_threshold")
        pass_threshold = float(recorded) if recorded is not None else DEFAULT_PASS_THRESHOLD

    groups = group_samples(run)
    samples = run.sample_count

    cases: List[CaseStability] = [
        analyze_case(case_id, model, results, pass_threshold)
        for (case_id, model), results in groups.items()
    ]

    models_seen: Dict[str, None] = dict.fromkeys(run.models_tested)
    for case in cases:
        models_seen.setdefault(case.model, None)

    model_stats = [
        _analyze_model(model, [c for c in cases if c.model == model], samples, pass_threshold)
        for model in models_seen
    ]

    return StabilityReport(
        run_id=run.run_id,
        samples=samples,
        pass_threshold=pass_threshold,
        models=model_stats,
        cases=cases,
    )


def format_scores(scores: List[int]) -> str:
    """Render a score list compactly, e.g. ``5, 2, 5``."""
    return ", ".join(str(s) for s in scores) if scores else "-"
