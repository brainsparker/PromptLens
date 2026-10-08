"""Aggregate repeated judge runs into one score with stability metadata.

LLM judges are not deterministic. The same response, judged twice by the
same model at the same temperature, can land on different scores. When an
eval gate is built on a single judge call, that noise shows up as a flaky
CI check that teams re-run until it passes. Sampling the judge several
times and aggregating makes the noise visible (as per-case agreement and a
per-model flip rate) and makes the final score more robust to it.

This module is pure: it has no I/O and no provider calls, so it is easy to
test and reuse from custom judges.
"""

import statistics
from collections import Counter
from typing import Dict, List, Sequence

from promptlens.models.result import JudgeScore


def aggregate_scores(samples: Sequence[int], method: str = "median") -> int:
    """Collapse sampled integer scores into one score on the 1-5 scale.

    Args:
        samples: Individual judge scores (each 1-5). Must be non-empty.
        method: "median", "mean", or "min".
            - median: statistics.median_high, so the result is always one of
              the sampled values (an even count never produces 3.5).
            - mean: arithmetic mean rounded half up to the nearest integer.
            - min: the lowest sample, the strictest reading.

    Returns:
        Aggregated integer score, clamped to 1-5.

    Raises:
        ValueError: If samples is empty or method is unknown.
    """
    if not samples:
        raise ValueError("cannot aggregate an empty list of judge scores")

    normalized = method.strip().lower()
    if normalized == "median":
        value = statistics.median_high(samples)
    elif normalized == "mean":
        # Round half up so 3.5 becomes 4, matching how people read a score.
        value = int(sum(samples) / len(samples) + 0.5)
    elif normalized == "min":
        value = min(samples)
    else:
        raise ValueError(f"unknown judge aggregation: {method}")

    return max(1, min(5, int(value)))


def compute_agreement(samples: Sequence[int], final_score: int) -> float:
    """Return the fraction of samples equal to the final score.

    Args:
        samples: Individual judge scores. Must be non-empty.
        final_score: The aggregated score the samples are compared against.

    Returns:
        A value in [0.0, 1.0].
    """
    if not samples:
        raise ValueError("cannot compute agreement for an empty list of judge scores")
    matching = sum(1 for s in samples if s == final_score)
    return matching / len(samples)


def _aggregate_criteria(scores: Sequence[JudgeScore], method: str) -> Dict[str, int]:
    """Aggregate per-criterion sub-scores across runs, key by key.

    Keys missing from some runs are aggregated over the runs that have them.
    """
    keys: List[str] = []
    for score in scores:
        for key in score.criteria_scores:
            if key not in keys:
                keys.append(key)

    aggregated: Dict[str, int] = {}
    for key in keys:
        values = [s.criteria_scores[key] for s in scores if key in s.criteria_scores]
        if values:
            aggregated[key] = aggregate_scores(values, method)
    return aggregated


def aggregate_judge_scores(scores: Sequence[JudgeScore], method: str = "median") -> JudgeScore:
    """Combine several JudgeScore objects from repeated runs into one.

    The returned score keeps the explanation (and tool evaluation details)
    from the first run that landed on the aggregated score, so the text a
    reader sees always justifies the number next to it. The explanation is
    prefixed with the full sample list so disagreement is never hidden.

    Args:
        scores: One JudgeScore per judge run. Must be non-empty.
        method: Aggregation method, see aggregate_scores.

    Returns:
        A single JudgeScore with runs, samples, and agreement populated.
    """
    if not scores:
        raise ValueError("cannot aggregate an empty list of judge scores")

    if len(scores) == 1:
        only = scores[0]
        return only.model_copy(update={"runs": 1, "samples": [only.score], "agreement": 1.0})

    samples = [s.score for s in scores]
    final_score = aggregate_scores(samples, method)
    agreement = compute_agreement(samples, final_score)

    representative = next((s for s in scores if s.score == final_score), scores[0])

    distribution = Counter(samples)
    sample_text = ", ".join(str(s) for s in samples)
    if len(distribution) == 1:
        prefix = f"[{len(samples)} judge runs, all scored {samples[0]}] "
    else:
        prefix = (
            f"[{len(samples)} judge runs scored {sample_text}; "
            f"{method} = {final_score}, agreement {agreement:.0%}] "
        )

    tool_usage = [s.tool_usage_score for s in scores if s.tool_usage_score is not None]
    tool_efficiency = [
        s.tool_efficiency_score for s in scores if s.tool_efficiency_score is not None
    ]

    return JudgeScore(
        score=final_score,
        explanation=prefix + representative.explanation,
        criteria_scores=_aggregate_criteria(scores, method),
        judge_model=representative.judge_model,
        judge_provider=representative.judge_provider,
        timestamp=scores[-1].timestamp,
        tool_evaluations=representative.tool_evaluations,
        tool_usage_score=(sum(tool_usage) / len(tool_usage)) if tool_usage else None,
        tool_efficiency_score=(
            sum(tool_efficiency) / len(tool_efficiency) if tool_efficiency else None
        ),
        runs=len(samples),
        samples=samples,
        agreement=agreement,
    )
