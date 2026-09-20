"""Judge stability: aggregate repeated judge samples and report disagreement.

LLM judges are not deterministic instruments. The same response, judged twice
with the same prompt and the same model, can receive different scores. When a
single judge verdict decides a CI gate, that noise turns into flaky builds:
the build passes on one run and fails on the next with no change to the code
under test.

This module implements the aggregation half of judge sampling:

- ``aggregate_judge_scores`` folds N independent JudgeScore samples into one
  JudgeScore whose ``score`` is the median verdict and which carries the full
  sample list, mean, standard deviation, min, max, and a disagreement flag.
- ``straddles_threshold`` tells whether a sampled score has verdicts on both
  sides of a pass threshold, which is exactly the case where the gate verdict
  depends on which judge roll you got.
- ``summarize_stability`` rolls the per-result flags up into a run-level
  summary for the CLI, exporters, and the ``--fail-on-judge-disagreement``
  gate.
"""

import math
from dataclasses import dataclass, field
from statistics import mean, median, pstdev
from typing import List, Optional, Sequence

from promptlens.models.result import EvaluationResult, JudgeScore, RunResult

# Default spread (max sample minus min sample) that counts as disagreement.
DEFAULT_DISAGREEMENT_RANGE = 2


def _round_half_up(value: float) -> int:
    """Round to the nearest integer with .5 going up (Python's round goes to even)."""
    return int(math.floor(value + 0.5))


def aggregate_judge_scores(
    samples: Sequence[JudgeScore],
    disagreement_range: int = DEFAULT_DISAGREEMENT_RANGE,
) -> JudgeScore:
    """Fold repeated judge samples into a single JudgeScore with stability stats.

    The aggregate ``score`` is the median sample (rounded half up when the
    median lands on .5), because the median is robust to one outlier verdict.
    ``score_mean`` keeps the exact mean for averages and reports. The
    explanation, criteria scores, and tool evaluation fields are taken from a
    representative sample: the first sample whose score equals the aggregate.

    Args:
        samples: Judge scores for the same response. Must not be empty.
        disagreement_range: Spread at which the samples count as disagreement.

    Returns:
        One JudgeScore. A single sample is returned unchanged.

    Raises:
        ValueError: If samples is empty.
    """
    if not samples:
        raise ValueError("aggregate_judge_scores requires at least one sample")
    if len(samples) == 1:
        return samples[0]

    scores = [s.score for s in samples]
    aggregate_score = _round_half_up(median(scores))
    aggregate_score = min(5, max(1, aggregate_score))

    representative = next((s for s in samples if s.score == aggregate_score), None)
    if representative is None:
        # Median fell between two distinct values, pick the closest sample.
        representative = min(samples, key=lambda s: abs(s.score - aggregate_score))

    score_min = min(scores)
    score_max = max(scores)

    return JudgeScore(
        score=aggregate_score,
        explanation=representative.explanation,
        criteria_scores=dict(representative.criteria_scores),
        judge_model=representative.judge_model,
        judge_provider=representative.judge_provider,
        timestamp=samples[-1].timestamp,
        tool_evaluations=list(representative.tool_evaluations),
        tool_usage_score=representative.tool_usage_score,
        tool_efficiency_score=representative.tool_efficiency_score,
        sample_scores=scores,
        sample_explanations=[s.explanation for s in samples],
        score_mean=float(mean(scores)),
        score_std=float(pstdev(scores)),
        score_min=score_min,
        score_max=score_max,
        disagreement=(score_max - score_min) >= disagreement_range,
    )


def straddles_threshold(score: JudgeScore, threshold: Optional[float]) -> bool:
    """Return True when the samples fall on both sides of a pass threshold.

    A sample passes when it is greater than or equal to the threshold, which
    matches the ``--fail-under`` and JUnit semantics (strictly below fails).
    Unsampled scores never straddle: there is only one verdict.

    Args:
        score: A judge score, sampled or not
        threshold: The gate threshold on the 1-5 scale, or None for no gate
    """
    if threshold is None or not score.is_sampled:
        return False
    if score.score_min is None or score.score_max is None:
        return False
    return score.score_min < threshold <= score.score_max


@dataclass
class UnstableCase:
    """One judged response whose judge verdicts did not agree."""

    test_case_id: str
    model: str
    score: int
    score_min: int
    score_max: int
    score_std: float
    disagreement: bool
    straddles_gate: bool

    @property
    def spread_label(self) -> str:
        """Compact "min-max" label for reports."""
        return f"{self.score_min}-{self.score_max}"


@dataclass
class StabilitySummary:
    """Run-level roll-up of judge stability."""

    samples_per_response: int
    judged_results: int
    disagreements: int
    gate_straddles: int
    mean_std: Optional[float]
    threshold: Optional[float]
    unstable_cases: List[UnstableCase] = field(default_factory=list)

    @property
    def sampled(self) -> bool:
        """True when the run judged each response more than once."""
        return self.samples_per_response > 1

    @property
    def disagreement_rate(self) -> Optional[float]:
        """Share of judged results flagged as disagreement (None without results)."""
        if not self.judged_results:
            return None
        return self.disagreements / self.judged_results

    @property
    def has_gate_risk(self) -> bool:
        """True when at least one gate verdict depends on which judge sample won."""
        return self.gate_straddles > 0


def summarize_stability(
    result: RunResult,
    threshold: Optional[float] = None,
) -> StabilitySummary:
    """Roll per-result judge stability up to the run level.

    Args:
        result: A completed run
        threshold: The quality gate threshold, if one is in effect. Used to
            count responses whose samples straddle the gate.

    Returns:
        StabilitySummary. For an unsampled run every counter is zero.
    """
    judged: List[EvaluationResult] = [r for r in result.results if r.judge_score is not None]
    unstable: List[UnstableCase] = []
    stds: List[float] = []
    disagreements = 0
    straddles = 0

    for eval_result in judged:
        score = eval_result.judge_score
        assert score is not None  # for type checkers
        if not score.is_sampled:
            continue
        if score.score_std is not None:
            stds.append(score.score_std)
        straddle = straddles_threshold(score, threshold)
        if score.disagreement:
            disagreements += 1
        if straddle:
            straddles += 1
        if score.disagreement or straddle:
            unstable.append(
                UnstableCase(
                    test_case_id=eval_result.test_case_id,
                    model=eval_result.model_response.model,
                    score=score.score,
                    score_min=score.score_min if score.score_min is not None else score.score,
                    score_max=score.score_max if score.score_max is not None else score.score,
                    score_std=score.score_std or 0.0,
                    disagreement=score.disagreement,
                    straddles_gate=straddle,
                )
            )

    return StabilitySummary(
        samples_per_response=result.judge_samples,
        judged_results=len(judged),
        disagreements=disagreements,
        gate_straddles=straddles,
        mean_std=(sum(stds) / len(stds)) if stds else None,
        threshold=threshold,
        unstable_cases=unstable,
    )
