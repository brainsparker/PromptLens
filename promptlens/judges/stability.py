"""Multi-sample judge orchestration and score aggregation.

LLM-as-judge scores are noisy: the same judge model, prompt, and response can
produce different scores across runs, even at low temperature. This module
runs the judge multiple times per response and aggregates the samples into a
single JudgeScore that carries stability metadata (mean, standard deviation,
and the raw sample scores), so downstream consumers such as quality gates can
reason about score confidence instead of trusting a single sample.
"""

import asyncio
import logging
import statistics
from typing import List

from promptlens.judges.base import BaseJudge
from promptlens.models.result import JudgeScore, ModelResponse
from promptlens.models.test_case import TestCase

logger = logging.getLogger(__name__)

# A per-case sample standard deviation at or above this value marks the case
# as unstable: on a 1-5 scale it means samples disagree by a full point.
UNSTABLE_STDEV_THRESHOLD = 1.0


def aggregate_judge_scores(samples: List[JudgeScore]) -> JudgeScore:
    """Aggregate multiple judge samples into a single stability-aware score.

    The aggregate score is the median of the sample scores (rounded to the
    nearest integer and clamped to 1-5), which is robust to a single outlier
    sample. The mean and sample standard deviation are attached as stability
    metadata. Explanation, criteria scores, and tool evaluation fields are
    taken from the representative sample: the first sample whose score is
    closest to the median.

    Args:
        samples: Judge scores from repeated evaluations of the same response.
            Must contain at least one score.

    Returns:
        A JudgeScore with sample_scores, score_mean, and score_stdev populated.

    Raises:
        ValueError: If samples is empty.
    """
    if not samples:
        raise ValueError("aggregate_judge_scores requires at least one sample")

    if len(samples) == 1:
        single = samples[0]
        return single.model_copy(
            update={
                "sample_scores": [single.score],
                "score_mean": float(single.score),
                "score_stdev": 0.0,
            }
        )

    raw_scores = [s.score for s in samples]
    median = statistics.median(raw_scores)
    aggregate_score = max(1, min(5, int(round(median))))
    mean = statistics.mean(raw_scores)
    stdev = statistics.stdev(raw_scores)

    # Representative sample: first sample closest to the median.
    representative = min(samples, key=lambda s: abs(s.score - median))

    explanation = representative.explanation
    if stdev > 0:
        explanation = (
            f"{explanation}\n\n[Judge stability: {len(samples)} samples, "
            f"scores {sorted(raw_scores)}, mean {mean:.2f}, stdev {stdev:.2f}]"
        )

    return representative.model_copy(
        update={
            "score": aggregate_score,
            "explanation": explanation,
            "sample_scores": raw_scores,
            "score_mean": mean,
            "score_stdev": stdev,
        }
    )


async def sample_judge_scores(
    judge: BaseJudge,
    test_case: TestCase,
    model_response: ModelResponse,
    samples: int,
) -> JudgeScore:
    """Run the judge one or more times and return an aggregated score.

    Samples are evaluated concurrently. Failed samples are logged and
    dropped; as long as at least one sample succeeds, an aggregate score is
    returned. If every sample fails, the first failure is re-raised so the
    caller's existing error handling applies.

    Args:
        judge: The judge to run.
        test_case: The test case with expected behavior.
        model_response: The model's response to evaluate.
        samples: Number of independent judge evaluations to run (>= 1).

    Returns:
        Aggregated JudgeScore with stability metadata.

    Raises:
        ValueError: If samples is less than 1.
        Exception: The first sample failure, when all samples fail.
    """
    if samples < 1:
        raise ValueError("samples must be at least 1")

    if samples == 1:
        return aggregate_judge_scores(
            [await judge.evaluate(test_case, model_response)]
        )

    outcomes = await asyncio.gather(
        *(judge.evaluate(test_case, model_response) for _ in range(samples)),
        return_exceptions=True,
    )

    scores: List[JudgeScore] = []
    failures: List[BaseException] = []
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            failures.append(outcome)
        else:
            scores.append(outcome)

    if failures:
        logger.warning(
            f"{len(failures)}/{samples} judge samples failed for test case "
            f"'{test_case.id}': {failures[0]}"
        )

    if not scores:
        raise failures[0]

    return aggregate_judge_scores(scores)
