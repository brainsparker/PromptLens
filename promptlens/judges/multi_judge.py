"""Multi-judge consensus scoring.

Runs a panel of LLM judges in parallel over each response and reports
the panel median as the consensus score, along with an inter-judge
agreement signal. A single judge is an uncalibrated number; a small
panel with an agreement gap makes low-confidence scores visible.
"""

import asyncio
import logging
import math
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from promptlens.judges.base import BaseJudge
from promptlens.judges.llm_judge import LLMJudge
from promptlens.models.config import JudgeConfig
from promptlens.models.result import IndividualJudgeScore, JudgeScore, ModelResponse
from promptlens.models.test_case import TestCase

logger = logging.getLogger(__name__)


class MultiJudge(BaseJudge):
    """Consensus judge that aggregates a panel of LLM judges.

    Each judge on the panel evaluates the response independently and in
    parallel. The consensus score is the panel median (rounded half up),
    which is robust to a single outlier judge. Results where the panel
    spread exceeds the configured agreement threshold are flagged as
    low confidence so they can be surfaced in reports.
    """

    def __init__(self, config: JudgeConfig) -> None:
        """Initialize the multi-judge panel.

        Args:
            config: Judge configuration with a non-empty judges panel

        Raises:
            ValueError: If the config does not define a judge panel
        """
        if len(config.judges) < 2:
            raise ValueError("MultiJudge requires at least two judges in config.judges")

        self.config = config
        self.judges: List[LLMJudge] = []

        for instance in config.judges:
            judge_config = JudgeConfig(
                provider=instance.provider,
                model=instance.model,
                temperature=instance.temperature,
                custom_prompt=config.custom_prompt,
                criteria=config.criteria,
            )
            self.judges.append(LLMJudge(judge_config))

    async def evaluate(
        self,
        test_case: TestCase,
        model_response: ModelResponse,
    ) -> JudgeScore:
        """Evaluate a model response with the full judge panel.

        Args:
            test_case: The test case with expected behavior
            model_response: The model's response to evaluate

        Returns:
            Consensus JudgeScore with per-judge scores and agreement signal
        """
        panel_scores = await asyncio.gather(
            *(judge.evaluate(test_case, model_response) for judge in self.judges)
        )

        scores = [s.score for s in panel_scores]
        consensus = self._median_score(scores)
        gap = max(scores) - min(scores)
        low_confidence = gap > self.config.agreement_threshold

        individual = [
            IndividualJudgeScore(
                score=s.score,
                explanation=s.explanation,
                judge_model=s.judge_model,
                judge_provider=s.judge_provider,
            )
            for s in panel_scores
        ]

        explanation_lines = [
            f"Consensus of {len(panel_scores)} judges: median {consensus}/5, "
            f"agreement gap {gap}"
            + (" (low confidence: judges disagree)" if low_confidence else "")
            + "."
        ]
        for s in panel_scores:
            explanation_lines.append(f"[{s.judge_model}: {s.score}/5] {s.explanation}")

        # Tool evaluation fields: the automatic stage is deterministic, so
        # every judge produces identical tool_evaluations; take the first
        # non-empty set. Usage/efficiency scores are averaged across judges.
        tool_evaluations = next(
            (s.tool_evaluations for s in panel_scores if s.tool_evaluations), []
        )
        usage_scores = [
            s.tool_usage_score for s in panel_scores if s.tool_usage_score is not None
        ]
        efficiency_scores = [
            s.tool_efficiency_score
            for s in panel_scores
            if s.tool_efficiency_score is not None
        ]

        return JudgeScore(
            score=consensus,
            explanation="\n".join(explanation_lines),
            criteria_scores=self._aggregate_criteria(panel_scores),
            judge_model=self.judge_model,
            judge_provider=self.judge_provider,
            timestamp=datetime.utcnow(),
            tool_evaluations=tool_evaluations,
            tool_usage_score=(
                sum(usage_scores) / len(usage_scores) if usage_scores else None
            ),
            tool_efficiency_score=(
                sum(efficiency_scores) / len(efficiency_scores)
                if efficiency_scores
                else None
            ),
            individual_scores=individual,
            agreement_gap=gap,
            low_confidence=low_confidence,
        )

    @staticmethod
    def _median_score(scores: List[int]) -> int:
        """Compute the panel median, rounded half up, clamped to 1-5.

        Args:
            scores: Individual judge scores

        Returns:
            Consensus score (1-5)
        """
        median = statistics.median(scores)
        return max(1, min(5, math.floor(median + 0.5)))

    @staticmethod
    def _aggregate_criteria(panel_scores: List[JudgeScore]) -> Dict[str, int]:
        """Average per-criterion scores across the panel.

        Only criteria reported by at least one judge are included; each
        criterion is averaged over the judges that reported it.

        Args:
            panel_scores: Scores from each judge on the panel

        Returns:
            Mapping of criterion name to rounded average score
        """
        totals: Dict[str, List[int]] = defaultdict(list)
        for judge_score in panel_scores:
            for criterion, value in judge_score.criteria_scores.items():
                totals[criterion].append(value)

        return {
            criterion: max(1, min(5, math.floor(sum(values) / len(values) + 0.5)))
            for criterion, values in totals.items()
        }

    @property
    def judge_model(self) -> str:
        """Return a combined identifier for the panel.

        Returns:
            Plus-joined list of panel model identifiers
        """
        return "+".join(j.model for j in self.config.judges)

    @property
    def judge_provider(self) -> str:
        """Return a combined provider name for the panel.

        Returns:
            "consensus" plus the plus-joined provider list
        """
        return "consensus:" + "+".join(j.provider for j in self.config.judges)
