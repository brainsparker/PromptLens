"""Base judge interface."""

import asyncio
from abc import ABC, abstractmethod

from promptlens.judges.stability import DEFAULT_DISAGREEMENT_RANGE, aggregate_judge_scores
from promptlens.models.result import JudgeScore, ModelResponse
from promptlens.models.test_case import TestCase


class BaseJudge(ABC):
    """Abstract base class for judges.

    All judge implementations must inherit from this class and implement
    the abstract methods.
    """

    @abstractmethod
    async def evaluate(
        self,
        test_case: TestCase,
        model_response: ModelResponse,
    ) -> JudgeScore:
        """Evaluate a model response against a test case.

        Args:
            test_case: The test case with expected behavior
            model_response: The model's response to evaluate

        Returns:
            JudgeScore with score (1-5) and explanation

        Raises:
            Exception: If evaluation fails
        """
        pass

    async def evaluate_sampled(
        self,
        test_case: TestCase,
        model_response: ModelResponse,
        samples: int = 1,
        disagreement_range: int = DEFAULT_DISAGREEMENT_RANGE,
    ) -> JudgeScore:
        """Judge a response ``samples`` times and aggregate the verdicts.

        With ``samples == 1`` this is exactly ``evaluate``. With more samples
        the judge calls run concurrently and are folded into one JudgeScore
        that carries the individual verdicts and stability statistics (see
        ``promptlens.judges.stability.aggregate_judge_scores``).

        Args:
            test_case: The test case with expected behavior
            model_response: The model's response to evaluate
            samples: Number of independent judge calls (at least 1)
            disagreement_range: Spread at which samples count as disagreement

        Returns:
            Aggregated JudgeScore
        """
        if samples < 1:
            raise ValueError("samples must be at least 1")
        if samples == 1:
            return await self.evaluate(test_case, model_response)

        verdicts = await asyncio.gather(
            *(self.evaluate(test_case, model_response) for _ in range(samples))
        )
        return aggregate_judge_scores(list(verdicts), disagreement_range=disagreement_range)

    @property
    @abstractmethod
    def judge_model(self) -> str:
        """Return the judge model identifier.

        Returns:
            Model identifier (e.g., "claude-3-5-sonnet-20241022")
        """
        pass

    @property
    @abstractmethod
    def judge_provider(self) -> str:
        """Return the judge provider name.

        Returns:
            Provider name (e.g., "anthropic", "openai")
        """
        pass
