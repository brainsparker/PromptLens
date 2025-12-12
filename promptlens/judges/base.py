"""Base judge interface."""

from abc import ABC, abstractmethod

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
