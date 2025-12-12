"""LLM-as-judge implementation."""

import logging
from datetime import datetime

from promptlens.judges.base import BaseJudge
from promptlens.judges.parser import parse_judge_response, validate_score
from promptlens.judges.prompts import format_judge_prompt
from promptlens.models.config import JudgeConfig, ModelConfig, ProviderConfig
from promptlens.models.result import JudgeScore, ModelResponse
from promptlens.models.test_case import TestCase
from promptlens.providers.factory import get_provider

logger = logging.getLogger(__name__)


class LLMJudge(BaseJudge):
    """LLM-as-judge implementation using another LLM to evaluate responses.

    Uses a separate LLM (typically Claude or GPT-4) to evaluate model responses.
    """

    def __init__(self, config: JudgeConfig) -> None:
        """Initialize the LLM judge.

        Args:
            config: Judge configuration
        """
        self.config = config

        # Create a model config for the judge
        judge_model_config = ModelConfig(
            name=f"Judge ({config.model})",
            provider=config.provider,
            model=config.model,
            temperature=config.temperature,
            max_tokens=2048,  # Judges need more tokens for explanations
        )

        # Get the provider for the judge
        self.provider = get_provider(judge_model_config)

    async def evaluate(
        self,
        test_case: TestCase,
        model_response: ModelResponse,
    ) -> JudgeScore:
        """Evaluate a model response using LLM-as-judge.

        Args:
            test_case: The test case with expected behavior
            model_response: The model's response to evaluate

        Returns:
            JudgeScore with score (1-5) and explanation
        """
        # Format the judge prompt
        prompt = format_judge_prompt(
            query=test_case.query,
            expected_behavior=test_case.expected_behavior,
            response=model_response.content,
            custom_prompt=self.config.custom_prompt,
        )

        # Get judge's evaluation
        try:
            judge_response = await self.provider.generate(prompt)

            # Parse the response
            score, explanation = parse_judge_response(judge_response.content)

            # Validate score
            validated_score = validate_score(score)

            # If parsing failed, add note to explanation
            if score is None:
                explanation = (
                    f"[Score parsing failed, using default score of {validated_score}] "
                    + explanation
                )

            return JudgeScore(
                score=validated_score,
                explanation=explanation,
                judge_model=self.config.model,
                judge_provider=self.config.provider,
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            # Return a default score with error explanation
            return JudgeScore(
                score=3,  # Default middle score
                explanation=f"Judge evaluation failed: {str(e)}",
                judge_model=self.config.model,
                judge_provider=self.config.provider,
                timestamp=datetime.utcnow(),
            )

    @property
    def judge_model(self) -> str:
        """Return the judge model identifier.

        Returns:
            Model identifier
        """
        return self.config.model

    @property
    def judge_provider(self) -> str:
        """Return the judge provider name.

        Returns:
            Provider name
        """
        return self.config.provider
