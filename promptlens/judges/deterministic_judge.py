"""Deterministic judge: scores responses from assertions alone.

No judge model, no API key, no sampling. The same response and the same
assertions always produce the same score, which is what a CI merge gate needs.
LLM-as-judge scores remain the right tool for open-ended quality questions;
this judge covers the decidable properties (required phrases, formats, length,
overlap with a reference) that can safely block a build.
"""

import logging
from datetime import datetime
from typing import List

from promptlens.judges.assertions import evaluate_assertions
from promptlens.judges.base import BaseJudge
from promptlens.models.config import JudgeConfig
from promptlens.models.result import AssertionResult, JudgeScore, ModelResponse
from promptlens.models.test_case import TestCase

logger = logging.getLogger(__name__)

DETERMINISTIC_JUDGE_MODEL = "deterministic"
DETERMINISTIC_JUDGE_PROVIDER = "promptlens"


def score_from_assertions(results: List[AssertionResult]) -> int:
    """Map assertion outcomes onto the 1-5 judge scale.

    All assertions passing scores 5, none passing scores 1, and partial
    passes scale linearly in between, so existing --fail-under gates and
    report score distributions keep working unchanged.

    Args:
        results: Assertion outcomes for one response

    Returns:
        Integer score from 1 to 5
    """
    if not results:
        raise ValueError("cannot score a response with no assertion results")
    pass_ratio = sum(1 for r in results if r.passed) / len(results)
    return 1 + int(round(pass_ratio * 4))


def explain_assertions(results: List[AssertionResult]) -> str:
    """Render assertion outcomes as a compact, human-readable explanation."""
    passed = sum(1 for r in results if r.passed)
    lines = [f"Deterministic assertions: {passed}/{len(results)} passed."]
    for r in results:
        marker = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{marker}] {r.label}: {r.detail}")
    return "\n".join(lines)


class DeterministicJudge(BaseJudge):
    """Judge that scores purely from a test case's deterministic assertions.

    Test cases without assertions are skipped (see can_evaluate) rather than
    assigned a placeholder score, so a partially annotated golden set reports
    what it can measure and nothing else.
    """

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config

    def can_evaluate(self, test_case: TestCase) -> bool:
        return bool(test_case.assertions)

    async def evaluate(
        self,
        test_case: TestCase,
        model_response: ModelResponse,
    ) -> JudgeScore:
        """Evaluate a response by running its test case's assertions.

        Args:
            test_case: Test case carrying one or more assertions
            model_response: The response to check

        Returns:
            JudgeScore derived from the assertion pass ratio

        Raises:
            ValueError: If the test case declares no assertions
        """
        if not test_case.assertions:
            raise ValueError(
                f"Test case '{test_case.id}' has no assertions; the deterministic "
                "judge cannot score it"
            )

        results = evaluate_assertions(test_case, model_response)
        return JudgeScore(
            score=score_from_assertions(results),
            explanation=explain_assertions(results),
            judge_model=DETERMINISTIC_JUDGE_MODEL,
            judge_provider=DETERMINISTIC_JUDGE_PROVIDER,
            timestamp=datetime.utcnow(),
            assertion_results=results,
        )

    @property
    def judge_model(self) -> str:
        return DETERMINISTIC_JUDGE_MODEL

    @property
    def judge_provider(self) -> str:
        return DETERMINISTIC_JUDGE_PROVIDER
