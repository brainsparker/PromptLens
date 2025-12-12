"""Parser for judge responses."""

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def parse_judge_response(response: str) -> Tuple[Optional[int], str]:
    """Parse score and explanation from judge response.

    Expected format:
        SCORE: [1-5]
        EXPLANATION: [explanation text]

    Args:
        response: Raw response from judge model

    Returns:
        Tuple of (score, explanation)
        Score is None if parsing fails
    """
    score = None
    explanation = "Unable to parse explanation"

    # Extract score
    score_match = re.search(r"SCORE:\s*(\d+)", response, re.IGNORECASE)
    if score_match:
        try:
            score_value = int(score_match.group(1))
            # Clamp to 1-5 range
            score = max(1, min(5, score_value))
        except ValueError:
            logger.warning(f"Failed to parse score from: {score_match.group(1)}")

    # Extract explanation
    explanation_match = re.search(
        r"EXPLANATION:\s*(.+)", response, re.IGNORECASE | re.DOTALL
    )
    if explanation_match:
        explanation = explanation_match.group(1).strip()

    if score is None:
        logger.warning(f"Failed to parse score from judge response: {response[:200]}")

    return score, explanation


def validate_score(score: Optional[int]) -> int:
    """Validate and return a score, using default if invalid.

    Args:
        score: Score to validate (may be None)

    Returns:
        Valid score (1-5), defaults to 3 if invalid
    """
    if score is None:
        logger.warning("No score found, using default score of 3")
        return 3

    if not (1 <= score <= 5):
        logger.warning(f"Score {score} out of range, clamping to 1-5")
        return max(1, min(5, score))

    return score
