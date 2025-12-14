"""Parser for judge responses."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from promptlens.models.tools import ExpectedToolCall, ToolCall, ToolCallEvaluation

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


def parse_tool_judge_response(response: str) -> Tuple[Dict[str, int], str]:
    """Parse multi-criteria scores from tool evaluation judge response.

    Expected format:
        PARAMETER_CORRECTNESS: [1-5]
        TOOL_SELECTION: [1-5]
        TOOL_EFFICIENCY: [1-5]
        FINAL_ANSWER: [1-5]
        OVERALL_SCORE: [1-5]
        EXPLANATION: [explanation text]

    Args:
        response: Raw response from judge model

    Returns:
        Tuple of (criteria_scores dict, explanation)
    """
    criteria_scores = {}
    explanation = "Unable to parse explanation"

    # Criteria to extract
    criteria = [
        "parameter_correctness",
        "tool_selection",
        "tool_efficiency",
        "final_answer",
        "overall_score",
    ]

    # Extract each criterion score
    for criterion in criteria:
        pattern = rf"{criterion.upper().replace('_', '_')}:\s*(\d+)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                score_value = int(match.group(1))
                # Clamp to 1-5 range
                criteria_scores[criterion] = max(1, min(5, score_value))
            except ValueError:
                logger.warning(f"Failed to parse {criterion} score from: {match.group(1)}")

    # Extract explanation
    explanation_match = re.search(
        r"EXPLANATION:\s*(.+)", response, re.IGNORECASE | re.DOTALL
    )
    if explanation_match:
        explanation = explanation_match.group(1).strip()

    if not criteria_scores:
        logger.warning(f"Failed to parse any criteria scores from judge response: {response[:200]}")

    return criteria_scores, explanation


def evaluate_tool_call_accuracy(
    expected: ExpectedToolCall,
    actual: Optional[ToolCall],
    index: int = 0
) -> ToolCallEvaluation:
    """Automatically evaluate a tool call by comparing expected vs actual.

    This provides objective metrics before the LLM judge evaluation.

    Args:
        expected: Expected tool call from test case
        actual: Actual tool call from model response (None if not called)
        index: Index of this tool call in the sequence

    Returns:
        ToolCallEvaluation with accuracy metrics
    """
    # Case 1: Tool wasn't called at all
    if actual is None:
        return ToolCallEvaluation(
            tool_call_index=index,
            expected_tool=expected.name,
            actual_tool=None,
            correct_tool=False,
            parameter_accuracy=0.0,
            missing_parameters=list(expected.arguments.keys()),
            incorrect_parameters=[],
            extra_parameters=[],
            explanation=f"Expected tool '{expected.name}' was not called",
        )

    # Case 2: Wrong tool was called
    if actual.name != expected.name:
        return ToolCallEvaluation(
            tool_call_index=index,
            expected_tool=expected.name,
            actual_tool=actual.name,
            correct_tool=False,
            parameter_accuracy=0.0,
            missing_parameters=list(expected.arguments.keys()),
            incorrect_parameters=[],
            extra_parameters=list(actual.arguments.keys()),
            explanation=f"Called '{actual.name}' instead of '{expected.name}'",
        )

    # Case 3: Correct tool - evaluate parameters
    missing_params: List[str] = []
    incorrect_params: List[str] = []
    extra_params: List[str] = []
    correct_params = 0
    total_expected = len(expected.arguments)

    parameter_details: Dict[str, Any] = {}

    # Check each expected parameter
    for param_name, expected_value in expected.arguments.items():
        if param_name not in actual.arguments:
            missing_params.append(param_name)
            parameter_details[param_name] = {
                "expected": expected_value,
                "actual": None,
                "status": "missing",
            }
        else:
            actual_value = actual.arguments[param_name]
            # Compare values (semantic comparison for strings, exact for others)
            if _values_match(expected_value, actual_value, expected.strict_matching):
                correct_params += 1
                parameter_details[param_name] = {
                    "expected": expected_value,
                    "actual": actual_value,
                    "status": "correct",
                }
            else:
                incorrect_params.append(param_name)
                parameter_details[param_name] = {
                    "expected": expected_value,
                    "actual": actual_value,
                    "status": "incorrect",
                }

    # Check for extra parameters
    if not expected.allow_extra_params:
        for param_name in actual.arguments:
            if param_name not in expected.arguments:
                extra_params.append(param_name)
                parameter_details[param_name] = {
                    "expected": None,
                    "actual": actual.arguments[param_name],
                    "status": "extra",
                }

    # Calculate accuracy
    if total_expected == 0:
        parameter_accuracy = 1.0 if not extra_params else 0.5
    else:
        parameter_accuracy = correct_params / total_expected
        # Penalize for extra params if not allowed
        if extra_params and not expected.allow_extra_params:
            parameter_accuracy *= 0.8

    # Build explanation
    explanation_parts = [f"Tool '{actual.name}' called correctly"]
    if correct_params == total_expected and not extra_params:
        explanation_parts.append("All parameters correct")
    else:
        if missing_params:
            explanation_parts.append(f"Missing: {', '.join(missing_params)}")
        if incorrect_params:
            explanation_parts.append(f"Incorrect: {', '.join(incorrect_params)}")
        if extra_params:
            explanation_parts.append(f"Extra: {', '.join(extra_params)}")

    return ToolCallEvaluation(
        tool_call_index=index,
        expected_tool=expected.name,
        actual_tool=actual.name,
        correct_tool=True,
        parameter_accuracy=parameter_accuracy,
        missing_parameters=missing_params,
        incorrect_parameters=incorrect_params,
        extra_parameters=extra_params,
        parameter_details=parameter_details,
        explanation=". ".join(explanation_parts),
    )


def _values_match(expected: Any, actual: Any, strict: bool = False) -> bool:
    """Check if two values match (with semantic comparison for strings).

    Args:
        expected: Expected value
        actual: Actual value
        strict: If True, use exact matching

    Returns:
        True if values match
    """
    if strict:
        return expected == actual

    # Type mismatch (except number types)
    if type(expected) != type(actual):
        # Allow int/float equivalence
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(expected - actual) < 1e-6
        return False

    # For strings, do case-insensitive comparison and strip whitespace
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()

    # For lists, check length and elements
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        return all(_values_match(e, a, strict) for e, a in zip(expected, actual))

    # For dicts, check keys and values
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False
        return all(_values_match(expected[k], actual[k], strict) for k in expected)

    # Default: exact match
    return expected == actual
