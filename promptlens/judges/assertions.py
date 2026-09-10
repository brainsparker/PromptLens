"""Deterministic assertion evaluation.

Every function here is pure: same inputs, same outputs, no network, no model.
"""

import json
import re
import string
from typing import List, Optional

from promptlens.models.assertions import (
    DEFAULT_F1_THRESHOLD,
    LENGTH_ASSERTIONS,
    STRING_ASSERTIONS,
    Assertion,
)
from promptlens.models.result import AssertionResult, ModelResponse
from promptlens.models.test_case import TestCase

_ARTICLES = frozenset({"a", "an", "the"})
_PUNCTUATION = frozenset(string.punctuation)


def normalize_text(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation and articles,
    collapse whitespace.

    Args:
        text: Raw text

    Returns:
        Normalized text suitable for exact-match and token overlap
    """
    lowered = text.lower()
    no_punct = "".join(ch for ch in lowered if ch not in _PUNCTUATION)
    tokens = [tok for tok in no_punct.split() if tok not in _ARTICLES]
    return " ".join(tokens)


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between a prediction and a reference after normalization.

    Gives partial credit when the prediction contains most of the reference
    without demanding an exact match.

    Args:
        prediction: Model output
        reference: Expected text

    Returns:
        F1 score between 0.0 and 1.0
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in ref_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_assertion(
    assertion: Assertion,
    content: str,
    reference_answer: Optional[str] = None,
) -> AssertionResult:
    """Evaluate one assertion against response content.

    Args:
        assertion: The assertion to check
        content: Model response text
        reference_answer: Fallback reference for token_f1 when the assertion
            carries no value of its own

    Returns:
        AssertionResult with pass/fail and a short detail string
    """
    kind = assertion.type
    label = assertion.label()

    if kind in STRING_ASSERTIONS:
        expected = str(assertion.value)
        haystack = content if assertion.case_sensitive else content.lower()
        needle = expected if assertion.case_sensitive else expected.lower()

        if kind == "equals":
            passed = haystack.strip() == needle.strip()
        elif kind == "contains":
            passed = needle in haystack
        elif kind == "not_contains":
            passed = needle not in haystack
        elif kind == "starts_with":
            passed = haystack.lstrip().startswith(needle)
        else:  # ends_with
            passed = haystack.rstrip().endswith(needle)

        detail = "matched" if passed else f"response did not satisfy {kind} {expected!r}"
        return AssertionResult(type=kind, label=label, passed=passed, detail=detail)

    if kind in ("regex", "not_regex"):
        flags = 0 if assertion.case_sensitive else re.IGNORECASE
        pattern = re.compile(str(assertion.value), flags)
        found = pattern.search(content) is not None
        passed = found if kind == "regex" else not found
        if passed:
            detail = "pattern matched" if kind == "regex" else "pattern absent"
        else:
            detail = (
                f"pattern {assertion.value!r} not found"
                if kind == "regex"
                else f"pattern {assertion.value!r} matched but must be absent"
            )
        return AssertionResult(type=kind, label=label, passed=passed, detail=detail)

    if kind == "is_json":
        try:
            json.loads(_strip_code_fence(content))
            return AssertionResult(type=kind, label=label, passed=True, detail="valid JSON")
        except (ValueError, TypeError) as exc:
            return AssertionResult(
                type=kind, label=label, passed=False, detail=f"invalid JSON: {exc}"
            )

    if kind in LENGTH_ASSERTIONS:
        limit = int(assertion.value)
        length = len(content)
        if kind == "min_length":
            passed = length >= limit
            detail = f"length {length} >= {limit}" if passed else f"length {length} < {limit}"
        else:
            passed = length <= limit
            detail = f"length {length} <= {limit}" if passed else f"length {length} > {limit}"
        return AssertionResult(type=kind, label=label, passed=passed, detail=detail)

    if kind == "token_f1":
        reference = assertion.value if assertion.value is not None else reference_answer
        if reference is None:
            return AssertionResult(
                type=kind,
                label=label,
                passed=False,
                detail=(
                    "no reference text: set a value on the assertion or "
                    "reference_answer on the test case"
                ),
            )
        threshold = assertion.threshold if assertion.threshold is not None else DEFAULT_F1_THRESHOLD
        score = token_f1(content, str(reference))
        passed = score >= threshold
        return AssertionResult(
            type=kind,
            label=label,
            passed=passed,
            detail=f"token F1 {score:.2f} (threshold {threshold:g})",
            score=score,
        )

    # The Assertion model validates types, so this is unreachable in practice.
    return AssertionResult(
        type=kind, label=label, passed=False, detail=f"unknown assertion type {kind}"
    )


def evaluate_assertions(test_case: TestCase, response: ModelResponse) -> List[AssertionResult]:
    """Evaluate every assertion declared on a test case.

    Args:
        test_case: Test case carrying assertions and an optional reference_answer
        response: The model response to check

    Returns:
        One AssertionResult per assertion, in declaration order
    """
    return [
        evaluate_assertion(assertion, response.content, test_case.reference_answer)
        for assertion in test_case.assertions
    ]


def _strip_code_fence(text: str) -> str:
    """Remove a surrounding markdown code fence so fenced JSON still validates."""
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[: -3]
    return stripped.strip()
