"""Evaluate deterministic assertions against a model response.

Every check here is a pure function of the response and the assertion: no
model call, no network, no randomness. The same response always produces
the same verdict, which is what makes these safe to put in a blocking CI
gate while judge scores stay advisory.
"""

import json
import logging
import re
from typing import Any, List, Optional, Tuple

from promptlens.models.assertions import Assertion, AssertionResult
from promptlens.models.result import ModelResponse

logger = logging.getLogger(__name__)

_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# How much of the response to echo back in failure messages.
_SNIPPET_CHARS = 120


def _snippet(text: str, limit: int = _SNIPPET_CHARS) -> str:
    """Return a one-line, truncated preview of the text for messages."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


def _fold(text: str, ignore_case: bool) -> str:
    return text.lower() if ignore_case else text


def _parse_json(content: str) -> Tuple[Optional[Any], Optional[str]]:
    """Parse the response as JSON.

    Accepts the whole response, or a single fenced ```json block when the
    model wrapped its answer in markdown. Returns (parsed, error).
    """
    stripped = content.strip()
    candidates = [stripped]
    fenced = _FENCED_JSON.findall(stripped)
    if len(fenced) == 1:
        candidates.append(fenced[0].strip())

    last_error: Optional[str] = None
    for candidate in candidates:
        if not candidate:
            last_error = "response is empty"
            continue
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError as exc:
            last_error = f"{exc.msg} at line {exc.lineno} column {exc.colno}"
    return None, last_error or "response is not valid JSON"


def _validate_schema(instance: Any, schema: dict) -> Optional[str]:
    """Validate instance against a JSON Schema. Returns an error or None."""
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            "The json_schema assertion requires the 'jsonschema' package. "
            "Install it with: pip install jsonschema"
        ) from exc

    validator_cls = jsonschema.validators.validator_for(schema)
    try:
        validator_cls.check_schema(schema)
    except jsonschema.SchemaError as exc:
        return f"invalid JSON Schema: {exc.message}"

    errors = sorted(validator_cls(schema).iter_errors(instance), key=lambda e: list(e.path))
    if not errors:
        return None
    first = errors[0]
    location = "/".join(str(p) for p in first.absolute_path) or "(root)"
    suffix = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
    return f"schema violation at {location}: {first.message}{suffix}"


def evaluate_assertion(assertion: Assertion, response: ModelResponse) -> AssertionResult:
    """Evaluate one assertion against one model response.

    Args:
        assertion: The assertion to check
        response: The model response to check it against

    Returns:
        AssertionResult with pass/fail and an explanation
    """
    kind = assertion.type
    content = response.content or ""
    passed, detail = _evaluate(kind, assertion, content, response)

    if passed:
        message = detail or "passed"
    else:
        message = assertion.message or detail or "failed"

    return AssertionResult(
        type=kind,
        name=assertion.label,
        passed=passed,
        message=message,
    )


def _evaluate(
    kind: str,
    assertion: Assertion,
    content: str,
    response: ModelResponse,
) -> Tuple[bool, str]:
    """Return (passed, detail) for the assertion."""
    ignore_case = assertion.ignore_case
    haystack = _fold(content, ignore_case)

    if kind == "equals":
        expected = _fold(str(assertion.value), ignore_case).strip()
        actual = haystack.strip()
        if actual == expected:
            return True, "response matches expected text"
        return False, f"expected exactly {assertion.value!r}, got {_snippet(content)!r}"

    if kind == "contains":
        missing = [v for v in assertion.values if _fold(v, ignore_case) not in haystack]
        if not missing:
            return True, "all required text present"
        return False, f"missing required text: {', '.join(repr(m) for m in missing)}"

    if kind == "not_contains":
        found = [v for v in assertion.values if _fold(v, ignore_case) in haystack]
        if not found:
            return True, "no forbidden text present"
        return False, f"found forbidden text: {', '.join(repr(f) for f in found)}"

    if kind == "contains_any":
        if any(_fold(v, ignore_case) in haystack for v in assertion.values):
            return True, "at least one expected phrase present"
        options = ", ".join(repr(v) for v in assertion.values)
        return False, f"none of the expected phrases present: {options}"

    if kind == "starts_with":
        prefix = _fold(str(assertion.value), ignore_case)
        if haystack.lstrip().startswith(prefix):
            return True, "response starts with expected text"
        return False, f"expected response to start with {assertion.value!r}"

    if kind == "ends_with":
        suffix = _fold(str(assertion.value), ignore_case)
        if haystack.rstrip().endswith(suffix):
            return True, "response ends with expected text"
        return False, f"expected response to end with {assertion.value!r}"

    if kind in ("regex", "not_regex"):
        flags = re.IGNORECASE if ignore_case else 0
        pattern = re.compile(str(assertion.value), flags)
        matched = pattern.search(content) is not None
        if kind == "regex":
            if matched:
                return True, "pattern matched"
            return False, f"pattern {assertion.value!r} did not match"
        if not matched:
            return True, "pattern did not match"
        return False, f"pattern {assertion.value!r} matched but must not"

    if kind == "is_json":
        parsed, error = _parse_json(content)
        if error is None:
            return True, "response is valid JSON"
        return False, f"response is not valid JSON: {error}"

    if kind == "json_schema":
        parsed, error = _parse_json(content)
        if error is not None:
            return False, f"response is not valid JSON: {error}"
        schema_error = _validate_schema(parsed, assertion.value)
        if schema_error is None:
            return True, "response matches JSON Schema"
        return False, schema_error

    if kind == "max_chars":
        limit = int(assertion.value)
        length = len(content)
        if length <= limit:
            return True, f"{length} chars (limit {limit})"
        return False, f"response is {length} chars, limit is {limit}"

    if kind == "min_chars":
        minimum = int(assertion.value)
        length = len(content.strip())
        if length >= minimum:
            return True, f"{length} chars (minimum {minimum})"
        return False, f"response is {length} chars, minimum is {minimum}"

    if kind == "max_latency_ms":
        limit = float(assertion.value)
        latency = float(response.latency_ms or 0.0)
        if latency <= limit:
            return True, f"{latency:.0f} ms (limit {limit:g} ms)"
        return False, f"latency {latency:.0f} ms exceeds limit {limit:g} ms"

    if kind == "max_cost_usd":
        limit = float(assertion.value)
        cost = response.cost_usd
        if cost is None:
            return False, "cost is unknown for this response, cannot enforce max_cost_usd"
        if cost <= limit:
            return True, f"${cost:.6f} (limit ${limit:g})"
        return False, f"cost ${cost:.6f} exceeds limit ${limit:g}"

    raise ValueError(f"Unknown assertion type: {kind}")  # pragma: no cover


def run_assertions(
    assertions: List[Assertion],
    response: ModelResponse,
) -> List[AssertionResult]:
    """Evaluate every assertion against the response.

    All assertions run, even after a failure, so the report shows the full
    picture rather than the first problem only.

    Args:
        assertions: Assertions declared on the test case
        response: The model response to check

    Returns:
        One AssertionResult per assertion, in declaration order
    """
    results: List[AssertionResult] = []
    for assertion in assertions:
        try:
            results.append(evaluate_assertion(assertion, response))
        except Exception as exc:  # defensive: a bad assertion must not kill the run
            logger.error(f"Assertion '{assertion.label}' raised: {exc}")
            results.append(
                AssertionResult(
                    type=assertion.type,
                    name=assertion.label,
                    passed=False,
                    message=f"assertion could not be evaluated: {exc}",
                )
            )
    return results
