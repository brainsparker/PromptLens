"""Tests for deterministic assertions: models and evaluator."""

import pytest
from pydantic import ValidationError

from promptlens.assertions import evaluate_assertion, run_assertions
from promptlens.models.assertions import (
    ASSERTION_TYPES,
    Assertion,
    AssertionSummary,
)
from promptlens.models.result import ModelResponse
from promptlens.models.test_case import GoldenSet, TestCase


def _response(content="Hello world. Refunds are accepted within 30 days.", **kwargs):
    defaults = {
        "model": "model-a",
        "provider": "anthropic",
        "latency_ms": 850.0,
        "cost_usd": 0.0012,
    }
    defaults.update(kwargs)
    return ModelResponse(content=content, **defaults)


def _run(kind, value=None, content=None, **fields):
    assertion = Assertion(type=kind, value=value, **fields)
    response = _response() if content is None else _response(content=content)
    return evaluate_assertion(assertion, response)


class TestAssertionModel:
    def test_every_documented_type_is_constructible(self):
        samples = {
            "equals": "x",
            "contains": "x",
            "not_contains": ["x"],
            "contains_any": ["x", "y"],
            "starts_with": "x",
            "ends_with": "x",
            "regex": r"\d+",
            "not_regex": r"\d+",
            "is_json": None,
            "json_schema": {"type": "object"},
            "max_chars": 10,
            "min_chars": 1,
            "max_latency_ms": 1000,
            "max_cost_usd": 0.01,
        }
        assert set(samples) == set(ASSERTION_TYPES)
        for kind, value in samples.items():
            Assertion(type=kind, value=value)

    def test_type_is_normalized(self):
        assert Assertion(type=" Not-Contains ", value="x").type == "not_contains"

    def test_unknown_type_rejected(self):
        with pytest.raises(ValidationError, match="Unknown assertion type"):
            Assertion(type="looks_good", value="x")

    def test_missing_value_rejected(self):
        with pytest.raises(ValidationError, match="requires a value"):
            Assertion(type="contains")

    def test_is_json_rejects_value(self):
        with pytest.raises(ValidationError, match="does not take a value"):
            Assertion(type="is_json", value="x")

    def test_invalid_regex_rejected_at_load_time(self):
        with pytest.raises(ValidationError, match="invalid pattern"):
            Assertion(type="regex", value="(unclosed")

    def test_list_types_reject_empty_items(self):
        with pytest.raises(ValidationError):
            Assertion(type="contains", value=["ok", ""])
        with pytest.raises(ValidationError):
            Assertion(type="contains_any", value=[])

    def test_number_types_reject_non_positive_and_bools(self):
        with pytest.raises(ValidationError):
            Assertion(type="max_chars", value=0)
        with pytest.raises(ValidationError):
            Assertion(type="max_latency_ms", value=True)
        with pytest.raises(ValidationError):
            Assertion(type="max_cost_usd", value="cheap")

    def test_json_schema_requires_mapping(self):
        with pytest.raises(ValidationError, match="JSON Schema mapping"):
            Assertion(type="json_schema", value="object")

    def test_label_prefers_name_then_summarizes_value(self):
        assert Assertion(type="contains", value="x", name="mentions x").label == "mentions x"
        assert Assertion(type="contains", value="x").label == "contains 'x'"
        assert Assertion(type="is_json").label == "is_json"
        label = Assertion(type="not_contains", value=["a", "b", "c", "d"]).label
        assert label.startswith("not_contains [") and label.endswith(", ...]")

    def test_test_case_accepts_assertions_and_defaults_empty(self):
        plain = TestCase(id="t", query="q", expected_behavior="e")
        assert plain.assertions == []
        with_assertions = TestCase(
            id="t",
            query="q",
            expected_behavior="e",
            assertions=[{"type": "contains", "value": "x"}],
        )
        assert with_assertions.assertions[0].type == "contains"

    def test_golden_set_rejects_bad_assertion_with_case_context(self):
        with pytest.raises(ValidationError):
            GoldenSet(
                name="gs",
                test_cases=[
                    {
                        "id": "t",
                        "query": "q",
                        "expected_behavior": "e",
                        "assertions": [{"type": "regex", "value": "["}],
                    }
                ],
            )


class TestTextAssertions:
    def test_equals_trims_whitespace(self):
        assert _run("equals", "hi", content="  hi \n").passed
        assert not _run("equals", "hi", content="hi there").passed

    def test_equals_ignore_case(self):
        assert _run("equals", "HELLO", content="hello", ignore_case=True).passed

    def test_contains_single_and_list(self):
        assert _run("contains", "30 days").passed
        assert _run("contains", ["Refunds", "30 days"]).passed
        result = _run("contains", ["Refunds", "store credit"])
        assert not result.passed
        assert "store credit" in result.message
        assert "Refunds" not in result.message

    def test_contains_ignore_case(self):
        assert not _run("contains", "refunds").passed
        assert _run("contains", "refunds", ignore_case=True).passed

    def test_not_contains(self):
        assert _run("not_contains", ["As an AI", "I cannot"]).passed
        result = _run("not_contains", "Refunds")
        assert not result.passed
        assert "Refunds" in result.message

    def test_contains_any(self):
        assert _run("contains_any", ["nope", "30 days"]).passed
        assert not _run("contains_any", ["nope", "never"]).passed

    def test_starts_with_and_ends_with(self):
        assert _run("starts_with", "Hello").passed
        assert _run("starts_with", "hello", ignore_case=True).passed
        assert not _run("starts_with", "Refunds").passed
        assert _run("ends_with", "30 days.").passed
        assert not _run("ends_with", "Hello").passed

    def test_regex_and_not_regex(self):
        assert _run("regex", r"\b\d+ days\b").passed
        assert not _run("regex", r"^\d+$").passed
        assert _run("not_regex", r"^\d+$").passed
        result = _run("not_regex", r"\d+ days")
        assert not result.passed
        assert "must not" in result.message

    def test_regex_ignore_case(self):
        assert not _run("regex", "REFUNDS").passed
        assert _run("regex", "REFUNDS", ignore_case=True).passed

    def test_max_and_min_chars(self):
        assert _run("max_chars", 100).passed
        result = _run("max_chars", 10)
        assert not result.passed
        assert "limit is 10" in result.message
        assert _run("min_chars", 10).passed
        assert not _run("min_chars", 1000).passed

    def test_custom_failure_message_replaces_detail(self):
        result = _run("contains", "warranty", message="Policy answers must mention warranty")
        assert not result.passed
        assert result.message == "Policy answers must mention warranty"

    def test_passing_result_reports_detail_not_custom_message(self):
        result = _run("contains", "30 days", message="custom")
        assert result.passed
        assert result.message != "custom"


class TestJsonAssertions:
    def test_is_json_accepts_object_and_array(self):
        assert _run("is_json", content='{"ok": true}').passed
        assert _run("is_json", content="[1, 2, 3]").passed

    def test_is_json_rejects_prose(self):
        result = _run("is_json", content="Sure! Here is the answer.")
        assert not result.passed
        assert "not valid JSON" in result.message

    def test_is_json_accepts_single_fenced_block(self):
        content = 'Here you go:\n```json\n{"city": "Paris"}\n```\nAnything else?'
        assert _run("is_json", content=content).passed

    def test_is_json_rejects_empty(self):
        assert not _run("is_json", content="   ").passed

    def test_json_schema_pass_and_fail(self):
        schema = {
            "type": "object",
            "required": ["city", "temperature"],
            "properties": {
                "city": {"type": "string"},
                "temperature": {"type": "number", "minimum": -100, "maximum": 100},
            },
        }
        assert _run("json_schema", schema, content='{"city": "Paris", "temperature": 21}').passed

        missing = _run("json_schema", schema, content='{"city": "Paris"}')
        assert not missing.passed
        assert "temperature" in missing.message

        out_of_range = _run("json_schema", schema, content='{"city": "Paris", "temperature": 500}')
        assert not out_of_range.passed
        assert "temperature" in out_of_range.message

    def test_json_schema_on_non_json_reports_parse_error(self):
        result = _run("json_schema", {"type": "object"}, content="not json")
        assert not result.passed
        assert "not valid JSON" in result.message

    def test_json_schema_reports_invalid_schema(self):
        result = _run("json_schema", {"type": "not-a-type"}, content="{}")
        assert not result.passed
        assert "invalid JSON Schema" in result.message


class TestMetadataAssertions:
    def test_max_latency(self):
        assert _run("max_latency_ms", 1000).passed
        result = _run("max_latency_ms", 500)
        assert not result.passed
        assert "850 ms" in result.message

    def test_max_cost(self):
        assert _run("max_cost_usd", 0.01).passed
        assert not _run("max_cost_usd", 0.001).passed

    def test_max_cost_fails_closed_when_cost_unknown(self):
        assertion = Assertion(type="max_cost_usd", value=0.01)
        result = evaluate_assertion(assertion, _response(cost_usd=None))
        assert not result.passed
        assert "unknown" in result.message


class TestRunAssertions:
    def test_runs_every_assertion_in_order_even_after_failure(self):
        assertions = [
            Assertion(type="contains", value="missing"),
            Assertion(type="contains", value="30 days"),
            Assertion(type="max_chars", value=5),
        ]
        results = run_assertions(assertions, _response())
        assert [r.passed for r in results] == [False, True, False]
        assert [r.type for r in results] == ["contains", "contains", "max_chars"]

    def test_empty_assertions_yield_empty_results(self):
        assert run_assertions([], _response()) == []

    def test_evaluator_exception_becomes_failed_result(self, monkeypatch):
        import promptlens.assertions.evaluator as evaluator_module

        def boom(*_args, **_kwargs):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(evaluator_module, "evaluate_assertion", boom)
        results = evaluator_module.run_assertions(
            [Assertion(type="contains", value="x")], _response()
        )
        assert len(results) == 1
        assert not results[0].passed
        assert "kaboom" in results[0].message


class TestAssertionSummary:
    def test_pass_rate(self):
        assert AssertionSummary().pass_rate is None
        summary = AssertionSummary(cases_with_assertions=4, cases_passed=3, cases_failed=1)
        assert summary.pass_rate == 0.75
