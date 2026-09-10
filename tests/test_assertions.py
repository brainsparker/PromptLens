"""Tests for deterministic assertions: model validation and evaluation."""

import pytest
from pydantic import ValidationError

from promptlens.judges.assertions import (
    evaluate_assertion,
    evaluate_assertions,
    normalize_text,
    token_f1,
)
from promptlens.models.assertions import Assertion
from promptlens.models.result import ModelResponse
from promptlens.models.test_case import TestCase


def _response(content: str) -> ModelResponse:
    return ModelResponse(content=content, model="m", provider="p", latency_ms=1.0)


class TestAssertionModel:
    def test_type_is_normalized(self):
        assert Assertion(type=" Contains ", value="x").type == "contains"

    def test_unknown_type_rejected(self):
        with pytest.raises(ValidationError, match="unsupported assertion type"):
            Assertion(type="fuzzy", value="x")

    @pytest.mark.parametrize(
        "kind",
        ["equals", "contains", "not_contains", "starts_with", "ends_with", "regex", "not_regex"],
    )
    def test_text_types_require_string_value(self, kind):
        with pytest.raises(ValidationError, match="requires a string value"):
            Assertion(type=kind)
        with pytest.raises(ValidationError, match="requires a string value"):
            Assertion(type=kind, value=3)

    def test_invalid_regex_rejected(self):
        with pytest.raises(ValidationError, match="invalid regex"):
            Assertion(type="regex", value="(unclosed")

    def test_length_types_require_non_negative_integer(self):
        assert Assertion(type="max_length", value=10).value == 10
        assert Assertion(type="min_length", value=10.0).value == 10
        with pytest.raises(ValidationError, match="integer"):
            Assertion(type="max_length", value="10")
        with pytest.raises(ValidationError, match="non-negative integer"):
            Assertion(type="max_length", value=-1)
        with pytest.raises(ValidationError, match="non-negative integer"):
            Assertion(type="min_length", value=2.5)
        with pytest.raises(ValidationError, match="integer"):
            Assertion(type="max_length", value=True)

    def test_is_json_takes_no_value(self):
        assert Assertion(type="is_json").value is None
        with pytest.raises(ValidationError, match="does not take a value"):
            Assertion(type="is_json", value="{}")

    def test_threshold_only_for_scored_types(self):
        assert Assertion(type="token_f1", threshold=0.7).threshold == 0.7
        with pytest.raises(ValidationError, match="threshold is only valid"):
            Assertion(type="contains", value="x", threshold=0.5)
        with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
            Assertion(type="token_f1", threshold=1.5)

    def test_label_uses_name_when_given(self):
        assert Assertion(type="contains", value="x", name="mentions x").label() == "mentions x"
        assert Assertion(type="is_json").label() == "is_json"
        assert Assertion(type="token_f1").label() == "token_f1 >= 0.5"
        assert Assertion(type="contains", value="30 days").label() == "contains '30 days'"

    def test_golden_set_test_case_accepts_assertions_from_dicts(self):
        tc = TestCase(
            id="tc",
            query="q",
            expected_behavior="e",
            assertions=[{"type": "contains", "value": "30 days"}, {"type": "is_json"}],
        )
        assert [a.type for a in tc.assertions] == ["contains", "is_json"]

    def test_test_case_without_assertions_still_valid(self):
        tc = TestCase(id="tc", query="q", expected_behavior="e")
        assert tc.assertions == []


class TestNormalizationAndF1:
    def test_normalize_strips_case_punctuation_articles_and_whitespace(self):
        assert normalize_text("The  Answer, is: 42!") == "answer is 42"

    def test_token_f1_exact_match(self):
        assert token_f1("Paris is the capital.", "the capital is Paris") == 1.0

    def test_token_f1_partial_overlap(self):
        score = token_f1("refunds within 30 days of purchase", "30 days")
        assert 0.0 < score < 1.0

    def test_token_f1_no_overlap(self):
        assert token_f1("apples", "oranges") == 0.0

    def test_token_f1_empty_edge_cases(self):
        assert token_f1("", "") == 1.0
        assert token_f1("", "something") == 0.0
        assert token_f1("something", "") == 0.0

    def test_token_f1_is_deterministic(self):
        a = token_f1("Reset your password from the account page", "account page password reset")
        b = token_f1("Reset your password from the account page", "account page password reset")
        assert a == b


class TestEvaluateAssertion:
    def test_equals(self):
        assert evaluate_assertion(Assertion(type="equals", value="42"), " 42 ").passed
        assert not evaluate_assertion(Assertion(type="equals", value="42"), "forty-two").passed

    def test_contains_case_sensitivity(self):
        sensitive = Assertion(type="contains", value="Refund")
        insensitive = Assertion(type="contains", value="Refund", case_sensitive=False)
        assert not evaluate_assertion(sensitive, "we offer a refund").passed
        assert evaluate_assertion(insensitive, "we offer a refund").passed

    def test_not_contains(self):
        assertion = Assertion(type="not_contains", value="I don't know")
        result = evaluate_assertion(assertion, "I don't know, sorry")
        assert not result.passed
        assert "not_contains" in result.detail

    def test_starts_with_and_ends_with(self):
        starts = Assertion(type="starts_with", value="Sure")
        assert evaluate_assertion(starts, "  Sure, here you go").passed
        assert evaluate_assertion(Assertion(type="ends_with", value="."), "Done.\n").passed
        assert not evaluate_assertion(Assertion(type="ends_with", value="?"), "Done.").passed

    def test_regex_and_not_regex(self):
        digits_days = Assertion(type="regex", value=r"\b\d+ days\b")
        assert evaluate_assertion(digits_days, "within 30 days").passed
        assert not evaluate_assertion(Assertion(type="regex", value=r"^\d+$"), "abc").passed
        no_ai_disclaimer = Assertion(type="not_regex", value=r"(?i)as an ai")
        assert evaluate_assertion(no_ai_disclaimer, "Here is the answer").passed
        assert not evaluate_assertion(no_ai_disclaimer, "As an AI, I cannot").passed

    def test_regex_case_insensitive_flag(self):
        assertion = Assertion(type="regex", value="refund", case_sensitive=False)
        assert evaluate_assertion(assertion, "REFUND policy").passed

    def test_is_json_including_fenced(self):
        assert evaluate_assertion(Assertion(type="is_json"), '{"a": 1}').passed
        assert evaluate_assertion(Assertion(type="is_json"), '```json\n{"a": 1}\n```').passed
        bad = evaluate_assertion(Assertion(type="is_json"), "not json")
        assert not bad.passed
        assert "invalid JSON" in bad.detail

    def test_length_bounds(self):
        assert evaluate_assertion(Assertion(type="max_length", value=5), "12345").passed
        assert not evaluate_assertion(Assertion(type="max_length", value=5), "123456").passed
        assert evaluate_assertion(Assertion(type="min_length", value=3), "abc").passed
        assert not evaluate_assertion(Assertion(type="min_length", value=3), "ab").passed

    def test_token_f1_uses_value_or_reference_answer(self):
        with_value = Assertion(type="token_f1", value="Paris", threshold=0.5)
        result = evaluate_assertion(with_value, "Paris")
        assert result.passed and result.score == 1.0

        fallback = Assertion(type="token_f1", threshold=0.5)
        result = evaluate_assertion(fallback, "Paris", reference_answer="Paris")
        assert result.passed

        missing = evaluate_assertion(fallback, "Paris")
        assert not missing.passed
        assert "no reference text" in missing.detail

    def test_token_f1_threshold_applied(self):
        assertion = Assertion(type="token_f1", value="alpha beta gamma delta", threshold=0.9)
        result = evaluate_assertion(assertion, "alpha beta")
        assert not result.passed
        assert "threshold 0.9" in result.detail


class TestEvaluateAssertions:
    def test_runs_in_declaration_order_with_reference_fallback(self):
        tc = TestCase(
            id="tc",
            query="q",
            expected_behavior="e",
            reference_answer="Refunds are accepted within 30 days",
            assertions=[
                {"type": "contains", "value": "30 days"},
                {"type": "not_contains", "value": "90 days"},
                {"type": "token_f1", "threshold": 0.4},
            ],
        )
        response = _response("Refunds are accepted within 30 days of purchase.")
        results = evaluate_assertions(tc, response)
        assert [r.type for r in results] == ["contains", "not_contains", "token_f1"]
        assert all(r.passed for r in results)

    def test_no_assertions_gives_empty_list(self):
        tc = TestCase(id="tc", query="q", expected_behavior="e")
        assert evaluate_assertions(tc, _response("anything")) == []
