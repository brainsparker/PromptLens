"""Tests for the deterministic check engine and check models."""

import pytest
from pydantic import ValidationError

from promptlens.checks import run_checks
from promptlens.models.checks import Check


def _run_one(check_kwargs, text):
    results = run_checks([Check(**check_kwargs)], text)
    assert len(results) == 1
    return results[0]


class TestContains:
    def test_passes_when_substring_present(self):
        result = _run_one({"type": "contains", "value": "refund"}, "We offer a refund.")
        assert result.passed

    def test_fails_when_substring_missing(self):
        result = _run_one({"type": "contains", "value": "refund"}, "No returns.")
        assert not result.passed
        assert "refund" in result.detail

    def test_case_insensitive_by_default(self):
        result = _run_one({"type": "contains", "value": "Refund"}, "full REFUND policy")
        assert result.passed

    def test_case_sensitive_option(self):
        result = _run_one(
            {"type": "contains", "value": "Refund", "case_sensitive": True},
            "full REFUND policy",
        )
        assert not result.passed

    def test_list_all_mode_requires_every_value(self):
        check = {"type": "contains", "value": ["alpha", "beta"]}
        assert _run_one(check, "alpha and beta").passed
        assert not _run_one(check, "alpha only").passed

    def test_list_any_mode_requires_one_value(self):
        check = {"type": "contains", "value": ["alpha", "beta"], "mode": "any"}
        assert _run_one(check, "beta only").passed
        assert not _run_one(check, "gamma only").passed


class TestNotContains:
    def test_passes_when_absent(self):
        result = _run_one({"type": "not_contains", "value": "guarantee"}, "We try hard.")
        assert result.passed

    def test_fails_when_present(self):
        result = _run_one({"type": "not_contains", "value": "guarantee"}, "We guarantee it.")
        assert not result.passed
        assert "guarantee" in result.detail

    def test_list_fails_if_any_present(self):
        check = {"type": "not_contains", "value": ["Python", "Java"]}
        assert not _run_one(check, "Use Python for this.").passed
        assert _run_one(check, "Use a typed language.").passed


class TestRegex:
    def test_matches_pattern(self):
        result = _run_one({"type": "regex", "pattern": r"ORD-\d{5}"}, "Order ORD-12345 confirmed")
        assert result.passed

    def test_fails_without_match(self):
        result = _run_one({"type": "regex", "pattern": r"ORD-\d{5}"}, "Order 12345 confirmed")
        assert not result.passed

    def test_invalid_pattern_rejected_at_model_level(self):
        with pytest.raises(ValidationError):
            Check(type="regex", pattern="[unclosed")


class TestExactMatch:
    def test_exact_match_with_strip_and_case_folding(self):
        result = _run_one({"type": "exact_match", "value": "positive"}, "  Positive \n")
        assert result.passed

    def test_exact_match_case_sensitive(self):
        check = {"type": "exact_match", "value": "positive", "case_sensitive": True}
        assert not _run_one(check, "Positive").passed
        assert _run_one(check, "positive").passed

    def test_exact_match_fails_on_extra_content(self):
        result = _run_one(
            {"type": "exact_match", "value": "positive"},
            "The sentiment is positive.",
        )
        assert not result.passed

    def test_exact_match_without_strip(self):
        check = {"type": "exact_match", "value": "positive", "strip": False}
        assert not _run_one(check, " positive ").passed


class TestJsonValid:
    def test_valid_raw_json(self):
        assert _run_one({"type": "json_valid"}, '{"a": 1}').passed

    def test_valid_fenced_json(self):
        text = 'Here you go:\n```json\n{"a": 1}\n```'
        assert _run_one({"type": "json_valid"}, text).passed

    def test_invalid_json(self):
        assert not _run_one({"type": "json_valid"}, "not json at all").passed


class TestJsonSchema:
    SCHEMA = {
        "type": "object",
        "required": ["name", "age"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
    }

    def test_conforming_object_passes(self):
        result = _run_one(
            {"type": "json_schema", "json_schema": self.SCHEMA},
            '{"name": "Ada", "age": 36, "tags": ["math"]}',
        )
        assert result.passed

    def test_missing_required_property_fails(self):
        result = _run_one(
            {"type": "json_schema", "json_schema": self.SCHEMA},
            '{"name": "Ada"}',
        )
        assert not result.passed
        assert "age" in result.detail

    def test_wrong_type_fails(self):
        result = _run_one(
            {"type": "json_schema", "json_schema": self.SCHEMA},
            '{"name": "Ada", "age": "thirty-six"}',
        )
        assert not result.passed
        assert "age" in result.detail

    def test_boolean_is_not_integer(self):
        result = _run_one(
            {"type": "json_schema", "json_schema": self.SCHEMA},
            '{"name": "Ada", "age": true}',
        )
        assert not result.passed

    def test_array_items_validated(self):
        result = _run_one(
            {"type": "json_schema", "json_schema": self.SCHEMA},
            '{"name": "Ada", "age": 36, "tags": ["ok", 5]}',
        )
        assert not result.passed

    def test_enum_constraint(self):
        schema = {"type": "string", "enum": ["positive", "negative", "neutral"]}
        assert _run_one({"type": "json_schema", "json_schema": schema}, '"positive"').passed
        assert not _run_one({"type": "json_schema", "json_schema": schema}, '"mixed"').passed

    def test_fenced_json_accepted(self):
        result = _run_one(
            {"type": "json_schema", "json_schema": self.SCHEMA},
            '```json\n{"name": "Ada", "age": 36}\n```',
        )
        assert result.passed

    def test_non_json_response_fails(self):
        result = _run_one(
            {"type": "json_schema", "json_schema": self.SCHEMA},
            "I cannot produce JSON.",
        )
        assert not result.passed


class TestCheckModelValidation:
    def test_unknown_type_rejected(self):
        with pytest.raises(ValidationError):
            Check(type="fuzzy_match", value="x")

    def test_contains_requires_value(self):
        with pytest.raises(ValidationError):
            Check(type="contains")

    def test_regex_requires_pattern(self):
        with pytest.raises(ValidationError):
            Check(type="regex")

    def test_json_schema_requires_schema(self):
        with pytest.raises(ValidationError):
            Check(type="json_schema")

    def test_exact_match_rejects_list_value(self):
        with pytest.raises(ValidationError):
            Check(type="exact_match", value=["a", "b"])

    def test_empty_value_list_rejected(self):
        with pytest.raises(ValidationError):
            Check(type="contains", value=[])

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValidationError):
            Check(type="contains", value="x", mode="some")

    def test_type_normalized(self):
        check = Check(type="  Contains ", value="x")
        assert check.type == "contains"

    def test_multiple_checks_run_in_order(self):
        checks = [
            Check(type="contains", value="alpha"),
            Check(type="regex", pattern=r"\d+"),
        ]
        results = run_checks(checks, "alpha 42")
        assert [r.check_type for r in results] == ["contains", "regex"]
        assert all(r.passed for r in results)
