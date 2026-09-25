"""Tests for deterministic assertions: models, checker, and golden-set loading."""

import textwrap

import pytest
from pydantic import ValidationError

from promptlens.assertions import check_assertion, extract_json, run_assertions
from promptlens.loaders.yaml_loader import get_loader
from promptlens.models.assertions import Assertion
from promptlens.models.result import ModelResponse
from promptlens.models.test_case import TestCase


def _response(content, latency_ms=100.0, error=None):
    return ModelResponse(
        content=content,
        model="model-a",
        provider="anthropic",
        latency_ms=latency_ms,
        error=error,
    )


def _check(kind, content, value=None, **kwargs):
    assertion = Assertion(type=kind, value=value, **kwargs)
    return check_assertion(assertion, _response(content))


class TestAssertionModel:
    def test_type_is_normalized(self):
        assert Assertion(type=" Not-Contains ", value="x").type == "not_contains"

    def test_unknown_type_rejected(self):
        with pytest.raises(ValidationError):
            Assertion(type="fuzzy", value="x")

    @pytest.mark.parametrize("kind", ["contains", "not_contains", "starts_with", "ends_with"])
    def test_text_assertions_require_non_empty_string(self, kind):
        with pytest.raises(ValidationError):
            Assertion(type=kind, value="")
        with pytest.raises(ValidationError):
            Assertion(type=kind, value=3)

    def test_invalid_regex_rejected_at_load_time(self):
        with pytest.raises(ValidationError, match="invalid pattern"):
            Assertion(type="regex", value="(unclosed")

    @pytest.mark.parametrize("kind", ["min_chars", "max_chars", "max_latency_ms"])
    def test_numeric_assertions_require_non_negative_number(self, kind):
        assert Assertion(type=kind, value=0).value == 0
        with pytest.raises(ValidationError):
            Assertion(type=kind, value="10")
        with pytest.raises(ValidationError):
            Assertion(type=kind, value=-1)
        with pytest.raises(ValidationError, match="cannot be a boolean"):
            Assertion(type=kind, value=True)

    def test_json_valid_takes_no_value(self):
        assert Assertion(type="json_valid").value is None
        with pytest.raises(ValidationError):
            Assertion(type="json_valid", value="x")

    def test_json_schema_requires_mapping(self):
        with pytest.raises(ValidationError):
            Assertion(type="json_schema", value="not a schema")
        assert Assertion(type="json_schema", value={"type": "object"}).type == "json_schema"

    def test_label_prefers_description_and_truncates(self):
        assert Assertion(type="contains", value="x", description="Mentions x").label() == "Mentions x"
        long_value = "a" * 100
        label = Assertion(type="contains", value=long_value).label()
        assert label.startswith("contains: ")
        assert label.endswith("...")
        assert len(label) < 80
        assert Assertion(type="json_valid").label() == "json_valid"


class TestTextChecks:
    def test_contains(self):
        assert _check("contains", "Paris is the capital.", "Paris").passed
        failed = _check("contains", "Lyon is nice.", "Paris")
        assert not failed.passed
        assert "not found" in failed.message

    def test_contains_case_insensitive(self):
        assert not _check("contains", "paris", "PARIS").passed
        assert _check("contains", "paris", "PARIS", case_sensitive=False).passed

    def test_not_contains(self):
        assert _check("not_contains", "Sure thing.", "I don't know").passed
        assert not _check("not_contains", "I DON'T KNOW", "i don't know", case_sensitive=False).passed

    def test_regex_and_not_regex(self):
        assert _check("regex", "Order #12345 shipped", r"#\d{5}").passed
        assert not _check("regex", "Order shipped", r"#\d{5}").passed
        assert _check("regex", "ERROR: boom", r"^error", case_sensitive=False).passed
        assert _check("not_regex", "All good", r"error").passed
        assert not _check("not_regex", "An error occurred", r"error").passed

    def test_equals_trims_whitespace(self):
        assert _check("equals", "  OK\n", "OK").passed
        assert _check("equals", "ok", "OK", case_sensitive=False).passed
        failed = _check("equals", "OK!", "OK")
        assert not failed.passed
        assert "expected 'OK'" in failed.message

    def test_starts_with_and_ends_with(self):
        assert _check("starts_with", "\nHello there", "Hello").passed
        assert not _check("starts_with", "Well, hello", "Hello").passed
        assert _check("ends_with", "Thanks!\n", "!").passed
        assert not _check("ends_with", "Thanks", "!").passed


class TestJsonChecks:
    def test_extract_json_tolerates_code_fence(self):
        parsed, error = extract_json('```json\n{"a": 1}\n```')
        assert error is None
        assert parsed == {"a": 1}
        parsed, error = extract_json("```\n[1, 2]\n```")
        assert error is None
        assert parsed == [1, 2]

    def test_extract_json_reports_error(self):
        parsed, error = extract_json("{not json")
        assert parsed is None
        assert error is not None

    def test_json_valid(self):
        assert _check("json_valid", '{"ok": true}').passed
        failed = _check("json_valid", "Sure! Here you go: {oops")
        assert not failed.passed
        assert failed.message.startswith("invalid JSON")

    def test_json_schema_passes_and_fails(self):
        pytest.importorskip("jsonschema")
        schema = {
            "type": "object",
            "required": ["sentiment"],
            "properties": {"sentiment": {"enum": ["positive", "negative"]}},
        }
        assert _check("json_schema", '{"sentiment": "positive"}', schema).passed
        failed = _check("json_schema", '{"sentiment": "meh"}', schema)
        assert not failed.passed
        assert "schema violation at sentiment" in failed.message
        missing = _check("json_schema", "{}", schema)
        assert not missing.passed
        assert "schema violation at $" in missing.message

    def test_json_schema_on_invalid_json(self):
        failed = _check("json_schema", "nope", {"type": "object"})
        assert not failed.passed
        assert failed.message.startswith("invalid JSON")

    def test_json_schema_without_dependency_fails_with_install_hint(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "jsonschema":
                raise ImportError("no jsonschema")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        failed = _check("json_schema", "{}", {"type": "object"})
        assert not failed.passed
        assert "promptlens[schema]" in failed.message


class TestBoundsChecks:
    def test_min_and_max_chars(self):
        assert _check("min_chars", "hello", 5).passed
        assert not _check("min_chars", "hi", 5).passed
        assert _check("max_chars", "hello", 5).passed
        failed = _check("max_chars", "hello!", 5)
        assert not failed.passed
        assert failed.message == "6 chars (max 5)"

    def test_max_latency_ms(self):
        fast = check_assertion(Assertion(type="max_latency_ms", value=500), _response("x", latency_ms=200))
        slow = check_assertion(Assertion(type="max_latency_ms", value=500), _response("x", latency_ms=900))
        assert fast.passed
        assert not slow.passed
        assert "900 ms (max 500 ms)" in slow.message


class TestRunAssertions:
    def _case(self, assertions, **kwargs):
        return TestCase(
            id="tc-1",
            query="q",
            expected_behavior="e",
            assertions=assertions,
            **kwargs,
        )

    def test_runs_in_declaration_order(self):
        case = self._case([
            {"type": "contains", "value": "b"},
            {"type": "max_chars", "value": 1},
        ])
        results = run_assertions(case, _response("abc"))
        assert [r.type for r in results] == ["contains", "max_chars"]
        assert [r.passed for r in results] == [True, False]

    def test_no_assertions_returns_empty(self):
        assert run_assertions(self._case([]), _response("abc")) == []

    def test_errored_response_skips_checks(self):
        case = self._case([{"type": "contains", "value": "x"}])
        assert run_assertions(case, _response("", error="timeout")) == []


class TestTestCaseIntegration:
    def test_assertions_only_requires_assertions(self):
        with pytest.raises(ValidationError, match="declares no assertions"):
            TestCase(id="t", query="q", expected_behavior="e", evaluation_mode="assertions_only")

    def test_unknown_evaluation_mode_rejected(self):
        with pytest.raises(ValidationError, match="evaluation_mode must be one of"):
            TestCase(id="t", query="q", expected_behavior="e", evaluation_mode="vibes")

    def test_existing_modes_still_accepted(self):
        for mode in ("standard", "tool_only", "tool_and_answer"):
            assert TestCase(id="t", query="q", expected_behavior="e", evaluation_mode=mode).evaluation_mode == mode

    def test_yaml_golden_set_with_assertions_loads(self, tmp_path):
        golden = tmp_path / "golden.yaml"
        golden.write_text(textwrap.dedent(
            """
            name: "Assertions"
            test_cases:
              - id: "a-1"
                query: "Say OK"
                expected_behavior: "Says OK"
                evaluation_mode: assertions_only
                assertions:
                  - type: equals
                    value: "OK"
                  - type: json_valid
              - id: "a-2"
                query: "Return JSON"
                expected_behavior: "JSON object"
                assertions:
                  - type: json_schema
                    value:
                      type: object
                      required: ["id"]
            """
        ))
        golden_set = get_loader(str(golden)).load(str(golden))
        assert len(golden_set.test_cases) == 2
        assert golden_set.test_cases[0].assertions[0].type == "equals"
        assert golden_set.test_cases[0].assertions[1].type == "json_valid"
        assert golden_set.test_cases[1].assertions[0].value["required"] == ["id"]

    def test_yaml_golden_set_with_bad_regex_fails_validation(self, tmp_path):
        golden = tmp_path / "golden.yaml"
        golden.write_text(textwrap.dedent(
            """
            name: "Broken"
            test_cases:
              - id: "a-1"
                query: "q"
                expected_behavior: "e"
                assertions:
                  - type: regex
                    value: "(("
            """
        ))
        with pytest.raises(ValueError, match="invalid pattern"):
            get_loader(str(golden)).load(str(golden))

    def test_shipped_example_golden_set_validates(self):
        from pathlib import Path

        example = Path(__file__).parent.parent / "examples" / "golden_sets" / "assertions.yaml"
        golden_set = get_loader(str(example)).load(str(example))
        assert sum(len(tc.assertions) for tc in golden_set.test_cases) >= 8
        assert any(tc.evaluation_mode == "assertions_only" for tc in golden_set.test_cases)
