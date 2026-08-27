"""Tests for the OpenTelemetry (OTLP/JSON) exporter."""

import json
import re
import urllib.error
from datetime import datetime

import pytest

from promptlens.exporters.otel_exporter import (
    DEFAULT_FAIL_UNDER,
    OTLP_ENDPOINT_ENV_VAR,
    SPAN_KIND_CLIENT,
    SPAN_KIND_INTERNAL,
    STATUS_CODE_ERROR,
    STATUS_CODE_OK,
    OTelExporter,
)
from promptlens.models.result import (
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)


def _make_response(model="model-a", provider="anthropic", error=None, **kwargs):
    defaults = {
        "content": "The answer is 42.",
        "model": model,
        "provider": provider,
        "latency_ms": 1234.5,
        "cost_usd": 0.0021,
        "tokens_used": 150,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "stop_reason": "end_turn",
        "timestamp": datetime(2026, 8, 27, 12, 0, 5),
        "error": error,
    }
    defaults.update(kwargs)
    return ModelResponse(**defaults)


def _make_score(score, explanation="Looks correct.", **kwargs):
    defaults = {
        "score": score,
        "explanation": explanation,
        "judge_model": "judge-model",
        "judge_provider": "anthropic",
        "timestamp": datetime(2026, 8, 27, 12, 0, 6),
    }
    defaults.update(kwargs)
    return JudgeScore(**defaults)


def _make_eval(test_case_id, model="model-a", score=None, error=None, judge=None):
    if judge is None and score is not None:
        judge = _make_score(score)
    return EvaluationResult(
        test_case_id=test_case_id,
        query="What is the answer?",
        expected_behavior="Answers correctly",
        model_response=_make_response(model=model, error=error),
        judge_score=judge,
    )


def _make_run(results, models=None, run_name="ci-run"):
    models = models or ["model-a"]
    return RunResult(
        run_id="run-123",
        run_name=run_name,
        timestamp=datetime(2026, 8, 27, 12, 0, 0),
        golden_set_name="golden-set",
        models_tested=models,
        results=results,
        total_cost_usd=0.0042,
        total_time_ms=2500.0,
    )


def _export(run_result, tmp_path, fail_under=None):
    exporter = OTelExporter(fail_under=fail_under)
    output = tmp_path / "traces.otlp.json"
    exporter.export(run_result, str(output))
    with open(output) as handle:
        return json.load(handle)


def _spans(payload):
    return payload["resourceSpans"][0]["scopeSpans"][0]["spans"]


def _attr_map(attributes):
    result = {}
    for item in attributes:
        value = item["value"]
        result[item["key"]] = next(iter(value.values()))
    return result


class TestOTelExporterStructure:
    def test_document_shape(self, tmp_path):
        payload = _export(_make_run([_make_eval("tc-1", score=4)]), tmp_path)

        resource_spans = payload["resourceSpans"]
        assert len(resource_spans) == 1
        resource_attrs = _attr_map(resource_spans[0]["resource"]["attributes"])
        assert resource_attrs["service.name"] == "promptlens"
        assert "service.version" in resource_attrs

        scope_spans = resource_spans[0]["scopeSpans"]
        assert scope_spans[0]["scope"]["name"] == "promptlens"

    def test_root_span_and_parenting(self, tmp_path):
        payload = _export(_make_run([_make_eval("tc-1", score=4)]), tmp_path)
        spans = _spans(payload)
        assert len(spans) == 2

        root, child = spans[0], spans[1]
        assert root["name"] == "promptlens.run golden-set"
        assert root["kind"] == SPAN_KIND_INTERNAL
        assert "parentSpanId" not in root
        assert child["parentSpanId"] == root["spanId"]
        assert child["traceId"] == root["traceId"]
        assert re.fullmatch(r"[0-9a-f]{32}", root["traceId"])
        assert re.fullmatch(r"[0-9a-f]{16}", root["spanId"])
        assert re.fullmatch(r"[0-9a-f]{16}", child["spanId"])
        assert child["spanId"] != root["spanId"]

    def test_root_span_run_attributes(self, tmp_path):
        payload = _export(_make_run([_make_eval("tc-1", score=4)]), tmp_path)
        attrs = _attr_map(_spans(payload)[0]["attributes"])
        assert attrs["promptlens.run.id"] == "run-123"
        assert attrs["promptlens.run.name"] == "ci-run"
        assert attrs["promptlens.golden_set.name"] == "golden-set"
        assert attrs["promptlens.cost_usd"] == 0.0042

    def test_timestamps_are_nano_strings(self, tmp_path):
        payload = _export(_make_run([_make_eval("tc-1", score=4)]), tmp_path)
        for span in _spans(payload):
            assert re.fullmatch(r"\d+", span["startTimeUnixNano"])
            assert re.fullmatch(r"\d+", span["endTimeUnixNano"])
            assert int(span["startTimeUnixNano"]) <= int(span["endTimeUnixNano"])


class TestGenAISpanAttributes:
    def test_gen_ai_semconv_attributes(self, tmp_path):
        payload = _export(_make_run([_make_eval("tc-1", score=4)]), tmp_path)
        span = _spans(payload)[1]

        assert span["name"] == "chat model-a"
        assert span["kind"] == SPAN_KIND_CLIENT
        attrs = _attr_map(span["attributes"])
        assert attrs["gen_ai.operation.name"] == "chat"
        assert attrs["gen_ai.provider.name"] == "anthropic"
        assert attrs["gen_ai.request.model"] == "model-a"
        # int64 attributes are proto3 JSON string-encoded
        assert attrs["gen_ai.usage.input_tokens"] == "100"
        assert attrs["gen_ai.usage.output_tokens"] == "50"
        finish = attrs["gen_ai.response.finish_reasons"]
        assert finish["values"][0]["stringValue"] == "end_turn"
        assert attrs["promptlens.test_case.id"] == "tc-1"
        assert span["status"]["code"] == STATUS_CODE_OK

    def test_error_sets_status_and_error_type(self, tmp_path):
        payload = _export(_make_run([_make_eval("tc-1", error="rate limited")]), tmp_path)
        span = _spans(payload)[1]
        assert span["status"]["code"] == STATUS_CODE_ERROR
        assert span["status"]["message"] == "rate limited"
        attrs = _attr_map(span["attributes"])
        assert attrs["error.type"] == "provider_error"

    def test_none_attributes_are_dropped(self, tmp_path):
        run = _make_run(
            [
                EvaluationResult(
                    test_case_id="tc-1",
                    query="q",
                    expected_behavior="e",
                    model_response=_make_response(
                        prompt_tokens=None,
                        completion_tokens=None,
                        cost_usd=None,
                        stop_reason=None,
                    ),
                )
            ]
        )
        payload = _export(run, tmp_path)
        attrs = _attr_map(_spans(payload)[1]["attributes"])
        assert "gen_ai.usage.input_tokens" not in attrs
        assert "gen_ai.usage.output_tokens" not in attrs
        assert "promptlens.cost_usd" not in attrs
        assert "gen_ai.response.finish_reasons" not in attrs


class TestEvaluationEvents:
    def _events(self, payload):
        return _spans(payload)[1].get("events", [])

    def test_overall_judge_event(self, tmp_path):
        payload = _export(_make_run([_make_eval("tc-1", score=4)]), tmp_path)
        events = self._events(payload)
        assert len(events) == 1

        event = events[0]
        assert event["name"] == "gen_ai.evaluation.result"
        attrs = _attr_map(event["attributes"])
        assert attrs["gen_ai.evaluation.name"] == "overall"
        assert attrs["gen_ai.evaluation.score.value"] == 4.0
        assert attrs["gen_ai.evaluation.score.label"] == "pass"
        assert attrs["gen_ai.evaluation.explanation"] == "Looks correct."

    def test_fail_label_below_threshold(self, tmp_path):
        payload = _export(_make_run([_make_eval("tc-1", score=2)]), tmp_path)
        attrs = _attr_map(self._events(payload)[0]["attributes"])
        assert attrs["gen_ai.evaluation.score.label"] == "fail"

    def test_custom_fail_under_threshold(self, tmp_path):
        payload = _export(_make_run([_make_eval("tc-1", score=4)]), tmp_path, fail_under=4.5)
        attrs = _attr_map(self._events(payload)[0]["attributes"])
        assert attrs["gen_ai.evaluation.score.label"] == "fail"

    def test_default_threshold_constant(self):
        assert OTelExporter().fail_under == DEFAULT_FAIL_UNDER

    def test_criteria_scores_become_events(self, tmp_path):
        judge = _make_score(4, criteria_scores={"accuracy": 5, "tone": 2})
        payload = _export(_make_run([_make_eval("tc-1", judge=judge)]), tmp_path)
        events = self._events(payload)
        assert len(events) == 3

        by_name = {
            _attr_map(e["attributes"])["gen_ai.evaluation.name"]: _attr_map(e["attributes"])
            for e in events
        }
        assert by_name["accuracy"]["gen_ai.evaluation.score.value"] == 5.0
        assert by_name["accuracy"]["gen_ai.evaluation.score.label"] == "pass"
        assert by_name["tone"]["gen_ai.evaluation.score.value"] == 2.0
        assert by_name["tone"]["gen_ai.evaluation.score.label"] == "fail"

    def test_tool_scores_become_unlabeled_events(self, tmp_path):
        judge = _make_score(4, tool_usage_score=0.9, tool_efficiency_score=0.5)
        payload = _export(_make_run([_make_eval("tc-1", judge=judge)]), tmp_path)
        by_name = {
            _attr_map(e["attributes"])["gen_ai.evaluation.name"]: _attr_map(e["attributes"])
            for e in self._events(payload)
        }
        assert by_name["tool_usage"]["gen_ai.evaluation.score.value"] == 0.9
        assert "gen_ai.evaluation.score.label" not in by_name["tool_usage"]
        assert by_name["tool_efficiency"]["gen_ai.evaluation.score.value"] == 0.5

    def test_unjudged_result_has_no_events(self, tmp_path):
        payload = _export(_make_run([_make_eval("tc-1")]), tmp_path)
        assert self._events(payload) == []


class TestEndpointPush:
    def test_no_push_without_endpoint(self, tmp_path, monkeypatch):
        monkeypatch.delenv(OTLP_ENDPOINT_ENV_VAR, raising=False)
        calls = []
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **k: calls.append(a) or pytest.fail("should not push"),
        )
        OTelExporter().export(
            _make_run([_make_eval("tc-1", score=4)]),
            str(tmp_path / "traces.otlp.json"),
        )
        assert calls == []

    def test_push_appends_traces_path(self, tmp_path, monkeypatch):
        requests = []

        class _FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_urlopen(request, timeout=None):
            requests.append(request)
            return _FakeResponse()

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        exporter = OTelExporter(endpoint="http://localhost:4318")
        exporter.export(
            _make_run([_make_eval("tc-1", score=4)]),
            str(tmp_path / "traces.otlp.json"),
        )

        assert len(requests) == 1
        request = requests[0]
        assert request.full_url == "http://localhost:4318/v1/traces"
        assert request.get_header("Content-type") == "application/json"
        body = json.loads(request.data.decode("utf-8"))
        assert "resourceSpans" in body

    def test_push_keeps_explicit_traces_path(self, tmp_path, monkeypatch):
        requests = []

        class _FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda request, timeout=None: requests.append(request) or _FakeResponse(),
        )
        OTelExporter(endpoint="http://collector:4318/v1/traces").export(
            _make_run([_make_eval("tc-1", score=4)]),
            str(tmp_path / "traces.otlp.json"),
        )
        assert requests[0].full_url == "http://collector:4318/v1/traces"

    def test_endpoint_from_environment(self, monkeypatch):
        monkeypatch.setenv(OTLP_ENDPOINT_ENV_VAR, "http://env-collector:4318")
        assert OTelExporter().endpoint == "http://env-collector:4318"

    def test_push_failure_does_not_raise(self, tmp_path, monkeypatch):
        def failing_urlopen(request, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
        output = tmp_path / "traces.otlp.json"
        OTelExporter(endpoint="http://localhost:4318").export(
            _make_run([_make_eval("tc-1", score=4)]), str(output)
        )
        # File is still written even when the collector is down.
        assert output.exists()


class TestConfigAndWiring:
    def test_otel_is_a_valid_output_format(self):
        from promptlens.models.config import OutputConfig

        config = OutputConfig(formats=["otel"])
        assert config.formats == ["otel"]

    def test_junit_is_a_valid_output_format(self):
        from promptlens.models.config import OutputConfig

        config = OutputConfig(formats=["junit"])
        assert config.formats == ["junit"]

    def test_file_extension(self):
        assert OTelExporter().file_extension == ".json"

    def test_nested_output_dir_created(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "traces.otlp.json"
        OTelExporter().export(_make_run([_make_eval("tc-1", score=4)]), str(nested))
        assert nested.exists()
