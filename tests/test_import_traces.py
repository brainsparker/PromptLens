"""Tests for importing production traces into golden sets."""

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from promptlens.cli import cli
from promptlens.importers import ImportOptions, build_golden_set, detect_format, write_golden_set
from promptlens.importers.builder import (
    DROP_DUPLICATE,
    DROP_INPUT_LENGTH,
    DROP_NO_INPUT,
    DROP_NO_OUTPUT,
    DROP_TEST_TRAFFIC,
    TAG_ERROR,
    TAG_MULTI_TURN,
    TAG_NO_OUTPUT,
    TAG_REPEATED_TOOL_CALL,
    TAG_SCRUBBED,
    TAG_THRASH,
    TAG_TOOL_ERROR,
    TAG_TRUNCATED,
    normalize_query,
)
from promptlens.importers.jsonl import load_generic_rows
from promptlens.importers.openai_log import load_openai_logs
from promptlens.importers.otel import load_otel_traces
from promptlens.importers.scrub import scrub_text
from promptlens.loaders.yaml_loader import get_loader

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "traces"
OTEL_EXAMPLE = EXAMPLES / "otel_genai_spans.json"
OPENAI_EXAMPLE = EXAMPLES / "openai_requests.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attr(key, value):
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, list):
        return {
            "key": key,
            "value": {"arrayValue": {"values": [{"stringValue": str(v)} for v in value]}},
        }
    return {"key": key, "value": {"stringValue": value}}


def _semconv_messages(role, text):
    return json.dumps([{"role": role, "parts": [{"type": "text", "content": text}]}])


def _otlp_span(trace_id, span_id, name, attrs, parent=None, error=False, start=0, end=1000):
    span = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "startTimeUnixNano": str(1_760_000_000_000_000_000 + start * 1_000_000),
        "endTimeUnixNano": str(1_760_000_000_000_000_000 + end * 1_000_000),
        "attributes": [_attr(k, v) for k, v in attrs.items()],
        "status": {"code": 2 if error else 1},
    }
    if parent:
        span["parentSpanId"] = parent
    return span


def _otlp_document(spans):
    return {
        "resourceSpans": [
            {"resource": {"attributes": []}, "scopeSpans": [{"scope": {}, "spans": spans}]}
        ]
    }


def _write(tmp_path, name, payload, jsonl=False):
    path = tmp_path / name
    if jsonl:
        path.write_text("\n".join(json.dumps(row) for row in payload) + "\n", encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _options(source, **overrides):
    defaults = dict(name="Imported", source_file=str(source))
    defaults.update(overrides)
    return ImportOptions(**defaults)


# ---------------------------------------------------------------------------
# OpenTelemetry adapter
# ---------------------------------------------------------------------------


class TestOtelAdapter:
    def test_groups_spans_by_trace_and_builds_trajectory(self):
        records = {r.source_id: r for r in load_otel_traces(str(OTEL_EXAMPLE))}

        refund = records["b2"]
        assert refund.last_user_message.startswith("Please refund order A-1")
        assert refund.output == "Your refund has been issued."
        assert [c.name for c in refund.tool_calls] == [
            "search_kb",
            "search_kb",
            "lookup_order",
            "issue_refund",
        ]
        assert refund.tool_errors == 2
        assert refund.tool_calls[3].arguments == {"order_id": "A-9"}
        assert refund.input_tokens == 9800 and refund.output_tokens == 120
        assert refund.latency_ms == pytest.approx(9400.0)
        assert refund.error is False
        assert "error_message" not in refund.metadata

    def test_errored_chat_span_without_output(self):
        records = {r.source_id: r for r in load_otel_traces(str(OTEL_EXAMPLE))}
        failed = records["c3"]
        assert failed.error is True
        assert failed.output is None
        assert failed.metadata["error_message"] == "429 rate limited"

    def test_http_root_wrapping_chat_spans_uses_last_output_and_finish_reason(self):
        records = {r.source_id: r for r in load_otel_traces(str(OTEL_EXAMPLE))}
        trip = records["f6"]
        assert trip.last_user_message.startswith("Plan a 3 day trip to Lisbon")
        assert trip.output.startswith("Day 1: Alfama")
        assert trip.finish_reasons == ["length"]
        assert trip.system_instructions == "You are a travel assistant."
        assert trip.input_tokens == 720
        assert trip.metadata["span_count"] == 3

    def test_flat_sdk_span_shape_and_openllmetry_attributes(self, tmp_path):
        spans = [
            {
                "name": "openai.chat",
                "context": {"trace_id": "0xabc", "span_id": "0x1"},
                "parent_id": None,
                "start_time": "2026-10-20T10:00:00Z",
                "end_time": "2026-10-20T10:00:02Z",
                "status": {"status_code": "OK"},
                "attributes": {
                    "gen_ai.system": "openai",
                    "gen_ai.request.model": "gpt-4.1",
                    "gen_ai.prompt.0.role": "system",
                    "gen_ai.prompt.0.content": "Be brief.",
                    "gen_ai.prompt.1.role": "user",
                    "gen_ai.prompt.1.content": "Summarize the Q3 report",
                    "gen_ai.completion.0.role": "assistant",
                    "gen_ai.completion.0.content": "Revenue up 12 percent.",
                    "gen_ai.usage.prompt_tokens": 50,
                    "gen_ai.usage.completion_tokens": 10,
                },
            }
        ]
        path = _write(tmp_path, "flat.json", spans)
        records = load_otel_traces(str(path))
        assert len(records) == 1
        record = records[0]
        assert record.source_id == "0xabc"
        assert record.last_user_message == "Summarize the Q3 report"
        assert record.output == "Revenue up 12 percent."
        assert record.system_instructions == "Be brief."
        assert record.provider == "openai"
        assert record.total_tokens == 60
        assert record.latency_ms == pytest.approx(2000.0)

    def test_openinference_attributes(self, tmp_path):
        spans = [
            _otlp_span(
                "t1",
                "s1",
                "ChatCompletion",
                {
                    "openinference.span.kind": "LLM",
                    "llm.model_name": "claude-sonnet-4-5",
                    "llm.input_messages.0.message.role": "user",
                    "llm.input_messages.0.message.content": "Translate hello to French",
                    "llm.output_messages.0.message.role": "assistant",
                    "llm.output_messages.0.message.content": "Bonjour",
                    "llm.token_count.prompt": 12,
                    "llm.token_count.completion": 3,
                },
            ),
            _otlp_span(
                "t1",
                "s2",
                "lookup",
                {
                    "openinference.span.kind": "TOOL",
                    "tool.name": "dictionary",
                    "input.value": json.dumps({"word": "hello"}),
                },
                parent="s1",
            ),
        ]
        path = _write(tmp_path, "oi.json", _otlp_document(spans))
        record = load_otel_traces(str(path))[0]
        assert record.last_user_message == "Translate hello to French"
        assert record.output == "Bonjour"
        assert record.model == "claude-sonnet-4-5"
        assert record.total_tokens == 15
        assert record.tool_calls[0].name == "dictionary"
        assert record.tool_calls[0].arguments == {"word": "hello"}

    def test_legacy_event_based_content(self, tmp_path):
        span = _otlp_span(
            "t2", "s1", "chat", {"gen_ai.operation.name": "chat", "gen_ai.system": "anthropic"}
        )
        span["events"] = [
            {
                "name": "gen_ai.content.prompt",
                "attributes": [
                    _attr("gen_ai.prompt", json.dumps([{"role": "user", "content": "ping"}]))
                ],
            },
            {
                "name": "gen_ai.content.completion",
                "attributes": [
                    _attr(
                        "gen_ai.completion", json.dumps([{"role": "assistant", "content": "pong"}])
                    )
                ],
            },
        ]
        path = _write(tmp_path, "events.json", _otlp_document([span]))
        record = load_otel_traces(str(path))[0]
        assert record.last_user_message == "ping"
        assert record.output == "pong"

    def test_tool_calls_fall_back_to_output_message_parts(self, tmp_path):
        output = json.dumps(
            [
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool_call",
                            "id": "c1",
                            "name": "get_weather",
                            "arguments": {"city": "Lisbon"},
                        }
                    ],
                }
            ]
        )
        span = _otlp_span(
            "t3",
            "s1",
            "chat",
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.input.messages": _semconv_messages("user", "Weather in Lisbon?"),
                "gen_ai.output.messages": output,
            },
        )
        path = _write(tmp_path, "parts.json", _otlp_document([span]))
        record = load_otel_traces(str(path))[0]
        assert record.output is None
        assert record.tool_calls[0].name == "get_weather"
        assert record.tool_calls[0].arguments == {"city": "Lisbon"}

    def test_traces_without_genai_spans_are_ignored(self, tmp_path):
        path = _write(
            tmp_path,
            "http.json",
            _otlp_document([_otlp_span("t4", "s1", "GET /", {"http.request.method": "GET"})]),
        )
        assert load_otel_traces(str(path)) == []

    def test_jsonl_of_otlp_documents(self, tmp_path):
        docs = [
            _otlp_document(
                [
                    _otlp_span(
                        "t5",
                        "s1",
                        "chat",
                        {
                            "gen_ai.operation.name": "chat",
                            "gen_ai.input.messages": _semconv_messages("user", "one"),
                            "gen_ai.output.messages": _semconv_messages("assistant", "1"),
                        },
                    )
                ]
            ),
            _otlp_document(
                [
                    _otlp_span(
                        "t6",
                        "s1",
                        "chat",
                        {
                            "gen_ai.operation.name": "chat",
                            "gen_ai.input.messages": _semconv_messages("user", "two"),
                            "gen_ai.output.messages": _semconv_messages("assistant", "2"),
                        },
                    )
                ]
            ),
        ]
        path = _write(tmp_path, "spans.jsonl", docs, jsonl=True)
        assert sorted(r.source_id for r in load_otel_traces(str(path))) == ["t5", "t6"]

    def test_invalid_json_line_reports_line_number(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"name": "chat", "attributes": {}}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="bad.jsonl:2"):
            load_otel_traces(str(path))


# ---------------------------------------------------------------------------
# OpenAI-style log adapter
# ---------------------------------------------------------------------------


class TestOpenAILogAdapter:
    def test_request_response_rows(self):
        records = {r.source_id: r for r in load_openai_logs(str(OPENAI_EXAMPLE))}
        billing = records["chatcmpl-1"]
        assert billing.last_user_message.startswith("Why was I charged twice")
        assert billing.system_instructions == "You are a billing assistant."
        assert billing.output.startswith("I see two charges")
        assert billing.model == "gpt-4.1-2026-04-14"
        assert billing.input_tokens == 210 and billing.output_tokens == 44
        assert billing.latency_ms == 1830.0
        assert billing.finish_reasons == ["stop"]

    def test_tool_call_only_response_and_multi_turn(self):
        records = {r.source_id: r for r in load_openai_logs(str(OPENAI_EXAMPLE))}
        cancel = records["chatcmpl-2"]
        assert cancel.output is None
        assert cancel.tool_calls[0].name == "schedule_cancellation"
        assert cancel.tool_calls[0].arguments == {"when": "period_end"}
        assert [m.role for m in cancel.messages] == ["user", "assistant", "user"]
        assert cancel.last_user_message == "End of period please"

    def test_error_row(self):
        records = {r.source_id: r for r in load_openai_logs(str(OPENAI_EXAMPLE))}
        failed = records["chatcmpl-4"]
        assert failed.error is True
        assert failed.output is None

    def test_bare_request_with_trailing_assistant_turn(self, tmp_path):
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "2+2?"},
                    {"role": "assistant", "content": "4"},
                ],
                "model": "local",
            }
        ]
        path = _write(tmp_path, "bare.jsonl", rows, jsonl=True)
        record = load_openai_logs(str(path))[0]
        assert record.source_id == "line-1"
        assert record.last_user_message == "2+2?"
        assert record.output == "4"
        assert record.model == "local"

    def test_anthropic_messages_shape_with_tool_use_and_result(self, tmp_path):
        rows = [
            {
                "id": "msg_1",
                "request": {
                    "model": "claude-sonnet-4-5",
                    "system": "You are terse.",
                    "messages": [
                        {"role": "user", "content": "What is the weather in Oslo?"},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "tu1",
                                    "name": "weather",
                                    "input": {"city": "Oslo"},
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "tu1",
                                    "content": "Error: upstream down",
                                    "is_error": True,
                                }
                            ],
                        },
                    ],
                },
                "response": {
                    "id": "msg_1",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [{"type": "text", "text": "I could not reach the weather service."}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 90, "output_tokens": 12},
                },
            }
        ]
        path = _write(tmp_path, "anthropic.jsonl", rows, jsonl=True)
        record = load_openai_logs(str(path))[0]
        assert record.system_instructions == "You are terse."
        assert record.last_user_message == "What is the weather in Oslo?"
        assert record.output == "I could not reach the weather service."
        assert record.finish_reasons == ["end_turn"]
        assert record.tool_calls[0].name == "weather"
        assert record.tool_calls[0].error is True
        assert record.total_tokens == 102

    def test_responses_api_shape(self, tmp_path):
        rows = [
            {
                "request": {
                    "model": "gpt-4.1",
                    "instructions": "Be helpful.",
                    "input": "Name a prime above 10",
                },
                "response": {
                    "id": "resp_1",
                    "created": 1760000000,
                    "output": [
                        {"type": "function_call", "name": "math", "arguments": '{"op": "prime"}'},
                        {"type": "message", "content": [{"type": "output_text", "text": "11"}]},
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                },
            }
        ]
        path = _write(tmp_path, "responses.jsonl", rows, jsonl=True)
        record = load_openai_logs(str(path))[0]
        assert record.source_id == "resp_1"
        assert record.system_instructions == "Be helpful."
        assert record.output == "11"
        assert record.tool_calls[0].arguments == {"op": "prime"}
        assert record.timestamp.startswith("2025-10-09")


# ---------------------------------------------------------------------------
# Generic JSONL adapter
# ---------------------------------------------------------------------------


class TestGenericAdapter:
    def test_autodetects_common_columns(self, tmp_path):
        rows = [
            {
                "id": "r1",
                "question": "Where is my order?",
                "answer": "On its way.",
                "model": "m",
                "total_tokens": 40,
                "cost_usd": 0.001,
                "tags": ["orders"],
            },
            {"prompt": "no output here"},
        ]
        path = _write(tmp_path, "rows.jsonl", rows, jsonl=True)
        records = load_generic_rows(str(path))
        assert len(records) == 2
        assert records[0].source_id == "r1"
        assert records[0].output == "On its way."
        assert records[0].input_tokens == 40
        assert records[0].metadata["tags"] == ["orders"]
        assert records[1].source_id == "line-2"
        assert records[1].output is None

    def test_explicit_dotted_fields_and_conversation_input(self, tmp_path):
        rows = [
            {
                "meta": {"trace": "x9"},
                "data": {
                    "input": [
                        {"role": "user", "content": "first"},
                        {"role": "assistant", "content": "ok"},
                        {"role": "user", "content": "second"},
                    ],
                    "target": "done",
                },
            }
        ]
        path = _write(tmp_path, "nested.json", rows)
        record = load_generic_rows(
            str(path), input_field="data.input", output_field="data.target", id_field="meta.trace"
        )[0]
        assert record.source_id == "x9"
        assert record.last_user_message == "second"
        assert record.output == "done"


# ---------------------------------------------------------------------------
# Scrubbing
# ---------------------------------------------------------------------------


class TestScrub:
    def test_replaces_personal_data_with_placeholders(self):
        text = (
            "Mail jane.doe@example.com or call (440) 555-0133. Card 4111 1111 1111 1111, SSN 123-45-6789, "
            "host 10.0.0.12, key sk-abcdefghijklmnopqrstuvwxyz0123"
        )
        scrubbed, report = scrub_text(text)
        assert "[EMAIL]" in scrubbed and "jane.doe" not in scrubbed
        assert "[PHONE]" in scrubbed and "555-0133" not in scrubbed
        assert "[CARD]" in scrubbed and "4111" not in scrubbed
        assert "[SSN]" in scrubbed and "123-45-6789" not in scrubbed
        assert "[IP]" in scrubbed and "10.0.0.12" not in scrubbed
        assert "[SECRET]" in scrubbed and "sk-abc" not in scrubbed
        assert report.total == 6

    def test_non_luhn_digit_runs_and_order_numbers_survive(self):
        scrubbed, report = scrub_text("Order 1234 5678 9012 3456 shipped; ticket 98765")
        assert scrubbed == "Order 1234 5678 9012 3456 shipped; ticket 98765"
        assert report.total == 0

    def test_empty_and_none(self):
        assert scrub_text(None) == (None, scrub_text(None)[1])
        assert scrub_text("")[0] == ""


# ---------------------------------------------------------------------------
# Builder pipeline
# ---------------------------------------------------------------------------


class TestBuilder:
    def test_otel_example_end_to_end(self):
        golden_set, summary = build_golden_set(
            _options(OTEL_EXAMPLE, exclude_id_patterns=["^test-"])
        )
        assert summary.source_format == "otel"
        assert summary.candidates_seen == 6
        assert summary.dropped[DROP_DUPLICATE] == 1
        assert summary.dropped[DROP_TEST_TRAFFIC] == 1
        assert summary.kept == 4
        assert len(golden_set.test_cases) == 4

        by_source = {tc.metadata["source_id"]: tc for tc in golden_set.test_cases}

        # Signal sort: the failed run leads, then the tool-error run.
        assert golden_set.test_cases[0].metadata["source_id"] == "c3"
        assert golden_set.test_cases[1].metadata["source_id"] == "b2"

        failed = by_source["c3"]
        assert TAG_ERROR in failed.tags and TAG_NO_OUTPUT in failed.tags
        assert failed.reference_answer is None
        assert failed.expected_behavior.startswith("REVIEW: production failed")
        assert failed.metadata["production_error_message"] == "429 rate limited"

        refund = by_source["b2"]
        assert (
            TAG_TOOL_ERROR in refund.tags
            and TAG_REPEATED_TOOL_CALL in refund.tags
            and TAG_SCRUBBED in refund.tags
        )
        assert refund.query == "Please refund order A-1, it arrived broken. Call me at [PHONE]."
        assert (
            refund.reference_answer is None
        ), "a run with tool errors must not become the reference"
        assert refund.metadata["production_output"] == "Your refund has been issued."
        assert refund.metadata["trajectory"][3] == {
            "name": "issue_refund",
            "arguments": {"order_id": "A-9"},
            "result": "refund ok",
        }
        assert refund.metadata["scrubbed"] == {"[PHONE]": 1}
        assert refund.metadata["agent"] == "support-bot"

        trip = by_source["f6"]
        assert TAG_TRUNCATED in trip.tags
        assert trip.metadata["system_instructions"] == "You are a travel assistant."

        clean = by_source["a1"]
        assert clean.query == "How do I reset my password? My email is [EMAIL]"
        assert clean.reference_answer.startswith("Open Settings")
        assert clean.metadata["needs_review"] is True
        assert clean.metadata["production_model"] == "claude-sonnet-4-5"
        assert clean.category == "production"

        imports = golden_set.metadata["imports"]
        assert imports[0]["kept"] == 4
        assert imports[0]["scrubbed"] == {"[EMAIL]": 1, "[PHONE]": 1}

    def test_ids_are_stable_and_prefixed(self):
        first, _ = build_golden_set(_options(OTEL_EXAMPLE, prefix="support"))
        second, _ = build_golden_set(_options(OTEL_EXAMPLE, prefix="support"))
        assert [tc.id for tc in first.test_cases] == [tc.id for tc in second.test_cases]
        assert all(tc.id.startswith("support-") for tc in first.test_cases)

    def test_limit_and_sort_modes(self):
        by_tokens, _ = build_golden_set(_options(OTEL_EXAMPLE, sort="tokens", limit=2))
        assert [tc.metadata["source_id"] for tc in by_tokens.test_cases] == ["b2", "f6"]

        random_a, _ = build_golden_set(_options(OTEL_EXAMPLE, sort="random", seed=7))
        random_b, _ = build_golden_set(_options(OTEL_EXAMPLE, sort="random", seed=7))
        assert [tc.id for tc in random_a.test_cases] == [tc.id for tc in random_b.test_cases]

        recent, _ = build_golden_set(_options(OPENAI_EXAMPLE, sort="recent"))
        observed = [tc.metadata["observed_at"] for tc in recent.test_cases]
        assert observed == sorted(observed, reverse=True)

    def test_reference_answer_modes(self):
        never, _ = build_golden_set(_options(OTEL_EXAMPLE, reference_answer="never"))
        assert all(tc.reference_answer is None for tc in never.test_cases)

        always, _ = build_golden_set(_options(OTEL_EXAMPLE, reference_answer="always"))
        refund = next(tc for tc in always.test_cases if tc.metadata["source_id"] == "b2")
        assert refund.reference_answer == "Your refund has been issued."

    def test_drop_errors_and_no_scrub(self):
        golden_set, summary = build_golden_set(
            _options(OTEL_EXAMPLE, drop_errors=True, scrub=False)
        )
        assert summary.dropped["error"] == 1
        clean = next(tc for tc in golden_set.test_cases if tc.metadata["source_id"] == "a1")
        assert "jane.doe@example.com" in clean.query
        assert TAG_SCRUBBED not in clean.tags

    def test_include_history_and_expected_tool_calls(self):
        golden_set, summary = build_golden_set(
            _options(OPENAI_EXAMPLE, include_history=True, expected_tool_calls=True)
        )
        assert summary.dropped[DROP_INPUT_LENGTH] == 1
        cancel = next(
            tc for tc in golden_set.test_cases if tc.metadata["source_id"] == "chatcmpl-2"
        )
        assert cancel.query.startswith("Earlier in the conversation:\nUser: Cancel my subscription")
        assert cancel.query.endswith("Current message:\nEnd of period please")
        assert TAG_MULTI_TURN in cancel.tags
        assert TAG_NO_OUTPUT not in cancel.tags
        assert cancel.expected_tool_calls[0].name == "schedule_cancellation"
        assert cancel.expected_tool_calls[0].arguments == {"when": "period_end"}

    def test_card_scrubbed_in_openai_example(self):
        golden_set, _ = build_golden_set(_options(OPENAI_EXAMPLE))
        billing = next(
            tc for tc in golden_set.test_cases if tc.metadata["source_id"] == "chatcmpl-1"
        )
        assert "[CARD]" in billing.query and "4111" not in billing.query

    def test_drop_reasons_for_missing_input_and_output(self, tmp_path):
        rows = [
            {"id": "a", "input": "", "output": "x"},
            {"id": "b", "input": "has input but no output"},
            {"id": "c", "input": "fine", "output": "ok"},
        ]
        path = _write(tmp_path, "rows.jsonl", rows, jsonl=True)
        golden_set, summary = build_golden_set(_options(path))
        assert summary.dropped[DROP_NO_INPUT] == 1
        assert summary.dropped[DROP_NO_OUTPUT] == 1
        assert [tc.metadata["source_id"] for tc in golden_set.test_cases] == ["c"]

    def test_thrash_tag_from_token_percentile(self, tmp_path):
        rows = [
            {"id": f"r{i}", "input": f"question number {i}", "output": "a", "total_tokens": 100}
            for i in range(11)
        ]
        rows.append({"id": "heavy", "input": "the heavy one", "output": "a", "total_tokens": 50000})
        path = _write(tmp_path, "rows.jsonl", rows, jsonl=True)
        golden_set, _ = build_golden_set(_options(path))
        heavy = golden_set.test_cases[0]
        assert heavy.metadata["source_id"] == "heavy"
        assert TAG_THRASH in heavy.tags
        assert all(TAG_THRASH not in tc.tags for tc in golden_set.test_cases[1:])

    def test_append_skips_known_sources_and_duplicate_queries(self, tmp_path):
        output = tmp_path / "golden.yaml"
        first, _ = build_golden_set(_options(OTEL_EXAMPLE, limit=2))
        write_golden_set(first, str(output))
        existing = get_loader(str(output)).load(str(output))

        merged, summary = build_golden_set(_options(OTEL_EXAMPLE), existing=existing)
        assert summary.skipped_existing == 2
        assert summary.dropped[DROP_DUPLICATE] == 1
        assert len(merged.test_cases) == 2 + summary.kept
        assert len({tc.id for tc in merged.test_cases}) == len(merged.test_cases)
        assert merged.name == existing.name
        assert len(merged.metadata["imports"]) == 2

        # Appending a different source with an overlapping query dedupes against existing cases.
        rows = [
            {
                "id": "dup",
                "input": "HOW DO I RESET MY PASSWORD? my email is jane.doe@example.com",
                "output": "x",
            }
        ]
        path = _write(tmp_path, "dup.jsonl", rows, jsonl=True)
        merged_again, summary_again = build_golden_set(_options(path), existing=merged)
        assert summary_again.dropped[DROP_DUPLICATE] == 1
        assert len(merged_again.test_cases) == len(merged.test_cases)

    def test_written_file_roundtrips_through_the_loader(self, tmp_path):
        golden_set, _ = build_golden_set(_options(OTEL_EXAMPLE, expected_tool_calls=True))
        yaml_path = tmp_path / "out.yaml"
        json_path = tmp_path / "out.json"
        write_golden_set(golden_set, str(yaml_path))
        write_golden_set(golden_set, str(json_path))

        for path in (yaml_path, json_path):
            loaded = get_loader(str(path)).load(str(path))
            assert [tc.id for tc in loaded.test_cases] == [tc.id for tc in golden_set.test_cases]
            assert (
                loaded.test_cases[1].metadata["trajectory"]
                == golden_set.test_cases[1].metadata["trajectory"]
            )

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert list(data["test_cases"][0].keys())[:3] == ["id", "query", "expected_behavior"]

    def test_option_validation(self):
        with pytest.raises(ValueError, match="sort"):
            _options(OTEL_EXAMPLE, sort="nope").validate()
        with pytest.raises(ValueError, match="prefix"):
            _options(OTEL_EXAMPLE, prefix="-bad").validate()
        with pytest.raises(ValueError, match="limit"):
            _options(OTEL_EXAMPLE, limit=0).validate()
        with pytest.raises(ValueError, match="reference_answer"):
            _options(OTEL_EXAMPLE, reference_answer="maybe").validate()
        with pytest.raises(Exception):
            _options(OTEL_EXAMPLE, exclude_id_patterns=["("]).validate()

    def test_detect_format(self, tmp_path):
        assert detect_format(str(OTEL_EXAMPLE)) == "otel"
        assert detect_format(str(OPENAI_EXAMPLE)) == "openai"
        generic = _write(tmp_path, "g.jsonl", [{"input": "a", "output": "b"}], jsonl=True)
        assert detect_format(str(generic)) == "jsonl"
        flat = _write(tmp_path, "flat.json", [{"name": "chat", "attributes": {}}])
        assert detect_format(str(flat)) == "otel"

    def test_normalize_query(self):
        assert normalize_query("  Reset   my PASSWORD!?") == normalize_query("reset my password")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestImportCommand:
    def test_writes_golden_set_and_validate_accepts_it(self, tmp_path):
        output = tmp_path / "prod.yaml"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "import",
                str(OTEL_EXAMPLE),
                "-o",
                str(output),
                "--exclude-id",
                "^test-",
                "--limit",
                "3",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Kept" in result.output
        assert "Wrote 3 test cases" in result.output
        assert output.exists()

        validated = runner.invoke(cli, ["validate", str(output)])
        assert validated.exit_code == 0, validated.output
        assert "Test Cases: 3" in validated.output

    def test_dry_run_writes_nothing(self, tmp_path):
        output = tmp_path / "prod.yaml"
        result = CliRunner().invoke(
            cli, ["import", str(OPENAI_EXAMPLE), "-o", str(output), "--dry-run"]
        )
        assert result.exit_code == 0, result.output
        assert "Dry run" in result.output
        assert not output.exists()

    def test_append_extends_existing_file(self, tmp_path):
        output = tmp_path / "prod.yaml"
        runner = CliRunner()
        assert runner.invoke(cli, ["import", str(OPENAI_EXAMPLE), "-o", str(output)]).exit_code == 0
        result = runner.invoke(
            cli,
            ["import", str(OTEL_EXAMPLE), "-o", str(output), "--append", "--exclude-id", "^test-"],
        )
        assert result.exit_code == 0, result.output
        assert "(4 new)" in result.output
        loaded = get_loader(str(output)).load(str(output))
        assert len(loaded.test_cases) == 7
        assert loaded.name == "Production traces: openai_requests"

    def test_append_requires_existing_file(self, tmp_path):
        result = CliRunner().invoke(
            cli, ["import", str(OTEL_EXAMPLE), "-o", str(tmp_path / "missing.yaml"), "--append"]
        )
        assert result.exit_code == 2
        assert "does not exist" in result.output

    def test_no_usable_cases_exits_nonzero(self, tmp_path):
        source = _write(tmp_path, "empty.jsonl", [{"input": "", "output": ""}], jsonl=True)
        output = tmp_path / "prod.yaml"
        result = CliRunner().invoke(cli, ["import", str(source), "-o", str(output)])
        assert result.exit_code == 1
        assert "No usable cases" in result.output
        assert not output.exists()

    def test_generic_format_with_explicit_fields(self, tmp_path):
        source = _write(tmp_path, "rows.json", [{"q": "What is 2+2?", "a": "4", "key": "k1"}])
        output = tmp_path / "prod.json"
        result = CliRunner().invoke(
            cli,
            [
                "import",
                str(source),
                "-o",
                str(output),
                "--format",
                "jsonl",
                "--input-field",
                "q",
                "--output-field",
                "a",
                "--id-field",
                "key",
                "--category",
                "math",
                "--prefix",
                "m",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(output.read_text(encoding="utf-8"))
        case = data["test_cases"][0]
        assert case["id"].startswith("m-")
        assert case["query"] == "What is 2+2?"
        assert case["reference_answer"] == "4"
        assert case["category"] == "math"
        assert case["metadata"]["source_id"] == "k1"
