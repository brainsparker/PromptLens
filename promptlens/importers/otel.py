"""Import OpenTelemetry GenAI spans as trace records.

Reads span exports produced by OpenTelemetry instrumentation that follows the
GenAI semantic conventions (``gen_ai.*`` attributes), plus the two most common
dialects seen in the wild:

* OpenLLMetry / Traceloop indexed attributes (``gen_ai.prompt.0.content``)
* OpenInference, used by Arize Phoenix (``llm.input_messages.0.message.content``)

Accepted file shapes:

* OTLP/JSON as written by the OTLP file exporter or the collector
  (``{"resourceSpans": [...]}``), one document or one document per line
* A JSON array of flat spans, or a JSONL file with one flat span per line,
  as written by the Python SDK console exporter and similar tools

Spans are grouped by trace id. One trace becomes one record: the user input
comes from the top-most GenAI span, the final output from the last GenAI span,
and ``execute_tool`` spans become the tool-call trajectory.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from promptlens.importers.records import Message, ToolCallRecord, TraceRecord, content_to_text

logger = logging.getLogger(__name__)

GENAI_OPERATIONS = {"chat", "invoke_agent", "text_completion", "generate_content", "embeddings"}
TOOL_OPERATIONS = {"execute_tool"}
OPENINFERENCE_LLM_KINDS = {"LLM", "AGENT", "CHAIN"}
OPENINFERENCE_TOOL_KINDS = {"TOOL"}

STATUS_ERROR_CODES = {2, "2", "ERROR", "STATUS_CODE_ERROR", "Error"}


# ---------------------------------------------------------------------------
# File reading and span flattening
# ---------------------------------------------------------------------------


def _load_documents(path: Path) -> List[Any]:
    """Load a JSON file or a JSONL file into a list of top-level documents."""
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    try:
        return [json.loads(stripped)]
    except json.JSONDecodeError:
        pass

    documents = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            documents.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_number}: invalid JSON ({e.msg})") from e
    return documents


def _decode_any_value(value: Any) -> Any:
    """Decode an OTLP ``AnyValue`` into a plain Python value."""
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        try:
            return int(value["intValue"])
        except (TypeError, ValueError):
            return value["intValue"]
    if "doubleValue" in value:
        return value["doubleValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "arrayValue" in value:
        return [_decode_any_value(v) for v in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {
            kv.get("key"): _decode_any_value(kv.get("value"))
            for kv in value["kvlistValue"].get("values", [])
        }
    if "bytesValue" in value:
        return value["bytesValue"]
    return value


def _decode_attributes(attributes: Any) -> Dict[str, Any]:
    """Normalize OTLP key/value lists and plain dicts into one dict."""
    if attributes is None:
        return {}
    if isinstance(attributes, dict):
        return dict(attributes)
    decoded: Dict[str, Any] = {}
    if isinstance(attributes, list):
        for item in attributes:
            if isinstance(item, dict) and "key" in item:
                decoded[item["key"]] = _decode_any_value(item.get("value"))
    return decoded


def _to_nanos(value: Any) -> Optional[int]:
    """Convert a timestamp field to integer nanoseconds when possible."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
        try:
            from datetime import datetime

            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(parsed.timestamp() * 1_000_000_000)
        except ValueError:
            return None
    return None


def _flatten_span(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce an OTLP span or a flat SDK span to one common dict."""
    context = raw.get("context") or {}
    trace_id = raw.get("traceId") or raw.get("trace_id") or context.get("trace_id") or ""
    span_id = raw.get("spanId") or raw.get("span_id") or context.get("span_id") or ""
    parent_id = raw.get("parentSpanId") or raw.get("parent_span_id") or raw.get("parent_id") or ""
    if isinstance(parent_id, dict):
        parent_id = parent_id.get("span_id", "")

    status = raw.get("status") or {}
    status_code = status.get("code", status.get("status_code"))
    error = status_code in STATUS_ERROR_CODES

    events = []
    for event in raw.get("events") or []:
        if isinstance(event, dict):
            events.append(
                {
                    "name": event.get("name", ""),
                    "attributes": _decode_attributes(event.get("attributes")),
                }
            )

    return {
        "trace_id": str(trace_id),
        "span_id": str(span_id),
        "parent_id": str(parent_id or ""),
        "name": raw.get("name", ""),
        "attributes": _decode_attributes(raw.get("attributes")),
        "events": events,
        "error": error,
        "status_message": status.get("message") or status.get("description"),
        "start": _to_nanos(raw.get("startTimeUnixNano") or raw.get("start_time")),
        "end": _to_nanos(raw.get("endTimeUnixNano") or raw.get("end_time")),
    }


def _iter_raw_spans(document: Any) -> Iterable[Dict[str, Any]]:
    """Yield raw span dicts from any accepted document shape."""
    if isinstance(document, list):
        for item in document:
            for span in _iter_raw_spans(item):
                yield span
        return
    if not isinstance(document, dict):
        return
    if "resourceSpans" in document:
        for resource_span in document.get("resourceSpans") or []:
            for scope_span in resource_span.get("scopeSpans") or []:
                for span in scope_span.get("spans") or []:
                    if isinstance(span, dict):
                        yield span
        return
    if "spans" in document and isinstance(document["spans"], list):
        for span in document["spans"]:
            if isinstance(span, dict):
                yield span
        return
    if "name" in document and ("attributes" in document or "context" in document):
        yield document


def load_spans(path: str) -> List[Dict[str, Any]]:
    """Load and flatten every span in an OTel export file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    spans = []
    for document in _load_documents(file_path):
        for raw in _iter_raw_spans(document):
            spans.append(_flatten_span(raw))
    return spans


# ---------------------------------------------------------------------------
# Message extraction across GenAI dialects
# ---------------------------------------------------------------------------


def _maybe_json(value: Any) -> Any:
    """Parse JSON-encoded attribute strings, leaving other values alone."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in ("[", "{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _messages_from_semconv(value: Any) -> List[Message]:
    """Parse ``gen_ai.input.messages`` / ``gen_ai.output.messages``."""
    parsed = _maybe_json(value)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    messages = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user")
        parts = item.get("parts", item.get("content"))
        messages.append(Message(role=role, content=content_to_text(parts)))
    return messages


def _tool_calls_from_semconv_output(value: Any) -> List[ToolCallRecord]:
    """Pull ``tool_call`` parts out of ``gen_ai.output.messages``."""
    parsed = _maybe_json(value)
    calls = []
    if not isinstance(parsed, list):
        return calls
    for item in parsed:
        if not isinstance(item, dict):
            continue
        for part in item.get("parts") or []:
            if isinstance(part, dict) and part.get("type") in (
                "tool_call",
                "tool_use",
                "function_call",
            ):
                arguments = _maybe_json(part.get("arguments", part.get("input", {})))
                calls.append(
                    ToolCallRecord(
                        name=str(part.get("name", "")),
                        arguments=(
                            arguments if isinstance(arguments, dict) else {"value": arguments}
                        ),
                    )
                )
    return calls


def _messages_from_indexed(
    attributes: Dict[str, Any], prefix: str, suffix_role: str, suffix_content: str
) -> List[Message]:
    """Parse indexed attributes such as ``gen_ai.prompt.0.content``."""
    indexed: Dict[int, Dict[str, Any]] = defaultdict(dict)
    for key, value in attributes.items():
        if not key.startswith(prefix + "."):
            continue
        remainder = key[len(prefix) + 1 :]
        index_text, _, field_name = remainder.partition(".")
        if not index_text.isdigit():
            continue
        indexed[int(index_text)][field_name] = value
    messages = []
    for index in sorted(indexed):
        fields = indexed[index]
        content = fields.get(suffix_content)
        if content is None:
            continue
        role = str(
            fields.get(suffix_role)
            or ("assistant" if "completion" in prefix or "output" in prefix else "user")
        )
        messages.append(Message(role=role, content=content_to_text(_maybe_json(content))))
    return messages


def _extract_messages(span: Dict[str, Any]) -> Tuple[List[Message], List[Message], Optional[str]]:
    """Return (input messages, output messages, system instructions) for a span."""
    attributes = span["attributes"]
    inputs: List[Message] = []
    outputs: List[Message] = []
    system: Optional[str] = None

    # Current GenAI semantic conventions.
    if "gen_ai.input.messages" in attributes:
        inputs = _messages_from_semconv(attributes["gen_ai.input.messages"])
    if "gen_ai.output.messages" in attributes:
        outputs = _messages_from_semconv(attributes["gen_ai.output.messages"])
    if "gen_ai.system_instructions" in attributes:
        system = content_to_text(_maybe_json(attributes["gen_ai.system_instructions"])) or None

    # Earlier semconv drafts: whole conversation as JSON on one attribute or event.
    if not inputs and "gen_ai.prompt" in attributes:
        inputs = _messages_from_semconv(attributes["gen_ai.prompt"])
    if not outputs and "gen_ai.completion" in attributes:
        outputs = _messages_from_semconv(attributes["gen_ai.completion"])
    for event in span["events"]:
        event_attributes = event["attributes"]
        if (
            not inputs
            and event["name"] == "gen_ai.content.prompt"
            and "gen_ai.prompt" in event_attributes
        ):
            inputs = _messages_from_semconv(event_attributes["gen_ai.prompt"])
        if (
            not outputs
            and event["name"] == "gen_ai.content.completion"
            and "gen_ai.completion" in event_attributes
        ):
            outputs = _messages_from_semconv(event_attributes["gen_ai.completion"])
        if not inputs and event["name"] == "gen_ai.user.message" and "content" in event_attributes:
            inputs = [Message(role="user", content=content_to_text(event_attributes["content"]))]
        if not outputs and event["name"] == "gen_ai.choice":
            message = _maybe_json(event_attributes.get("message"))
            if isinstance(message, dict):
                outputs = [
                    Message(role="assistant", content=content_to_text(message.get("content")))
                ]

    # OpenLLMetry / Traceloop indexed attributes.
    if not inputs:
        inputs = _messages_from_indexed(attributes, "gen_ai.prompt", "role", "content")
    if not outputs:
        outputs = _messages_from_indexed(attributes, "gen_ai.completion", "role", "content")

    # OpenInference (Arize Phoenix).
    if not inputs:
        inputs = _messages_from_indexed(
            attributes, "llm.input_messages", "message.role", "message.content"
        )
    if not outputs:
        outputs = _messages_from_indexed(
            attributes, "llm.output_messages", "message.role", "message.content"
        )
    if not inputs and isinstance(attributes.get("input.value"), (str, dict, list)):
        inputs = [
            Message(role="user", content=content_to_text(_maybe_json(attributes["input.value"])))
        ]
    if not outputs and isinstance(attributes.get("output.value"), (str, dict, list)):
        outputs = [
            Message(
                role="assistant", content=content_to_text(_maybe_json(attributes["output.value"]))
            )
        ]

    # System prompts sometimes travel as the first input message.
    if system is None:
        for message in inputs:
            if message.role == "system" and message.content.strip():
                system = message.content
                break

    return inputs, outputs, system


def _span_kind(span: Dict[str, Any]) -> str:
    """Classify a span as 'genai', 'tool', or 'other'."""
    attributes = span["attributes"]
    operation = attributes.get("gen_ai.operation.name")
    if operation in TOOL_OPERATIONS:
        return "tool"
    if operation in GENAI_OPERATIONS:
        return "genai"
    oi_kind = attributes.get("openinference.span.kind")
    if oi_kind in OPENINFERENCE_TOOL_KINDS:
        return "tool"
    if oi_kind in OPENINFERENCE_LLM_KINDS:
        return "genai"
    if "gen_ai.tool.name" in attributes or "tool.name" in attributes:
        return "tool"
    if any(key.startswith(("gen_ai.", "llm.")) for key in attributes):
        return "genai"
    return "other"


def _tool_call_from_span(span: Dict[str, Any]) -> ToolCallRecord:
    attributes = span["attributes"]
    name = attributes.get("gen_ai.tool.name") or attributes.get("tool.name") or span["name"]
    arguments = _maybe_json(
        attributes.get("gen_ai.tool.call.arguments", attributes.get("input.value", {}))
    )
    result = attributes.get("gen_ai.tool.call.result", attributes.get("output.value"))
    if result is not None and not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False)
    return ToolCallRecord(
        name=str(name),
        arguments=(
            arguments
            if isinstance(arguments, dict)
            else ({"value": arguments} if arguments else {})
        ),
        result=result,
        error=bool(span["error"]),
    )


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _depth(span_id: str, parents: Dict[str, str]) -> int:
    depth = 0
    seen = set()
    current = span_id
    while current in parents and parents[current] and parents[current] not in seen:
        seen.add(current)
        current = parents[current]
        depth += 1
        if depth > 64:
            break
    return depth


def _record_from_trace(trace_id: str, spans: List[Dict[str, Any]]) -> Optional[TraceRecord]:
    """Collapse the spans of one trace into a single record."""
    spans = sorted(spans, key=lambda s: (s["start"] is None, s["start"] or 0))
    parents = {s["span_id"]: s["parent_id"] for s in spans if s["span_id"]}
    known_ids = set(parents)

    genai_spans = [s for s in spans if _span_kind(s) == "genai"]
    tool_spans = [s for s in spans if _span_kind(s) == "tool"]
    if not genai_spans:
        return None

    def local_depth(span: Dict[str, Any]) -> int:
        if span["parent_id"] and span["parent_id"] not in known_ids:
            return 0
        return _depth(span["span_id"], parents)

    min_depth = min(local_depth(s) for s in genai_spans)
    top_level = [s for s in genai_spans if local_depth(s) == min_depth]

    inputs: List[Message] = []
    outputs: List[Message] = []
    system: Optional[str] = None
    output_tool_calls: List[ToolCallRecord] = []
    output_span: Optional[Dict[str, Any]] = None
    for span in top_level:
        span_inputs, span_outputs, span_system = _extract_messages(span)
        if span_inputs and not inputs:
            inputs = span_inputs
        if span_outputs:
            outputs = span_outputs
            output_span = span
            output_tool_calls = _tool_calls_from_semconv_output(
                span["attributes"].get("gen_ai.output.messages")
            )
        if span_system and system is None:
            system = span_system

    # An agent span without recorded content often wraps chat spans that have it.
    if not inputs or not outputs:
        for span in genai_spans:
            span_inputs, span_outputs, span_system = _extract_messages(span)
            if span_inputs and not inputs:
                inputs = span_inputs
            if span_outputs:
                outputs = span_outputs
                output_span = span
            if span_system and system is None:
                system = span_system

    primary = top_level[0]
    attributes = primary["attributes"]
    model = (
        attributes.get("gen_ai.request.model")
        or attributes.get("gen_ai.response.model")
        or attributes.get("llm.model_name")
    )
    provider = attributes.get("gen_ai.provider.name") or attributes.get("gen_ai.system")

    input_tokens = 0
    output_tokens = 0
    saw_tokens = False
    for span in genai_spans:
        a = span["attributes"]
        for key in (
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.prompt_tokens",
            "llm.token_count.prompt",
        ):
            value = _as_int(a.get(key))
            if value is not None:
                input_tokens += value
                saw_tokens = True
                break
        for key in (
            "gen_ai.usage.output_tokens",
            "gen_ai.usage.completion_tokens",
            "llm.token_count.completion",
        ):
            value = _as_int(a.get(key))
            if value is not None:
                output_tokens += value
                saw_tokens = True
                break

    finish_reasons: List[str] = []
    finish_source = (output_span or primary)["attributes"]
    raw_finish = finish_source.get("gen_ai.response.finish_reasons")
    if isinstance(raw_finish, list):
        finish_reasons = [str(r) for r in raw_finish]
    elif isinstance(raw_finish, str) and raw_finish:
        finish_reasons = [raw_finish]

    starts = [s["start"] for s in spans if s["start"] is not None]
    ends = [s["end"] for s in spans if s["end"] is not None]
    latency_ms = None
    if starts and ends:
        latency_ms = max(0.0, (max(ends) - min(starts)) / 1_000_000)

    timestamp = None
    if starts:
        from datetime import datetime, timezone

        timestamp = datetime.fromtimestamp(min(starts) / 1_000_000_000, tz=timezone.utc).isoformat()

    tool_calls = [_tool_call_from_span(s) for s in tool_spans] or output_tool_calls
    error = any(s["error"] for s in genai_spans)
    status_messages = [
        s["status_message"] for s in genai_spans if s["error"] and s["status_message"]
    ]

    output_text = None
    for message in reversed(outputs):
        if message.role in ("assistant", "model", "ai") and message.content.strip():
            output_text = message.content
            break
    if output_text is None and outputs and outputs[-1].content.strip():
        output_text = outputs[-1].content

    metadata: Dict[str, Any] = {
        "source_platform": "otel",
        "span_name": primary["name"],
        "span_count": len(spans),
    }
    operation = attributes.get("gen_ai.operation.name") or attributes.get("openinference.span.kind")
    if operation:
        metadata["operation"] = operation
    agent_name = attributes.get("gen_ai.agent.name")
    if agent_name:
        metadata["agent"] = agent_name
    if status_messages:
        metadata["error_message"] = status_messages[0]

    return TraceRecord(
        source_id=trace_id,
        messages=[m for m in inputs if m.role != "system"],
        output=output_text,
        system_instructions=system,
        model=model,
        provider=provider,
        tool_calls=tool_calls,
        input_tokens=input_tokens if saw_tokens else None,
        output_tokens=output_tokens if saw_tokens else None,
        latency_ms=latency_ms,
        error=error,
        finish_reasons=finish_reasons,
        timestamp=timestamp,
        metadata=metadata,
    )


def load_otel_traces(path: str) -> List[TraceRecord]:
    """Read an OTel span export and return one record per trace."""
    spans = load_spans(path)
    by_trace: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for index, span in enumerate(spans):
        by_trace[span["trace_id"] or f"span-{index}"].append(span)

    records = []
    for trace_id, trace_spans in by_trace.items():
        record = _record_from_trace(trace_id, trace_spans)
        if record is not None:
            records.append(record)
    logger.info(f"Loaded {len(records)} traces from {len(spans)} spans in {path}")
    records.sort(key=lambda r: (r.timestamp is None, r.timestamp or ""))
    return records
