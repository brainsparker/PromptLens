"""Import OpenAI-style chat completion logs as trace records.

Most teams that log their own LLM traffic write one JSON object per request.
This adapter accepts the common shapes, one per line (JSONL) or as a JSON
array:

* ``{"request": {...}, "response": {...}}`` where ``request`` is the chat
  completions body (``messages``, ``model``, ``tools``) and ``response`` is
  the chat completion object (``choices``, ``usage``, ``id``)
* A merged object with ``messages`` and ``choices`` side by side
* A bare request with ``messages`` whose last assistant turn is the output
* The same for the Responses API: ``input`` plus ``output`` items

Anthropic Messages API logs (``messages`` plus ``content`` blocks and
``stop_reason``) are handled by the same code path.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from promptlens.importers.records import Message, ToolCallRecord, TraceRecord, content_to_text

logger = logging.getLogger(__name__)


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSON array or a JSONL file into a list of dicts."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    text = file_path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []

    if stripped.startswith("["):
        try:
            data = json.loads(stripped)
            return [item for item in data if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass

    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_number}: invalid JSON ({e.msg})") from e
        if isinstance(item, dict):
            item.setdefault("_line", line_number)
            rows.append(item)
    return rows


def _parse_arguments(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"value": value}
    return {}


def _tool_calls_from_message(message: Dict[str, Any]) -> List[ToolCallRecord]:
    """Extract tool calls from an assistant message in OpenAI or Anthropic shape."""
    calls = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        calls.append(
            ToolCallRecord(
                name=str(function.get("name") or call.get("name") or ""),
                arguments=_parse_arguments(function.get("arguments", call.get("arguments"))),
            )
        )
    function_call = message.get("function_call")
    if isinstance(function_call, dict):
        calls.append(
            ToolCallRecord(
                name=str(function_call.get("name", "")),
                arguments=_parse_arguments(function_call.get("arguments")),
            )
        )
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") in (
                "tool_use",
                "tool_call",
                "function_call",
            ):
                calls.append(
                    ToolCallRecord(
                        name=str(block.get("name", "")),
                        arguments=_parse_arguments(block.get("input", block.get("arguments"))),
                    )
                )
    return calls


def _attach_tool_results(calls: List[ToolCallRecord], messages: List[Dict[str, Any]]) -> None:
    """Fill tool results from ``tool`` role messages, matched by order."""
    results = []
    for message in messages:
        if message.get("role") == "tool":
            results.append(content_to_text(message.get("content")))
        elif message.get("role") == "user" and isinstance(message.get("content"), list):
            for block in message["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    results.append(content_to_text(block.get("content")))
                    if block.get("is_error"):
                        results[-1] = "ERROR: " + results[-1]
    for call, result in zip(calls, results):
        call.result = result
        if result.lower().startswith(("error", "exception", "traceback")):
            call.error = True


def _is_tool_result_turn(message: Dict[str, Any]) -> bool:
    """True for Anthropic-style user turns that only carry tool results."""
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return False
    return all(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)


def _record_from_row(row: Dict[str, Any], index: int) -> Optional[TraceRecord]:
    request = row.get("request") if isinstance(row.get("request"), dict) else row
    response = row.get("response") if isinstance(row.get("response"), dict) else row

    raw_messages = request.get("messages")
    if raw_messages is None and isinstance(request.get("input"), list):
        raw_messages = request["input"]
    if raw_messages is None and isinstance(request.get("input"), str):
        raw_messages = [{"role": "user", "content": request["input"]}]
    if not isinstance(raw_messages, list):
        return None
    raw_messages = [m for m in raw_messages if isinstance(m, dict)]

    system = None
    if isinstance(request.get("system"), (str, list)):
        system = content_to_text(request["system"]) or None
    if system is None and isinstance(request.get("instructions"), str):
        system = request["instructions"] or None

    messages: List[Message] = []
    for message in raw_messages:
        role = str(message.get("role") or "user")
        if role == "system" or role == "developer":
            if system is None:
                system = content_to_text(message.get("content")) or None
            continue
        if role == "tool":
            continue
        if _is_tool_result_turn(message):
            continue
        messages.append(Message(role=role, content=content_to_text(message.get("content"))))

    # Output: response choices, Responses API output items, Anthropic content,
    # or a trailing assistant message in the request log.
    output = None
    finish_reasons: List[str] = []
    tool_calls: List[ToolCallRecord] = []
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        if isinstance(message, dict):
            output = content_to_text(message.get("content")) or None
            tool_calls = _tool_calls_from_message(message)
        for item in choices:
            if isinstance(item, dict) and item.get("finish_reason"):
                finish_reasons.append(str(item["finish_reason"]))
    elif isinstance(response.get("output"), list):
        texts = []
        for item in response["output"]:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                tool_calls.append(
                    ToolCallRecord(
                        name=str(item.get("name", "")),
                        arguments=_parse_arguments(item.get("arguments")),
                    )
                )
            elif item.get("type") == "message":
                texts.append(content_to_text(item.get("content")))
        output = "\n".join(t for t in texts if t) or None
    elif isinstance(response.get("output_text"), str):
        output = response["output_text"] or None
    elif isinstance(response.get("content"), list) and response.get("role") == "assistant":
        output = content_to_text(response["content"]) or None
        tool_calls = _tool_calls_from_message(response)
        if response.get("stop_reason"):
            finish_reasons.append(str(response["stop_reason"]))

    if output is None and messages and messages[-1].role == "assistant":
        output = messages[-1].content or None
        trailing = raw_messages[-1] if raw_messages else {}
        tool_calls = tool_calls or _tool_calls_from_message(trailing)
        messages = messages[:-1]

    if not tool_calls:
        for message in raw_messages:
            if message.get("role") == "assistant":
                tool_calls.extend(_tool_calls_from_message(message))
    _attach_tool_results(tool_calls, raw_messages)

    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))

    error = bool(row.get("error")) or bool(response.get("error"))
    if isinstance(row.get("status_code"), int) and row["status_code"] >= 400:
        error = True

    source_id = (
        row.get("id")
        or row.get("trace_id")
        or row.get("request_id")
        or response.get("id")
        or f"line-{row.get('_line', index + 1)}"
    )

    timestamp = row.get("timestamp") or row.get("created_at")
    if timestamp is None and isinstance(response.get("created"), (int, float)):
        from datetime import datetime, timezone

        timestamp = datetime.fromtimestamp(response["created"], tz=timezone.utc).isoformat()

    latency = row.get("latency_ms", row.get("duration_ms"))
    cost = row.get("cost_usd", row.get("cost"))

    metadata: Dict[str, Any] = {"source_platform": "openai-log"}
    for key in ("user", "session_id", "conversation_id", "environment"):
        if key in row and isinstance(row[key], (str, int, float)):
            metadata[key] = row[key]

    return TraceRecord(
        source_id=str(source_id),
        messages=messages,
        output=output,
        system_instructions=system,
        model=response.get("model") or request.get("model"),
        provider=row.get("provider"),
        tool_calls=tool_calls,
        input_tokens=int(input_tokens) if isinstance(input_tokens, (int, float)) else None,
        output_tokens=int(output_tokens) if isinstance(output_tokens, (int, float)) else None,
        cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
        latency_ms=float(latency) if isinstance(latency, (int, float)) else None,
        error=error,
        finish_reasons=finish_reasons,
        timestamp=str(timestamp) if timestamp is not None else None,
        metadata=metadata,
    )


def load_openai_logs(path: str) -> List[TraceRecord]:
    """Read a chat completion log file and return one record per request."""
    rows = load_json_or_jsonl(path)
    records = []
    for index, row in enumerate(rows):
        record = _record_from_row(row, index)
        if record is not None:
            records.append(record)
    logger.info(f"Loaded {len(records)} requests from {len(rows)} rows in {path}")
    return records
