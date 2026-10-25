"""Import generic JSON or JSONL rows as trace records.

For logs that are neither OTel spans nor provider request logs: one object
per row with an input column and (optionally) an output column. Field names
are auto-detected from the common candidates below, or set explicitly with
``--input-field`` / ``--output-field`` / ``--id-field``.
"""

import logging
from typing import Any, Dict, List, Optional

from promptlens.importers.openai_log import load_json_or_jsonl
from promptlens.importers.records import Message, TraceRecord, content_to_text

logger = logging.getLogger(__name__)

INPUT_FIELDS = ("input", "query", "prompt", "question", "user_message", "message", "text")
OUTPUT_FIELDS = ("output", "response", "answer", "completion", "assistant_message", "reply")
ID_FIELDS = ("id", "trace_id", "conversation_id", "request_id", "uuid")
TIMESTAMP_FIELDS = ("timestamp", "created_at", "time", "ts")
MODEL_FIELDS = ("model", "model_name", "llm")
ERROR_FIELDS = ("error", "failed", "is_error")
TOKEN_FIELDS = ("total_tokens", "tokens")
COST_FIELDS = ("cost_usd", "cost")
LATENCY_FIELDS = ("latency_ms", "duration_ms", "latency")


def _first_present(row: Dict[str, Any], names: Any) -> Optional[Any]:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _first_key(row: Dict[str, Any], names: Any) -> Optional[str]:
    """Name of the first candidate column present in the row, even if empty."""
    for name in names:
        if name in row:
            return name
    return None


def _get_path(row: Dict[str, Any], dotted: str) -> Optional[Any]:
    """Resolve ``a.b.c`` against nested dicts."""
    current: Any = row
    for part in dotted.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def load_generic_rows(
    path: str,
    input_field: Optional[str] = None,
    output_field: Optional[str] = None,
    id_field: Optional[str] = None,
) -> List[TraceRecord]:
    """Read a generic JSON/JSONL log and return one record per row."""
    rows = load_json_or_jsonl(path)
    records = []
    for index, row in enumerate(rows):
        if input_field:
            raw_input = _get_path(row, input_field)
        else:
            input_key = _first_key(row, INPUT_FIELDS)
            if input_key is None:
                continue
            raw_input = row[input_key]
        if raw_input is None and input_field:
            continue
        raw_output = (
            _get_path(row, output_field) if output_field else _first_present(row, OUTPUT_FIELDS)
        )
        raw_id = _get_path(row, id_field) if id_field else _first_present(row, ID_FIELDS)

        if (
            isinstance(raw_input, list)
            and raw_input
            and isinstance(raw_input[0], dict)
            and "role" in raw_input[0]
        ):
            messages = [
                Message(role=str(m.get("role", "user")), content=content_to_text(m.get("content")))
                for m in raw_input
                if isinstance(m, dict) and m.get("role") != "system"
            ]
        else:
            messages = [Message(role="user", content=content_to_text(raw_input))]

        error_value = _first_present(row, ERROR_FIELDS)
        error = (
            bool(error_value)
            if not isinstance(error_value, str)
            else error_value.lower() not in ("", "false", "0", "none")
        )

        tokens = _first_present(row, TOKEN_FIELDS)
        cost = _first_present(row, COST_FIELDS)
        latency = _first_present(row, LATENCY_FIELDS)
        timestamp = _first_present(row, TIMESTAMP_FIELDS)

        metadata: Dict[str, Any] = {"source_platform": "jsonl"}
        for key in ("tags", "category", "user", "session_id", "environment", "agent"):
            if key in row and isinstance(row[key], (str, int, float, list)):
                metadata[key] = row[key]

        records.append(
            TraceRecord(
                source_id=(
                    str(raw_id) if raw_id is not None else f"line-{row.get('_line', index + 1)}"
                ),
                messages=messages,
                output=content_to_text(raw_output) or None if raw_output is not None else None,
                model=_first_present(row, MODEL_FIELDS),
                provider=row.get("provider") if isinstance(row.get("provider"), str) else None,
                input_tokens=int(tokens) if isinstance(tokens, (int, float)) else None,
                cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
                latency_ms=float(latency) if isinstance(latency, (int, float)) else None,
                error=error,
                timestamp=str(timestamp) if timestamp is not None else None,
                metadata=metadata,
            )
        )
    logger.info(f"Loaded {len(records)} rows from {path}")
    return records
