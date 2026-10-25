"""Importers that turn production traces into golden sets.

Supported sources:

- OpenTelemetry GenAI spans (OTLP/JSON, flat SDK spans, OpenLLMetry and
  OpenInference attribute dialects)
- OpenAI-style chat completion and Responses API request logs
- Generic JSON/JSONL rows with input and output columns
"""

from promptlens.importers.builder import (
    REFERENCE_MODES,
    SORT_MODES,
    SOURCE_FORMATS,
    ImportOptions,
    ImportSummary,
    build_golden_set,
    detect_format,
    write_golden_set,
)
from promptlens.importers.records import Message, ToolCallRecord, TraceRecord

__all__ = [
    "REFERENCE_MODES",
    "SORT_MODES",
    "SOURCE_FORMATS",
    "ImportOptions",
    "ImportSummary",
    "Message",
    "ToolCallRecord",
    "TraceRecord",
    "build_golden_set",
    "detect_format",
    "write_golden_set",
]
