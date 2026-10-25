"""Normalized trace records shared by every import source.

Each source adapter (OpenTelemetry GenAI spans, OpenAI-style request logs,
generic JSONL) turns its native shape into ``TraceRecord`` objects. The
builder then filters, deduplicates, scrubs, tags, and samples those records
into a golden set without knowing where they came from.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """One turn of a conversation."""

    role: str
    content: str


@dataclass
class ToolCallRecord:
    """A tool call observed in a production trace."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    error: bool = False


@dataclass
class TraceRecord:
    """A single production interaction, normalized across sources.

    Attributes:
        source_id: Stable identifier in the source system (trace id, response
            id, or a line number fallback). Used for provenance and stable
            test case ids.
        messages: The conversation as seen by the model, in order. The last
            user message becomes the test case query.
        output: The final assistant response, if recorded.
        system_instructions: System prompt, if recorded.
        model: Model name reported by the source.
        provider: Provider name reported by the source.
        tool_calls: Tool calls made during the interaction, in order.
        input_tokens: Prompt tokens, if reported.
        output_tokens: Completion tokens, if reported.
        cost_usd: Cost, if the source computed it.
        latency_ms: End-to-end latency, if the source recorded it.
        error: True when the source marked the interaction as failed.
        finish_reasons: Finish reasons reported by the model.
        timestamp: ISO 8601 timestamp of the interaction, if known.
        metadata: Extra source-specific fields worth keeping for provenance.
    """

    source_id: str
    messages: List[Message] = field(default_factory=list)
    output: Optional[str] = None
    system_instructions: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[float] = None
    error: bool = False
    finish_reasons: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def last_user_message(self) -> Optional[str]:
        """Return the content of the final user turn, if any."""
        for message in reversed(self.messages):
            if message.role == "user" and message.content.strip():
                return message.content
        return None

    @property
    def total_tokens(self) -> Optional[int]:
        """Prompt plus completion tokens when at least one side is known."""
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)

    @property
    def tool_errors(self) -> int:
        """Number of tool calls that failed."""
        return sum(1 for call in self.tool_calls if call.error)


def content_to_text(content: Any) -> str:
    """Flatten a message content field into plain text.

    Handles the common shapes: a string, a list of content parts (OpenAI and
    Anthropic style, plus the OTel GenAI ``parts`` shape), or an object with
    a ``text`` or ``content`` key. Anything else is JSON-ish repr'd so that
    nothing silently disappears.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            text = content_to_text(part)
            if text:
                pieces.append(text)
        return "\n".join(pieces)
    if isinstance(content, dict):
        for key in ("text", "content", "input_text", "output_text"):
            if key in content and content[key] is not None:
                return content_to_text(content[key])
        # Tool call parts carry no user-facing text; skip them.
        part_type = content.get("type")
        if part_type in (
            "tool_call",
            "tool_use",
            "tool_call_response",
            "tool_result",
            "function_call",
        ):
            return ""
        return str(content)
    return str(content)
