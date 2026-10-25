"""Turn normalized trace records into a reviewable golden set.

Pipeline: filter out unusable rows, deduplicate near-identical inputs, scrub
personal data, tag each case from telemetry alone (errors, tool failures,
thrash, truncation), rank by how interesting the case is, cut to the
requested size, then write PromptLens test cases with full provenance.

Every imported case is marked ``needs_review`` because the expected behavior
has to be written by a person. The production response is kept as
``reference_answer`` only when the run was clean, so a broken answer never
becomes the thing a judge compares against.
"""

import hashlib
import json
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Tuple

import yaml

from promptlens.importers.jsonl import load_generic_rows
from promptlens.importers.openai_log import load_openai_logs
from promptlens.importers.otel import load_otel_traces
from promptlens.importers.records import TraceRecord
from promptlens.importers.scrub import ScrubReport, scrub_text
from promptlens.models.test_case import GoldenSet, TestCase
from promptlens.models.tools import ExpectedToolCall

SOURCE_FORMATS = ("auto", "otel", "openai", "jsonl")
SORT_MODES = ("signal", "tokens", "recent", "random")
REFERENCE_MODES = ("clean", "always", "never")

DROP_NO_INPUT = "no_input"
DROP_NO_OUTPUT = "no_output"
DROP_INPUT_LENGTH = "input_length"
DROP_TEST_TRAFFIC = "test_traffic"
DROP_DUPLICATE = "duplicate"
DROP_ERROR = "error"

TAG_ERROR = "error"
TAG_TOOL_ERROR = "tool-error"
TAG_REPEATED_TOOL_CALL = "repeated-tool-call"
TAG_THRASH = "thrash"
TAG_TRUNCATED = "truncated"
TAG_MULTI_TURN = "multi-turn"
TAG_NO_OUTPUT = "no-output"
TAG_SCRUBBED = "scrubbed"

DEFAULT_EXPECTED_BEHAVIOR = (
    "REVIEW: describe what a correct response does for this input. "
    "The production response is kept in reference_answer for comparison."
)
DEFAULT_EXPECTED_BEHAVIOR_ERROR = (
    "REVIEW: production failed on this input. Describe what a correct response does; "
    "the failing run is recorded in metadata."
)

TRUNCATED_FINISH_REASONS = {"length", "max_tokens", "MAX_TOKENS"}
TOOL_RESULT_PREVIEW_CHARS = 500


@dataclass
class ImportOptions:
    """User-facing knobs for the import pipeline."""

    name: str
    source_file: str
    source_format: str = "auto"
    prefix: str = "prod"
    limit: Optional[int] = None
    sort: str = "signal"
    seed: int = 0
    include_history: bool = False
    reference_answer: str = "clean"
    expected_tool_calls: bool = False
    scrub: bool = True
    drop_errors: bool = False
    min_input_chars: int = 3
    max_input_chars: int = 8000
    exclude_id_patterns: List[str] = field(default_factory=list)
    thrash_tool_calls: int = 6
    category: Optional[str] = None
    expected_behavior: Optional[str] = None
    input_field: Optional[str] = None
    output_field: Optional[str] = None
    id_field: Optional[str] = None

    def validate(self) -> None:
        if self.source_format not in SOURCE_FORMATS:
            raise ValueError(f"source_format must be one of {SOURCE_FORMATS}")
        if self.sort not in SORT_MODES:
            raise ValueError(f"sort must be one of {SORT_MODES}")
        if self.reference_answer not in REFERENCE_MODES:
            raise ValueError(f"reference_answer must be one of {REFERENCE_MODES}")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be at least 1")
        if self.min_input_chars < 0 or self.max_input_chars < self.min_input_chars:
            raise ValueError("max_input_chars must be greater than or equal to min_input_chars")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.prefix):
            raise ValueError("prefix must start with a letter or digit and use only [A-Za-z0-9_.-]")
        for pattern in self.exclude_id_patterns:
            re.compile(pattern)


@dataclass
class ImportSummary:
    """What happened during an import, for the CLI and the golden set metadata."""

    source_file: str
    source_format: str
    candidates_seen: int = 0
    kept: int = 0
    dropped: Counter = field(default_factory=Counter)
    tags: Counter = field(default_factory=Counter)
    scrub: ScrubReport = field(default_factory=ScrubReport)
    skipped_existing: int = 0

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "source_format": self.source_format,
            "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "candidates_seen": self.candidates_seen,
            "kept": self.kept,
            "dropped": dict(sorted(self.dropped.items())),
            "tags": dict(sorted(self.tags.items())),
            "scrubbed": dict(sorted(self.scrub.replacements.items())),
            "skipped_existing": self.skipped_existing,
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def detect_format(path: str) -> str:
    """Guess the source format from the file's first non-empty document."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    text = file_path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return "jsonl"

    sample: Any = None
    try:
        sample = json.loads(stripped)
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    sample = None
                break
    if isinstance(sample, list) and sample:
        sample = sample[0]
    if not isinstance(sample, dict):
        return "jsonl"

    if "resourceSpans" in sample or "spans" in sample:
        return "otel"
    if "name" in sample and ("attributes" in sample or "context" in sample):
        return "otel"
    request = sample.get("request") if isinstance(sample.get("request"), dict) else sample
    response = sample.get("response") if isinstance(sample.get("response"), dict) else sample
    if isinstance(request.get("messages"), list) or isinstance(response.get("choices"), list):
        return "openai"
    # Responses API: a list of input items, an instructions string, or output items.
    if isinstance(request.get("input"), list) or isinstance(request.get("instructions"), str):
        return "openai"
    if isinstance(response.get("output"), list) and "input" in request:
        return "openai"
    return "jsonl"


def load_records(options: ImportOptions) -> Tuple[List[TraceRecord], str]:
    """Load trace records using the requested or detected adapter."""
    source_format = options.source_format
    if source_format == "auto":
        source_format = detect_format(options.source_file)
    if source_format == "otel":
        records = load_otel_traces(options.source_file)
    elif source_format == "openai":
        records = load_openai_logs(options.source_file)
    else:
        records = load_generic_rows(
            options.source_file,
            input_field=options.input_field,
            output_field=options.output_field,
            id_field=options.id_field,
        )
    return records, source_format


# ---------------------------------------------------------------------------
# Filtering, dedupe, tagging, ranking
# ---------------------------------------------------------------------------


def normalize_query(text: str) -> str:
    """Normalization key for duplicate detection: case, whitespace, punctuation."""
    lowered = text.lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _tool_signature(name: str, arguments: Dict[str, Any]) -> str:
    try:
        return name + ":" + json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return name + ":" + repr(arguments)


def tag_record(
    record: TraceRecord, token_threshold: Optional[int], thrash_tool_calls: int
) -> List[str]:
    """Derive tags from telemetry alone, no labeling required."""
    tags = []
    if record.error:
        tags.append(TAG_ERROR)
    if record.tool_errors:
        tags.append(TAG_TOOL_ERROR)
    signatures = Counter(_tool_signature(c.name, c.arguments) for c in record.tool_calls)
    if any(count > 1 for count in signatures.values()):
        tags.append(TAG_REPEATED_TOOL_CALL)
    total = record.total_tokens
    if len(record.tool_calls) >= thrash_tool_calls or (
        token_threshold is not None and total is not None and total >= token_threshold
    ):
        tags.append(TAG_THRASH)
    if any(reason in TRUNCATED_FINISH_REASONS for reason in record.finish_reasons):
        tags.append(TAG_TRUNCATED)
    if sum(1 for m in record.messages if m.role == "user") > 1:
        tags.append(TAG_MULTI_TURN)
    if not record.output and not record.tool_calls:
        tags.append(TAG_NO_OUTPUT)
    return tags


def _signal_score(record: TraceRecord, tags: List[str]) -> Tuple[int, int, int, int]:
    """Higher is more interesting. Errors first, then tool trouble, then size."""
    return (
        1 if TAG_ERROR in tags else 0,
        1 if (TAG_TOOL_ERROR in tags or TAG_REPEATED_TOOL_CALL in tags) else 0,
        1 if (TAG_THRASH in tags or TAG_TRUNCATED in tags) else 0,
        record.total_tokens or 0,
    )


def _token_threshold(records: List[TraceRecord]) -> Optional[int]:
    """90th percentile of total tokens, when enough records report tokens."""
    totals = sorted(r.total_tokens for r in records if r.total_tokens is not None)
    if len(totals) < 10:
        return None
    quantiles = statistics.quantiles(totals, n=10)
    return int(quantiles[-1])


def _compile_excludes(patterns: List[str]) -> List[Pattern[str]]:
    return [re.compile(p) for p in patterns]


def select_records(
    records: List[TraceRecord],
    options: ImportOptions,
    summary: ImportSummary,
    existing_queries: Optional[set] = None,
) -> List[Tuple[TraceRecord, List[str]]]:
    """Filter, dedupe, tag, rank, and cut the records. Returns (record, tags) pairs."""
    summary.candidates_seen = len(records)
    excludes = _compile_excludes(options.exclude_id_patterns)
    seen_queries = set(existing_queries or set())
    token_threshold = _token_threshold(records)

    kept: List[Tuple[TraceRecord, List[str]]] = []
    for record in records:
        if any(p.search(record.source_id) for p in excludes):
            summary.dropped[DROP_TEST_TRAFFIC] += 1
            continue
        query = record.last_user_message
        if not query:
            summary.dropped[DROP_NO_INPUT] += 1
            continue
        if not record.output and not record.tool_calls and not record.error:
            summary.dropped[DROP_NO_OUTPUT] += 1
            continue
        if record.error and options.drop_errors:
            summary.dropped[DROP_ERROR] += 1
            continue
        length = len(query.strip())
        if length < options.min_input_chars or length > options.max_input_chars:
            summary.dropped[DROP_INPUT_LENGTH] += 1
            continue
        # Dedupe on the scrubbed text so "[EMAIL]" placeholders match across imports.
        dedupe_text = scrub_text(query)[0] if options.scrub else query
        key = normalize_query(dedupe_text or "")
        if key in seen_queries:
            summary.dropped[DROP_DUPLICATE] += 1
            continue
        seen_queries.add(key)
        kept.append((record, tag_record(record, token_threshold, options.thrash_tool_calls)))

    if options.sort == "signal":
        kept.sort(key=lambda pair: _signal_score(pair[0], pair[1]), reverse=True)
    elif options.sort == "tokens":
        kept.sort(key=lambda pair: pair[0].total_tokens or 0, reverse=True)
    elif options.sort == "recent":
        kept.sort(key=lambda pair: pair[0].timestamp or "", reverse=True)
    elif options.sort == "random":
        random.Random(options.seed).shuffle(kept)

    if options.limit is not None:
        kept = kept[: options.limit]

    summary.kept = len(kept)
    for _, tags in kept:
        summary.tags.update(tags)
    return kept


# ---------------------------------------------------------------------------
# Test case construction
# ---------------------------------------------------------------------------


def stable_case_id(prefix: str, source_id: str, taken: set) -> str:
    """Short, stable id derived from the source id; suffixed on collision."""
    digest = hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:8]
    candidate = f"{prefix}-{digest}"
    suffix = 2
    while candidate in taken:
        candidate = f"{prefix}-{digest}-{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def render_query(record: TraceRecord, include_history: bool) -> str:
    """The test case query: the last user turn, optionally with earlier turns."""
    last = record.last_user_message or ""
    if not include_history:
        return last
    earlier = []
    last_index = None
    for index in range(len(record.messages) - 1, -1, -1):
        if record.messages[index].role == "user" and record.messages[index].content.strip():
            last_index = index
            break
    for message in record.messages[: last_index or 0]:
        if not message.content.strip():
            continue
        speaker = "User" if message.role == "user" else "Assistant"
        earlier.append(f"{speaker}: {message.content.strip()}")
    if not earlier:
        return last
    return "Earlier in the conversation:\n" + "\n\n".join(earlier) + "\n\nCurrent message:\n" + last


def _trajectory(record: TraceRecord) -> List[Dict[str, Any]]:
    steps = []
    for call in record.tool_calls:
        step: Dict[str, Any] = {"name": call.name, "arguments": call.arguments}
        if call.result is not None:
            preview = call.result
            if len(preview) > TOOL_RESULT_PREVIEW_CHARS:
                preview = preview[:TOOL_RESULT_PREVIEW_CHARS] + "..."
            step["result"] = preview
        if call.error:
            step["error"] = True
        steps.append(step)
    return steps


def _apply_scrub(
    record: TraceRecord, query: str, options: ImportOptions, summary: ImportSummary
) -> Tuple[str, Optional[str], ScrubReport]:
    report = ScrubReport()
    if not options.scrub:
        return query, record.output, report
    query, query_report = scrub_text(query)
    report.merge(query_report)
    output, output_report = scrub_text(record.output)
    report.merge(output_report)
    for call in record.tool_calls:
        if call.result:
            call.result, result_report = scrub_text(call.result)
            report.merge(result_report)
        scrubbed_arguments = {}
        for key, value in call.arguments.items():
            if isinstance(value, str):
                value, argument_report = scrub_text(value)
                report.merge(argument_report)
            scrubbed_arguments[key] = value
        call.arguments = scrubbed_arguments
    summary.scrub.merge(report)
    return query, output, report


def build_test_case(
    record: TraceRecord,
    tags: List[str],
    options: ImportOptions,
    summary: ImportSummary,
    taken_ids: set,
) -> TestCase:
    """Map one record to a PromptLens test case with provenance metadata."""
    query = render_query(record, options.include_history)
    query, output, scrub_report = _apply_scrub(record, query, options, summary)
    case_tags = list(tags)
    if scrub_report.total:
        case_tags.append(TAG_SCRUBBED)
        summary.tags[TAG_SCRUBBED] += 1

    if options.expected_behavior:
        expected_behavior = options.expected_behavior
    elif record.error:
        expected_behavior = DEFAULT_EXPECTED_BEHAVIOR_ERROR
    else:
        expected_behavior = DEFAULT_EXPECTED_BEHAVIOR

    reference_answer = None
    run_was_clean = not record.error and TAG_TOOL_ERROR not in tags
    if output and (
        options.reference_answer == "always"
        or (options.reference_answer == "clean" and run_was_clean)
    ):
        reference_answer = output

    metadata: Dict[str, Any] = {
        "needs_review": True,
        "source_id": record.source_id,
        "source_platform": record.metadata.get("source_platform", "unknown"),
        "source_file": Path(options.source_file).name,
    }
    if record.timestamp:
        metadata["observed_at"] = record.timestamp
    if record.model:
        metadata["production_model"] = record.model
    if record.provider:
        metadata["production_provider"] = record.provider
    if record.input_tokens is not None:
        metadata["input_tokens"] = record.input_tokens
    if record.output_tokens is not None:
        metadata["output_tokens"] = record.output_tokens
    if record.total_tokens is not None:
        metadata["total_tokens"] = record.total_tokens
    if record.cost_usd is not None:
        metadata["cost_usd"] = record.cost_usd
    if record.latency_ms is not None:
        metadata["latency_ms"] = round(record.latency_ms, 1)
    if record.finish_reasons:
        metadata["finish_reasons"] = record.finish_reasons
    if record.tool_calls:
        metadata["tool_calls"] = len(record.tool_calls)
        metadata["tool_errors"] = record.tool_errors
        metadata["trajectory"] = _trajectory(record)
    metadata["production_error"] = bool(record.error)
    if record.metadata.get("error_message"):
        metadata["production_error_message"] = record.metadata["error_message"]
    if output and reference_answer is None:
        metadata["production_output"] = output
    if record.system_instructions:
        metadata["system_instructions"] = record.system_instructions
    if scrub_report.total:
        metadata["scrubbed"] = dict(sorted(scrub_report.replacements.items()))
    for key in ("agent", "operation", "user", "session_id", "conversation_id", "environment"):
        if key in record.metadata:
            metadata[key] = record.metadata[key]

    expected_tool_calls = []
    if options.expected_tool_calls:
        expected_tool_calls = [
            ExpectedToolCall(name=call.name, arguments=call.arguments)
            for call in record.tool_calls
            if call.name and not call.error
        ]

    category = options.category or record.metadata.get("category")
    if not isinstance(category, str) or not category:
        category = "production"

    extra_tags = record.metadata.get("tags")
    if isinstance(extra_tags, list):
        case_tags.extend(str(t) for t in extra_tags if str(t) not in case_tags)

    return TestCase(
        id=stable_case_id(options.prefix, record.source_id, taken_ids),
        query=query,
        expected_behavior=expected_behavior,
        category=category,
        tags=case_tags,
        metadata=metadata,
        reference_answer=reference_answer,
        expected_tool_calls=expected_tool_calls,
    )


# ---------------------------------------------------------------------------
# Golden set assembly and writing
# ---------------------------------------------------------------------------


def build_golden_set(
    options: ImportOptions,
    existing: Optional[GoldenSet] = None,
) -> Tuple[GoldenSet, ImportSummary]:
    """Run the whole pipeline and return the golden set plus a summary.

    When ``existing`` is given, its cases are kept, cases whose source id or
    normalized query already exists are skipped, and new cases are appended.
    """
    options.validate()
    records, source_format = load_records(options)
    summary = ImportSummary(source_file=options.source_file, source_format=source_format)

    taken_ids: set = set()
    existing_queries: set = set()
    existing_source_ids: set = set()
    if existing is not None:
        for case in existing.test_cases:
            taken_ids.add(case.id)
            existing_queries.add(normalize_query(case.query))
            source_id = case.metadata.get("source_id")
            if source_id:
                existing_source_ids.add(str(source_id))

    if existing_source_ids:
        fresh = []
        for record in records:
            if record.source_id in existing_source_ids:
                summary.skipped_existing += 1
            else:
                fresh.append(record)
        records = fresh

    selected = select_records(records, options, summary, existing_queries)
    new_cases = [
        build_test_case(record, tags, options, summary, taken_ids) for record, tags in selected
    ]

    if existing is not None:
        metadata = dict(existing.metadata)
        imports = list(metadata.get("imports", []))
        imports.append(summary.to_metadata())
        metadata["imports"] = imports
        golden_set = GoldenSet(
            name=existing.name,
            description=existing.description,
            version=existing.version,
            test_cases=list(existing.test_cases) + new_cases,
            metadata=metadata,
        )
    else:
        golden_set = GoldenSet(
            name=options.name,
            description=(
                f"Imported from production traces ({source_format}: "
                f"{Path(options.source_file).name}). Every case needs a reviewed "
                "expected_behavior before it is used as a regression test."
            ),
            version="1.0",
            test_cases=new_cases,
            metadata={"imports": [summary.to_metadata()]},
        )
    return golden_set, summary


def _case_to_dict(case: TestCase) -> Dict[str, Any]:
    """Serialize a test case with stable key order and no empty defaults."""
    data: Dict[str, Any] = {
        "id": case.id,
        "query": case.query,
        "expected_behavior": case.expected_behavior,
    }
    if case.category:
        data["category"] = case.category
    if case.tags:
        data["tags"] = list(case.tags)
    if case.reference_answer:
        data["reference_answer"] = case.reference_answer
    if case.tools:
        data["tools"] = [tool.model_dump(exclude_none=True) for tool in case.tools]
    if case.expected_tool_calls:
        data["expected_tool_calls"] = [
            call.model_dump(exclude_defaults=True) for call in case.expected_tool_calls
        ]
    if case.evaluation_mode != "standard":
        data["evaluation_mode"] = case.evaluation_mode
    if case.tool_execution:
        data["tool_execution"] = True
    if case.metadata:
        data["metadata"] = case.metadata
    return data


def golden_set_to_dict(golden_set: GoldenSet) -> Dict[str, Any]:
    data: Dict[str, Any] = {"name": golden_set.name}
    if golden_set.description:
        data["description"] = golden_set.description
    data["version"] = golden_set.version
    if golden_set.metadata:
        data["metadata"] = golden_set.metadata
    data["test_cases"] = [_case_to_dict(case) for case in golden_set.test_cases]
    return data


class _LiteralDumper(yaml.SafeDumper):
    """YAML dumper that writes multi-line strings as readable literal blocks."""


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    if "\n" in value:
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", value)


_LiteralDumper.add_representer(str, _represent_str)


def write_golden_set(golden_set: GoldenSet, output_path: str) -> None:
    """Write the golden set as YAML or JSON depending on the extension."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = golden_set_to_dict(golden_set)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    text = yaml.dump(
        data,
        Dumper=_LiteralDumper,
        sort_keys=False,
        allow_unicode=True,
        width=100,
        default_flow_style=False,
    )
    path.write_text(text, encoding="utf-8")
