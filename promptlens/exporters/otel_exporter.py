"""OpenTelemetry (OTLP/JSON) exporter for run results.

Exports an evaluation run as an OTLP JSON trace document following the
OpenTelemetry GenAI semantic conventions, so eval results can live in the
same observability backend as production traces (Grafana Tempo, Jaeger,
Datadog, Honeycomb, AWS, or any OTLP-compatible collector).

Mapping rules:
    - The run is a root span named "promptlens.run <golden set>".
    - Each evaluated test case becomes a child CLIENT span named
      "chat <model>" carrying gen_ai.* request/response attributes
      (provider, model, token usage, finish reasons).
    - Each judge verdict becomes a "gen_ai.evaluation.result" span event
      with gen_ai.evaluation.name, gen_ai.evaluation.score.value,
      gen_ai.evaluation.score.label (pass/fail vs. the failure threshold),
      and gen_ai.evaluation.explanation. Per-criteria sub-scores and tool
      usage scores are emitted as additional evaluation events.
    - Provider errors set span status ERROR and the error.type attribute.

PromptLens-specific fields that have no GenAI convention yet (run id, test
case id, cost) are namespaced under "promptlens.*".

The exporter is dependency-free: it emits the OTLP JSON protobuf encoding
directly and can optionally push the payload to an OTLP/HTTP endpoint
(e.g. http://localhost:4318) using the standard library.

Note: the GenAI semantic conventions are still marked "Development"
upstream; attribute names follow the spec as of August 2026.
"""

import json
import logging
import os
import secrets
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from promptlens.exporters.base import BaseExporter
from promptlens.models.result import EvaluationResult, RunResult

logger = logging.getLogger(__name__)

# Judge scores are on a 1-5 scale. Scores below this value are labeled
# "fail" on evaluation events unless a different threshold is provided.
DEFAULT_FAIL_UNDER = 3.0

# OTLP enum values (proto: opentelemetry.proto.trace.v1)
SPAN_KIND_INTERNAL = 1
SPAN_KIND_CLIENT = 3
STATUS_CODE_OK = 1
STATUS_CODE_ERROR = 2

# Environment variable honored when no endpoint is passed explicitly.
# Matches the standard OTel SDK variable so existing collector setups work.
OTLP_ENDPOINT_ENV_VAR = "OTEL_EXPORTER_OTLP_ENDPOINT"

_TRACES_PATH = "/v1/traces"


def _attr(key: str, value: Any) -> Optional[Dict[str, Any]]:
    """Encode a single attribute as an OTLP JSON KeyValue.

    Returns None for values that cannot be represented (None values),
    so callers can filter them out.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        typed: Dict[str, Any] = {"boolValue": value}
    elif isinstance(value, int):
        # int64 values are encoded as strings in proto3 JSON mapping.
        typed = {"intValue": str(value)}
    elif isinstance(value, float):
        typed = {"doubleValue": value}
    elif isinstance(value, (list, tuple)):
        typed = {"arrayValue": {"values": [{"stringValue": str(item)} for item in value]}}
    else:
        typed = {"stringValue": str(value)}
    return {"key": key, "value": typed}


def _attrs(pairs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Encode a dict of attributes, dropping None values."""
    encoded = []
    for key, value in pairs.items():
        item = _attr(key, value)
        if item is not None:
            encoded.append(item)
    return encoded


def _unix_nano(moment: datetime) -> str:
    """Encode a datetime as OTLP timeUnixNano (string of nanoseconds)."""
    return str(int(moment.timestamp() * 1_000_000_000))


def _new_trace_id() -> str:
    return secrets.token_hex(16)


def _new_span_id() -> str:
    return secrets.token_hex(8)


class OTelExporter(BaseExporter):
    """Exporter for OpenTelemetry OTLP/JSON traces.

    Writes an ExportTraceServiceRequest JSON document and optionally pushes
    it to an OTLP/HTTP collector endpoint.

    Args:
        fail_under: Judge score threshold (1-5 scale) used to label
            evaluation events "pass" or "fail". Defaults to
            DEFAULT_FAIL_UNDER.
        endpoint: Optional OTLP/HTTP base endpoint (e.g.
            "http://localhost:4318"). When set, the trace payload is POSTed
            to <endpoint>/v1/traces after the file is written. Falls back to
            the OTEL_EXPORTER_OTLP_ENDPOINT environment variable. A push
            failure logs a warning but does not fail the export, so a down
            collector never breaks a CI run.
        timeout: HTTP timeout in seconds for the optional push.
    """

    def __init__(
        self,
        fail_under: Optional[float] = None,
        endpoint: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        self.fail_under = DEFAULT_FAIL_UNDER if fail_under is None else float(fail_under)
        self.endpoint = endpoint or os.environ.get(OTLP_ENDPOINT_ENV_VAR) or None
        self.timeout = timeout

    @property
    def file_extension(self) -> str:
        return ".json"

    def export(self, result: RunResult, output_path: str) -> None:
        """Export results to an OTLP JSON file and optionally push them.

        Args:
            result: The run result to export
            output_path: Path to write the JSON file
        """
        path = self.ensure_output_dir(output_path)
        payload = self.build_payload(result)

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        logger.info("Exported OTLP traces to %s", path)

        if self.endpoint:
            self._push(payload)

    def build_payload(self, result: RunResult) -> Dict[str, Any]:
        """Build the ExportTraceServiceRequest JSON document for a run."""
        trace_id = _new_trace_id()
        root_span_id = _new_span_id()

        run_start = result.timestamp
        run_end = run_start + timedelta(milliseconds=result.total_time_ms or 0.0)

        root_span = {
            "traceId": trace_id,
            "spanId": root_span_id,
            "name": f"promptlens.run {result.golden_set_name}",
            "kind": SPAN_KIND_INTERNAL,
            "startTimeUnixNano": _unix_nano(run_start),
            "endTimeUnixNano": _unix_nano(run_end),
            "attributes": _attrs(
                {
                    "promptlens.run.id": result.run_id,
                    "promptlens.run.name": result.run_name,
                    "promptlens.golden_set.name": result.golden_set_name,
                    "promptlens.models_tested": result.models_tested,
                    "promptlens.cost_usd": result.total_cost_usd,
                }
            ),
            "status": {"code": STATUS_CODE_OK},
        }

        spans = [root_span]
        for evaluation in result.results:
            spans.append(self._evaluation_span(evaluation, trace_id, root_span_id, result))

        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": _attrs(
                            {
                                "service.name": "promptlens",
                                "service.version": _package_version(),
                            }
                        )
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "promptlens"},
                            "spans": spans,
                        }
                    ],
                }
            ]
        }

    def _evaluation_span(
        self,
        evaluation: EvaluationResult,
        trace_id: str,
        parent_span_id: str,
        result: RunResult,
    ) -> Dict[str, Any]:
        """Build the gen_ai span for one evaluated test case."""
        response = evaluation.model_response

        end = response.timestamp
        start = end - timedelta(milliseconds=response.latency_ms or 0.0)

        attributes: Dict[str, Any] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": response.provider,
            "gen_ai.request.model": response.model,
            "gen_ai.response.model": response.model,
            "gen_ai.usage.input_tokens": response.prompt_tokens,
            "gen_ai.usage.output_tokens": response.completion_tokens,
            "promptlens.run.id": result.run_id,
            "promptlens.test_case.id": evaluation.test_case_id,
            "promptlens.cost_usd": response.cost_usd,
        }
        if response.stop_reason:
            attributes["gen_ai.response.finish_reasons"] = [response.stop_reason]
        if response.error:
            attributes["error.type"] = "provider_error"

        span: Dict[str, Any] = {
            "traceId": trace_id,
            "spanId": _new_span_id(),
            "parentSpanId": parent_span_id,
            "name": f"chat {response.model}",
            "kind": SPAN_KIND_CLIENT,
            "startTimeUnixNano": _unix_nano(start),
            "endTimeUnixNano": _unix_nano(end),
            "attributes": _attrs(attributes),
        }

        if response.error:
            span["status"] = {
                "code": STATUS_CODE_ERROR,
                "message": response.error,
            }
        else:
            span["status"] = {"code": STATUS_CODE_OK}

        events = self._evaluation_events(evaluation, end)
        if events:
            span["events"] = events

        return span

    def _evaluation_events(
        self, evaluation: EvaluationResult, moment: datetime
    ) -> List[Dict[str, Any]]:
        """Build gen_ai.evaluation.result events for a judge verdict."""
        judge = evaluation.judge_score
        if judge is None:
            return []

        when = judge.timestamp or moment
        events = [
            self._evaluation_event(
                name="overall",
                score=float(judge.score),
                label=self._label(float(judge.score)),
                explanation=judge.explanation,
                when=when,
            )
        ]

        for criterion, score in sorted(judge.criteria_scores.items()):
            events.append(
                self._evaluation_event(
                    name=criterion,
                    score=float(score),
                    label=self._label(float(score)),
                    when=when,
                )
            )

        # Tool scores are on a 0.0-1.0 scale; pass/fail labeling with the
        # 1-5 judge threshold would be misleading, so no label is emitted.
        if judge.tool_usage_score is not None:
            events.append(
                self._evaluation_event(
                    name="tool_usage",
                    score=float(judge.tool_usage_score),
                    when=when,
                )
            )
        if judge.tool_efficiency_score is not None:
            events.append(
                self._evaluation_event(
                    name="tool_efficiency",
                    score=float(judge.tool_efficiency_score),
                    when=when,
                )
            )

        return events

    def _evaluation_event(
        self,
        name: str,
        score: float,
        when: datetime,
        label: Optional[str] = None,
        explanation: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "timeUnixNano": _unix_nano(when),
            "name": "gen_ai.evaluation.result",
            "attributes": _attrs(
                {
                    "gen_ai.evaluation.name": name,
                    "gen_ai.evaluation.score.value": score,
                    "gen_ai.evaluation.score.label": label,
                    "gen_ai.evaluation.explanation": explanation,
                }
            ),
        }

    def _label(self, score: float) -> str:
        return "pass" if score >= self.fail_under else "fail"

    def _push(self, payload: Dict[str, Any]) -> None:
        """POST the payload to the configured OTLP/HTTP endpoint.

        Push failures are logged and swallowed: the file on disk is the
        source of truth, and a down collector should not fail a CI run.
        """
        url = self.endpoint.rstrip("/")
        if not url.endswith(_TRACES_PATH):
            url = url + _TRACES_PATH

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                logger.info("Pushed OTLP traces to %s (HTTP %s)", url, response.status)
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("Failed to push OTLP traces to %s: %s", url, exc)


def _package_version() -> str:
    """Best-effort promptlens version for the resource attributes."""
    try:
        from promptlens import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - defensive
        return "unknown"
