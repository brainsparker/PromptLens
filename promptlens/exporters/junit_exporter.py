"""JUnit XML exporter for run results.

Emits a JUnit-style XML report so evaluation runs plug directly into CI
systems (GitHub Actions test summaries, GitLab, Jenkins, CircleCI) without
any custom glue. One test suite is emitted per model, and one test case per
evaluated golden-set entry.

Mapping rules:
    - A test case whose model response errored is reported as an <error>.
    - A test case with any failed deterministic check is reported as a
      <failure> (checks take precedence over the judge score, since they
      are reproducible hard expectations).
    - A test case whose judge score is below the failure threshold is
      reported as a <failure>.
    - A test case with neither a judge score nor deterministic checks is
      reported as <skipped>, so CI does not report a false pass. A case
      whose checks all passed counts as a pass even without a judge score,
      because a deterministic verdict exists.
    - Everything else is a pass.
"""

import logging
import xml.etree.ElementTree as ET
from typing import List, Optional

from promptlens.exporters.base import BaseExporter
from promptlens.models.result import EvaluationResult, RunResult

logger = logging.getLogger(__name__)

# Judge scores are on a 1-5 scale. Scores below this value count as failures
# unless a different threshold is provided.
DEFAULT_FAIL_UNDER = 3.0


class JUnitXMLExporter(BaseExporter):
    """Exporter for JUnit XML format.

    Produces a <testsuites> document with one <testsuite> per model tested,
    suitable for CI test-report ingestion.

    Args:
        fail_under: Judge score threshold (1-5 scale). Test cases scoring
            strictly below this value are marked as failures. Defaults to
            DEFAULT_FAIL_UNDER.
    """

    def __init__(self, fail_under: Optional[float] = None) -> None:
        self.fail_under = DEFAULT_FAIL_UNDER if fail_under is None else float(fail_under)

    def export(self, result: RunResult, output_path: str) -> None:
        """Export results to a JUnit XML file.

        Args:
            result: The run result to export
            output_path: Path to write the XML file
        """
        path = self.ensure_output_dir(output_path)

        testsuites = ET.Element("testsuites")
        testsuites.set("name", result.run_name or result.golden_set_name)
        testsuites.set("timestamp", result.timestamp.isoformat())

        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0
        total_time = 0.0

        for model in result.models_tested:
            model_results = [
                r for r in result.results if r.model_response.model == model
            ]
            suite, stats = self._build_suite(result, model, model_results)
            testsuites.append(suite)
            total_tests += stats["tests"]
            total_failures += stats["failures"]
            total_errors += stats["errors"]
            total_skipped += stats["skipped"]
            total_time += stats["time"]

        testsuites.set("tests", str(total_tests))
        testsuites.set("failures", str(total_failures))
        testsuites.set("errors", str(total_errors))
        testsuites.set("skipped", str(total_skipped))
        testsuites.set("time", f"{total_time:.3f}")

        tree = ET.ElementTree(testsuites)
        try:
            ET.indent(tree, space="  ")  # Python 3.9+
        except AttributeError:  # pragma: no cover
            pass
        tree.write(str(path), encoding="utf-8", xml_declaration=True)

        logger.info(f"Exported results to {path}")

    def _build_suite(
        self,
        result: RunResult,
        model: str,
        model_results: List[EvaluationResult],
    ) -> tuple:
        """Build a <testsuite> element for one model.

        Args:
            result: The full run result (for run-level metadata)
            model: Model identifier for this suite
            model_results: Evaluation results belonging to this model

        Returns:
            Tuple of (testsuite Element, stats dict)
        """
        suite = ET.Element("testsuite")
        suite.set("name", model)

        failures = 0
        errors = 0
        skipped = 0
        suite_time = 0.0

        provider = None
        for eval_result in model_results:
            provider = eval_result.model_response.provider
            case_time = (eval_result.model_response.latency_ms or 0.0) / 1000.0
            suite_time += case_time

            testcase = ET.SubElement(suite, "testcase")
            testcase.set("name", eval_result.test_case_id)
            testcase.set("classname", f"{result.golden_set_name}.{model}")
            testcase.set("time", f"{case_time:.3f}")

            response_error = eval_result.model_response.error
            judge_score = eval_result.judge_score
            failed_checks = eval_result.failed_checks
            has_checks = bool(eval_result.check_results)

            if response_error:
                errors += 1
                error_el = ET.SubElement(testcase, "error")
                error_el.set("message", _truncate(response_error, 300))
                error_el.set("type", "ModelResponseError")
                error_el.text = response_error
            elif failed_checks:
                failures += 1
                failure_el = ET.SubElement(testcase, "failure")
                failure_el.set(
                    "message",
                    f"{len(failed_checks)} deterministic check(s) failed",
                )
                failure_el.set("type", "CheckFailure")
                failure_lines = [
                    f"Query: {_truncate(eval_result.query, 500)}",
                ]
                for check_result in failed_checks:
                    failure_lines.append(
                        f"FAILED {check_result.description}: "
                        f"{_truncate(check_result.detail, 500)}"
                    )
                failure_el.text = "\n".join(failure_lines)
            elif judge_score is None and not has_checks:
                skipped += 1
                skipped_el = ET.SubElement(testcase, "skipped")
                skipped_el.set(
                    "message",
                    "No judge score available (judging disabled or judge failed)",
                )
            elif judge_score is not None and judge_score.score < self.fail_under:
                failures += 1
                failure_el = ET.SubElement(testcase, "failure")
                failure_el.set(
                    "message",
                    f"Judge score {judge_score.score} is below "
                    f"threshold {self.fail_under:g}",
                )
                failure_el.set("type", "JudgeScoreBelowThreshold")
                failure_el.text = (
                    f"Query: {_truncate(eval_result.query, 500)}\n"
                    f"Expected: {_truncate(eval_result.expected_behavior, 500)}\n"
                    f"Score: {judge_score.score}\n"
                    f"Explanation: {_truncate(judge_score.explanation, 1000)}"
                )

            system_out = ET.SubElement(testcase, "system-out")
            out_lines = [
                f"provider: {eval_result.model_response.provider}",
                f"latency_ms: {eval_result.model_response.latency_ms}",
                f"cost_usd: {eval_result.model_response.cost_usd or 0.0}",
                f"tokens_used: {eval_result.model_response.tokens_used or 0}",
            ]
            if judge_score is not None:
                out_lines.append(f"judge_score: {judge_score.score}")
                out_lines.append(
                    f"judge_explanation: {_truncate(judge_score.explanation, 500)}"
                )
            if has_checks:
                passed_count = len(eval_result.check_results) - len(failed_checks)
                out_lines.append(
                    f"checks: {passed_count}/{len(eval_result.check_results)} passed"
                )
            system_out.text = "\n".join(out_lines)

        suite.set("tests", str(len(model_results)))
        suite.set("failures", str(failures))
        suite.set("errors", str(errors))
        suite.set("skipped", str(skipped))
        suite.set("time", f"{suite_time:.3f}")

        properties = ET.Element("properties")
        _add_property(properties, "model", model)
        if provider:
            _add_property(properties, "provider", provider)
        _add_property(properties, "run_id", result.run_id)
        _add_property(properties, "golden_set", result.golden_set_name)
        avg_score = result.get_average_score(model)
        if avg_score is not None:
            _add_property(properties, "average_judge_score", f"{avg_score:.2f}")
        _add_property(
            properties, "total_cost_usd", f"{result.get_total_cost(model):.6f}"
        )
        _add_property(properties, "fail_under", f"{self.fail_under:g}")
        suite.insert(0, properties)

        stats = {
            "tests": len(model_results),
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "time": suite_time,
        }
        return suite, stats

    @property
    def file_extension(self) -> str:
        """Return the file extension.

        Returns:
            The .xml extension
        """
        return ".xml"


def _add_property(parent: ET.Element, name: str, value: str) -> None:
    """Append a <property> element to a <properties> parent."""
    prop = ET.SubElement(parent, "property")
    prop.set("name", name)
    prop.set("value", value)


def _truncate(text: str, limit: int) -> str:
    """Truncate text to a character limit, marking the cut."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
