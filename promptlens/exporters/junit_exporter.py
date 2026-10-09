"""JUnit XML exporter for run results.

Emits a JUnit-style XML report so evaluation runs plug directly into CI
systems (GitHub Actions test summaries, GitLab, Jenkins, CircleCI) without
any custom glue. One test suite is emitted per model, and one test case per
evaluated golden-set entry.

Mapping rules:
    - A test case whose model response errored is reported as an <error>.
    - A test case whose judge score is below the failure threshold is
      reported as a <failure>.
    - A test case that was never judged (judging disabled or judge failed)
      is reported as <skipped>, so CI does not report a false pass.
    - Everything else is a pass.

Multi-sample runs (``--samples N``) still emit one <testcase> per golden-set
entry. The entry passes only when every scored sample passed (pass^k), so a
flaky case shows up as a failure with type FlakyJudgeScore and the full list
of per-sample scores. A case errors only when every sample errored; errored
samples alongside scored ones are listed in system-out and left out of the
pass/fail decision, so infrastructure noise is never reported as quality.
"""

import logging
import xml.etree.ElementTree as ET
from typing import List, Optional

from promptlens.exporters.base import BaseExporter
from promptlens.models.result import EvaluationResult, RunResult
from promptlens.stability import DEFAULT_PASS_THRESHOLD, analyze_case, group_samples

logger = logging.getLogger(__name__)

# Judge scores are on a 1-5 scale. Scores below this value count as failures
# unless a different threshold is provided. Shared with the stability report
# and the --fail-on-flaky gate so every surface draws the same pass line.
DEFAULT_FAIL_UNDER = DEFAULT_PASS_THRESHOLD


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

        multi_sample = result.sample_count > 1
        for model in result.models_tested:
            model_results = [
                r for r in result.results if r.model_response.model == model
            ]
            if multi_sample:
                suite, stats = self._build_multi_sample_suite(result, model, model_results)
            else:
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

            if response_error:
                errors += 1
                error_el = ET.SubElement(testcase, "error")
                error_el.set("message", _truncate(response_error, 300))
                error_el.set("type", "ModelResponseError")
                error_el.text = response_error
            elif judge_score is None:
                skipped += 1
                skipped_el = ET.SubElement(testcase, "skipped")
                skipped_el.set(
                    "message",
                    "No judge score available (judging disabled or judge failed)",
                )
            elif judge_score.score < self.fail_under:
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
            system_out.text = "\n".join(out_lines)

        self._finish_suite(
            suite, result, model, provider, len(model_results), failures, errors, skipped, suite_time
        )

        stats = {
            "tests": len(model_results),
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "time": suite_time,
        }
        return suite, stats

    def _build_multi_sample_suite(
        self,
        result: RunResult,
        model: str,
        model_results: List[EvaluationResult],
    ) -> tuple:
        """Build a <testsuite> for one model of a multi-sample run.

        One <testcase> per golden-set entry, judged on pass^k: the entry
        passes only when every scored sample is at or above the threshold.

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
        tests = 0

        groups = group_samples(result)
        for (case_id, group_model), samples in groups.items():
            if group_model != model:
                continue
            tests += 1
            stability = analyze_case(case_id, model, samples, self.fail_under)
            first = samples[0]
            provider = first.model_response.provider
            case_time = sum((r.model_response.latency_ms or 0.0) for r in samples) / 1000.0
            suite_time += case_time

            testcase = ET.SubElement(suite, "testcase")
            testcase.set("name", case_id)
            testcase.set("classname", f"{result.golden_set_name}.{model}")
            testcase.set("time", f"{case_time:.3f}")

            errored = [r for r in samples if r.model_response.error]
            scores_text = ", ".join(str(score) for score in stability.scores) or "-"

            if len(errored) == len(samples):
                errors += 1
                messages = "; ".join(
                    _truncate(r.model_response.error or "", 120) for r in errored
                )
                error_el = ET.SubElement(testcase, "error")
                error_el.set("message", _truncate(f"All {len(samples)} samples errored: {messages}", 300))
                error_el.set("type", "ModelResponseError")
                error_el.text = "\n".join(
                    f"sample {r.sample_index + 1}: {r.model_response.error}" for r in errored
                )
            elif stability.scored_samples == 0:
                skipped += 1
                skipped_el = ET.SubElement(testcase, "skipped")
                skipped_el.set(
                    "message",
                    "No judge score available in any sample (judging disabled or judge failed)",
                )
            elif not stability.pass_hat_k:
                failures += 1
                failed = stability.scored_samples - stability.passed_samples
                failure_el = ET.SubElement(testcase, "failure")
                if stability.flaky:
                    failure_el.set("type", "FlakyJudgeScore")
                    failure_el.set(
                        "message",
                        f"Flaky: {failed} of {stability.scored_samples} samples scored below "
                        f"threshold {self.fail_under:g} (scores: {scores_text})",
                    )
                else:
                    failure_el.set("type", "JudgeScoreBelowThreshold")
                    failure_el.set(
                        "message",
                        f"All {stability.scored_samples} samples scored below "
                        f"threshold {self.fail_under:g} (scores: {scores_text})",
                    )
                explanations = "\n".join(
                    f"sample {r.sample_index + 1} (score {r.judge_score.score}): "
                    f"{_truncate(r.judge_score.explanation, 400)}"
                    for r in samples
                    if r.judge_score is not None
                )
                failure_el.text = (
                    f"Query: {_truncate(first.query, 500)}\n"
                    f"Expected: {_truncate(first.expected_behavior, 500)}\n"
                    f"Scores: {scores_text}\n"
                    f"Passed: {stability.passed_samples}/{stability.scored_samples}\n"
                    f"{explanations}"
                )

            system_out = ET.SubElement(testcase, "system-out")
            out_lines = [
                f"provider: {provider}",
                f"samples: {len(samples)}",
                f"scored_samples: {stability.scored_samples}",
                f"errored_samples: {stability.errored_samples}",
                f"judge_scores: {scores_text}",
                f"passed_samples: {stability.passed_samples}",
                f"pass_at_k: {str(stability.pass_at_k).lower()}",
                f"pass_hat_k: {str(stability.pass_hat_k).lower()}",
                f"flaky: {str(stability.flaky).lower()}",
                f"total_latency_ms: {sum((r.model_response.latency_ms or 0.0) for r in samples)}",
                f"total_cost_usd: {sum((r.model_response.cost_usd or 0.0) for r in samples)}",
            ]
            if stability.mean_score is not None:
                out_lines.append(f"mean_judge_score: {stability.mean_score:.2f}")
            if stability.std_dev is not None:
                out_lines.append(f"judge_score_std_dev: {stability.std_dev:.2f}")
            system_out.text = "\n".join(out_lines)

        self._finish_suite(
            suite, result, model, provider, tests, failures, errors, skipped, suite_time
        )
        _add_property(suite.find("properties"), "samples", str(result.sample_count))

        stats = {
            "tests": tests,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "time": suite_time,
        }
        return suite, stats

    def _finish_suite(
        self,
        suite: ET.Element,
        result: RunResult,
        model: str,
        provider: Optional[str],
        tests: int,
        failures: int,
        errors: int,
        skipped: int,
        suite_time: float,
    ) -> None:
        """Set suite counters and insert the run-level <properties> block."""
        suite.set("tests", str(tests))
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
