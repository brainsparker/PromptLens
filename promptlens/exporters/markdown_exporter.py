"""Markdown exporter for run results."""

import logging
from pathlib import Path
from typing import Optional

from promptlens.exporters.base import BaseExporter
from promptlens.models.result import RunResult

logger = logging.getLogger(__name__)


class MarkdownExporter(BaseExporter):
    """Exporter for Markdown format.

    Creates a formatted markdown report suitable for GitHub, Notion, etc.
    """

    def export(self, result: RunResult, output_path: str) -> None:
        """Export results to Markdown file.

        Args:
            result: The run result to export
            output_path: Path to write the markdown file
        """
        path = self.ensure_output_dir(output_path)

        # Generate markdown
        markdown = self._generate_markdown(result)

        with open(path, "w", encoding="utf-8") as f:
            f.write(markdown)

        logger.info(f"Exported results to {path}")

    def _generate_markdown(self, result: RunResult) -> str:
        """Generate markdown content from results.

        Args:
            result: The run result

        Returns:
            Markdown string
        """
        lines = []

        # Header
        lines.append(f"# {result.run_name or 'PromptLens Evaluation'}")
        lines.append("")
        lines.append(f"**Run ID:** `{result.run_id}`  ")
        lines.append(f"**Timestamp:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC  ")
        lines.append(f"**Golden Set:** {result.golden_set_name}  ")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Cost | ${result.total_cost_usd:.4f} |")
        lines.append(f"| Total Time | {result.total_time_ms:.0f}ms |")
        lines.append(
            f"| Test Cases | {len(result.results) // len(result.models_tested)} |"
        )
        lines.append(f"| Models | {len(result.models_tested)} |")
        lines.append("")

        # Per-model results
        lines.append("## Model Results")
        lines.append("")

        for model in result.models_tested:
            lines.append(f"### {model}")
            lines.append("")

            avg_score = result.get_average_score(model)
            total_cost = result.get_total_cost(model)
            total_latency = result.get_total_latency(model)

            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            if avg_score is not None:
                lines.append(f"| Average Score | {avg_score:.2f}/5.0 |")
            lines.append(f"| Total Cost | ${total_cost:.4f} |")
            lines.append(f"| Total Time | {total_latency:.0f}ms |")
            lines.append("")

        # Detailed results
        lines.append("## Detailed Results")
        lines.append("")

        # Group by test case
        test_cases = {}
        for eval_result in result.results:
            if eval_result.test_case_id not in test_cases:
                test_cases[eval_result.test_case_id] = []
            test_cases[eval_result.test_case_id].append(eval_result)

        for test_case_id, evals in test_cases.items():
            lines.append(f"### Test Case: `{test_case_id}`")
            lines.append("")
            lines.append(f"**Query:** {evals[0].query}")
            lines.append("")
            lines.append(f"**Expected:** {evals[0].expected_behavior}")
            lines.append("")

            has_trajectory = any(e.trajectory_result is not None for e in evals)

            # Results table
            if has_trajectory:
                lines.append("| Model | Score | Trajectory | Latency | Cost | Response |")
                lines.append("|-------|-------|------------|---------|------|----------|")
            else:
                lines.append("| Model | Score | Latency | Cost | Response |")
                lines.append("|-------|-------|---------|------|----------|")

            for eval_result in evals:
                score = (
                    f"{eval_result.judge_score.score}/5"
                    if eval_result.judge_score
                    else "N/A"
                )
                latency = f"{eval_result.model_response.latency_ms:.0f}ms"
                cost = f"${eval_result.model_response.cost_usd:.4f}" if eval_result.model_response.cost_usd else "$0.00"
                response = eval_result.model_response.content[:100].replace("\n", " ")
                if eval_result.model_response.error:
                    response = f"ERROR: {eval_result.model_response.error}"

                if has_trajectory:
                    if eval_result.trajectory_result is None:
                        trajectory = "N/A"
                    elif eval_result.trajectory_result.passed:
                        trajectory = "✅ pass"
                    else:
                        trajectory = (
                            f"❌ {len(eval_result.trajectory_result.failed_checks)} failed"
                        )
                    lines.append(
                        f"| {eval_result.model_response.model} | {score} | {trajectory} | {latency} | {cost} | {response}... |"
                    )
                else:
                    lines.append(
                        f"| {eval_result.model_response.model} | {score} | {latency} | {cost} | {response}... |"
                    )

            # Failed assertion details
            for eval_result in evals:
                tr = eval_result.trajectory_result
                if tr is not None and not tr.passed:
                    lines.append("")
                    lines.append(
                        f"**Failed trajectory assertions "
                        f"({eval_result.model_response.model}):**"
                    )
                    lines.append("")
                    observed = " -> ".join(tr.observed_calls) or "(none)"
                    lines.append(f"- Observed tool calls: `{observed}`")
                    for check in tr.failed_checks:
                        lines.append(f"- {check.detail}")

            lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by [PromptLens](https://github.com/sparker/promptlens)*")

        return "\n".join(lines)

    @property
    def file_extension(self) -> str:
        """Return the file extension.

        Returns:
            ".md"
        """
        return ".md"
