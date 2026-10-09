"""Markdown exporter for run results."""

import logging
from pathlib import Path
from typing import Optional

from promptlens.exporters.base import BaseExporter
from promptlens.models.result import RunResult
from promptlens.stability import compute_stability, format_scores

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
        lines.append(f"| Test Cases | {result.test_case_count} |")
        lines.append(f"| Models | {len(result.models_tested)} |")
        if result.sample_count > 1:
            lines.append(f"| Samples per Test Case | {result.sample_count} |")
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

        # Stability (multi-sample runs only)
        multi_sample = result.sample_count > 1
        if multi_sample:
            lines.extend(self._stability_section(result))

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

            # Results table
            if multi_sample:
                lines.append("| Model | Sample | Score | Latency | Cost | Response |")
                lines.append("|-------|--------|-------|---------|------|----------|")
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

                sample_col = f" {eval_result.sample_index + 1} |" if multi_sample else ""
                lines.append(
                    f"| {eval_result.model_response.model} |{sample_col} {score} | {latency} | {cost} | {response}... |"
                )

            lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by [PromptLens](https://github.com/sparker/promptlens)*")

        return "\n".join(lines)

    def _stability_section(self, result: RunResult) -> list:
        """Render the pass@k / pass^k / flaky-case section for a multi-sample run."""
        report = compute_stability(result)
        k = report.samples
        lines = []
        lines.append(f"## Stability ({k} samples per test case)")
        lines.append("")
        lines.append(
            f"A sample passes when its judge score is at least {report.pass_threshold:g}. "
            f"pass@{k} is the share of test cases that passed at least once (optimistic); "
            f"pass^{k} is the share that passed every time (what the model reliably does). "
            "Flaky cases both passed and failed."
        )
        lines.append("")
        lines.append(f"| Model | pass@{k} | pass^{k} | Flaky Cases | Mean Score (95% CI) | Mean Within-Case Std Dev |")
        lines.append("|-------|--------|--------|-------------|---------------------|--------------------------|")
        for model in report.models:
            if model.scored_cases == 0:
                lines.append(f"| {model.model} | - | - | - | no judge scores | - |")
                continue
            ci = (
                f"{model.mean_score:.2f} ({model.score_ci_low:.2f} to {model.score_ci_high:.2f})"
                if model.mean_score is not None and model.score_ci_low is not None
                else "-"
            )
            std = f"{model.mean_case_std_dev:.2f}" if model.mean_case_std_dev is not None else "-"
            lines.append(
                f"| {model.model} | {model.pass_at_k:.0%} ({model.pass_at_k_cases}/{model.scored_cases}) "
                f"| {model.pass_hat_k:.0%} ({model.pass_hat_k_cases}/{model.scored_cases}) "
                f"| {model.flaky_cases} | {ci} | {std} |"
            )
        lines.append("")

        flaky = report.flaky_cases
        if flaky:
            lines.append(f"### Flaky Cases ({len(flaky)})")
            lines.append("")
            lines.append("| Test Case | Model | Scores | Passed | Std Dev |")
            lines.append("|-----------|-------|--------|--------|---------|")
            for case in flaky:
                std = f"{case.std_dev:.2f}" if case.std_dev is not None else "-"
                lines.append(
                    f"| `{case.test_case_id}` | {case.model} | {format_scores(case.scores)} "
                    f"| {case.passed_samples}/{case.scored_samples} | {std} |"
                )
            lines.append("")
        else:
            lines.append("No flaky cases: every scored test case was either always passing or always failing.")
            lines.append("")
        return lines

    @property
    def file_extension(self) -> str:
        """Return the file extension.

        Returns:
            ".md"
        """
        return ".md"
