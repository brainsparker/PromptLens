"""Markdown exporter for run results."""

import logging
from pathlib import Path
from typing import Optional

from promptlens.exporters.base import BaseExporter
from promptlens.judges.stability import summarize_stability
from promptlens.models.result import JudgeScore, RunResult

logger = logging.getLogger(__name__)


def _format_score(judge_score: Optional[JudgeScore]) -> str:
    """Render a score cell, adding the sample spread for sampled judgements."""
    if judge_score is None:
        return "N/A"
    if not judge_score.is_sampled:
        return f"{judge_score.score}/5"
    spread = f"{judge_score.score_min}-{judge_score.score_max}"
    marker = " (disagreement)" if judge_score.disagreement else ""
    return f"{judge_score.score}/5 (samples {spread}){marker}"


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

        # Judge stability (sampled runs only)
        stability = summarize_stability(result)
        if stability.sampled:
            lines.append("## Judge Stability")
            lines.append("")
            lines.append(
                f"Each response was judged {stability.samples_per_response} times. "
                "The score column below shows the median verdict and the sample spread."
            )
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Judge Samples per Response | {stability.samples_per_response} |")
            if stability.mean_std is not None:
                lines.append(f"| Mean Score Std Dev | {stability.mean_std:.2f} |")
            rate = stability.disagreement_rate
            rate_text = f" ({rate:.0%})" if rate is not None else ""
            lines.append(
                f"| Disagreements | {stability.disagreements}/{stability.judged_results}{rate_text} |"
            )
            lines.append("")
            if stability.unstable_cases:
                lines.append("| Test Case | Model | Median | Spread | Std Dev |")
                lines.append("|-----------|-------|--------|--------|---------|")
                for case in stability.unstable_cases:
                    lines.append(
                        f"| `{case.test_case_id}` | {case.model} | {case.score}/5 | "
                        f"{case.spread_label} | {case.score_std:.2f} |"
                    )
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

            # Results table
            lines.append("| Model | Score | Latency | Cost | Response |")
            lines.append("|-------|-------|---------|------|----------|")

            for eval_result in evals:
                score = _format_score(eval_result.judge_score)
                latency = f"{eval_result.model_response.latency_ms:.0f}ms"
                cost = f"${eval_result.model_response.cost_usd:.4f}" if eval_result.model_response.cost_usd else "$0.00"
                response = eval_result.model_response.content[:100].replace("\n", " ")
                if eval_result.model_response.error:
                    response = f"ERROR: {eval_result.model_response.error}"

                lines.append(
                    f"| {eval_result.model_response.model} | {score} | {latency} | {cost} | {response}... |"
                )

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
