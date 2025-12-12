"""HTML exporter for run results."""

import json
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from promptlens.exporters.base import BaseExporter
from promptlens.models.result import RunResult

logger = logging.getLogger(__name__)


class HTMLExporter(BaseExporter):
    """Exporter for HTML format.

    Creates an interactive HTML report with charts and comparisons.
    """

    def __init__(self) -> None:
        """Initialize the HTML exporter."""
        # Set up Jinja2 environment
        template_dir = Path(__file__).parent.parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def export(self, result: RunResult, output_path: str) -> None:
        """Export results to HTML file.

        Args:
            result: The run result to export
            output_path: Path to write the HTML file
        """
        path = self.ensure_output_dir(output_path)

        # Prepare data for template
        template_data = self._prepare_template_data(result)

        # Render template
        template = self.env.get_template("report.html")
        html = template.render(**template_data)

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Exported HTML report to {path}")

    def _prepare_template_data(self, result: RunResult) -> dict:
        """Prepare data for the HTML template.

        Args:
            result: The run result

        Returns:
            Dictionary of template variables
        """
        # Calculate per-model stats
        model_stats = []
        for model in result.models_tested:
            avg_score = result.get_average_score(model)
            total_cost = result.get_total_cost(model)
            total_latency = result.get_total_latency(model)

            # Get results for this model
            model_results = [
                r for r in result.results if r.model_response.model == model
            ]

            # Score distribution
            scores = [r.judge_score.score for r in model_results if r.judge_score]
            score_dist = {i: scores.count(i) for i in range(1, 6)}

            model_stats.append({
                "name": model,
                "average_score": avg_score,
                "total_cost": total_cost,
                "total_latency": total_latency,
                "score_distribution": score_dist,
                "result_count": len(model_results),
            })

        # Group results by test case
        test_case_results = {}
        for eval_result in result.results:
            if eval_result.test_case_id not in test_case_results:
                test_case_results[eval_result.test_case_id] = {
                    "id": eval_result.test_case_id,
                    "query": eval_result.query,
                    "expected_behavior": eval_result.expected_behavior,
                    "model_results": [],
                }
            test_case_results[eval_result.test_case_id]["model_results"].append(
                {
                    "model": eval_result.model_response.model,
                    "provider": eval_result.model_response.provider,
                    "response": eval_result.model_response.content,
                    "score": eval_result.judge_score.score if eval_result.judge_score else None,
                    "explanation": (
                        eval_result.judge_score.explanation
                        if eval_result.judge_score
                        else "No score available"
                    ),
                    "latency_ms": eval_result.model_response.latency_ms,
                    "cost_usd": eval_result.model_response.cost_usd or 0.0,
                    "error": eval_result.model_response.error,
                }
            )

        return {
            "run_id": result.run_id,
            "run_name": result.run_name or "Unnamed Run",
            "timestamp": result.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "golden_set_name": result.golden_set_name,
            "total_cost": result.total_cost_usd,
            "total_time": result.total_time_ms,
            "test_case_count": len(test_case_results),
            "model_count": len(result.models_tested),
            "model_stats": model_stats,
            "test_cases": list(test_case_results.values()),
            "models": result.models_tested,
        }

    @property
    def file_extension(self) -> str:
        """Return the file extension.

        Returns:
            ".html"
        """
        return ".html"
