"""CSV exporter for run results."""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from promptlens.exporters.base import BaseExporter
from promptlens.models.result import JudgeScore, RunResult

logger = logging.getLogger(__name__)


class CSVExporter(BaseExporter):
    """Exporter for CSV format.

    Flattens results into a tabular format suitable for spreadsheets.
    """

    def export(self, result: RunResult, output_path: str) -> None:
        """Export results to CSV file.

        Args:
            result: The run result to export
            output_path: Path to write the CSV file
        """
        path = self.ensure_output_dir(output_path)

        # Flatten results into rows
        rows = self._flatten_results(result)

        # Write CSV
        with open(path, "w", encoding="utf-8", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        logger.info(f"Exported results to {path}")

    def _flatten_results(self, result: RunResult) -> List[Dict[str, Any]]:
        """Flatten RunResult into flat dict rows.

        Args:
            result: The run result

        Returns:
            List of flat dictionaries
        """
        rows = []

        for eval_result in result.results:
            row = {
                "run_id": result.run_id,
                "run_name": result.run_name or "",
                "test_case_id": eval_result.test_case_id,
                "query": eval_result.query[:100],  # Truncate for CSV
                "expected_behavior": eval_result.expected_behavior[:100],
                "model": eval_result.model_response.model,
                "provider": eval_result.model_response.provider,
                "response": eval_result.model_response.content[:200],  # Truncate
                "score": eval_result.judge_score.score if eval_result.judge_score else None,
                "explanation": (
                    eval_result.judge_score.explanation[:200]
                    if eval_result.judge_score
                    else ""
                ),
                "latency_ms": eval_result.model_response.latency_ms,
                "cost_usd": eval_result.model_response.cost_usd or 0.0,
                "tokens_used": eval_result.model_response.tokens_used or 0,
                "error": eval_result.model_response.error or "",
            }
            row.update(self._stability_columns(eval_result.judge_score))
            rows.append(row)

        return rows

    @staticmethod
    def _stability_columns(judge_score: Optional[JudgeScore]) -> Dict[str, Any]:
        """Judge stability columns. Blank when the judge ran once."""
        if judge_score is None or not judge_score.is_sampled:
            return {
                "judge_samples": judge_score.sample_count if judge_score else None,
                "score_mean": "",
                "score_std": "",
                "score_min": "",
                "score_max": "",
                "score_samples": "",
                "judge_disagreement": "",
            }
        return {
            "judge_samples": judge_score.sample_count,
            "score_mean": f"{judge_score.score_mean:.3f}",
            "score_std": f"{judge_score.score_std:.3f}",
            "score_min": judge_score.score_min,
            "score_max": judge_score.score_max,
            "score_samples": "|".join(str(s) for s in judge_score.sample_scores),
            "judge_disagreement": judge_score.disagreement,
        }

    @property
    def file_extension(self) -> str:
        """Return the file extension.

        Returns:
            ".csv"
        """
        return ".csv"
