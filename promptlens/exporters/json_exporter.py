"""JSON exporter for run results."""

import json
import logging
from pathlib import Path

from promptlens.exporters.base import BaseExporter
from promptlens.models.result import RunResult
from promptlens.stability import compute_stability

logger = logging.getLogger(__name__)


class JSONExporter(BaseExporter):
    """Exporter for JSON format."""

    def export(self, result: RunResult, output_path: str) -> None:
        """Export results to JSON file.

        Args:
            result: The run result to export
            output_path: Path to write the JSON file
        """
        path = self.ensure_output_dir(output_path)

        # Convert to dict and serialize
        data = result.model_dump(mode="json")

        # Multi-sample runs carry their stability analysis alongside the raw
        # results so downstream tools do not have to recompute it. RunResult
        # ignores the extra key when the file is loaded back.
        if result.sample_count > 1:
            data["stability"] = compute_stability(result).model_dump(mode="json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Exported results to {path}")

    @property
    def file_extension(self) -> str:
        """Return the file extension.

        Returns:
            ".json"
        """
        return ".json"
