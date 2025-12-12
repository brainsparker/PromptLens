"""JSON exporter for run results."""

import json
import logging
from pathlib import Path

from promptlens.exporters.base import BaseExporter
from promptlens.models.result import RunResult

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
