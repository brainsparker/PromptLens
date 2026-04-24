"""JSON loader for golden sets."""

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from promptlens.loaders.base import BaseLoader
from promptlens.models.test_case import GoldenSet

logger = logging.getLogger(__name__)


class JSONLoader(BaseLoader):
    """Loader for JSON golden sets."""

    def load(self, path: str) -> GoldenSet:
        """Load a golden set from a JSON file.

        Args:
            path: Path to the JSON file

        Returns:
            Parsed GoldenSet object

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the JSON is invalid or doesn't match the schema
        """
        file_path = self.validate_path(path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}")

        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid golden set format in {path}: root value must be a JSON object"
            )

        try:
            golden_set = GoldenSet(**data)
            logger.info(
                f"Loaded golden set '{golden_set.name}' with "
                f"{len(golden_set.test_cases)} test cases from {path}"
            )
            return golden_set
        except ValidationError as e:
            raise ValueError(f"Invalid golden set format in {path}: {e}")
