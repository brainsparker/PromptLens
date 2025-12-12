"""YAML loader for golden sets."""

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from promptlens.loaders.base import BaseLoader
from promptlens.models.test_case import GoldenSet

logger = logging.getLogger(__name__)


class YAMLLoader(BaseLoader):
    """Loader for YAML golden sets."""

    def load(self, path: str) -> GoldenSet:
        """Load a golden set from a YAML file.

        Args:
            path: Path to the YAML file

        Returns:
            Parsed GoldenSet object

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the YAML is invalid or doesn't match the schema
        """
        file_path = self.validate_path(path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")

        if data is None:
            raise ValueError(f"Empty YAML file: {path}")

        try:
            golden_set = GoldenSet(**data)
            logger.info(
                f"Loaded golden set '{golden_set.name}' with "
                f"{len(golden_set.test_cases)} test cases from {path}"
            )
            return golden_set
        except ValidationError as e:
            raise ValueError(f"Invalid golden set format in {path}: {e}")


def get_loader(file_path: str) -> BaseLoader:
    """Get the appropriate loader based on file extension.

    Args:
        file_path: Path to the file

    Returns:
        Appropriate loader instance

    Raises:
        ValueError: If file extension is not supported
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".json":
        from promptlens.loaders.json_loader import JSONLoader
        return JSONLoader()
    elif extension in [".yaml", ".yml"]:
        return YAMLLoader()
    else:
        raise ValueError(
            f"Unsupported file extension: {extension}. "
            "Supported extensions: .json, .yaml, .yml"
        )
