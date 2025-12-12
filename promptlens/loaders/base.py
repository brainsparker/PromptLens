"""Base loader interface for golden sets."""

from abc import ABC, abstractmethod
from pathlib import Path

from promptlens.models.test_case import GoldenSet


class BaseLoader(ABC):
    """Abstract base class for golden set loaders.

    All loader implementations must inherit from this class and implement
    the abstract methods.
    """

    @abstractmethod
    def load(self, path: str) -> GoldenSet:
        """Load a golden set from a file.

        Args:
            path: Path to the golden set file

        Returns:
            Parsed GoldenSet object

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        pass

    def validate_path(self, path: str) -> Path:
        """Validate that the file path exists.

        Args:
            path: Path to validate

        Returns:
            Validated Path object

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return file_path
