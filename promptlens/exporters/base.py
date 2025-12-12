"""Base exporter interface."""

from abc import ABC, abstractmethod
from pathlib import Path

from promptlens.models.result import RunResult


class BaseExporter(ABC):
    """Abstract base class for exporters.

    All exporter implementations must inherit from this class and implement
    the abstract methods.
    """

    @abstractmethod
    def export(self, result: RunResult, output_path: str) -> None:
        """Export results to a file.

        Args:
            result: The run result to export
            output_path: Path to write the exported file

        Raises:
            IOError: If writing fails
        """
        pass

    def ensure_output_dir(self, output_path: str) -> Path:
        """Ensure the output directory exists.

        Args:
            output_path: Path to the output file

        Returns:
            Path object for the output file
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Return the file extension for this exporter.

        Returns:
            File extension (e.g., ".html", ".json")
        """
        pass
