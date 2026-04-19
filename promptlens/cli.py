"""Command-line interface for PromptLens."""

import asyncio
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from promptlens import __version__
from promptlens.exporters.csv_exporter import CSVExporter
from promptlens.exporters.html_exporter import HTMLExporter
from promptlens.exporters.json_exporter import JSONExporter
from promptlens.exporters.markdown_exporter import MarkdownExporter
from promptlens.models.config import RunConfig
from promptlens.runners.runner import Runner

# Load environment variables
load_dotenv()

console = Console()


def _remove_path_if_exists(path: Path) -> None:
    """Remove a file/symlink/directory path if it exists."""
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Set logging level",
)
def cli(log_level: str) -> None:
    """PromptLens - Lightweight LLM evaluation tool.

    Evaluate prompts, agents, and LLM workflows with ease.
    """
    setup_logging(log_level.upper())


@cli.command()
@click.argument("config", type=click.Path(exists=True))
@click.option(
    "--golden-set",
    type=click.Path(exists=True),
    help="Override golden set path from config",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    help="Override output directory from config",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate config without running evaluation",
)
def run(
    config: str,
    golden_set: Optional[str],
    output_dir: Optional[str],
    dry_run: bool,
) -> None:
    """Run evaluation with the given configuration file.

    CONFIG: Path to YAML configuration file

    Examples:
        promptlens run config.yaml
        promptlens run config.yaml --output-dir ./results
        promptlens run config.yaml --dry-run
    """
    try:
        # Load config
        console.print(f"\n[cyan]Loading configuration from {config}...[/cyan]")
        with open(config, "r") as f:
            config_data = yaml.safe_load(f)

        # Override with CLI options
        if golden_set:
            config_data["golden_set"] = golden_set
        if output_dir:
            config_data.setdefault("output", {})
            config_data["output"]["directory"] = output_dir

        # Parse config
        try:
            run_config = RunConfig(**config_data)
        except Exception as e:
            console.print(f"[red]Invalid configuration: {e}[/red]")
            sys.exit(1)

        console.print("[green]✓[/green] Configuration loaded successfully")

        if dry_run:
            console.print("\n[yellow]Dry run mode - configuration is valid[/yellow]")
            console.print(f"  Golden set: {run_config.golden_set}")
            console.print(f"  Models: {len(run_config.models)}")
            console.print(f"  Output: {run_config.output.directory}")
            return

        # Run evaluation
        console.print(f"\n[bold cyan]Starting evaluation...[/bold cyan]\n")
        runner = Runner(run_config)
        result = asyncio.run(runner.run())

        # Export results
        console.print(f"\n[yellow]Exporting results...[/yellow]")
        output_dir_path = Path(run_config.output.directory)
        run_output_dir = output_dir_path / result.run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)

        exporters = {
            "json": (JSONExporter(), "results.json"),
            "csv": (CSVExporter(), "results.csv"),
            "md": (MarkdownExporter(), "results.md"),
            "html": (HTMLExporter(), "report.html"),
        }

        exported_files = []
        for format_name in run_config.output.formats:
            if format_name in exporters:
                exporter, filename = exporters[format_name]
                output_path = run_output_dir / filename
                exporter.export(result, str(output_path))
                exported_files.append(str(output_path))
                console.print(f"[green]✓[/green] Exported {format_name.upper()}: {output_path}")

        # Create symlink to latest
        latest_link = output_dir_path / "latest"
        _remove_path_if_exists(latest_link)
        try:
            latest_link.symlink_to(result.run_id)
        except OSError:
            # Windows may not support symlinks
            pass

        console.print(f"\n[bold green]✓ Evaluation complete![/bold green]")
        console.print(f"Results saved to: {run_output_dir}")

        if "html" in run_config.output.formats:
            html_path = run_output_dir / "report.html"
            console.print(f"\n[cyan]View report: file://{html_path.absolute()}[/cyan]")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logging.exception("Evaluation failed")
        sys.exit(1)


@cli.command()
@click.argument("golden_set", type=click.Path(exists=True))
def validate(golden_set: str) -> None:
    """Validate a golden set file.

    GOLDEN_SET: Path to JSON or YAML golden set file

    Examples:
        promptlens validate tests.yaml
        promptlens validate tests.json
    """
    try:
        from promptlens.loaders.yaml_loader import get_loader

        console.print(f"\n[cyan]Validating {golden_set}...[/cyan]")

        loader = get_loader(golden_set)
        golden_set_obj = loader.load(golden_set)

        console.print(f"[green]✓[/green] Golden set is valid!")
        console.print(f"\n  Name: {golden_set_obj.name}")
        console.print(f"  Version: {golden_set_obj.version}")
        console.print(f"  Test Cases: {len(golden_set_obj.test_cases)}")

        if golden_set_obj.description:
            console.print(f"  Description: {golden_set_obj.description}")

        # Show test case IDs
        console.print(f"\n  Test Case IDs:")
        for tc in golden_set_obj.test_cases:
            console.print(f"    - {tc.id}")

    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./promptlens_results",
    help="Results directory to list",
)
def list_runs(output_dir: str) -> None:
    """List all evaluation runs.

    Examples:
        promptlens list-runs
        promptlens list-runs --output-dir ./results
    """
    try:
        output_path = Path(output_dir)
        if not output_path.exists():
            console.print(f"[yellow]No results found in {output_dir}[/yellow]")
            return

        # Find all run directories
        runs = [d for d in output_path.iterdir() if d.is_dir() and d.name != "latest"]

        if not runs:
            console.print(f"[yellow]No runs found in {output_dir}[/yellow]")
            return

        console.print(f"\n[cyan]Evaluation Runs in {output_dir}:[/cyan]\n")

        for run_dir in sorted(runs, key=lambda x: x.stat().st_mtime, reverse=True):
            # Try to load results.json
            json_path = run_dir / "results.json"
            if json_path.exists():
                import json

                with open(json_path) as f:
                    data = json.load(f)
                    run_name = data.get("run_name", "Unnamed")
                    timestamp = data.get("timestamp", "Unknown")
                    console.print(f"  [green]{run_dir.name}[/green]")
                    console.print(f"    Name: {run_name}")
                    console.print(f"    Time: {timestamp}")
            else:
                console.print(f"  [green]{run_dir.name}[/green]")

            console.print()

    except Exception as e:
        console.print(f"[red]Error listing runs: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("run_id")
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["json", "csv", "md", "html"], case_sensitive=False),
    required=True,
    help="Export format",
)
@click.option(
    "--output",
    type=click.Path(),
    help="Output file path (default: auto-generated)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./promptlens_results",
    help="Results directory",
)
def export(run_id: str, export_format: str, output: Optional[str], output_dir: str) -> None:
    """Export a run's results to a different format.

    RUN_ID: Run identifier

    Examples:
        promptlens export abc123 --format json
        promptlens export abc123 --format html --output report.html
    """
    try:
        import json

        # Load run results
        run_path = Path(output_dir) / run_id
        results_file = run_path / "results.json"

        if not results_file.exists():
            console.print(f"[red]Run {run_id} not found in {output_dir}[/red]")
            sys.exit(1)

        console.print(f"\n[cyan]Loading run {run_id}...[/cyan]")
        with open(results_file) as f:
            data = json.load(f)

        from promptlens.models.result import RunResult

        result = RunResult(**data)

        # Determine output path
        if not output:
            extensions = {"json": ".json", "csv": ".csv", "md": ".md", "html": ".html"}
            output = f"export_{run_id}{extensions[export_format]}"

        # Export
        exporters = {
            "json": JSONExporter(),
            "csv": CSVExporter(),
            "md": MarkdownExporter(),
            "html": HTMLExporter(),
        }

        exporter = exporters[export_format]
        exporter.export(result, output)

        console.print(f"[green]✓[/green] Exported to {output}")

    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
