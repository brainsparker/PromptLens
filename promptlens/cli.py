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
from promptlens.exporters.junit_exporter import JUnitXMLExporter
from promptlens.exporters.markdown_exporter import MarkdownExporter
from promptlens.models.config import RunConfig
from promptlens.models.result import RunResult
from promptlens.runners.runner import Runner


def _load_config_data(config_path: str) -> dict:
    """Load and validate top-level config structure from YAML."""
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    if config_data is None:
        raise ValueError("Configuration file is empty")
    if not isinstance(config_data, dict):
        raise ValueError(
            f"Configuration must be a YAML object at top level, got {type(config_data).__name__}"
        )

    return config_data

# Load environment variables
load_dotenv()

console = Console()


def _remove_path_if_exists(path: Path) -> None:
    """Remove a file/symlink/directory path if it exists."""
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _load_run(output_dir: str, run_id: str) -> "RunResult":
    """Load a saved run's results.json as a RunResult.

    Args:
        output_dir: Results directory (e.g. ./promptlens_results)
        run_id: Run identifier, or "latest" to follow the latest symlink

    Returns:
        The parsed RunResult

    Raises:
        FileNotFoundError: If the run directory or results.json is missing
    """
    import json

    results_file = Path(output_dir) / run_id / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Run {run_id} not found in {output_dir}")
    with open(results_file) as f:
        data = json.load(f)
    return RunResult(**data)


def _check_fail_under(result: "RunResult", fail_under: float) -> list:
    """Return models whose average judge score falls below the gate.

    A model with no judge scores at all also fails the gate, since the gate
    cannot be evaluated without scores and a silent pass would be misleading.

    Args:
        result: The completed run result
        fail_under: Minimum acceptable average judge score (1-5 scale)

    Returns:
        List of (model, average_score_or_None) tuples that fail the gate
    """
    failing = []
    for model in result.models_tested:
        avg = result.get_average_score(model)
        if avg is None or avg < fail_under:
            failing.append((model, avg))
    return failing


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
@click.option(
    "--fail-under",
    type=click.FloatRange(1.0, 5.0),
    default=None,
    help=(
        "Quality gate for CI: exit with code 2 if any model's average judge "
        "score falls below this value (1-5 scale). Also sets the per-test "
        "failure threshold used by the junit export format."
    ),
)
def run(
    config: str,
    golden_set: Optional[str],
    output_dir: Optional[str],
    dry_run: bool,
    fail_under: Optional[float],
) -> None:
    """Run evaluation with the given configuration file.

    CONFIG: Path to YAML configuration file

    Examples:
        promptlens run config.yaml
        promptlens run config.yaml --output-dir ./results
        promptlens run config.yaml --dry-run
        promptlens run config.yaml --fail-under 3.5
    """
    try:
        # Load config
        console.print(f"\n[cyan]Loading configuration from {config}...[/cyan]")
        config_data = _load_config_data(config)

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
            "junit": (JUnitXMLExporter(fail_under=fail_under), "junit.xml"),
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

        # Quality gate for CI
        if fail_under is not None:
            failing_models = _check_fail_under(result, fail_under)
            if failing_models:
                console.print(
                    f"\n[bold red]✗ Quality gate failed (--fail-under {fail_under:g}):[/bold red]"
                )
                for model, avg in failing_models:
                    avg_display = f"{avg:.2f}" if avg is not None else "no scores"
                    console.print(f"  {model}: average judge score {avg_display}")
                sys.exit(2)
            console.print(
                f"\n[bold green]✓ Quality gate passed (--fail-under {fail_under:g})[/bold green]"
            )

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
    type=click.Choice(["json", "csv", "md", "html", "junit"], case_sensitive=False),
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
            extensions = {
                "json": ".json",
                "csv": ".csv",
                "md": ".md",
                "html": ".html",
                "junit": ".xml",
            }
            output = f"export_{run_id}{extensions[export_format]}"

        # Export
        exporters = {
            "json": JSONExporter(),
            "csv": CSVExporter(),
            "md": MarkdownExporter(),
            "html": HTMLExporter(),
            "junit": JUnitXMLExporter(),
        }

        exporter = exporters[export_format]
        exporter.export(result, output)

        console.print(f"[green]✓[/green] Exported to {output}")

    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("baseline_run")
@click.argument("candidate_run")
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./promptlens_results",
    help="Results directory containing both runs",
)
@click.option(
    "--threshold",
    type=click.FloatRange(0.0, 4.0),
    default=0.0,
    help=(
        "Minimum judge-score drop (in points, 1-5 scale) for a case to "
        "count as a regression. 0 means any drop regresses."
    ),
)
@click.option(
    "--baseline-model",
    default=None,
    help="Compare only this model from the baseline run (requires --candidate-model)",
)
@click.option(
    "--candidate-model",
    default=None,
    help="Compare only this model from the candidate run (requires --baseline-model)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["md", "json"], case_sensitive=False),
    default=None,
    help="Also write the comparison to a file in this format",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path for --format (default: auto-generated)",
)
@click.option(
    "--fail-on-regression",
    is_flag=True,
    help="Quality gate for CI: exit with code 2 if any case regressed",
)
def compare(
    baseline_run: str,
    candidate_run: str,
    output_dir: str,
    threshold: float,
    baseline_model: Optional[str],
    candidate_model: Optional[str],
    output_format: Optional[str],
    output: Optional[str],
    fail_on_regression: bool,
) -> None:
    """Compare two runs and report per-case regressions and improvements.

    BASELINE_RUN: Run ID of the reference run (or "latest")

    CANDIDATE_RUN: Run ID of the new run to evaluate (or "latest")

    Examples:
        promptlens compare run_abc run_def
        promptlens compare run_abc latest --fail-on-regression
        promptlens compare run_abc run_def --threshold 1 --format md
        promptlens compare run_abc run_def --baseline-model gpt-4o --candidate-model claude-sonnet-4-5
    """
    try:
        from rich.table import Table

        from promptlens.comparison import (
            STATUS_IMPROVED,
            STATUS_REGRESSED,
            compare_runs,
            comparison_to_markdown,
        )

        if (baseline_model is None) != (candidate_model is None):
            console.print(
                "[red]--baseline-model and --candidate-model must be used together[/red]"
            )
            sys.exit(1)

        console.print(f"\n[cyan]Loading baseline run {baseline_run}...[/cyan]")
        baseline = _load_run(output_dir, baseline_run)
        console.print(f"[cyan]Loading candidate run {candidate_run}...[/cyan]")
        candidate = _load_run(output_dir, candidate_run)

        result = compare_runs(
            baseline,
            candidate,
            regression_threshold=threshold,
            baseline_model=baseline_model,
            candidate_model=candidate_model,
        )

        # Summary
        console.print(
            f"\n[bold]Baseline:[/bold] {result.baseline_run_name or result.baseline_run_id}"
            f"  [bold]Candidate:[/bold] {result.candidate_run_name or result.candidate_run_id}"
        )
        if result.avg_score_delta is not None:
            console.print(
                f"Average judge score: {result.baseline_avg_score:.2f} → "
                f"{result.candidate_avg_score:.2f} ({result.avg_score_delta:+.2f})"
            )
        console.print(
            f"Cost delta (shared cases): {result.total_cost_delta_usd:+.4f} USD"
        )
        if result.avg_latency_delta_ms is not None:
            console.print(f"Average latency delta: {result.avg_latency_delta_ms:+.1f} ms")
        console.print(
            f"\n[red]{result.regressed} regressed[/red] / "
            f"[green]{result.improved} improved[/green] / "
            f"{result.unchanged} unchanged / {result.unscored} unscored / "
            f"{result.added} added / {result.removed} removed"
        )

        # Per-case table for regressions and improvements
        notable = [
            c for c in result.cases if c.status in (STATUS_REGRESSED, STATUS_IMPROVED)
        ]
        if notable:
            table = Table(title="Changed Cases")
            table.add_column("Test Case")
            table.add_column("Model")
            table.add_column("Baseline")
            table.add_column("Candidate")
            table.add_column("Delta")
            table.add_column("Status")
            for case in notable:
                model_cell = case.candidate_model or case.baseline_model or "-"
                if (
                    case.baseline_model
                    and case.candidate_model
                    and case.baseline_model != case.candidate_model
                ):
                    model_cell = f"{case.baseline_model} → {case.candidate_model}"
                baseline_cell = (
                    str(case.baseline_score)
                    if case.baseline_score is not None
                    else ("error" if case.baseline_error else "-")
                )
                candidate_cell = (
                    str(case.candidate_score)
                    if case.candidate_score is not None
                    else ("error" if case.candidate_error else "-")
                )
                delta_cell = (
                    f"{case.score_delta:+d}" if case.score_delta is not None else "-"
                )
                status_cell = (
                    f"[red]{case.status}[/red]"
                    if case.status == STATUS_REGRESSED
                    else f"[green]{case.status}[/green]"
                )
                table.add_row(
                    case.test_case_id,
                    model_cell,
                    baseline_cell,
                    candidate_cell,
                    delta_cell,
                    status_cell,
                )
            console.print(table)

        # Optional file output
        if output_format:
            output_format = output_format.lower()
            if not output:
                output = (
                    f"compare_{result.baseline_run_id}_vs_"
                    f"{result.candidate_run_id}.{output_format}"
                )
            if output_format == "md":
                content = comparison_to_markdown(result)
            else:
                content = result.model_dump_json(indent=2)
            with open(output, "w") as f:
                f.write(content)
            console.print(f"\n[green]✓[/green] Comparison written to {output}")

        # Quality gate for CI
        if fail_on_regression:
            if result.has_regressions:
                console.print(
                    f"\n[bold red]✗ Regression gate failed: "
                    f"{result.regressed} case(s) regressed[/bold red]"
                )
                sys.exit(2)
            console.print("\n[bold green]✓ Regression gate passed[/bold green]")

    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Comparison failed: {e}[/red]")
        logging.exception("Comparison failed")
        sys.exit(1)


if __name__ == "__main__":
    cli()
