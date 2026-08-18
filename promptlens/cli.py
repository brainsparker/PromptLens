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


def _resolve_run_reference(run_ref: str, output_dir: str) -> Path:
    """Resolve a run reference to a results.json path.

    Accepts, in order of precedence:
    - a direct path to a results.json file
    - a path to a run directory containing results.json
    - a run ID (or "latest") inside the output directory

    Raises:
        FileNotFoundError: If no results.json can be located
    """
    direct = Path(run_ref)
    if direct.is_file():
        return direct
    if direct.is_dir() and (direct / "results.json").is_file():
        return direct / "results.json"

    in_output_dir = Path(output_dir) / run_ref / "results.json"
    if in_output_dir.is_file():
        return in_output_dir

    raise FileNotFoundError(
        f"Could not find results for '{run_ref}'. Pass a run ID from "
        f"'promptlens list-runs', a run directory, or a results.json path."
    )


def _load_run_result(results_file: Path) -> RunResult:
    """Load a RunResult from a results.json file."""
    import json

    with open(results_file, encoding="utf-8") as f:
        data = json.load(f)
    return RunResult(**data)


@cli.command()
@click.argument("baseline")
@click.argument("candidate")
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./promptlens_results",
    help="Results directory used to resolve run IDs",
)
@click.option(
    "--score-threshold",
    type=click.IntRange(min=0),
    default=0,
    help="Judge-score change (1-5 scale) a paired case must exceed to count "
    "as regressed or improved. 0 flags any decrease.",
)
@click.option(
    "--fail-on-regression",
    is_flag=True,
    default=False,
    help="Exit with a non-zero status if any paired case regressed. "
    "Use as a CI gate alongside 'promptlens run --fail-under'.",
)
@click.option(
    "--output",
    type=click.Path(),
    help="Write a Markdown comparison report to this path",
)
@click.option(
    "--json-output",
    type=click.Path(),
    help="Write the full comparison as JSON to this path",
)
def compare(
    baseline: str,
    candidate: str,
    output_dir: str,
    score_threshold: int,
    fail_on_regression: bool,
    output: Optional[str],
    json_output: Optional[str],
) -> None:
    """Compare two runs and detect regressions.

    BASELINE and CANDIDATE are run IDs (see 'promptlens list-runs'),
    run directories, or paths to results.json files. 'latest' resolves
    to the most recent run in the output directory.

    Test cases are paired by (test case ID, model), so comparisons
    survive reordered, added, and removed cases.

    Examples:
        promptlens compare 20260810_1200 latest
        promptlens compare baseline/results.json candidate/results.json
        promptlens compare main-run pr-run --fail-on-regression
    """
    from promptlens.comparison import compare_runs, render_markdown

    try:
        baseline_file = _resolve_run_reference(baseline, output_dir)
        candidate_file = _resolve_run_reference(candidate, output_dir)

        baseline_run = _load_run_result(baseline_file)
        candidate_run = _load_run_result(candidate_file)

        comparison = compare_runs(
            baseline_run, candidate_run, score_threshold=score_threshold
        )
    except Exception as e:
        console.print(f"[red]Comparison failed: {e}[/red]")
        sys.exit(1)

    from rich.table import Table

    summary_table = Table(title="Comparison Summary")
    summary_table.add_column("Model")
    summary_table.add_column("Baseline avg", justify="right")
    summary_table.add_column("Candidate avg", justify="right")
    summary_table.add_column("Delta", justify="right")
    summary_table.add_column("Regressed", justify="right")
    summary_table.add_column("Improved", justify="right")

    for model_summary in comparison.model_summaries:
        baseline_avg = (
            f"{model_summary.baseline_avg_score:.2f}"
            if model_summary.baseline_avg_score is not None
            else "-"
        )
        candidate_avg = (
            f"{model_summary.candidate_avg_score:.2f}"
            if model_summary.candidate_avg_score is not None
            else "-"
        )
        delta = (
            f"{model_summary.avg_score_delta:+.2f}"
            if model_summary.avg_score_delta is not None
            else "-"
        )
        regressed_cell = (
            f"[red]{model_summary.regressed}[/red]"
            if model_summary.regressed
            else "0"
        )
        summary_table.add_row(
            model_summary.model,
            baseline_avg,
            candidate_avg,
            delta,
            regressed_cell,
            str(model_summary.improved),
        )

    console.print()
    console.print(summary_table)

    regressions = comparison.regressions
    if regressions:
        console.print(f"\n[red]✗ {len(regressions)} regression(s) detected:[/red]")
        for case in regressions:
            if case.candidate_error and not case.baseline_error:
                detail = f"now errors: {case.candidate_error}"
            else:
                detail = (
                    f"score {case.baseline_score} → {case.candidate_score}"
                )
            console.print(f"  [red]{case.test_case_id}[/red] ({case.model}): {detail}")
    else:
        console.print("\n[green]✓ No regressions detected[/green]")

    try:
        if output:
            report_path = Path(output)
            if report_path.parent != Path("."):
                report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(render_markdown(comparison), encoding="utf-8")
            console.print(f"[green]✓[/green] Markdown report written to {output}")

        if json_output:
            import json

            json_path = Path(json_output)
            if json_path.parent != Path("."):
                json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    comparison.model_dump(mode="json"),
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            console.print(f"[green]✓[/green] JSON comparison written to {json_output}")
    except Exception as e:
        console.print(f"[red]Failed to write comparison report: {e}[/red]")
        sys.exit(1)

    if fail_on_regression and comparison.has_regressions():
        console.print(
            f"\n[red]Failing: --fail-on-regression is set and "
            f"{len(regressions)} case(s) regressed[/red]"
        )
        sys.exit(1)


if __name__ == "__main__":
    cli()
