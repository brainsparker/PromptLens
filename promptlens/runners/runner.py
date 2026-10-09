"""Main runner orchestration for executing evaluations."""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import List, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from promptlens.judges.llm_judge import LLMJudge
from promptlens.loaders.yaml_loader import get_loader
from promptlens.models.config import RunConfig
from promptlens.models.result import EvaluationResult, RunResult
from promptlens.models.test_case import GoldenSet, TestCase
from promptlens.providers.base import BaseProvider
from promptlens.providers.factory import get_provider
from promptlens.stability import compute_stability, format_scores

logger = logging.getLogger(__name__)
console = Console()


class Runner:
    """Main orchestration engine for running evaluations.

    Coordinates loading golden sets, running models, judging responses,
    and aggregating results.
    """

    def __init__(self, config: RunConfig) -> None:
        """Initialize the runner.

        Args:
            config: Run configuration
        """
        self.config = config
        self.run_id = str(uuid.uuid4())[:8]

        # Initialize judge
        self.judge = LLMJudge(config.judge)

        # Initialize providers for each model
        self.providers: List[BaseProvider] = []
        for model_config in config.models:
            try:
                provider = get_provider(model_config)
                self.providers.append(provider)
                logger.info(f"Initialized provider for {model_config.name}")
            except Exception as e:
                logger.error(f"Failed to initialize provider for {model_config.name}: {e}")
                console.print(
                    f"[red]Warning: Skipping {model_config.name} - {e}[/red]"
                )

        if not self.providers:
            raise ValueError("No providers successfully initialized")

        # Semaphore for rate limiting. Created inside run() so it binds to the
        # event loop that actually executes the evaluation; on Python 3.9 a
        # semaphore built here would attach to whatever loop (if any) is
        # current at construction time.
        self.semaphore: Optional[asyncio.Semaphore] = None

        # Repeats per (test case, model). More than one enables stability
        # analysis (pass@k, pass^k, flaky cases).
        self.samples: int = config.execution.samples

    async def run(self) -> RunResult:
        """Run the complete evaluation.

        Returns:
            RunResult with all evaluation results

        Raises:
            Exception: If evaluation fails
        """
        console.print(f"\n[bold cyan]PromptLens Evaluation Run[/bold cyan]")
        console.print(f"Run ID: {self.run_id}")
        console.print(f"Config: {self.config.output.run_name or 'Unnamed'}")
        if self.samples > 1:
            console.print(f"Samples per test case: {self.samples}")
        console.print()

        # Load golden set
        console.print(f"[yellow]Loading golden set...[/yellow]")
        loader = get_loader(self.config.golden_set)
        golden_set = loader.load(self.config.golden_set)
        console.print(
            f"[green]✓[/green] Loaded '{golden_set.name}' "
            f"with {len(golden_set.test_cases)} test cases\n"
        )

        # Run evaluations
        console.print(f"[yellow]Running evaluations...[/yellow]")
        self.semaphore = asyncio.Semaphore(self.config.execution.parallel_requests)
        results = await self._run_evaluations(golden_set)

        # Calculate totals
        total_cost = sum(r.model_response.cost_usd or 0.0 for r in results)
        total_time = sum(r.model_response.latency_ms for r in results)

        # Create run result
        run_result = RunResult(
            run_id=self.run_id,
            run_name=self.config.output.run_name,
            timestamp=datetime.utcnow(),
            golden_set_name=golden_set.name,
            models_tested=[p.config.model for p in self.providers],
            results=results,
            total_cost_usd=total_cost,
            total_time_ms=total_time,
            metadata={
                "golden_set_path": self.config.golden_set,
                "test_case_count": len(golden_set.test_cases),
                "provider_count": len(self.providers),
                "samples": self.samples,
            },
        )

        # Print summary
        self._print_summary(run_result)

        return run_result

    async def _run_evaluations(
        self,
        golden_set: GoldenSet,
    ) -> List[EvaluationResult]:
        """Run evaluations for all test cases and models.

        Args:
            golden_set: The golden set to evaluate

        Returns:
            List of evaluation results
        """
        results: List[EvaluationResult] = []

        # Calculate total tasks
        total_tasks = len(golden_set.test_cases) * len(self.providers) * self.samples

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Evaluating {len(self.providers)} model(s)...",
                total=total_tasks,
            )

            # Create tasks for all combinations. Samples are independent
            # requests; nothing is shared between repeats of the same case.
            tasks = []
            for test_case in golden_set.test_cases:
                for provider in self.providers:
                    for sample_index in range(self.samples):
                        tasks.append(
                            self._evaluate_single(
                                test_case=test_case,
                                provider=provider,
                                progress=progress,
                                task_id=task,
                                sample_index=sample_index,
                            )
                        )

            # Run all tasks with concurrency control
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed: {result}")
                    console.print(f"[red]Error in evaluation: {result}[/red]")
                else:
                    valid_results.append(result)

        console.print(
            f"[green]✓[/green] Completed {len(valid_results)}/{total_tasks} evaluations\n"
        )

        return valid_results

    async def _evaluate_single(
        self,
        test_case: TestCase,
        provider: BaseProvider,
        progress: Progress,
        task_id: any,
        sample_index: int = 0,
    ) -> EvaluationResult:
        """Evaluate a single test case with a single provider.

        Args:
            test_case: Test case to evaluate
            provider: Provider to use
            progress: Progress bar instance
            task_id: Task ID for progress updates
            sample_index: Which repeat of this (test case, provider) pair

        Returns:
            EvaluationResult
        """
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.config.execution.parallel_requests)
        async with self.semaphore:
            # Check if tools are requested but provider doesn't support them
            if test_case.tools and not provider.supports_tools():
                logger.warning(
                    f"Test case '{test_case.id}' requires tools, but provider "
                    f"'{provider.provider_name}' does not support tool calling. "
                    "Tool evaluation will not work properly."
                )

            # Generate response (pass tools if provided)
            model_response = await provider.generate(
                test_case.query,
                tools=test_case.tools if test_case.tools else None,
                retry_attempts=self.config.execution.retry_attempts,
                retry_delay_seconds=self.config.execution.retry_delay_seconds,
                timeout_seconds=self.config.execution.timeout_seconds,
            )

            # Judge the response (only if generation succeeded)
            judge_score = None
            if not model_response.error:
                try:
                    judge_score = await self.judge.evaluate(test_case, model_response)
                except Exception as e:
                    logger.error(f"Judge evaluation failed: {e}")

            # Update progress
            progress.update(task_id, advance=1)

            return EvaluationResult(
                test_case_id=test_case.id,
                query=test_case.query,
                expected_behavior=test_case.expected_behavior,
                model_response=model_response,
                judge_score=judge_score,
                timestamp=datetime.utcnow(),
                sample_index=sample_index,
            )

    def _print_summary(self, result: RunResult) -> None:
        """Print a summary of the run results.

        Args:
            result: The run result to summarize
        """
        console.print("[bold green]═══ Evaluation Summary ═══[/bold green]\n")

        # Per-model summary
        for model in result.models_tested:
            avg_score = result.get_average_score(model)
            total_cost = result.get_total_cost(model)
            total_latency = result.get_total_latency(model)

            console.print(f"[bold]{model}[/bold]")
            if avg_score is not None:
                console.print(f"  Average Score: {avg_score:.2f}/5.0")
            console.print(f"  Total Cost: ${total_cost:.4f}")
            console.print(f"  Total Time: {total_latency:.0f}ms")
            console.print()

        # Overall summary
        console.print(f"[bold]Overall[/bold]")
        console.print(f"  Total Cost: ${result.total_cost_usd:.4f}")
        console.print(f"  Total Time: {result.total_time_ms:.0f}ms")
        console.print(f"  Test Cases: {result.test_case_count}")
        if result.sample_count > 1:
            console.print(f"  Samples per Test Case: {result.sample_count}")
        console.print()

        if result.sample_count > 1:
            print_stability_summary(result)


def print_stability_summary(result: RunResult, pass_threshold: Optional[float] = None) -> None:
    """Print pass@k, pass^k and flaky cases for a multi-sample run.

    Args:
        result: The run result to summarize
        pass_threshold: Judge score at or above which a sample passes.
            Defaults to the run's recorded threshold, then 3.0.
    """
    report = compute_stability(result, pass_threshold)
    k = report.samples
    console.print(
        f"[bold green]═══ Stability ({k} samples per test case, "
        f"pass threshold {report.pass_threshold:g}) ═══[/bold green]\n"
    )

    for model in report.models:
        console.print(f"[bold]{model.model}[/bold]")
        if model.scored_cases == 0:
            console.print("  No judge scores available")
            console.print()
            continue
        console.print(
            f"  pass@{k}: {model.pass_at_k:.0%} "
            f"({model.pass_at_k_cases}/{model.scored_cases} cases passed at least once)"
        )
        console.print(
            f"  pass^{k}: {model.pass_hat_k:.0%} "
            f"({model.pass_hat_k_cases}/{model.scored_cases} cases passed every time)"
        )
        flaky_style = "red" if model.flaky_cases else "green"
        console.print(
            f"  Flaky cases: [{flaky_style}]{model.flaky_cases}[/{flaky_style}]"
            f" ({model.flaky_rate:.0%})"
        )
        if model.mean_score is not None and model.score_ci_low is not None:
            console.print(
                f"  Mean Score: {model.mean_score:.2f} "
                f"(95% CI {model.score_ci_low:.2f} to {model.score_ci_high:.2f})"
            )
        if model.mean_case_std_dev is not None:
            console.print(f"  Mean within-case std dev: {model.mean_case_std_dev:.2f}")
        console.print()

    flaky = report.flaky_cases
    if flaky:
        console.print("[bold]Flaky cases[/bold] (passed in some samples, failed in others)")
        for case in flaky:
            console.print(
                f"  [yellow]~[/yellow] {case.test_case_id} ({case.model}): "
                f"scores {format_scores(case.scores)}, "
                f"passed {case.passed_samples}/{case.scored_samples}"
            )
        console.print()
