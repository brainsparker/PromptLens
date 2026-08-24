"""Main runner orchestration for executing evaluations."""

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
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

from promptlens.judges.cache import JudgeCache
from promptlens.judges.llm_judge import LLMJudge
from promptlens.judges.spend import JudgeSpendTracker
from promptlens.loaders.yaml_loader import get_loader
from promptlens.models.checks import CheckResult, run_checks
from promptlens.models.config import RunConfig
from promptlens.models.result import (
    EvaluationResult,
    JudgeScore,
    ModelResponse,
    RunResult,
)
from promptlens.models.test_case import GoldenSet, TestCase
from promptlens.providers.base import BaseProvider
from promptlens.providers.factory import get_provider

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

        # Semaphore for rate limiting
        self.semaphore = asyncio.Semaphore(config.execution.parallel_requests)

        # Judge spend guard: budget tracker, result cache, and gate counter
        self.spend = JudgeSpendTracker(config.judge.budget_usd)
        self.judge_cache: Optional[JudgeCache] = None
        if config.judge.cache:
            cache_path = (
                Path(config.output.directory) / ".promptlens" / "judge_cache.json"
            )
            self.judge_cache = JudgeCache(cache_path)
        self.gated_count = 0

    async def run(self) -> RunResult:
        """Run the complete evaluation.

        Returns:
            RunResult with all evaluation results

        Raises:
            Exception: If evaluation fails
        """
        console.print(f"\n[bold cyan]PromptLens Evaluation Run[/bold cyan]")
        console.print(f"Run ID: {self.run_id}")
        console.print(f"Config: {self.config.output.run_name or 'Unnamed'}\n")

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
        results = await self._run_evaluations(golden_set)

        # Persist any new judge verdicts for future runs
        if self.judge_cache is not None:
            self.judge_cache.flush()

        # Calculate totals
        total_cost = sum(r.model_response.cost_usd or 0.0 for r in results)
        total_time = sum(r.model_response.latency_ms for r in results)
        judge_cost = sum(
            r.judge_score.cost_usd or 0.0 for r in results if r.judge_score
        )

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
            judge_cost_usd=judge_cost,
            metadata={
                "golden_set_path": self.config.golden_set,
                "test_case_count": len(golden_set.test_cases),
                "provider_count": len(self.providers),
                "judge_cache_hits": self.judge_cache.hits if self.judge_cache else 0,
                "judge_gated_by_checks": self.gated_count,
                "judge_budget_usd": self.spend.budget_usd,
                "judge_budget_skipped": self.spend.skipped_count,
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
        total_tasks = len(golden_set.test_cases) * len(self.providers)

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

            # Create tasks for all combinations
            tasks = []
            for test_case in golden_set.test_cases:
                for provider in self.providers:
                    tasks.append(
                        self._evaluate_single(
                            test_case=test_case,
                            provider=provider,
                            progress=progress,
                            task_id=task,
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
    ) -> EvaluationResult:
        """Evaluate a single test case with a single provider.

        Args:
            test_case: Test case to evaluate
            provider: Provider to use
            progress: Progress bar instance
            task_id: Task ID for progress updates

        Returns:
            EvaluationResult
        """
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
            check_results: List[CheckResult] = []
            judge_skipped_reason = None
            if not model_response.error:
                judge_score, check_results, judge_skipped_reason = (
                    await self._judge_with_guard(test_case, model_response)
                )

            # Update progress
            progress.update(task_id, advance=1)

            return EvaluationResult(
                test_case_id=test_case.id,
                query=test_case.query,
                expected_behavior=test_case.expected_behavior,
                model_response=model_response,
                judge_score=judge_score,
                timestamp=datetime.utcnow(),
                check_results=check_results,
                judge_skipped_reason=judge_skipped_reason,
            )

    async def _judge_with_guard(
        self,
        test_case: TestCase,
        model_response: ModelResponse,
    ):
        """Judge a response through the spend guard.

        Order of operations:
        1. Deterministic checks: if any fail, the case scores 1 and the
           LLM judge is never called (zero judge spend).
        2. Judge cache: identical judging contexts reuse the stored
           verdict from a previous run (zero judge spend).
        3. Judge budget: once the configured budget is exhausted, further
           judge calls are skipped and reported.
        4. Otherwise, call the LLM judge, record its cost, and cache the
           verdict for future runs.

        Args:
            test_case: Test case being evaluated
            model_response: Successful model response to judge

        Returns:
            Tuple of (judge_score, check_results, judge_skipped_reason)
        """
        # 1. Deterministic checks gate the judge call
        check_results: List[CheckResult] = []
        if test_case.checks:
            check_results = run_checks(test_case.checks, model_response.content)
            failed = [c for c in check_results if not c.passed]
            if failed:
                self.gated_count += 1
                failure_summary = "; ".join(c.detail for c in failed)
                judge_score = JudgeScore(
                    score=1,
                    explanation=(
                        "Failed deterministic checks (LLM judge skipped, no judge "
                        f"spend): {failure_summary}"
                    ),
                    judge_model="deterministic-checks",
                    judge_provider="promptlens",
                    timestamp=datetime.utcnow(),
                    cost_usd=0.0,
                )
                return judge_score, check_results, None

        # 2. Reuse a cached verdict when the judging context is identical
        cache_key = None
        if self.judge_cache is not None:
            cache_key = JudgeCache.key(self.config.judge, test_case, model_response)
            cached_score = self.judge_cache.get(cache_key)
            if cached_score is not None:
                return cached_score, check_results, None

        # 3. Enforce the judge budget
        if not self.spend.allows_call():
            self.spend.record_skip()
            reason = (
                f"judge budget of ${self.spend.budget_usd:g} exhausted "
                f"(spent ${self.spend.spent_usd:.4f})"
            )
            return None, check_results, reason

        # 4. Pay for a judge call
        try:
            judge_score = await self.judge.evaluate(test_case, model_response)
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return None, check_results, None

        self.spend.record_cost(judge_score.cost_usd)
        if self.judge_cache is not None and cache_key is not None:
            self.judge_cache.put(cache_key, judge_score)

        return judge_score, check_results, None

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
        console.print(f"  Judge Cost: ${result.judge_cost_usd:.4f}")
        console.print(f"  Total Time: {result.total_time_ms:.0f}ms")
        console.print(f"  Test Cases: {len(result.results) // len(result.models_tested)}")

        # Spend guard summary
        cache_hits = self.judge_cache.hits if self.judge_cache else 0
        if cache_hits or self.gated_count or self.spend.skipped_count:
            console.print(f"\n[bold]Judge Spend Guard[/bold]")
            if cache_hits:
                console.print(
                    f"  [green]✓[/green] {cache_hits} verdict(s) reused from cache ($0)"
                )
            if self.gated_count:
                console.print(
                    f"  [green]✓[/green] {self.gated_count} judge call(s) gated by "
                    f"deterministic checks ($0)"
                )
            if self.spend.skipped_count:
                console.print(
                    f"  [yellow]⚠[/yellow] {self.spend.skipped_count} judge call(s) "
                    f"skipped: budget of ${self.spend.budget_usd:g} exhausted"
                )
        console.print()
