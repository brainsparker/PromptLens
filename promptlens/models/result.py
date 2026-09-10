"""Result data models for evaluation runs."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from promptlens.models.tools import ToolCall, ToolCallEvaluation


class ModelResponse(BaseModel):
    """Response from a model.

    Attributes:
        content: The actual response text from the model
        model: Model identifier (e.g., "claude-3-5-sonnet-20241022")
        provider: Provider name (e.g., "anthropic", "openai")
        tokens_used: Total tokens used (if available)
        prompt_tokens: Number of prompt tokens (if available)
        completion_tokens: Number of completion tokens (if available)
        latency_ms: Round-trip latency in milliseconds
        cost_usd: Estimated cost in USD
        error: Error message if the request failed
        timestamp: When the response was generated
        tool_calls: Tool calls made by the model (if any)
        stop_reason: Reason the model stopped generating (e.g., "end_turn", "tool_use")
    """

    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: float
    cost_usd: Optional[float] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Tool calling fields (optional, for backward compatibility)
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="Tool calls made by the model in this response"
    )
    stop_reason: Optional[str] = Field(
        None,
        description="Reason the model stopped (e.g., 'end_turn', 'tool_use', 'max_tokens')"
    )


class AssertionResult(BaseModel):
    """Outcome of one deterministic assertion against a response.

    Attributes:
        type: Assertion type (contains, regex, is_json, token_f1, ...)
        label: Human-readable label for reports
        passed: Whether the assertion held
        detail: Short explanation of the outcome
        score: Numeric score for scored assertions such as token_f1 (0.0-1.0)
    """

    type: str
    label: str
    passed: bool
    detail: str = ""
    score: Optional[float] = None


class JudgeScore(BaseModel):
    """Score from a judge (LLM-as-judge or deterministic).

    Attributes:
        score: Integer score from 1-5
        explanation: Explanation of the score
        criteria_scores: Optional sub-scores for specific criteria
        judge_model: Model used for judging
        judge_provider: Provider of the judge model
        timestamp: When the score was generated
        tool_evaluations: Detailed evaluation of each tool call (if applicable)
        tool_usage_score: Overall score for tool usage correctness (1-5)
        tool_efficiency_score: Score for tool usage efficiency (1-5)
        assertion_results: Outcomes of the test case's deterministic assertions
    """

    score: int = Field(..., ge=1, le=5)  # Must be 1-5
    explanation: str
    criteria_scores: Dict[str, int] = Field(default_factory=dict)
    judge_model: str
    judge_provider: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Deterministic assertion results (optional, for backward compatibility)
    assertion_results: List[AssertionResult] = Field(
        default_factory=list,
        description="Results of deterministic assertions declared on the test case",
    )

    @property
    def failed_assertions(self) -> List[AssertionResult]:
        """Assertions that did not hold."""
        return [a for a in self.assertion_results if not a.passed]

    @property
    def assertions_passed(self) -> bool:
        """True when every recorded assertion held (vacuously true if none)."""
        return not self.failed_assertions

    # Tool evaluation fields (optional, for backward compatibility)
    tool_evaluations: List[ToolCallEvaluation] = Field(
        default_factory=list,
        description="Detailed evaluation results for each tool call"
    )
    tool_usage_score: Optional[float] = Field(
        None,
        description="Overall score for tool usage correctness (0.0-1.0)"
    )
    tool_efficiency_score: Optional[float] = Field(
        None,
        description="Score for tool usage efficiency (0.0-1.0)"
    )


class EvaluationResult(BaseModel):
    """Complete evaluation result for one test case + model.

    Attributes:
        test_case_id: ID of the test case that was evaluated
        query: The original query
        expected_behavior: What was expected
        model_response: The model's response with metadata
        judge_score: Score from the judge (if judging was performed)
        timestamp: When the evaluation was performed
    """

    test_case_id: str
    query: str
    expected_behavior: str
    model_response: ModelResponse
    judge_score: Optional[JudgeScore] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RunResult(BaseModel):
    """Results from a complete evaluation run.

    Attributes:
        run_id: Unique identifier for this run
        run_name: Optional human-readable name
        timestamp: When the run started
        golden_set_name: Name of the golden set used
        models_tested: List of model identifiers tested
        results: All evaluation results
        total_cost_usd: Total cost across all requests
        total_time_ms: Total time for all requests
        metadata: Additional run metadata
    """

    run_id: str
    run_name: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    golden_set_name: str
    models_tested: List[str]
    results: List[EvaluationResult]
    total_cost_usd: float = 0.0
    total_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_average_score(self, model: Optional[str] = None) -> Optional[float]:
        """Calculate average score for a specific model or all models.

        Args:
            model: Optional model name to filter by

        Returns:
            Average score or None if no scores available
        """
        filtered_results = self.results
        if model:
            filtered_results = [r for r in self.results if r.model_response.model == model]

        scores = [r.judge_score.score for r in filtered_results if r.judge_score]
        return sum(scores) / len(scores) if scores else None

    def get_total_cost(self, model: Optional[str] = None) -> float:
        """Calculate total cost for a specific model or all models.

        Args:
            model: Optional model name to filter by

        Returns:
            Total cost in USD
        """
        filtered_results = self.results
        if model:
            filtered_results = [r for r in self.results if r.model_response.model == model]

        return sum(
            r.model_response.cost_usd or 0.0 for r in filtered_results
        )

    def get_assertion_summary(self, model: Optional[str] = None) -> Dict[str, int]:
        """Count deterministic assertion outcomes for a model or all models.

        Args:
            model: Optional model name to filter by

        Returns:
            Dict with "passed", "failed", and "total" assertion counts, plus
            "cases_failed": the number of evaluations with at least one failure
        """
        filtered_results = self.results
        if model:
            filtered_results = [r for r in self.results if r.model_response.model == model]

        passed = 0
        failed = 0
        cases_failed = 0
        for r in filtered_results:
            if not r.judge_score or not r.judge_score.assertion_results:
                continue
            case_failed = 0
            for a in r.judge_score.assertion_results:
                if a.passed:
                    passed += 1
                else:
                    failed += 1
                    case_failed += 1
            if case_failed:
                cases_failed += 1

        return {
            "passed": passed,
            "failed": failed,
            "total": passed + failed,
            "cases_failed": cases_failed,
        }

    def get_total_latency(self, model: Optional[str] = None) -> float:
        """Calculate total latency for a specific model or all models.

        Args:
            model: Optional model name to filter by

        Returns:
            Total latency in milliseconds
        """
        filtered_results = self.results
        if model:
            filtered_results = [r for r in self.results if r.model_response.model == model]

        return sum(r.model_response.latency_ms for r in filtered_results)
