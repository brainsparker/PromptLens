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


class JudgeScore(BaseModel):
    """Score from LLM judge.

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
    """

    score: int = Field(..., ge=1, le=5)  # Must be 1-5
    explanation: str
    criteria_scores: Dict[str, int] = Field(default_factory=dict)
    judge_model: str
    judge_provider: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

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
