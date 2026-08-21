"""Data models for cross-run comparison."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

# Case-level comparison statuses
STATUS_REGRESSED = "regressed"
STATUS_IMPROVED = "improved"
STATUS_UNCHANGED = "unchanged"
STATUS_INCOMPARABLE = "incomparable"

# Reasons attached to a case status for extra context
REASON_SCORE_DROP = "score_drop"
REASON_SCORE_GAIN = "score_gain"
REASON_NEW_ERROR = "new_error"
REASON_ERROR_RESOLVED = "error_resolved"
REASON_SCORE_MISSING = "score_missing"
REASON_NO_BASELINE_SCORE = "no_baseline_score"
REASON_NO_SCORES = "no_scores"
REASON_BOTH_ERRORED = "both_errored"
REASON_WITHIN_THRESHOLD = "within_threshold"


class CaseComparison(BaseModel):
    """Comparison of a single (test case, model) pair across two runs.

    Attributes:
        test_case_id: ID of the test case
        model: Model identifier the pair was evaluated on
        query: The test case query (taken from the candidate run)
        baseline_score: Judge score in the baseline run (1-5), if judged
        candidate_score: Judge score in the candidate run (1-5), if judged
        score_delta: candidate_score - baseline_score, if both are present
        baseline_cost_usd: Cost of the baseline response
        candidate_cost_usd: Cost of the candidate response
        baseline_latency_ms: Latency of the baseline response
        candidate_latency_ms: Latency of the candidate response
        baseline_error: Error message from the baseline run, if any
        candidate_error: Error message from the candidate run, if any
        status: One of regressed / improved / unchanged / incomparable
        reason: Machine-readable reason for the status
    """

    test_case_id: str
    model: str
    query: str = ""
    baseline_score: Optional[int] = None
    candidate_score: Optional[int] = None
    score_delta: Optional[int] = None
    baseline_cost_usd: Optional[float] = None
    candidate_cost_usd: Optional[float] = None
    baseline_latency_ms: Optional[float] = None
    candidate_latency_ms: Optional[float] = None
    baseline_error: Optional[str] = None
    candidate_error: Optional[str] = None
    status: str
    reason: str


class ModelComparison(BaseModel):
    """Aggregate comparison for one model across two runs.

    Attributes:
        model: Model identifier
        baseline_avg_score: Average judge score in the baseline run
        candidate_avg_score: Average judge score in the candidate run
        avg_score_delta: candidate average - baseline average, if both exist
        regressed: Number of regressed cases
        improved: Number of improved cases
        unchanged: Number of unchanged cases
        incomparable: Number of cases that could not be compared
        baseline_cost_usd: Total baseline cost for this model
        candidate_cost_usd: Total candidate cost for this model
        cost_delta_usd: candidate cost - baseline cost
        baseline_avg_latency_ms: Average baseline latency per case
        candidate_avg_latency_ms: Average candidate latency per case
        latency_delta_ms: candidate average latency - baseline average latency
    """

    model: str
    baseline_avg_score: Optional[float] = None
    candidate_avg_score: Optional[float] = None
    avg_score_delta: Optional[float] = None
    regressed: int = 0
    improved: int = 0
    unchanged: int = 0
    incomparable: int = 0
    baseline_cost_usd: float = 0.0
    candidate_cost_usd: float = 0.0
    cost_delta_usd: float = 0.0
    baseline_avg_latency_ms: Optional[float] = None
    candidate_avg_latency_ms: Optional[float] = None
    latency_delta_ms: Optional[float] = None


class ComparisonResult(BaseModel):
    """Full comparison of two evaluation runs.

    Cases and models present in only one run are reported separately and
    excluded from the pairwise comparison.

    Attributes:
        baseline_run_id: Run ID of the baseline run
        candidate_run_id: Run ID of the candidate run
        baseline_run_name: Human-readable name of the baseline run
        candidate_run_name: Human-readable name of the candidate run
        baseline_timestamp: When the baseline run started
        candidate_timestamp: When the candidate run started
        golden_set_name: Golden set name (from the candidate run)
        threshold: Minimum absolute score delta counted as a change
        models_compared: Models present in both runs
        models_added: Models only in the candidate run
        models_removed: Models only in the baseline run
        cases_added: Test case IDs only in the candidate run
        cases_removed: Test case IDs only in the baseline run
        cases: Pairwise case comparisons
        model_summaries: Per-model aggregate comparisons
        timestamp: When the comparison was computed
    """

    baseline_run_id: str
    candidate_run_id: str
    baseline_run_name: Optional[str] = None
    candidate_run_name: Optional[str] = None
    baseline_timestamp: Optional[datetime] = None
    candidate_timestamp: Optional[datetime] = None
    golden_set_name: str = ""
    threshold: float = 1.0
    models_compared: List[str] = Field(default_factory=list)
    models_added: List[str] = Field(default_factory=list)
    models_removed: List[str] = Field(default_factory=list)
    cases_added: List[str] = Field(default_factory=list)
    cases_removed: List[str] = Field(default_factory=list)
    cases: List[CaseComparison] = Field(default_factory=list)
    model_summaries: List[ModelComparison] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def regressed_cases(self) -> List[CaseComparison]:
        """All cases whose status is regressed."""
        return [c for c in self.cases if c.status == STATUS_REGRESSED]

    @property
    def improved_cases(self) -> List[CaseComparison]:
        """All cases whose status is improved."""
        return [c for c in self.cases if c.status == STATUS_IMPROVED]

    @property
    def regression_count(self) -> int:
        """Number of regressed cases."""
        return len(self.regressed_cases)

    @property
    def has_regressions(self) -> bool:
        """Whether any case regressed."""
        return self.regression_count > 0
