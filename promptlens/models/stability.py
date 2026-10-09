"""Data models for multi-sample stability analysis."""

from typing import List, Optional

from pydantic import BaseModel, Field


class CaseStability(BaseModel):
    """Score stability of one (test case, model) pair across samples.

    Attributes:
        test_case_id: ID of the test case
        model: Model identifier
        query: The test case query
        samples: Number of samples run for this pair
        scored_samples: Samples that produced a judge score
        errored_samples: Samples whose model call failed
        scores: Judge scores (1-5) in sample order, scored samples only
        mean_score: Mean of scores, if any
        min_score: Lowest score, if any
        max_score: Highest score, if any
        std_dev: Sample standard deviation of scores (0.0 for one sample)
        passed_samples: Scored samples at or above the pass threshold
        pass_rate: passed_samples / scored_samples, if any were scored
        pass_at_k: Whether at least one sample passed
        pass_hat_k: Whether every scored sample passed
        flaky: Whether the pair both passed and failed across samples
    """

    test_case_id: str
    model: str
    query: str = ""
    samples: int = 0
    scored_samples: int = 0
    errored_samples: int = 0
    scores: List[int] = Field(default_factory=list)
    mean_score: Optional[float] = None
    min_score: Optional[int] = None
    max_score: Optional[int] = None
    std_dev: Optional[float] = None
    passed_samples: int = 0
    pass_rate: Optional[float] = None
    pass_at_k: bool = False
    pass_hat_k: bool = False
    flaky: bool = False


class ModelStability(BaseModel):
    """Aggregate stability for one model across all test cases.

    Attributes:
        model: Model identifier
        samples: Samples per test case in this run
        pass_threshold: Judge score at or above which a sample passes
        cases: Test cases evaluated for this model
        scored_cases: Test cases with at least one judge score
        mean_score: Mean judge score over every sample of every case
        score_ci_low: Lower bound of the 95% confidence interval on mean_score
        score_ci_high: Upper bound of the 95% confidence interval on mean_score
        mean_case_std_dev: Average within-case score standard deviation
        pass_at_k_cases: Cases that passed in at least one sample
        pass_hat_k_cases: Cases that passed in every sample
        flaky_cases: Cases that both passed and failed
        pass_at_k: pass_at_k_cases / scored_cases (optimistic bound)
        pass_hat_k: pass_hat_k_cases / scored_cases (pessimistic bound)
        flaky_rate: flaky_cases / scored_cases
    """

    model: str
    samples: int = 1
    pass_threshold: float = 3.0
    cases: int = 0
    scored_cases: int = 0
    mean_score: Optional[float] = None
    score_ci_low: Optional[float] = None
    score_ci_high: Optional[float] = None
    mean_case_std_dev: Optional[float] = None
    pass_at_k_cases: int = 0
    pass_hat_k_cases: int = 0
    flaky_cases: int = 0
    pass_at_k: Optional[float] = None
    pass_hat_k: Optional[float] = None
    flaky_rate: Optional[float] = None


class StabilityReport(BaseModel):
    """Stability analysis of a whole run.

    Attributes:
        run_id: Run the report was computed from
        samples: Samples per test case
        pass_threshold: Judge score at or above which a sample passes
        models: Per-model aggregates
        cases: Per-(test case, model) details
    """

    run_id: str
    samples: int = 1
    pass_threshold: float = 3.0
    models: List[ModelStability] = Field(default_factory=list)
    cases: List[CaseStability] = Field(default_factory=list)

    @property
    def flaky_cases(self) -> List[CaseStability]:
        """All (test case, model) pairs that both passed and failed."""
        return [c for c in self.cases if c.flaky]

    @property
    def has_flaky_cases(self) -> bool:
        """Whether any pair was flaky."""
        return bool(self.flaky_cases)

    @property
    def is_multi_sample(self) -> bool:
        """Whether the run had more than one sample per test case."""
        return self.samples > 1
