"""Configuration data models."""

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Upper bound on judge samples per response. Judge calls are the most
# expensive part of an eval run, so this guards against a typo like 100.
MAX_JUDGE_SAMPLES = 25


class ProviderConfig(BaseModel):
    """Configuration for a single provider.

    Attributes:
        name: Provider name (e.g., "anthropic", "openai", "google", "http")
        model: Model identifier
        api_key: API key (usually loaded from environment)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        endpoint: Custom endpoint URL (for HTTP provider)
        additional_params: Provider-specific additional parameters
    """

    name: str
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 60
    endpoint: Optional[str] = None
    additional_params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: Optional[str]) -> Optional[str]:
        """Validate endpoint URL when provided."""
        if value is None:
            return value

        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("endpoint must use http or https scheme")
        if not parsed.netloc:
            raise ValueError("endpoint must include a host")
        return value


class ModelConfig(BaseModel):
    """Configuration for a model to test.

    Attributes:
        name: Display name for the model
        provider: Provider name
        model: Model identifier
        endpoint: Optional endpoint URL (for HTTP/local providers)
        timeout: Optional request timeout in seconds
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        additional_params: Provider-specific parameters
    """

    name: str
    provider: str
    model: str
    endpoint: Optional[str] = None
    timeout: Optional[int] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    additional_params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return value

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_tokens must be greater than 0")
        return value


class JudgeConfig(BaseModel):
    """Configuration for the judge.

    Attributes:
        provider: Provider for judge model
        model: Model to use for judging
        temperature: Sampling temperature
        custom_prompt: Optional custom judge prompt template
        criteria: List of criteria to evaluate
        samples: Number of independent judge calls per response. With the
            default of 1 the judge runs once, exactly as before. With N > 1
            every response is judged N times and the samples are aggregated
            into one score plus stability statistics (spread, standard
            deviation, disagreement flag), so a single noisy verdict cannot
            silently decide a CI gate.
        disagreement_range: Minimum spread (max sample minus min sample, on
            the 1-5 scale) at which a judged response is flagged as a judge
            disagreement. Only meaningful when samples > 1.
    """

    provider: str = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.3
    custom_prompt: Optional[str] = None
    criteria: List[str] = Field(default_factory=lambda: ["accuracy", "helpfulness"])
    samples: int = 1
    disagreement_range: int = 2

    @field_validator("samples")
    @classmethod
    def validate_samples(cls, value: int) -> int:
        if value < 1:
            raise ValueError("judge samples must be at least 1")
        if value > MAX_JUDGE_SAMPLES:
            raise ValueError(
                f"judge samples must be at most {MAX_JUDGE_SAMPLES} "
                "(each sample is a separate judge API call per response)"
            )
        return value

    @field_validator("disagreement_range")
    @classmethod
    def validate_disagreement_range(cls, value: int) -> int:
        if not 1 <= value <= 4:
            raise ValueError("disagreement_range must be between 1 and 4 on the 1-5 scale")
        return value


class ExecutionConfig(BaseModel):
    """Configuration for execution settings.

    Attributes:
        parallel_requests: Number of parallel requests
        retry_attempts: Maximum retry attempts for failed requests
        retry_delay_seconds: Initial delay between retries
        timeout_seconds: Request timeout
    """

    parallel_requests: int = 3
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: int = 60

    @field_validator("parallel_requests")
    @classmethod
    def validate_parallel_requests(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("parallel_requests must be greater than 0")
        return value

    @field_validator("retry_attempts")
    @classmethod
    def validate_retry_attempts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("retry_attempts must be greater than or equal to 0")
        return value

    @field_validator("retry_delay_seconds")
    @classmethod
    def validate_retry_delay_seconds(cls, value: float) -> float:
        if value < 0:
            raise ValueError("retry_delay_seconds must be greater than or equal to 0")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout_seconds(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        return value


class OutputConfig(BaseModel):
    """Configuration for output settings.

    Attributes:
        directory: Output directory for results
        formats: List of export formats
        run_name: Optional name for this run
    """

    directory: str = "./promptlens_results"
    formats: List[str] = Field(default_factory=lambda: ["html", "json"])
    run_name: Optional[str] = None

    @field_validator("formats")
    @classmethod
    def validate_formats(cls, value: List[str]) -> List[str]:
        allowed = {"html", "json", "csv", "md", "junit"}
        normalized = [fmt.lower() for fmt in value]
        invalid = sorted({fmt for fmt in normalized if fmt not in allowed})
        if invalid:
            raise ValueError(f"unsupported output format(s): {', '.join(invalid)}")
        if not normalized:
            raise ValueError("output formats must contain at least one format")
        return normalized


class RunConfig(BaseModel):
    """Complete run configuration.

    Attributes:
        golden_set: Path to golden set file
        models: List of models to test
        judge: Judge configuration
        execution: Execution settings
        output: Output settings
    """

    golden_set: str
    models: List[ModelConfig]
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def validate_models(self) -> "RunConfig":
        if not self.models:
            raise ValueError("models must contain at least one model configuration")
        return self

    @field_validator("models")
    @classmethod
    def validate_models_unique(cls, models: List[ModelConfig]) -> List[ModelConfig]:
        """Ensure model display names are unique to avoid ambiguous reports."""
        seen = set()
        duplicates = set()

        for model in models:
            normalized = model.name.strip().lower()
            if normalized in seen:
                duplicates.add(model.name)
            else:
                seen.add(normalized)

        if duplicates:
            duplicate_list = ", ".join(sorted(duplicates))
            raise ValueError(
                f"Model names must be unique (case-insensitive). Duplicates: {duplicate_list}"
            )

        return models

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "golden_set": "./examples/golden_sets/customer_support.yaml",
                "models": [
                    {
                        "name": "Claude 3.5 Sonnet",
                        "provider": "anthropic",
                        "model": "claude-3-5-sonnet-20241022",
                        "temperature": 0.7,
                        "max_tokens": 1024,
                    }
                ],
                "judge": {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet-20241022",
                    "temperature": 0.3,
                },
                "execution": {
                    "parallel_requests": 3,
                    "retry_attempts": 3,
                },
                "output": {
                    "directory": "./promptlens_results",
                    "formats": ["html", "json"],
                },
            }
        })
