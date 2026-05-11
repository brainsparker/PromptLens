"""Configuration data models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


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

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        """Ensure temperature stays within a sane range."""
        if not 0 <= value <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return value

    @field_validator("max_tokens", "timeout")
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        """Ensure max_tokens/timeout are positive."""
        if value <= 0:
            raise ValueError("value must be greater than 0")
        return value


class ModelConfig(BaseModel):
    """Configuration for a model to test.

    Attributes:
        name: Display name for the model
        provider: Provider name
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        additional_params: Provider-specific parameters
    """

    name: str
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    additional_params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        """Ensure temperature stays within a sane range."""
        if not 0 <= value <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return value

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int) -> int:
        """Ensure token limits are positive."""
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
    """

    provider: str = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.3
    custom_prompt: Optional[str] = None
    criteria: List[str] = Field(default_factory=lambda: ["accuracy", "helpfulness"])

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        """Ensure judge temperature stays within a sane range."""
        if not 0 <= value <= 2:
            raise ValueError("temperature must be between 0 and 2")
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

    @field_validator("parallel_requests", "timeout_seconds")
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        """Ensure values are positive."""
        if value <= 0:
            raise ValueError("value must be greater than 0")
        return value

    @field_validator("retry_attempts")
    @classmethod
    def validate_retry_attempts(cls, value: int) -> int:
        """Allow zero retries, but disallow negative values."""
        if value < 0:
            raise ValueError("retry_attempts cannot be negative")
        return value

    @field_validator("retry_delay_seconds")
    @classmethod
    def validate_retry_delay(cls, value: float) -> float:
        """Allow immediate retries, but disallow negative delay."""
        if value < 0:
            raise ValueError("retry_delay_seconds cannot be negative")
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
    def validate_formats(cls, formats: List[str]) -> List[str]:
        """Validate formats and normalize to lowercase unique entries."""
        allowed_formats = {"html", "json", "csv", "md", "markdown"}
        normalized: List[str] = []

        for fmt in formats:
            fmt_lower = fmt.lower().strip()
            if fmt_lower == "markdown":
                fmt_lower = "md"

            if fmt_lower not in allowed_formats:
                allowed = ", ".join(sorted(allowed_formats))
                raise ValueError(f"Unsupported output format '{fmt}'. Allowed: {allowed}")

            if fmt_lower not in normalized:
                normalized.append(fmt_lower)

        if not normalized:
            raise ValueError("formats must include at least one value")

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
    def validate_models_non_empty(self) -> "RunConfig":
        """Ensure at least one model is configured."""
        if not self.models:
            raise ValueError("models must include at least one model configuration")
        return self

    class Config:
        """Pydantic config."""

        json_schema_extra = {
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
        }
