"""Configuration data models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


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
        """Ensure temperature stays within common model bounds."""
        if not 0 <= value <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return value

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int) -> int:
        """Ensure max_tokens is positive."""
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
        """Ensure at least one parallel request worker."""
        if value <= 0:
            raise ValueError("parallel_requests must be greater than 0")
        return value

    @field_validator("retry_attempts")
    @classmethod
    def validate_retry_attempts(cls, value: int) -> int:
        """Ensure retry attempts are non-negative."""
        if value < 0:
            raise ValueError("retry_attempts cannot be negative")
        return value

    @field_validator("retry_delay_seconds")
    @classmethod
    def validate_retry_delay(cls, value: float) -> float:
        """Ensure retry delay is non-negative."""
        if value < 0:
            raise ValueError("retry_delay_seconds cannot be negative")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout_seconds(cls, value: int) -> int:
        """Ensure timeout is positive."""
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
    def validate_formats(cls, formats: List[str]) -> List[str]:
        """Normalize and validate export formats."""
        if not formats:
            raise ValueError("output.formats must include at least one format")

        allowed_formats = {"json", "csv", "md", "html"}
        normalized: List[str] = []
        for format_name in formats:
            normalized_name = format_name.lower()
            if normalized_name not in allowed_formats:
                allowed = ", ".join(sorted(allowed_formats))
                raise ValueError(f"unsupported output format '{format_name}'. Allowed: {allowed}")
            if normalized_name not in normalized:
                normalized.append(normalized_name)

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

    @field_validator("models")
    @classmethod
    def validate_models(cls, models: List[ModelConfig]) -> List[ModelConfig]:
        """Require at least one model configuration."""
        if not models:
            raise ValueError("models must include at least one model configuration")
        return models


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
