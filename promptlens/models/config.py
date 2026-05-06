"""Configuration data models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    timeout: int = Field(default=60, gt=0)
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
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    additional_params: Dict[str, Any] = Field(default_factory=dict)


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
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
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

    parallel_requests: int = Field(default=3, gt=0)
    retry_attempts: int = Field(default=3, ge=0)
    retry_delay_seconds: float = Field(default=1.0, ge=0.0)
    timeout_seconds: int = Field(default=60, gt=0)


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
