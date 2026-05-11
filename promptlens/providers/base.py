"""Base provider interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from promptlens.models.config import ProviderConfig
from promptlens.models.result import ModelResponse
from promptlens.models.tools import ToolDefinition


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations must inherit from this class and implement
    the abstract methods.

    Attributes:
        config: Provider configuration
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize the provider.

        Args:
            config: Provider configuration
        """
        self.config = config

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: any
    ) -> ModelResponse:
        """Generate a response from the model.

        Args:
            prompt: The input prompt
            tools: Optional list of tools/functions the model can use
            **kwargs: Additional provider-specific parameters

        Returns:
            ModelResponse with the generated content and metadata

        Raises:
            Exception: If the request fails after retries
        """
        pass

    @abstractmethod
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Estimate cost in USD for the given token counts.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Estimated cost in USD
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name.

        Returns:
            Provider name (e.g., "anthropic", "openai")
        """
        pass

    def validate_config(self) -> None:
        """Validate the provider configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config.model:
            raise ValueError("Model is required in provider configuration")

    def supports_tools(self) -> bool:
        """Check if this provider supports tool/function calling.

        Returns:
            True if the provider supports tools, False otherwise
        """
        return False  # Default: most providers don't support tools yet

    def get_retry_settings(self, kwargs: dict) -> Tuple[int, float]:
        """Resolve retry settings from runtime kwargs.

        Args:
            kwargs: Runtime kwargs passed to ``generate``.

        Returns:
            Tuple of (max_attempts, initial_delay_seconds)
        """
        max_attempts = int(kwargs.get("retry_attempts", 3))
        initial_delay = float(kwargs.get("retry_delay_seconds", 1.0))

        # Guard rails to prevent invalid runtime values from breaking retries
        if max_attempts < 1:
            max_attempts = 1
        if initial_delay < 0:
            initial_delay = 0.0

        return max_attempts, initial_delay

    def get_timeout_seconds(self, kwargs: dict) -> int:
        """Resolve request timeout in seconds from runtime kwargs.

        Args:
            kwargs: Runtime kwargs passed to ``generate``.

        Returns:
            Request timeout in seconds.
        """
        timeout_seconds = int(kwargs.get("timeout_seconds", self.config.timeout))
        return timeout_seconds if timeout_seconds > 0 else self.config.timeout
