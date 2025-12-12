"""Base provider interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import Optional

from promptlens.models.config import ProviderConfig
from promptlens.models.result import ModelResponse


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
        **kwargs: any
    ) -> ModelResponse:
        """Generate a response from the model.

        Args:
            prompt: The input prompt
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
