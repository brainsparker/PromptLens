"""Google Gemini provider implementation."""

import logging
import os
from datetime import datetime
from typing import Any, List, Optional

import google.generativeai as genai

from promptlens.models.config import ProviderConfig
from promptlens.models.result import ModelResponse
from promptlens.models.tools import ToolDefinition
from promptlens.providers.base import BaseProvider
from promptlens.utils.cost import calculate_cost
from promptlens.utils.retry import retry_with_exponential_backoff
from promptlens.utils.timing import measure_time

logger = logging.getLogger(__name__)


class GoogleProvider(BaseProvider):
    """Provider for Google Gemini models.

    Supports Gemini Pro, Gemini Ultra, and Gemini 1.5 models.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize the Google provider.

        Args:
            config: Provider configuration
        """
        super().__init__(config)

        # Get API key from config or environment
        api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY environment variable "
                "or provide it in the config."
            )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.config.model)

    async def generate(
        self,
        prompt: str,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any
    ) -> ModelResponse:
        """Generate a response from Gemini.

        Args:
            prompt: The input prompt
            tools: Optional list of tools (not supported yet, will be ignored with warning)
            **kwargs: Additional parameters (overrides config)

        Returns:
            ModelResponse with generated content and metadata
        """
        # Log warning if tools are provided (not yet supported)
        if tools:
            logger.warning(
                f"{self.provider_name} provider does not support tool calling yet. "
                "Tools will be ignored. Consider using Anthropic or OpenAI providers for tool evaluation."
            )

        async def _make_request() -> ModelResponse:
            async with measure_time() as timer:
                # Configure generation settings
                generation_config = genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", self.config.temperature),
                    max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                    **self.config.additional_params,
                )

                # Generate response (note: SDK doesn't have async support yet)
                # We'll use sync call - this is a known limitation
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )

                # Extract content
                content = response.text if response and response.text else ""

                # Try to extract token usage
                prompt_tokens = 0
                completion_tokens = 0
                if hasattr(response, "usage_metadata"):
                    prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                    completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

                total_tokens = prompt_tokens + completion_tokens

                # Calculate cost
                cost = calculate_cost(
                    provider=self.provider_name,
                    model=self.config.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

                return ModelResponse(
                    content=content,
                    model=self.config.model,
                    provider=self.provider_name,
                    tokens_used=total_tokens if total_tokens > 0 else None,
                    prompt_tokens=prompt_tokens if prompt_tokens > 0 else None,
                    completion_tokens=completion_tokens if completion_tokens > 0 else None,
                    latency_ms=timer.elapsed_ms,
                    cost_usd=cost if cost > 0 else None,
                    timestamp=datetime.utcnow(),
                )

        try:
            return await retry_with_exponential_backoff(
                func=_make_request,
                max_attempts=3,
                initial_delay=1.0,
            )
        except Exception as e:
            logger.error(f"Google request failed: {e}")
            # Return error response
            return ModelResponse(
                content="",
                model=self.config.model,
                provider=self.provider_name,
                latency_ms=0.0,
                error=str(e),
                timestamp=datetime.utcnow(),
            )

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost for the given token counts.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Estimated cost in USD
        """
        return calculate_cost(
            provider=self.provider_name,
            model=self.config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @property
    def provider_name(self) -> str:
        """Return the provider name.

        Returns:
            "google"
        """
        return "google"
