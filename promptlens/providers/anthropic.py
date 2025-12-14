"""Anthropic Claude provider implementation."""

import logging
import os
from datetime import datetime
from typing import Any, List, Optional

from anthropic import AsyncAnthropic

from promptlens.models.config import ProviderConfig
from promptlens.models.result import ModelResponse
from promptlens.models.tools import ToolCall, ToolDefinition
from promptlens.providers.base import BaseProvider
from promptlens.utils.cost import calculate_cost
from promptlens.utils.retry import retry_with_exponential_backoff
from promptlens.utils.timing import measure_time

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude models.

    Supports all Claude 3 models (Opus, Sonnet, Haiku).
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize the Anthropic provider.

        Args:
            config: Provider configuration
        """
        super().__init__(config)

        # Get API key from config or environment
        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or provide it in the config."
            )

        self.client = AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any
    ) -> ModelResponse:
        """Generate a response from Claude.

        Args:
            prompt: The input prompt
            tools: Optional list of tools/functions the model can use
            **kwargs: Additional parameters (overrides config)

        Returns:
            ModelResponse with generated content and metadata
        """
        async def _make_request() -> ModelResponse:
            async with measure_time() as timer:
                # Build request parameters
                request_params = {
                    "model": self.config.model,
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    **self.config.additional_params,
                }

                # Add tools if provided
                if tools:
                    request_params["tools"] = [
                        tool.to_anthropic_format() for tool in tools
                    ]

                response = await self.client.messages.create(**request_params)

                # Extract content (text blocks only)
                content_parts = []
                tool_calls = []

                for block in response.content:
                    if block.type == "text":
                        content_parts.append(block.text)
                    elif block.type == "tool_use":
                        # Capture tool call
                        tool_calls.append(
                            ToolCall(
                                id=block.id,
                                name=block.name,
                                arguments=block.input,
                            )
                        )

                content = "\n".join(content_parts) if content_parts else ""

                # Extract token usage
                prompt_tokens = response.usage.input_tokens
                completion_tokens = response.usage.output_tokens
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
                    tokens_used=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=timer.elapsed_ms,
                    cost_usd=cost,
                    timestamp=datetime.utcnow(),
                    tool_calls=tool_calls,
                    stop_reason=response.stop_reason,
                )

        try:
            return await retry_with_exponential_backoff(
                func=_make_request,
                max_attempts=3,
                initial_delay=1.0,
            )
        except Exception as e:
            logger.error(f"Anthropic request failed: {e}")
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
            "anthropic"
        """
        return "anthropic"

    def supports_tools(self) -> bool:
        """Check if this provider supports tool/function calling.

        Returns:
            True - Anthropic Claude models support tool calling
        """
        return True
