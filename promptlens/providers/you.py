"""You.com AI API provider implementation."""

import logging
import os
from datetime import datetime
from typing import Any, List, Optional

import aiohttp

from promptlens.models.config import ProviderConfig
from promptlens.models.result import ModelResponse
from promptlens.models.tools import ToolDefinition
from promptlens.providers.base import BaseProvider
from promptlens.utils.retry import retry_with_exponential_backoff
from promptlens.utils.timing import measure_time

logger = logging.getLogger(__name__)


class YouProvider(BaseProvider):
    """Provider for You.com AI API.

    Supports various models through You.com's unified API including
    GPT-4, Claude, Llama, and more.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize the You.com provider.

        Args:
            config: Provider configuration
        """
        super().__init__(config)

        # Get API key from config or environment
        api_key = config.api_key or os.getenv("YOU_API_KEY")
        if not api_key:
            raise ValueError(
                "You.com API key not found. Set YOU_API_KEY environment variable "
                "or provide it in the config."
            )

        self.api_key = api_key
        self.base_url = "https://api.you.com/smart-api"

    async def generate(
        self,
        prompt: str,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any
    ) -> ModelResponse:
        """Generate a response from You.com AI.

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
                headers = {
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json",
                }

                payload = {
                    "query": prompt,
                    "chat_model": self.config.model,
                    **self.config.additional_params,
                }

                # Add optional parameters if not in additional_params
                if "temperature" not in payload:
                    payload["temperature"] = kwargs.get("temperature", self.config.temperature)
                if "max_tokens" not in payload:
                    payload["max_tokens"] = kwargs.get("max_tokens", self.config.max_tokens)

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()

                # Extract content from You.com response format
                content = ""
                if "answer" in data:
                    content = data["answer"]
                elif "message" in data:
                    content = data["message"]
                elif "text" in data:
                    content = data["text"]

                # You.com doesn't typically provide token counts in standard responses
                # But we can estimate or leave as None
                prompt_tokens = None
                completion_tokens = None
                total_tokens = None

                # You.com pricing varies by model - return 0 for now
                # Users can track via their You.com dashboard
                cost = 0.0

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
                )

        try:
            return await retry_with_exponential_backoff(
                func=_make_request,
                max_attempts=3,
                initial_delay=1.0,
            )
        except Exception as e:
            logger.error(f"You.com request failed: {e}")
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
        """Estimate cost for You.com API.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Cost estimate (0.0 as pricing varies by plan)
        """
        # You.com pricing varies by subscription plan and model
        # Users should track via their dashboard
        return 0.0

    @property
    def provider_name(self) -> str:
        """Return the provider name.

        Returns:
            "you"
        """
        return "you"
