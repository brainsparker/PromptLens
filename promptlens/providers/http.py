"""Generic HTTP provider for local models (Ollama, LM Studio, etc.)."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from promptlens.models.config import ProviderConfig
from promptlens.models.result import ModelResponse
from promptlens.models.tools import ToolDefinition
from promptlens.providers.base import BaseProvider
from promptlens.utils.retry import retry_with_exponential_backoff
from promptlens.utils.timing import measure_time

logger = logging.getLogger(__name__)


class HTTPProvider(BaseProvider):
    """Generic HTTP provider for local models.

    Supports Ollama, LM Studio, and any other HTTP-based inference server.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize the HTTP provider.

        Args:
            config: Provider configuration with endpoint URL
        """
        super().__init__(config)

        if not config.endpoint:
            raise ValueError("HTTP provider requires an endpoint URL in the config")

        self.endpoint = config.endpoint

    async def generate(
        self,
        prompt: str,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any
    ) -> ModelResponse:
        """Generate a response from the HTTP endpoint.

        Args:
            prompt: The input prompt
            tools: Optional list of tools (not supported yet, will be ignored with warning)
            **kwargs: Additional parameters

        Returns:
            ModelResponse with generated content and metadata
        """
        # Log warning if tools are provided (not yet supported for generic HTTP)
        if tools:
            logger.warning(
                f"{self.provider_name} provider does not support tool calling yet. "
                "Tools will be ignored. Consider using Anthropic or OpenAI providers for tool evaluation."
            )

        async def _make_request() -> ModelResponse:
            async with measure_time() as timer:
                # Build request payload
                # Default format works for Ollama - can be customized via additional_params
                payload: Dict[str, Any] = {
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    **self.config.additional_params,
                }

                # Add temperature and max_tokens if not in additional_params
                if "temperature" not in payload:
                    payload["temperature"] = kwargs.get("temperature", self.config.temperature)
                if "max_tokens" not in payload and "num_predict" not in payload:
                    payload["num_predict"] = kwargs.get("max_tokens", self.config.max_tokens)

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.endpoint,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()

                # Extract content (try common response formats)
                content = ""
                if "response" in data:
                    content = data["response"]  # Ollama format
                elif "text" in data:
                    content = data["text"]
                elif "content" in data:
                    content = data["content"]
                elif "choices" in data and len(data["choices"]) > 0:
                    # OpenAI-compatible format
                    content = data["choices"][0].get("text", "")

                # Local models typically don't provide token counts or cost
                return ModelResponse(
                    content=content,
                    model=self.config.model,
                    provider=self.provider_name,
                    tokens_used=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    latency_ms=timer.elapsed_ms,
                    cost_usd=0.0,  # Local models are free
                    timestamp=datetime.utcnow(),
                )

        try:
            return await retry_with_exponential_backoff(
                func=_make_request,
                max_attempts=3,
                initial_delay=1.0,
            )
        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
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
        """Estimate cost for local models.

        Args:
            prompt_tokens: Number of prompt tokens (ignored)
            completion_tokens: Number of completion tokens (ignored)

        Returns:
            Always 0.0 for local models
        """
        return 0.0

    @property
    def provider_name(self) -> str:
        """Return the provider name.

        Returns:
            "http"
        """
        return "http"
