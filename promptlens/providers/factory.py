"""Provider factory for creating provider instances."""

from typing import Dict, Type

from promptlens.models.config import ModelConfig, ProviderConfig
from promptlens.providers.anthropic import AnthropicProvider
from promptlens.providers.base import BaseProvider
from promptlens.providers.google import GoogleProvider
from promptlens.providers.http import HTTPProvider
from promptlens.providers.openai import OpenAIProvider
from promptlens.providers.you import YouProvider

# Registry of available providers
PROVIDER_REGISTRY: Dict[str, Type[BaseProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "google": GoogleProvider,
    "http": HTTPProvider,
    "you": YouProvider,
}


def get_provider(model_config: ModelConfig) -> BaseProvider:
    """Create a provider instance from model configuration.

    Args:
        model_config: Model configuration

    Returns:
        Initialized provider instance

    Raises:
        ValueError: If provider is not supported
    """
    provider_name = model_config.provider.lower()

    if provider_name not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Provider '{provider_name}' not supported. "
            f"Available providers: {available}"
        )

    # Convert ModelConfig to ProviderConfig
    provider_config = ProviderConfig(
        name=provider_name,
        model=model_config.model,
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
        additional_params=model_config.additional_params,
    )

    # Get provider class and instantiate
    provider_class = PROVIDER_REGISTRY[provider_name]
    return provider_class(provider_config)


def register_provider(name: str, provider_class: Type[BaseProvider]) -> None:
    """Register a custom provider.

    Args:
        name: Provider name
        provider_class: Provider class (must inherit from BaseProvider)
    """
    if not issubclass(provider_class, BaseProvider):
        raise ValueError(f"Provider class must inherit from BaseProvider")

    PROVIDER_REGISTRY[name.lower()] = provider_class
