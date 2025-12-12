"""Cost calculation utilities."""

from typing import Dict, Optional

# Pricing table in USD per million tokens
# Prices are approximate and should be updated periodically
PRICING_TABLE: Dict[str, Dict[str, Dict[str, float]]] = {
    "anthropic": {
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-5-sonnet-latest": {"input": 3.0, "output": 15.0},
        "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
        "claude-3-opus-latest": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "claude-3-haiku-latest": {"input": 0.25, "output": 1.25},
    },
    "openai": {
        "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-32k": {"input": 60.0, "output": 120.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "gpt-3.5-turbo-16k": {"input": 3.0, "output": 4.0},
        "o1-preview": {"input": 15.0, "output": 60.0},
        "o1-mini": {"input": 3.0, "output": 12.0},
    },
    "google": {
        "gemini-pro": {"input": 0.5, "output": 1.5},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-ultra": {"input": 2.0, "output": 6.0},
    },
}


def calculate_cost(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Calculate cost in USD for the given token usage.

    Args:
        provider: Provider name (e.g., "anthropic", "openai")
        model: Model identifier
        prompt_tokens: Number of prompt/input tokens
        completion_tokens: Number of completion/output tokens

    Returns:
        Estimated cost in USD (0.0 if pricing not available)
    """
    if provider not in PRICING_TABLE:
        return 0.0

    pricing = PRICING_TABLE[provider].get(model)
    if not pricing:
        return 0.0

    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


def get_pricing_info(provider: str, model: str) -> Optional[Dict[str, float]]:
    """Get pricing information for a specific provider and model.

    Args:
        provider: Provider name
        model: Model identifier

    Returns:
        Dict with 'input' and 'output' pricing per MTok, or None if not available
    """
    if provider not in PRICING_TABLE:
        return None
    return PRICING_TABLE[provider].get(model)
