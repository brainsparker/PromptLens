"""LLM provider implementations."""

from promptlens.providers.base import BaseProvider


def get_provider(*args, **kwargs):
    """Lazily import and dispatch provider factory.

    Keeps package import lightweight for environments/tests that don't have every
    optional provider dependency installed.
    """
    from promptlens.providers.factory import get_provider as _get_provider

    return _get_provider(*args, **kwargs)


__all__ = ["BaseProvider", "get_provider"]
