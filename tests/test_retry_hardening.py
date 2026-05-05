import pytest

from promptlens.utils.retry import retry_with_exponential_backoff


@pytest.mark.asyncio
async def test_retry_rejects_non_positive_attempts() -> None:
    async def always_fails() -> str:
        raise RuntimeError("boom")

    with pytest.raises(ValueError, match="max_attempts"):
        await retry_with_exponential_backoff(always_fails, max_attempts=0)


@pytest.mark.asyncio
async def test_retry_rejects_negative_initial_delay() -> None:
    async def always_fails() -> str:
        raise RuntimeError("boom")

    with pytest.raises(ValueError, match="initial_delay"):
        await retry_with_exponential_backoff(always_fails, initial_delay=-0.1)


@pytest.mark.asyncio
async def test_retry_rejects_invalid_backoff_factor() -> None:
    async def always_fails() -> str:
        raise RuntimeError("boom")

    with pytest.raises(ValueError, match="backoff_factor"):
        await retry_with_exponential_backoff(always_fails, backoff_factor=0)


@pytest.mark.asyncio
async def test_retry_rejects_non_positive_max_delay() -> None:
    async def always_fails() -> str:
        raise RuntimeError("boom")

    with pytest.raises(ValueError, match="max_delay"):
        await retry_with_exponential_backoff(always_fails, max_delay=0)
