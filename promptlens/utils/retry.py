"""Retry logic with exponential backoff."""

import asyncio
import logging
import random
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def retry_with_exponential_backoff(
    func: Callable[[], Any],
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    jitter_ratio: float = 0.1,
) -> T:
    """Retry a function with exponential backoff.

    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each retry
        max_delay: Maximum delay between retries in seconds
        retry_on: Tuple of exception types to retry on
        jitter_ratio: Random jitter as a fraction of delay (0.1 = +/-10%)

    Returns:
        Result from the function

    Raises:
        Exception: The last exception if all retries fail
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    if initial_delay < 0:
        raise ValueError("initial_delay must be >= 0")

    if backoff_factor <= 0:
        raise ValueError("backoff_factor must be > 0")

    if max_delay <= 0:
        raise ValueError("max_delay must be > 0")

    delay = initial_delay
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return await func()
        except retry_on as e:
            last_exception = e

            if attempt == max_attempts - 1:
                logger.error(f"All {max_attempts} attempts failed. Last error: {e}")
                raise

            jitter_ratio = max(0.0, jitter_ratio)
            jitter = random.uniform(-jitter_ratio, jitter_ratio) if jitter_ratio else 0.0
            sleep_for = max(0.0, delay * (1 + jitter))

            logger.warning(
                f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                f"Retrying in {sleep_for:.2f}s..."
            )
            await asyncio.sleep(sleep_for)
            delay = min(delay * backoff_factor, max_delay)

    if last_exception:
        raise last_exception
