"""Retry logic with exponential backoff."""

import asyncio
import logging
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
) -> T:
    """Retry a function with exponential backoff.

    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each retry
        max_delay: Maximum delay between retries in seconds
        retry_on: Tuple of exception types to retry on

    Returns:
        Result from the function

    Raises:
        Exception: The last exception if all retries fail
    """
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

            logger.warning(
                f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)

    if last_exception:
        raise last_exception
