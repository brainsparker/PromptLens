"""Timing and latency tracking utilities."""

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator


class Timer:
    """Simple timer for measuring elapsed time.

    Example:
        timer = Timer()
        timer.start()
        # ... do work ...
        elapsed_ms = timer.stop()
    """

    def __init__(self) -> None:
        """Initialize the timer."""
        self._start_time: float = 0.0
        self._end_time: float = 0.0

    def start(self) -> None:
        """Start the timer."""
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer and return elapsed time.

        Returns:
            Elapsed time in milliseconds
        """
        self._end_time = time.perf_counter()
        return self.elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds.

        Returns:
            Elapsed time in milliseconds
        """
        if self._end_time == 0.0:
            # Timer still running
            return (time.perf_counter() - self._start_time) * 1000
        return (self._end_time - self._start_time) * 1000


@asynccontextmanager
async def measure_time() -> AsyncIterator[Timer]:
    """Context manager for measuring execution time.

    Example:
        async with measure_time() as timer:
            await some_async_function()
        print(f"Took {timer.elapsed_ms}ms")

    Yields:
        Timer instance that tracks elapsed time
    """
    timer = Timer()
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()
