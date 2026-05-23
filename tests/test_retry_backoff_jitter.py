import pytest

from promptlens.utils import retry as retry_module
from promptlens.utils.retry import retry_with_exponential_backoff


@pytest.mark.asyncio
async def test_retry_applies_jitter_to_sleep(monkeypatch):
    calls = {"count": 0}
    sleeps = []

    async def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("boom")
        return "ok"

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(retry_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(retry_module.random, "uniform", lambda a, b: 0.1)

    result = await retry_with_exponential_backoff(
        flaky,
        max_attempts=3,
        initial_delay=1.0,
        backoff_factor=2.0,
        jitter_ratio=0.1,
    )

    assert result == "ok"
    assert sleeps == [1.1, 2.2]


@pytest.mark.asyncio
async def test_retry_negative_jitter_ratio_is_clamped(monkeypatch):
    calls = {"count": 0}
    sleeps = []

    async def flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise ValueError("boom")
        return "ok"

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(retry_module.asyncio, "sleep", fake_sleep)

    result = await retry_with_exponential_backoff(
        flaky,
        max_attempts=2,
        initial_delay=1.0,
        jitter_ratio=-0.5,
    )

    assert result == "ok"
    assert sleeps == [1.0]
