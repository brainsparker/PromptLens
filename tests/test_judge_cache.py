"""Tests for the local judge result cache."""

import json
from datetime import datetime

from promptlens.judges.cache import JudgeCache
from promptlens.models.config import JudgeConfig
from promptlens.models.result import JudgeScore, ModelResponse
from promptlens.models.test_case import TestCase as PLTestCase


def make_test_case(**overrides) -> PLTestCase:
    data = {
        "id": "tc-1",
        "query": "How do I reset my password?",
        "expected_behavior": "Provide clear instructions",
    }
    data.update(overrides)
    return PLTestCase(**data)


def make_response(content: str = "Step 1: open settings") -> ModelResponse:
    return ModelResponse(
        content=content,
        model="test-model",
        provider="test",
        latency_ms=10.0,
        cost_usd=0.001,
    )


def make_score(**overrides) -> JudgeScore:
    data = {
        "score": 4,
        "explanation": "Clear and accurate",
        "judge_model": "judge-model",
        "judge_provider": "anthropic",
        "timestamp": datetime.utcnow(),
        "cost_usd": 0.002,
    }
    data.update(overrides)
    return JudgeScore(**data)


def test_key_is_stable_for_identical_context() -> None:
    config = JudgeConfig()
    case = make_test_case()
    response = make_response()

    assert JudgeCache.key(config, case, response) == JudgeCache.key(
        config, case, response
    )


def test_key_changes_when_response_changes() -> None:
    config = JudgeConfig()
    case = make_test_case()

    key_a = JudgeCache.key(config, case, make_response("answer A"))
    key_b = JudgeCache.key(config, case, make_response("answer B"))

    assert key_a != key_b


def test_key_changes_when_judge_model_changes() -> None:
    case = make_test_case()
    response = make_response()

    key_a = JudgeCache.key(JudgeConfig(model="judge-a"), case, response)
    key_b = JudgeCache.key(JudgeConfig(model="judge-b"), case, response)

    assert key_a != key_b


def test_key_changes_when_expected_behavior_changes() -> None:
    config = JudgeConfig()
    response = make_response()

    key_a = JudgeCache.key(config, make_test_case(expected_behavior="be terse"), response)
    key_b = JudgeCache.key(config, make_test_case(expected_behavior="be verbose"), response)

    assert key_a != key_b


def test_put_get_roundtrip_marks_cached_and_zero_cost(tmp_path) -> None:
    cache = JudgeCache(tmp_path / "cache.json")
    score = make_score()

    cache.put("key-1", score)
    hit = cache.get("key-1")

    assert hit is not None
    assert hit.score == 4
    assert hit.cached is True
    assert hit.cost_usd == 0.0
    assert cache.hits == 1


def test_get_miss_returns_none(tmp_path) -> None:
    cache = JudgeCache(tmp_path / "cache.json")

    assert cache.get("missing") is None
    assert cache.misses == 1


def test_error_verdicts_are_not_cached(tmp_path) -> None:
    cache = JudgeCache(tmp_path / "cache.json")

    cache.put("key-1", make_score(error="judge timed out"))

    assert cache.get("key-1") is None


def test_flush_and_reload_roundtrip(tmp_path) -> None:
    path = tmp_path / "cache.json"
    cache = JudgeCache(path)
    cache.put("key-1", make_score())
    cache.flush()

    reloaded = JudgeCache(path)
    hit = reloaded.get("key-1")

    assert hit is not None
    assert hit.score == 4
    assert hit.cached is True


def test_flush_without_changes_writes_nothing(tmp_path) -> None:
    path = tmp_path / "cache.json"
    cache = JudgeCache(path)

    cache.flush()

    assert not path.exists()


def test_corrupt_cache_file_starts_fresh(tmp_path) -> None:
    path = tmp_path / "cache.json"
    path.write_text("not json at all {", encoding="utf-8")

    cache = JudgeCache(path)

    assert len(cache) == 0
    assert cache.get("anything") is None


def test_wrong_version_cache_starts_fresh(tmp_path) -> None:
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"version": 99, "entries": {"k": {}}}), encoding="utf-8")

    cache = JudgeCache(path)

    assert len(cache) == 0


def test_unreadable_entry_is_dropped(tmp_path) -> None:
    path = tmp_path / "cache.json"
    path.write_text(
        json.dumps({"version": 1, "entries": {"key-1": {"bogus": "entry"}}}),
        encoding="utf-8",
    )

    cache = JudgeCache(path)

    assert cache.get("key-1") is None
    assert len(cache) == 0
