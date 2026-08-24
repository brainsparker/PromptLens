"""Local judge result cache.

Caches LLM judge verdicts on disk, keyed by the full judging context
(judge model and settings, test case identity, and the exact response
text). Re-running an evaluation over unchanged responses reuses the
stored verdict instead of paying for another judge call.

The cache is a single JSON file, so it can be inspected, committed, or
deleted with normal tools. No cloud, no database.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from promptlens.models.config import JudgeConfig
from promptlens.models.result import JudgeScore, ModelResponse
from promptlens.models.test_case import TestCase

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 1


class JudgeCache:
    """JSON-file-backed cache of judge verdicts.

    Entries are keyed by a SHA-256 hash of the judging context. The cache
    is loaded once, mutated in memory during the run, and flushed to disk
    at the end of the run.
    """

    def __init__(self, path: Path) -> None:
        """Initialize the cache.

        Args:
            path: Path to the cache JSON file
        """
        self.path = Path(path)
        self.hits = 0
        self.misses = 0
        self._dirty = False
        self._entries: Dict[str, dict] = self._load()

    def _load(self) -> Dict[str, dict]:
        """Load cache entries from disk, tolerating missing/corrupt files."""
        if not self.path.exists():
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if (
                not isinstance(data, dict)
                or data.get("version") != CACHE_FORMAT_VERSION
                or not isinstance(data.get("entries"), dict)
            ):
                logger.warning(
                    f"Judge cache at {self.path} has an unexpected format; starting fresh"
                )
                return {}
            return data["entries"]
        except (ValueError, OSError) as e:
            logger.warning(f"Failed to read judge cache at {self.path}: {e}; starting fresh")
            return {}

    @staticmethod
    def key(
        judge_config: JudgeConfig,
        test_case: TestCase,
        model_response: ModelResponse,
    ) -> str:
        """Compute the cache key for one judging context.

        The key covers everything that changes a judge verdict: the judge
        model and its settings, the test case identity and expectations,
        and the exact response under evaluation (text and tool calls).

        Args:
            judge_config: Active judge configuration
            test_case: The test case being judged
            model_response: The response being judged

        Returns:
            Hex SHA-256 digest string
        """
        payload = {
            "v": CACHE_FORMAT_VERSION,
            "judge_provider": judge_config.provider,
            "judge_model": judge_config.model,
            "judge_temperature": judge_config.temperature,
            "custom_prompt": judge_config.custom_prompt,
            "criteria": list(judge_config.criteria),
            "test_case_id": test_case.id,
            "query": test_case.query,
            "expected_behavior": test_case.expected_behavior,
            "evaluation_mode": test_case.evaluation_mode,
            "expected_tool_calls": [
                {"name": ec.name, "arguments": ec.arguments}
                for ec in test_case.expected_tool_calls
            ],
            "response_content": model_response.content,
            "response_tool_calls": [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in model_response.tool_calls
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[JudgeScore]:
        """Look up a cached verdict.

        Args:
            key: Cache key from JudgeCache.key()

        Returns:
            A JudgeScore marked as cached (with zero cost for this run),
            or None on a miss or an unreadable entry
        """
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        try:
            score = JudgeScore(**entry)
        except Exception as e:
            logger.warning(f"Dropping unreadable judge cache entry: {e}")
            self._entries.pop(key, None)
            self._dirty = True
            self.misses += 1
            return None
        self.hits += 1
        # The verdict was paid for in a previous run; this run spends nothing.
        return score.model_copy(update={"cached": True, "cost_usd": 0.0})

    def put(self, key: str, score: JudgeScore) -> None:
        """Store a verdict. Verdicts that carry an error are never cached.

        Args:
            key: Cache key from JudgeCache.key()
            score: The judge verdict to store
        """
        if score.error:
            return
        self._entries[key] = score.model_dump(mode="json")
        self._dirty = True

    def flush(self) -> None:
        """Write the cache to disk if anything changed."""
        if not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": CACHE_FORMAT_VERSION, "entries": self._entries}
            tmp_path = self.path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, default=str)
            tmp_path.replace(self.path)
            self._dirty = False
        except OSError as e:
            logger.warning(f"Failed to write judge cache to {self.path}: {e}")

    def __len__(self) -> int:
        return len(self._entries)
