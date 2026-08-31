"""Factory for creating judge instances from configuration."""

from promptlens.judges.base import BaseJudge
from promptlens.judges.llm_judge import LLMJudge
from promptlens.judges.multi_judge import MultiJudge
from promptlens.models.config import JudgeConfig


def create_judge(config: JudgeConfig) -> BaseJudge:
    """Create the appropriate judge for a configuration.

    Returns a MultiJudge consensus panel when the config lists multiple
    judges, otherwise a single LLMJudge (fully backward compatible).

    Args:
        config: Judge configuration

    Returns:
        A BaseJudge implementation
    """
    if config.judges:
        return MultiJudge(config)
    return LLMJudge(config)
