"""Judge factory for creating judge instances from configuration."""

from promptlens.judges.base import BaseJudge
from promptlens.judges.deterministic_judge import DeterministicJudge
from promptlens.judges.llm_judge import LLMJudge
from promptlens.models.config import JudgeConfig


def get_judge(config: JudgeConfig) -> BaseJudge:
    """Create a judge from its configuration.

    Args:
        config: Judge configuration; config.type selects the implementation

    Returns:
        Initialized judge

    Raises:
        ValueError: If the judge type is not supported
    """
    if config.type == "deterministic":
        return DeterministicJudge(config)
    if config.type == "llm":
        return LLMJudge(config)
    raise ValueError(f"Unsupported judge type: {config.type}")
