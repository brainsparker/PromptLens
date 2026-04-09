"""Config loading helpers for CLI commands."""

from typing import Any, Dict, Optional

import yaml


def load_config_data(config_path: str) -> Dict[str, Any]:
    """Load and validate raw config data from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    if config_data is None:
        raise ValueError(
            "Configuration file is empty. "
            "Provide a YAML mapping with at least 'golden_set' and 'models'."
        )

    if not isinstance(config_data, dict):
        raise ValueError(
            "Configuration root must be a YAML mapping/object "
            f"(got {type(config_data).__name__})."
        )

    return config_data


def apply_run_overrides(
    config_data: Dict[str, Any],
    golden_set: Optional[str],
    output_dir: Optional[str],
) -> Dict[str, Any]:
    """Apply CLI overrides to loaded config data."""
    merged = dict(config_data)

    if golden_set:
        merged["golden_set"] = golden_set

    if output_dir:
        output_config = merged.get("output")
        if output_config is None:
            merged["output"] = {"directory": output_dir}
        elif isinstance(output_config, dict):
            output_config["directory"] = output_dir
        else:
            raise ValueError("'output' section must be a mapping/object if provided.")

    return merged
