"""Regression tests for Pydantic v2 model_config migration."""

from promptlens.models.config import RunConfig
from promptlens.models.test_case import GoldenSet, TestCase as PromptLensTestCase
from promptlens.models.tools import ToolParameter


def test_tool_parameter_allows_additional_json_schema_fields() -> None:
    param = ToolParameter(
        type="string",
        description="User locale",
        format="email",
        minLength=3,
    )

    dumped = param.model_dump()
    assert dumped["format"] == "email"
    assert dumped["minLength"] == 3


def test_test_case_schema_keeps_example_metadata() -> None:
    schema = PromptLensTestCase.model_json_schema()

    assert "example" in schema
    assert schema["example"]["id"] == "cs-001"


def test_run_config_schema_keeps_example_metadata() -> None:
    schema = RunConfig.model_json_schema()

    assert "example" in schema
    assert schema["example"]["golden_set"].endswith("customer_support.yaml")


def test_golden_set_schema_keeps_example_metadata() -> None:
    schema = GoldenSet.model_json_schema()

    assert "example" in schema
    assert schema["example"]["name"] == "Customer Support Tests"
