"""Test case and golden set data models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from promptlens.models.tools import ToolDefinition, ExpectedToolCall

# Assertion types that require a value, and the Python type(s) the value must have.
ASSERTION_VALUE_TYPES: Dict[str, tuple] = {
    "json_schema": (dict,),
    "contains": (str,),
    "not_contains": (str,),
    "regex": (str,),
    "starts_with": (str,),
}

SUPPORTED_ASSERTION_TYPES = ("is_json",) + tuple(ASSERTION_VALUE_TYPES.keys())


class Assertion(BaseModel):
    """A deterministic check applied to a model response before LLM judging.

    Assertions cost zero tokens: they run locally and, when any of them
    fails, the LLM judge call is skipped entirely for that test case.

    Attributes:
        type: Assertion type. One of: is_json, json_schema, contains,
            not_contains, regex, starts_with
        value: Value for the check. Required for every type except is_json:
            a JSON Schema dict for json_schema, a string for the rest.
    """

    type: str
    value: Optional[Any] = None

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "type": "contains",
                "value": "reset link",
            }
        })

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Ensure the assertion type is supported."""
        if v not in SUPPORTED_ASSERTION_TYPES:
            raise ValueError(
                f"Unsupported assertion type '{v}'. "
                f"Supported types: {', '.join(SUPPORTED_ASSERTION_TYPES)}"
            )
        return v

    @model_validator(mode="after")
    def validate_value(self) -> "Assertion":
        """Ensure the value is present and correctly typed for the assertion type."""
        expected = ASSERTION_VALUE_TYPES.get(self.type)
        if expected is None:
            return self
        if self.value is None:
            raise ValueError(f"Assertion type '{self.type}' requires a value")
        if not isinstance(self.value, expected):
            type_names = " or ".join(t.__name__ for t in expected)
            raise ValueError(
                f"Assertion type '{self.type}' requires a {type_names} value, "
                f"got {type(self.value).__name__}"
            )
        return self


class TestCase(BaseModel):
    """A single test case from the golden set.

    Attributes:
        id: Unique identifier for the test case
        query: The input prompt/query to test
        expected_behavior: Description of what the model should do
        category: Optional category for grouping (e.g., "summarization", "coding")
        tags: List of tags for filtering and organization
        metadata: Additional arbitrary metadata
        reference_answer: Optional reference answer for comparison
        tools: Tools/functions available to the LLM for this test case
        expected_tool_calls: Expected tool calls the LLM should make
        evaluation_mode: Evaluation mode (standard/tool_only/tool_and_answer)
        tool_execution: Whether to actually execute tools (default: False)
        assertions: Deterministic checks run before the LLM judge. Declared
            with the "assert" key in YAML/JSON golden sets. When any
            assertion fails, the judge call is skipped for that case.
    """

    id: str
    query: str
    expected_behavior: str
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    reference_answer: Optional[str] = None

    # Tool calling evaluation fields (optional, for backward compatibility)
    tools: List[ToolDefinition] = Field(
        default_factory=list,
        description="Tools/functions available to the LLM for this test case"
    )
    expected_tool_calls: List[ExpectedToolCall] = Field(
        default_factory=list,
        description="Expected tool calls the LLM should make"
    )
    evaluation_mode: str = Field(
        default="standard",
        description="Evaluation mode: 'standard' (no tools), 'tool_only' (only tool usage), or 'tool_and_answer' (both)"
    )
    tool_execution: bool = Field(
        default=False,
        description="Whether to actually execute tools (default: False, evaluation only)"
    )

    # Deterministic assertions (optional, for backward compatibility)
    assertions: List[Assertion] = Field(
        default_factory=list,
        alias="assert",
        description=(
            "Deterministic checks run against the response before the LLM judge. "
            "Declared with the 'assert' key in golden sets."
        ),
    )

    model_config = ConfigDict(populate_by_name=True, json_schema_extra={
            "example": {
                "id": "cs-001",
                "query": "How do I reset my password?",
                "expected_behavior": "Provide clear step-by-step instructions",
                "category": "account_management",
                "tags": ["password", "account"],
            }
        })


class GoldenSet(BaseModel):
    """Collection of test cases.

    Attributes:
        name: Name of the golden set
        description: Optional description of the test suite
        version: Version string for the golden set
        test_cases: List of test cases
        metadata: Additional arbitrary metadata
    """

    name: str
    description: Optional[str] = None
    version: str = "1.0"
    test_cases: List[TestCase]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "name": "Customer Support Tests",
                "description": "Test cases for customer support chatbot",
                "version": "1.0",
                "test_cases": [
                    {
                        "id": "cs-001",
                        "query": "How do I reset my password?",
                        "expected_behavior": "Provide clear instructions",
                        "category": "account",
                        "tags": ["password"],
                    }
                ],
            }
        })
