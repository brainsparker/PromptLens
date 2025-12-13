"""
Tool-related data models for PromptLens tool/function calling evaluation.

This module defines the data structures needed to:
1. Define tools/functions that LLMs can use
2. Capture tool calls from LLM responses
3. Evaluate tool usage correctness and efficiency
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """
    Schema for a single parameter in a tool definition.

    Follows JSON Schema conventions used by OpenAI and Anthropic.
    """
    type: str = Field(..., description="Parameter type: string, number, boolean, object, array")
    description: str = Field(..., description="Human-readable description of the parameter")
    enum: Optional[List[Any]] = Field(None, description="Allowed values for this parameter")
    required: bool = Field(False, description="Whether this parameter is required")
    properties: Optional[Dict[str, "ToolParameter"]] = Field(None, description="For object types, nested properties")
    items: Optional["ToolParameter"] = Field(None, description="For array types, the item schema")

    class Config:
        extra = "allow"  # Allow additional JSON Schema fields


class ToolDefinition(BaseModel):
    """
    Complete definition of a tool/function that an LLM can call.

    Provides methods to convert to provider-specific formats (Anthropic, OpenAI).
    """
    name: str = Field(..., description="Unique identifier for the tool")
    description: str = Field(..., description="What this tool does (shown to LLM)")
    parameters: Dict[str, ToolParameter] = Field(
        default_factory=dict,
        description="Map of parameter name to parameter schema"
    )

    def to_anthropic_format(self) -> Dict[str, Any]:
        """
        Convert to Anthropic's tool calling format.

        Anthropic format:
        {
            "name": "get_weather",
            "description": "Get weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
        """
        properties = {}
        required = []

        for param_name, param_schema in self.parameters.items():
            param_dict = {
                "type": param_schema.type,
                "description": param_schema.description
            }

            if param_schema.enum:
                param_dict["enum"] = param_schema.enum

            if param_schema.properties:
                param_dict["properties"] = {
                    k: v.model_dump(exclude_none=True)
                    for k, v in param_schema.properties.items()
                }

            if param_schema.items:
                param_dict["items"] = param_schema.items.model_dump(exclude_none=True)

            properties[param_name] = param_dict

            if param_schema.required:
                required.append(param_name)

        input_schema = {
            "type": "object",
            "properties": properties
        }

        if required:
            input_schema["required"] = required

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": input_schema
        }

    def to_openai_format(self) -> Dict[str, Any]:
        """
        Convert to OpenAI's function calling format.

        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }
        """
        properties = {}
        required = []

        for param_name, param_schema in self.parameters.items():
            param_dict = {
                "type": param_schema.type,
                "description": param_schema.description
            }

            if param_schema.enum:
                param_dict["enum"] = param_schema.enum

            if param_schema.properties:
                param_dict["properties"] = {
                    k: v.model_dump(exclude_none=True)
                    for k, v in param_schema.properties.items()
                }

            if param_schema.items:
                param_dict["items"] = param_schema.items.model_dump(exclude_none=True)

            properties[param_name] = param_dict

            if param_schema.required:
                required.append(param_name)

        parameters = {
            "type": "object",
            "properties": properties
        }

        if required:
            parameters["required"] = required

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters
            }
        }


class ToolCall(BaseModel):
    """
    A tool call made by an LLM in its response.

    Captured from provider responses (Anthropic, OpenAI, etc.)
    """
    id: str = Field(..., description="Unique identifier for this tool call")
    name: str = Field(..., description="Name of the tool being called")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed arguments passed to the tool"
    )
    raw_arguments: Optional[str] = Field(
        None,
        description="Raw JSON string of arguments (if available)"
    )


class ExpectedToolCall(BaseModel):
    """
    Expected tool call for test case evaluation.

    Defines what tool call(s) we expect the LLM to make for a given query.
    """
    name: str = Field(..., description="Name of the expected tool")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Expected arguments (exact or semantic match)"
    )
    allow_extra_params: bool = Field(
        False,
        description="Whether to allow parameters not in expected_arguments"
    )
    strict_matching: bool = Field(
        False,
        description="Use strict equality vs semantic similarity for values"
    )
    optional: bool = Field(
        False,
        description="Whether this tool call is optional (won't penalize if missing)"
    )


class ToolCallEvaluation(BaseModel):
    """
    Evaluation result for a single tool call.

    Combines automatic comparison (expected vs actual) with evaluation metrics.
    """
    tool_call_index: int = Field(..., description="Index of this tool call in the response")
    expected_tool: str = Field(..., description="Name of expected tool")
    actual_tool: Optional[str] = Field(None, description="Name of actual tool called")

    # Correctness metrics
    correct_tool: bool = Field(..., description="Whether the correct tool was called")
    parameter_accuracy: float = Field(
        ...,
        description="0.0-1.0 score for parameter correctness"
    )

    # Parameter analysis
    missing_parameters: List[str] = Field(
        default_factory=list,
        description="Required parameters that were not provided"
    )
    incorrect_parameters: List[str] = Field(
        default_factory=list,
        description="Parameters with incorrect values"
    )
    extra_parameters: List[str] = Field(
        default_factory=list,
        description="Parameters provided but not expected"
    )

    # Detailed feedback
    parameter_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed comparison for each parameter"
    )
    explanation: str = Field(
        default="",
        description="Human-readable explanation of the evaluation"
    )


# Update forward references for recursive models
ToolParameter.model_rebuild()
