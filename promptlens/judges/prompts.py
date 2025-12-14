"""Judge prompt templates."""

from typing import Any, Dict, List

DEFAULT_JUDGE_PROMPT = """You are an expert evaluator assessing an LLM's response.

## Task
Evaluate how well the model's response meets the specified expectations.

## Query
{query}

## Expected Behavior
{expected_behavior}

## Actual Response
{response}

## Evaluation Criteria
Rate the response on a scale of 1-5:
- **1**: Completely incorrect, unhelpful, or fails to address the query
- **2**: Partially correct but has significant issues or omissions
- **3**: Mostly correct with minor issues; meets basic expectations
- **4**: Good response that meets expectations well
- **5**: Excellent response that exceeds expectations

## Your Task
Provide your evaluation in this exact format:

SCORE: [1-5]
EXPLANATION: [Your detailed explanation of why you gave this score]

Be objective and specific in your explanation. Consider:
- Accuracy: Is the information correct?
- Completeness: Does it address all parts of the query?
- Clarity: Is it well-structured and easy to understand?
- Relevance: Does it stay on topic?
"""


def format_judge_prompt(
    query: str,
    expected_behavior: str,
    response: str,
    custom_prompt: str = None,
) -> str:
    """Format the judge prompt with the given inputs.

    Args:
        query: The original query
        expected_behavior: What was expected from the model
        response: The actual response from the model
        custom_prompt: Optional custom prompt template

    Returns:
        Formatted prompt string
    """
    template = custom_prompt or DEFAULT_JUDGE_PROMPT

    return template.format(
        query=query,
        expected_behavior=expected_behavior,
        response=response,
    )


TOOL_EVALUATION_PROMPT = """You are an expert evaluator assessing an LLM's tool/function calling capabilities.

## Task
Evaluate how well the model used the available tools and provided a final answer.

## Query
{query}

## Expected Behavior
{expected_behavior}

## Available Tools
{tools_description}

## Expected Tool Calls
{expected_tool_calls}

## Actual Model Response
{response}

## Tool Calls Made
{tool_calls_made}

## Automatic Evaluation Results
{automatic_evaluation}

## Your Task
Evaluate the model's performance across multiple criteria:

### 1. Parameter Correctness (1-5)
Did the model fill tool parameters correctly?
- **5**: All parameters perfect
- **4**: Minor parameter issues
- **3**: Some parameters incorrect
- **2**: Many parameters wrong
- **1**: Completely wrong parameters

### 2. Tool Selection (1-5)
Did the model choose the right tool(s)?
- **5**: Perfect tool selection
- **4**: Right tools, minor inefficiency
- **3**: Mostly right tools
- **2**: Some wrong tools
- **1**: Completely wrong tools

### 3. Tool Efficiency (1-5)
Did the model use tools efficiently (minimum necessary calls, no redundancy)?
- **5**: Optimal efficiency
- **4**: Very efficient
- **3**: Acceptable efficiency
- **2**: Some wasteful calls
- **1**: Very inefficient

### 4. Final Answer Quality (1-5)
After using tools, was the final response good?
- **5**: Excellent answer
- **4**: Good answer
- **3**: Acceptable answer
- **2**: Poor answer
- **1**: Wrong/missing answer

### 5. Overall Score (1-5)
Holistic assessment considering all factors

## Response Format
Provide your evaluation in this exact format:

PARAMETER_CORRECTNESS: [1-5]
TOOL_SELECTION: [1-5]
TOOL_EFFICIENCY: [1-5]
FINAL_ANSWER: [1-5]
OVERALL_SCORE: [1-5]
EXPLANATION: [Your detailed explanation covering all criteria]

Be objective and specific. The automatic evaluation provides objective metrics - use them but also consider nuances like whether incorrect parameters would still work, efficiency of the approach, and quality of the natural language response.
"""


def format_tool_judge_prompt(
    query: str,
    expected_behavior: str,
    response: str,
    tools: List[Dict[str, Any]],
    expected_tool_calls: List[Dict[str, Any]],
    actual_tool_calls: List[Dict[str, Any]],
    automatic_evaluations: List[Dict[str, Any]],
    custom_prompt: str = None,
) -> str:
    """Format the tool evaluation judge prompt with all context.

    Args:
        query: The original query
        expected_behavior: What was expected from the model
        response: The actual text response from the model
        tools: Available tool definitions
        expected_tool_calls: Expected tool calls
        actual_tool_calls: Actual tool calls made by the model
        automatic_evaluations: Results from automatic evaluation
        custom_prompt: Optional custom prompt template

    Returns:
        Formatted prompt string
    """
    template = custom_prompt or TOOL_EVALUATION_PROMPT

    # Format tools description
    tools_desc_parts = []
    for tool in tools:
        params_desc = ", ".join(
            f"{name}: {schema.get('type', 'any')}"
            for name, schema in tool.get("parameters", {}).items()
        )
        tools_desc_parts.append(
            f"- {tool['name']}: {tool['description']} ({params_desc})"
        )
    tools_description = "\n".join(tools_desc_parts) if tools_desc_parts else "No tools available"

    # Format expected tool calls
    expected_calls_parts = []
    for i, call in enumerate(expected_tool_calls, 1):
        args_str = ", ".join(f"{k}={v}" for k, v in call.get("arguments", {}).items())
        expected_calls_parts.append(f"{i}. {call['name']}({args_str})")
    expected_calls_str = "\n".join(expected_calls_parts) if expected_calls_parts else "None specified"

    # Format actual tool calls
    actual_calls_parts = []
    if actual_tool_calls:
        for i, call in enumerate(actual_tool_calls, 1):
            args_str = ", ".join(f"{k}={v}" for k, v in call.get("arguments", {}).items())
            actual_calls_parts.append(f"{i}. {call['name']}({args_str})")
        actual_calls_str = "\n".join(actual_calls_parts)
    else:
        actual_calls_str = "No tool calls made"

    # Format automatic evaluation results
    auto_eval_parts = []
    for i, eval_result in enumerate(automatic_evaluations, 1):
        auto_eval_parts.append(
            f"{i}. Tool: {eval_result.get('expected_tool', 'N/A')} → {eval_result.get('actual_tool', 'Not called')}\n"
            f"   Correct tool: {eval_result.get('correct_tool', False)}\n"
            f"   Parameter accuracy: {eval_result.get('parameter_accuracy', 0.0):.1%}\n"
            f"   {eval_result.get('explanation', '')}"
        )
    auto_eval_str = "\n\n".join(auto_eval_parts) if auto_eval_parts else "No automatic evaluation performed"

    return template.format(
        query=query,
        expected_behavior=expected_behavior,
        response=response,
        tools_description=tools_description,
        expected_tool_calls=expected_calls_str,
        tool_calls_made=actual_calls_str,
        automatic_evaluation=auto_eval_str,
    )
