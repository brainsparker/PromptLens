"""Judge prompt templates."""

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
