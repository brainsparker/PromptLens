"""Test case and golden set data models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
    """

    id: str
    query: str
    expected_behavior: str
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    reference_answer: Optional[str] = None

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "id": "cs-001",
                "query": "How do I reset my password?",
                "expected_behavior": "Provide clear step-by-step instructions",
                "category": "account_management",
                "tags": ["password", "account"],
            }
        }


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

    class Config:
        """Pydantic config."""
        json_schema_extra = {
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
        }
