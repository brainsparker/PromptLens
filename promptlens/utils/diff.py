"""Text diff utilities for comparing responses."""

from difflib import SequenceMatcher, unified_diff
from typing import List, Tuple


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity ratio between 0.0 and 1.0
    """
    return SequenceMatcher(None, text1, text2).ratio()


def get_unified_diff(
    text1: str,
    text2: str,
    label1: str = "Expected",
    label2: str = "Actual",
    n_context_lines: int = 3,
) -> List[str]:
    """Generate unified diff between two texts.

    Args:
        text1: First text (expected)
        text2: Second text (actual)
        label1: Label for first text
        label2: Label for second text
        n_context_lines: Number of context lines in diff

    Returns:
        List of diff lines
    """
    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)

    diff = unified_diff(
        lines1,
        lines2,
        fromfile=label1,
        tofile=label2,
        n=n_context_lines,
    )
    return list(diff)


def highlight_differences(text1: str, text2: str) -> Tuple[str, str]:
    """Highlight differences between two texts with markers.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Tuple of (marked_text1, marked_text2) with <<markers>> around differences
    """
    matcher = SequenceMatcher(None, text1, text2)
    marked1 = []
    marked2 = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            marked1.append(text1[i1:i2])
            marked2.append(text2[j1:j2])
        elif tag == "replace":
            marked1.append(f"<<{text1[i1:i2]}>>")
            marked2.append(f"<<{text2[j1:j2]}>>")
        elif tag == "delete":
            marked1.append(f"<<{text1[i1:i2]}>>")
        elif tag == "insert":
            marked2.append(f"<<{text2[j1:j2]}>>")

    return "".join(marked1), "".join(marked2)
