"""Tests for output format validation hardening."""

import pytest
from pydantic import ValidationError

from promptlens.models.config import OutputConfig


def test_output_formats_are_normalized() -> None:
    config = OutputConfig(formats=[" JSON ", "Html", "md"])

    assert config.formats == ["json", "html", "md"]


def test_output_formats_reject_unsupported_values() -> None:
    with pytest.raises(ValidationError, match="Unsupported output formats"):
        OutputConfig(formats=["json", "xml"])
