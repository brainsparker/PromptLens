"""Validation hardening tests for TrajectorySpec and golden set loading."""

import pytest
from pydantic import ValidationError

from promptlens.models.test_case import GoldenSet, TestCase
from promptlens.models.trajectory import TrajectorySpec


class TestTrajectorySpecValidation:
    def test_empty_spec_rejected(self):
        with pytest.raises(ValidationError, match="at least one check"):
            TrajectorySpec()

    def test_min_greater_than_max_rejected(self):
        with pytest.raises(ValidationError, match="cannot exceed"):
            TrajectorySpec(min_calls=5, max_calls=2)

    def test_min_equal_max_allowed(self):
        spec = TrajectorySpec(min_calls=2, max_calls=2)
        assert spec.min_calls == 2

    def test_negative_counts_rejected(self):
        with pytest.raises(ValidationError):
            TrajectorySpec(max_calls=-1)
        with pytest.raises(ValidationError):
            TrajectorySpec(min_calls=-1)

    def test_single_item_order_rejected(self):
        with pytest.raises(ValidationError, match="at least two tool names"):
            TrajectorySpec(order=[["only_one"]])

    def test_empty_string_entries_rejected(self):
        with pytest.raises(ValidationError, match="non-empty"):
            TrajectorySpec(forbid=[""])
        with pytest.raises(ValidationError, match="non-empty"):
            TrajectorySpec(require=["  "])
        with pytest.raises(ValidationError, match="non-empty"):
            TrajectorySpec(order=[["a", ""]])

    def test_no_repeat_calls_alone_is_a_valid_spec(self):
        spec = TrajectorySpec(no_repeat_calls=True)
        assert spec.no_repeat_calls

    def test_normalized_require_converts_strings(self):
        spec = TrajectorySpec(require=["a", {"name": "b", "args": {"k": 1}}])
        normalized = spec.normalized_require()
        assert normalized[0].name == "a"
        assert normalized[0].args == {}
        assert normalized[1].name == "b"
        assert normalized[1].args == {"k": 1}


class TestGoldenSetTrajectoryParsing:
    def test_test_case_without_trajectory_still_valid(self):
        case = TestCase(id="t1", query="q", expected_behavior="b")
        assert case.trajectory is None

    def test_golden_set_parses_trajectory_from_dict(self):
        data = {
            "name": "Trajectory tests",
            "test_cases": [
                {
                    "id": "refund-001",
                    "query": "Refund order 123",
                    "expected_behavior": "Verify identity before refunding",
                    "trajectory": {
                        "require": [
                            "verify_identity",
                            {"name": "issue_refund", "args": {"order_id": "123"}},
                        ],
                        "forbid": ["delete_account"],
                        "order": [["verify_identity", "issue_refund"]],
                        "max_calls": 4,
                    },
                }
            ],
        }
        golden_set = GoldenSet(**data)
        spec = golden_set.test_cases[0].trajectory
        assert spec is not None
        assert spec.forbid == ["delete_account"]
        assert spec.max_calls == 4
        assert spec.normalized_require()[1].args == {"order_id": "123"}

    def test_golden_set_rejects_empty_trajectory_block(self):
        data = {
            "name": "Bad set",
            "test_cases": [
                {
                    "id": "t1",
                    "query": "q",
                    "expected_behavior": "b",
                    "trajectory": {},
                }
            ],
        }
        with pytest.raises(ValidationError, match="at least one check"):
            GoldenSet(**data)

    def test_golden_set_rejects_invalid_order_shape(self):
        data = {
            "name": "Bad set",
            "test_cases": [
                {
                    "id": "t1",
                    "query": "q",
                    "expected_behavior": "b",
                    "trajectory": {"order": [["solo"]]},
                }
            ],
        }
        with pytest.raises(ValidationError, match="at least two tool names"):
            GoldenSet(**data)
