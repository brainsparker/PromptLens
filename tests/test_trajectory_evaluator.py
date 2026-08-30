"""Tests for the deterministic trajectory evaluator."""


from promptlens.evaluators.trajectory import evaluate_trajectory
from promptlens.models.tools import ToolCall
from promptlens.models.trajectory import TrajectorySpec, TrajectoryToolRef


def _call(name, arguments=None, call_id="call-1"):
    return ToolCall(id=call_id, name=name, arguments=arguments or {})


class TestRequireCheck:
    def test_require_passes_when_tool_called(self):
        spec = TrajectorySpec(require=["get_weather"])
        result = evaluate_trajectory(spec, [_call("get_weather")])
        assert result.passed
        assert result.checks[0].check == "require"

    def test_require_fails_when_tool_missing(self):
        spec = TrajectorySpec(require=["get_weather"])
        result = evaluate_trajectory(spec, [_call("search")])
        assert not result.passed
        assert "never called" in result.checks[0].detail

    def test_require_with_args_subset_match(self):
        spec = TrajectorySpec(
            require=[TrajectoryToolRef(name="issue_refund", args={"amount": 42})]
        )
        calls = [_call("issue_refund", {"amount": 42, "currency": "USD"})]
        result = evaluate_trajectory(spec, calls)
        assert result.passed

    def test_require_with_args_mismatch_fails(self):
        spec = TrajectorySpec(
            require=[TrajectoryToolRef(name="issue_refund", args={"amount": 42})]
        )
        calls = [_call("issue_refund", {"amount": 999})]
        result = evaluate_trajectory(spec, calls)
        assert not result.passed
        assert "never with the required arguments" in result.checks[0].detail

    def test_require_with_nested_args(self):
        spec = TrajectorySpec(
            require=[
                TrajectoryToolRef(
                    name="update_user", args={"profile": {"tier": "gold"}}
                )
            ]
        )
        matching = [_call("update_user", {"profile": {"tier": "gold"}, "id": 7})]
        assert evaluate_trajectory(spec, matching).passed

        mismatched = [_call("update_user", {"profile": {"tier": "silver"}})]
        assert not evaluate_trajectory(spec, mismatched).passed

    def test_require_string_and_ref_mixed_from_yaml_shape(self):
        spec = TrajectorySpec(
            require=["verify_identity", {"name": "issue_refund", "args": {"amount": 10}}]
        )
        calls = [
            _call("verify_identity", {"user": "u1"}, "c1"),
            _call("issue_refund", {"amount": 10}, "c2"),
        ]
        assert evaluate_trajectory(spec, calls).passed


class TestForbidCheck:
    def test_forbid_passes_when_absent(self):
        spec = TrajectorySpec(forbid=["delete_account"])
        result = evaluate_trajectory(spec, [_call("get_weather")])
        assert result.passed

    def test_forbid_fails_when_called(self):
        spec = TrajectorySpec(forbid=["delete_account"])
        result = evaluate_trajectory(spec, [_call("delete_account")])
        assert not result.passed
        assert "forbidden tool 'delete_account' was called 1 time(s)" in (
            result.checks[0].detail
        )

    def test_forbid_counts_multiple_calls(self):
        spec = TrajectorySpec(forbid=["rm"])
        calls = [_call("rm", call_id="c1"), _call("rm", call_id="c2")]
        result = evaluate_trajectory(spec, calls)
        assert "2 time(s)" in result.checks[0].detail


class TestOrderCheck:
    def test_order_passes_for_exact_sequence(self):
        spec = TrajectorySpec(order=[["verify_identity", "issue_refund"]])
        calls = [_call("verify_identity", call_id="c1"), _call("issue_refund", call_id="c2")]
        assert evaluate_trajectory(spec, calls).passed

    def test_order_passes_with_interleaved_calls(self):
        spec = TrajectorySpec(order=[["a", "c"]])
        calls = [
            _call("a", call_id="c1"),
            _call("b", call_id="c2"),
            _call("c", call_id="c3"),
        ]
        assert evaluate_trajectory(spec, calls).passed

    def test_order_fails_when_reversed(self):
        spec = TrajectorySpec(order=[["verify_identity", "issue_refund"]])
        calls = [_call("issue_refund", call_id="c1"), _call("verify_identity", call_id="c2")]
        result = evaluate_trajectory(spec, calls)
        assert not result.passed
        assert "order broken" in result.checks[0].detail

    def test_order_fails_when_step_missing(self):
        spec = TrajectorySpec(order=[["a", "b", "c"]])
        calls = [_call("a", call_id="c1"), _call("c", call_id="c2")]
        result = evaluate_trajectory(spec, calls)
        # 'c' arrives before 'b', so the subsequence match stalls at 'b'
        assert not result.passed
        assert "'b'" in result.checks[0].detail

    def test_order_with_repeated_tool_requires_distinct_calls(self):
        spec = TrajectorySpec(order=[["retry", "retry"]])
        single = [_call("retry", call_id="c1")]
        assert not evaluate_trajectory(spec, single).passed

        double = [_call("retry", call_id="c1"), _call("retry", call_id="c2")]
        assert evaluate_trajectory(spec, double).passed

    def test_order_failure_reports_observed_sequence(self):
        spec = TrajectorySpec(order=[["x", "y"]])
        result = evaluate_trajectory(spec, [])
        assert "no tool calls" in result.checks[0].detail


class TestCallCountChecks:
    def test_max_calls_budget(self):
        spec = TrajectorySpec(max_calls=2)
        within = [_call("a", call_id="c1"), _call("b", call_id="c2")]
        assert evaluate_trajectory(spec, within).passed

        over = within + [_call("c", call_id="c3")]
        result = evaluate_trajectory(spec, over)
        assert not result.passed
        assert "budget is 2" in result.checks[0].detail

    def test_min_calls(self):
        spec = TrajectorySpec(min_calls=1)
        assert not evaluate_trajectory(spec, []).passed
        assert evaluate_trajectory(spec, [_call("a")]).passed

    def test_zero_max_calls_asserts_no_tools_used(self):
        spec = TrajectorySpec(max_calls=0)
        assert evaluate_trajectory(spec, []).passed
        assert not evaluate_trajectory(spec, [_call("a")]).passed


class TestNoRepeatCheck:
    def test_no_repeats_passes_for_distinct_args(self):
        spec = TrajectorySpec(no_repeat_calls=True)
        calls = [
            _call("search", {"q": "one"}, "c1"),
            _call("search", {"q": "two"}, "c2"),
        ]
        assert evaluate_trajectory(spec, calls).passed

    def test_no_repeats_fails_for_identical_calls(self):
        spec = TrajectorySpec(no_repeat_calls=True)
        calls = [
            _call("search", {"q": "same"}, "c1"),
            _call("search", {"q": "same"}, "c2"),
        ]
        result = evaluate_trajectory(spec, calls)
        assert not result.passed
        assert "'search' x2" in result.checks[0].detail

    def test_no_repeats_key_order_insensitive(self):
        spec = TrajectorySpec(no_repeat_calls=True)
        calls = [
            _call("f", {"a": 1, "b": 2}, "c1"),
            _call("f", {"b": 2, "a": 1}, "c2"),
        ]
        assert not evaluate_trajectory(spec, calls).passed


class TestCombinedEvaluation:
    def test_all_checks_reported_individually(self):
        spec = TrajectorySpec(
            require=["verify_identity"],
            forbid=["delete_account"],
            order=[["verify_identity", "issue_refund"]],
            min_calls=1,
            max_calls=5,
            no_repeat_calls=True,
        )
        calls = [
            _call("verify_identity", {"user": "u1"}, "c1"),
            _call("issue_refund", {"amount": 10}, "c2"),
        ]
        result = evaluate_trajectory(spec, calls)
        assert result.passed
        assert len(result.checks) == 6
        assert result.call_count == 2
        assert result.calls_observed == ["verify_identity", "issue_refund"]

    def test_single_failure_fails_overall(self):
        spec = TrajectorySpec(require=["a"], max_calls=10)
        result = evaluate_trajectory(spec, [_call("b")])
        assert not result.passed
        assert len(result.failed_checks) == 1

    def test_summary_line(self):
        spec = TrajectorySpec(require=["a"])
        result = evaluate_trajectory(spec, [_call("a")])
        assert "trajectory passed" in result.summary()
        assert "1/1 checks passed" in result.summary()

    def test_empty_tool_calls_with_forbid_only_passes(self):
        spec = TrajectorySpec(forbid=["rm"])
        result = evaluate_trajectory(spec, [])
        assert result.passed
        assert result.call_count == 0
