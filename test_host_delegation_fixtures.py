"""Shared host-delegation fixtures declare topology and schema assumptions."""

import tempfile
import unittest
from unittest.mock import patch

import model_router as router


class HostDelegationFixtureTests(unittest.TestCase):
    def test_depth_one_keeps_direct_workers_but_skips_nested_conductors(self):
        from model_router.host_delegation_fixtures import host_delegation

        with host_delegation(depth=1):
            limits = router._host_delegation_limits()
        self.assertEqual(limits["max_spawn_depth"], 1)
        self.assertFalse(limits["conductor_available"])

    def test_depth_one_records_parent_direct_worker_topology(self):
        from model_router.host_delegation_fixtures import host_delegation
        from model_router.test_external_orchestrator import _cfg, _delegating_request

        with tempfile.TemporaryDirectory() as directory:
            cfg = _cfg(directory)
            cfg["orchestration"]["min_chars"] = 1
            kwargs = {"request": _delegating_request(), "api_call_count": 1, "turn_id": "root"}
            decision = router.RouteDecision("terra", cfg["models"]["terra"], "test", "test")
            with host_delegation(depth=1), patch("model_router._delegation_target_names", return_value=("terra",)):
                reason = router._orchestration_skip_reason(kwargs, cfg, decision)

        self.assertEqual(reason, "host_has_no_conductor_depth; parent_delegates_direct_workers")

    def test_depth_two_makes_the_same_parent_eligible_for_a_conductor(self):
        from model_router.host_delegation_fixtures import host_delegation
        from model_router.test_external_orchestrator import _cfg, _delegating_request

        with tempfile.TemporaryDirectory() as directory:
            cfg = _cfg(directory)
            cfg["orchestration"]["min_chars"] = 1
            kwargs = {"request": _delegating_request(), "api_call_count": 1, "turn_id": "root"}
            decision = router.RouteDecision("terra", cfg["models"]["terra"], "test", "test")
            with host_delegation(depth=2), patch("model_router._delegation_target_names", return_value=("terra",)):
                reason = router._orchestration_skip_reason(kwargs, cfg, decision)

        self.assertIsNone(reason)

    def test_batch_only_and_model_param_schemas_are_distinct_capabilities(self):
        from model_router.host_delegation_fixtures import delegate_task_request

        self.assertFalse(router._host_delegate_has_model(delegate_task_request("batch-only")))
        self.assertTrue(router._host_delegate_has_model(delegate_task_request("model-param")))

    def test_direct_and_deferred_tool_sets_are_declared(self):
        from model_router.host_delegation_fixtures import DEFERRED_CLAUDE_TOOLS, DIRECT_CLAUDE_TOOLS

        self.assertIn("delegate_claude", DIRECT_CLAUDE_TOOLS)
        self.assertIn("tool_call", DEFERRED_CLAUDE_TOOLS)


if __name__ == "__main__":
    unittest.main()
