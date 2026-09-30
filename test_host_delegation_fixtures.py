"""Shared host-delegation fixtures declare topology and schema assumptions."""

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

    def test_depth_one_still_permits_a_direct_worker_call(self):
        from model_router.host_delegation_fixtures import host_delegation

        cfg = {"callable": {"terra": True}}
        direct_worker = {"tasks": [{"goal": "Inspect the parser.", "context": "Read-only evidence."}]}
        with host_delegation(depth=1), patch("model_router._load_config", return_value=cfg):
            self.assertIsNone(router.on_pre_tool_call(tool_name="delegate_task", args=direct_worker))

    def test_depth_two_exposes_a_conductor(self):
        from model_router.host_delegation_fixtures import host_delegation

        with host_delegation(depth=2):
            limits = router._host_delegation_limits()
        self.assertTrue(limits["conductor_available"])

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
