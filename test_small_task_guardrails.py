"""Regression guards for the pricing-copy orchestration incident."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import model_router as router


class SmallTaskGuardrailTests(unittest.TestCase):
    def _cfg(self, directory):
        return {
            "enabled": True,
            "models": {"terra": "gpt-5.6-terra", "luna": "gpt-6-luna", "grok": "grok-4.7"},
            "callable": {"terra": True, "luna": True, "grok": True},
            "tier_providers": {"terra": "openai-codex", "luna": "openai-codex", "grok": "xai-oauth"},
            "preferences": {"code": ["grok", "terra"]},
            "task_budget": {
                "enabled": True,
                "low_risk_max_chars": 500,
                "max_routing_decisions": 1,
                "max_depth": 1,
                "require_evidence_for_second_worker": True,
                "path": str(Path(directory) / "task-budget.jsonl"),
            },
        }

    def test_pricing_copy_correction_gets_one_direct_worker_then_requires_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            first = {"goal": "Correct the pricing copy on the public pricing page.", "model": "grok"}
            second = {"goal": "Check another wording option.", "model": "terra"}
            with patch.object(router, "_load_config", return_value=cfg):
                self.assertIsNone(router.on_pre_tool_call("delegate_task", first, turn_id="pricing-copy"))
                blocked = router.on_pre_tool_call("delegate_task", second, turn_id="pricing-copy")
        self.assertEqual(blocked["action"], "block")
        self.assertIn("evidence", blocked["message"].casefold())
        self.assertIn("Nothing was spawned", blocked["message"])

    def test_real_incident_goal_mentioning_production_is_still_bounded(self):
        goal = ("Correct misleading pricing-page messaging in Booking SaaS production: beauty studios are not "
                "offerable, so the Beauty category-limit row must not be advertised. Update copy and a focused test.")
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            with patch.object(router, "_load_config", return_value=cfg):
                self.assertIsNone(router.on_pre_tool_call("delegate_task", {"goal": goal, "model": "grok"}, turn_id="root:turn"))
                # A child of the same root turn draws on the same budget, not a fresh one.
                blocked = router.on_pre_tool_call("delegate_task", {"goal": goal, "model": "terra"}, turn_id="root:turn")
        self.assertEqual(blocked["action"], "block")

    def test_second_worker_needs_escalation_evidence_and_gets_only_one(self):
        goal = "Correct the pricing copy on the public pricing page."
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            with patch.object(router, "_load_config", return_value=cfg):
                self.assertIsNone(router.on_pre_tool_call("delegate_task", {"goal": goal}, turn_id="root:turn"))
                self.assertEqual(router.on_pre_tool_call("delegate_task", {"goal": goal}, turn_id="root:turn")["action"], "block")
                self.assertIsNone(router.on_pre_tool_call(
                    "delegate_task", {"goal": goal, "escalation_evidence": "focused test still failing: pricing-offer-gate"}, turn_id="root:turn"))
                self.assertEqual(router.on_pre_tool_call(
                    "delegate_task", {"goal": goal, "escalation_evidence": "again"}, turn_id="root:turn")["action"], "block")

    def test_deploy_requests_keep_the_normal_unbounded_path(self):
        goal = "Deploy the pricing fix to production and run the database migration."
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            with patch.object(router, "_load_config", return_value=cfg):
                for _ in range(3):
                    self.assertIsNone(router.on_pre_tool_call("delegate_task", {"goal": goal}, turn_id="root:turn"))

    def test_low_risk_worker_cannot_delegate_recursively(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            with patch.object(router, "_load_config", return_value=cfg):
                blocked = router.on_pre_tool_call(
                    "delegate_task",
                    {"goal": "Correct the pricing copy on the public pricing page.", "model": "luna"},
                    turn_id="pricing-copy:sa-1:turn",
                )
        self.assertEqual(blocked["action"], "block")
        self.assertIn("depth budget", blocked["message"].casefold())

    def test_low_risk_task_cannot_evade_budget_with_claude_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            with patch.object(router, "_load_config", return_value=cfg):
                self.assertIsNone(router.on_pre_tool_call(
                    "delegate_task",
                    {"goal": "Correct the pricing copy on the public pricing page.", "model": "grok"},
                    turn_id="pricing-copy",
                ))
                blocked = router.on_pre_tool_call(
                    "delegate_claude",
                    {"tasks": [{"goal": "Correct the pricing copy on the public pricing page."}]},
                    turn_id="pricing-copy",
                )
        self.assertEqual(blocked["action"], "block")
        self.assertIn("cross-provider handoff", blocked["message"])

    def test_preference_order_falls_back_only_to_the_next_configured_target(self):
        cfg = self._cfg(tempfile.gettempdir())
        cfg["preferences"] = {"code": ["grok", "terra", "luna"]}
        self.assertEqual(router._preferred_route("code", cfg), "grok")
        cfg["callable"]["grok"] = False
        self.assertEqual(router._preferred_route("code", cfg), "terra")

    def test_duplicate_lifecycle_event_is_recorded_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "orchestration.jsonl"
            cfg = {"orchestration": {"path": str(path)}}
            event = {"event": "spark_child_started", "plan_id": "plan-1", "turn_id": "pricing-copy", "child_session_id": "child-1"}
            router._orchestration_event(cfg, event)
            router._orchestration_event(cfg, event)
            entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(entries, [entries[0]])

    def test_distinct_lifecycle_phases_are_not_collapsed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "orchestration.jsonl"
            cfg = {"orchestration": {"path": str(path)}}
            base = {"event": "preflight_skipped", "turn_id": "pricing-copy", "plan_id": None, "child_session_id": None}
            router._orchestration_event(cfg, {**base, "phase": "preflight", "api_call_count": 1})
            router._orchestration_event(cfg, {**base, "phase": "rescue", "api_call_count": 6})
            phases = [json.loads(line)["phase"] for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(phases, ["preflight", "rescue"])


if __name__ == "__main__":
    unittest.main()
