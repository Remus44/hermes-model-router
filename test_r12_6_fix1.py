import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import model_router as router
from model_router import execution_adapters, triage


def _cfg(path):
    return {
        "enabled": True,
        "models": {"terra": "gpt-5.6-terra", "sol": "gpt-6.1-sol"},
        "tier_providers": {"terra": "openai-codex", "sol": "openai-codex"},
        "callable": {"terra": True, "sol": True, "haiku": True, "sonnet5": True, "opus5": True},
        "default_model": "terra",
        "triage": {"enabled": True, "max_workers_per_turn": 4, "max_children_per_worker": 1,
                   "path": str(Path(path) / "triage.jsonl")},
        "orchestration": {"enabled": False}, "shadow": {"enabled": False},
    }


class R126Fix1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = _cfg(self.tmp.name)
        self.root = SimpleNamespace(session_id="r12-fix1-root", _delegate_depth=0)
        triage.reset_state()
        self.addCleanup(triage.reset_state)

    def hook(self, call_id, args, turn_id="r12-fix1:turn"):
        return router.on_pre_tool_call("delegate_claude", args, session_id="r12-fix1-root",
                                       turn_id=turn_id, tool_call_id=call_id)

    def guard(self, call_id, args, raw):
        return execution_adapters.guard_legacy_tool_execution(
            tool_name="delegate_claude", args=args, next_call=raw,
            session_id="r12-fix1-root", turn_id="r12-fix1:turn", tool_call_id=call_id)

    def test_i1_evicted_pre_hook_replay_captures_g1_not_g2(self):
        same = {"goal": "same", "tier": "haiku"}
        other = {"goal": "other", "tier": "haiku"}
        calls = []
        book = execution_adapters.ReservationBook()
        with patch.object(router, "_load_config", return_value=self.config), \
             patch.object(router, "_delegation_targets_detail", return_value={}), \
             patch.object(triage, "_caller", return_value=self.root), \
             patch.object(execution_adapters, "RESERVATIONS", book), \
             patch.object(execution_adapters, "_MAX_RECORDS", 1):
            self.assertIsNone(self.hook("same", same))
            first = self.guard("same", same, lambda _sent: calls.append("G1") or json.dumps({"delegation_id": "G1"}))
            self.assertIsNone(self.hook("same", same), "duplicate must pass the real pre-hook")
            self.guard("other", other, lambda _sent: calls.append("other") or json.dumps({"delegation_id": "other"}))
            replay = self.guard("same", same, lambda _sent: calls.append("G2") or json.dumps({"delegation_id": "G2"}))
        self.assertEqual(replay, first)
        self.assertNotIn("G2", calls, "I1: an evicted G1 pre-hook exemption launched G2")

    def test_i3_unbound_bookkeeping_is_capped_without_refund(self):
        book = execution_adapters.ReservationBook()
        with patch.object(router, "_load_config", return_value=self.config), \
             patch.object(router, "_delegation_targets_detail", return_value={}), \
             patch.object(triage, "_caller", return_value=self.root), \
             patch.object(execution_adapters, "RESERVATIONS", book), \
             patch.object(execution_adapters, "_MAX_RECORDS", 2):
            for number in range(5):
                self.assertIsNone(self.hook(f"refused-{number}", {"goal": str(number), "tier": "haiku"},
                                            turn_id=f"r12-fix1:refused-{number}"))
        self.assertLessEqual(len(triage._RETAINED_ADMISSIONS), 2)
        self.assertEqual(sum(triage._TURN_WORKERS.values()), 5)

    def test_i4_sealed_replay_bypasses_new_target_switch_but_new_work_is_refused(self):
        args = {"goal": "same", "tier": "haiku"}
        calls = []
        book = execution_adapters.ReservationBook()
        with patch.object(router, "_load_config", return_value=self.config), \
             patch.object(router, "_delegation_targets_detail", return_value={}), \
             patch.object(triage, "_caller", return_value=self.root), \
             patch.object(execution_adapters, "RESERVATIONS", book):
            self.assertIsNone(self.hook("same", args))
            first = self.guard("same", args, lambda _sent: calls.append(1) or json.dumps({"delegation_id": "G1"}))
            self.config["callable"]["haiku"] = False
            self.assertIsNone(self.hook("same", args))
            self.assertEqual(self.guard("same", args, lambda _sent: calls.append(2) or "wrong"), first)
            fresh = self.hook("fresh", {"goal": "fresh", "tier": "haiku"})
        self.assertEqual(len(calls), 1)
        self.assertEqual(fresh["action"], "block")


if __name__ == "__main__":
    unittest.main()
