"""Load balancing between the two delegation accounts, under Claude delegation.

The soft/hard guard reacts to one account reaching a limit. Balancing compares
the two: when the preferred account's tighter window (weekly or 5-hour) is busy
and the other account in the same chain is clearly freer, the freer one goes
first. It only reorders targets the chain already lists, never touches the
parent, and stays out of the way when a reading is missing or stale.
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import model_router
from model_router import claude_delegation, route_llm_request, usage_guard
from model_router.host_delegation_fixtures import host_delegation
from model_router.test_claude_preflight import CODE, REVIEW, _anthropic_request, _openai_request
from model_router.test_model_router import CALLABLE, MODELS
from model_router.usage_guard import Reading

PROVIDERS = {"luna": "openai-codex", "spark": "openai-codex", "terra": "openai-codex", "sol": "openai-codex",
             "opus5": "anthropic", "sonnet5": "anthropic", "haiku": "anthropic"}


def _cfg(temp_dir, *, balance=None):
    return {
        "enabled": True, "provider": "openai-codex", "models": MODELS,
        "callable": {**CALLABLE, "opus5": True, "sonnet5": True, "haiku": True},
        "tier_providers": dict(PROVIDERS),
        "default_model": "terra",
        "orchestration": {"enabled": True, "max_tasks": 3, "path": str(Path(temp_dir) / "orchestration.jsonl")},
        "shadow": {"enabled": False},
        "preferences": {"review": ["sonnet5", "opus5", "terra"], "code": ["terra", "sonnet5"],
                        "chat": ["luna", "spark"]},
        "usage_guard": {
            "cache_seconds": 300,
            "balance": {"enabled": True, "busy_percent": 60, "margin_percent": 40, **(balance or {})},
            "accounts": {
                "anthropic": {"soft_percent": 80, "hard_percent": 90, "step_down": {"opus5": "sonnet5"}},
                "openai-codex": {"soft_percent": 70, "hard_percent": 90, "step_down": {"sol": "terra"}},
            },
        },
    }


def _reading(weekly, session, age=0.0):
    return Reading(weekly, session, None, None, time.time() - age)


class LoadTests(unittest.TestCase):
    def test_load_is_the_tighter_window(self):
        self.assertEqual(usage_guard.load(_reading(34, 86)), (86, "5-hour"))
        self.assertEqual(usage_guard.load(_reading(75, 10)), (75, "weekly"))
        self.assertEqual(usage_guard.load(_reading(None, 20)), (20, "5-hour"))
        self.assertIsNone(usage_guard.load(_reading(None, None)))
        self.assertIsNone(usage_guard.load(None))

    def test_the_5_hour_window_alone(self):
        self.assertEqual(usage_guard.load(_reading(80, 23), "5-hour"), (23, "5-hour"))
        self.assertIsNone(usage_guard.load(_reading(80, None), "5-hour"))

    def test_balance_is_off_unless_configured(self):
        self.assertFalse(usage_guard.balance_config({})["enabled"])

    def test_the_window_defaults_to_5_hour_and_rejects_anything_else(self):
        self.assertEqual(usage_guard.balance_config({})["window"], "5-hour")
        for value, expected in (("tighter", "tighter"), ("5-hour", "5-hour"), ("monthly", "5-hour")):
            cfg = {"usage_guard": {"balance": {"window": value}}}
            self.assertEqual(usage_guard.balance_config(cfg)["window"], expected)

    def test_a_reading_older_than_twice_the_cache_is_not_fresh(self):
        cfg = {"usage_guard": {"cache_seconds": 300}}
        self.assertTrue(usage_guard.fresh(_reading(1, 1, age=500), cfg))
        self.assertFalse(usage_guard.fresh(_reading(1, 1, age=700), cfg))
        self.assertFalse(usage_guard.fresh(None, cfg))


class ChainTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, claude_delegation, "_ACTIVE", False)
        claude_delegation._ACTIVE = True
        self.addCleanup(usage_guard._reset_cache)

    def _chain(self, kind, claude, codex, **cfg_kwargs):
        readings = {"anthropic": claude, "openai-codex": codex}
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d, **cfg_kwargs)
            with patch("model_router._hermes_delegation_target_names", return_value=("sonnet5", "opus5")):
                states = model_router._account_states(cfg, readings)
                return model_router._advised_chain(kind, cfg, states, set(model_router._delegation_target_names()),
                                                   readings)

    def test_a_busy_claude_window_sends_review_to_codex_first(self):
        names, reason = self._chain("review", _reading(34, 86), _reading(1, 6))
        self.assertEqual(names, ["terra", "sonnet5", "opus5"])
        self.assertIn("Claude 5-hour 86%", reason)
        self.assertIn("Codex 6%", reason)

    def test_a_busy_codex_window_sends_code_to_claude_first(self):
        names, reason = self._chain("code", _reading(10, 12), _reading(20, 65))
        self.assertEqual(names, ["sonnet5", "terra"])
        self.assertIn("Codex 5-hour 65%", reason)

    def test_the_tighter_window_setting_also_counts_weekly(self):
        # 65% stays under Codex's own 70% soft limit, so this is balancing, not the guard.
        names, reason = self._chain("code", _reading(10, 12), _reading(65, 20), balance={"window": "tighter"})
        self.assertEqual(names, ["sonnet5", "terra"])
        self.assertIn("Codex weekly 65%", reason)

    def test_the_weekly_window_does_not_balance_by_default(self):
        """Claude's weekly lead (the parent's week) is the soft/hard limits' business."""
        self.assertEqual(self._chain("review", _reading(70, 30), _reading(1, 25)),
                         (["sonnet5", "opus5", "terra"], ""))

    def test_the_shipped_thresholds_even_out_a_modest_5_hour_gap(self):
        """Claude at 23% (parent included) and Codex at 12%: the next review goes to Codex."""
        names, reason = self._chain("review", _reading(39, 23), _reading(1, 12),
                                    balance={"busy_percent": 20, "margin_percent": 10})
        self.assertEqual(names, ["terra", "sonnet5", "opus5"])
        self.assertIn("Claude 5-hour 23% vs Codex 12%", reason)

    def test_the_shipped_thresholds_leave_a_small_gap_alone(self):
        self.assertEqual(self._chain("review", _reading(39, 23), _reading(1, 15),
                                     balance={"busy_percent": 20, "margin_percent": 10}),
                         (["sonnet5", "opus5", "terra"], ""))

    def test_below_the_busy_line_the_chain_is_kept(self):
        self.assertEqual(self._chain("review", _reading(40, 55), _reading(1, 1)),
                         (["sonnet5", "opus5", "terra"], ""))

    def test_without_the_margin_the_chain_is_kept(self):
        self.assertEqual(self._chain("review", _reading(40, 70), _reading(35, 40)),
                         (["sonnet5", "opus5", "terra"], ""))

    def test_a_chain_on_one_account_is_kept(self):
        self.assertEqual(self._chain("chat", _reading(1, 1), _reading(70, 88)), (["luna", "spark"], ""))

    def test_a_stale_reading_disables_balancing(self):
        self.assertEqual(self._chain("review", _reading(34, 86, age=3600), _reading(1, 6)),
                         (["sonnet5", "opus5", "terra"], ""))

    def test_a_missing_reading_disables_balancing(self):
        self.assertEqual(self._chain("review", _reading(34, 86), None), (["sonnet5", "opus5", "terra"], ""))

    def test_disabled_balancing_keeps_the_chain(self):
        self.assertEqual(self._chain("review", _reading(34, 86), _reading(1, 6), balance={"enabled": False}),
                         (["sonnet5", "opus5", "terra"], ""))

    def test_inactive_claude_delegation_keeps_the_chain(self):
        claude_delegation._ACTIVE = False
        names, reason = self._chain("code", _reading(10, 12), _reading(20, 70))
        self.assertEqual((names[0], reason), ("terra", ""))

    def test_the_soft_limit_still_applies_first(self):
        """Claude past its weekly soft limit is demoted by the guard, not by balancing."""
        # Codex at 50% is under its soft limit and within the margin: only the guard acts.
        names, reason = self._chain("review", _reading(85, 30), _reading(50, 30))
        self.assertEqual(names, ["terra", "sonnet5", "opus5"])
        self.assertEqual(reason, "")

    def test_balance_cannot_revive_a_weekly_held_account(self):
        for weekly in (85, 95):
            with self.subTest(weekly=weekly):
                names, reason = self._chain("code", _reading(weekly, 0), _reading(10, 65))
                self.assertEqual(names, ["terra", "sonnet5"])
                self.assertEqual(reason, "")

    def test_balance_cannot_promote_a_cooling_target(self):
        with patch("model_router._tier_cooldown_remaining", side_effect=lambda n, c: 60 if n == "sonnet5" else 0):
            names, reason = self._chain("code", _reading(10, 0), _reading(10, 65))
        self.assertEqual((names, reason), (["terra", "sonnet5"], ""))


class ForcedCallTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(usage_guard._reset_cache)

    def _route(self, text, claude, codex):
        readings = {"anthropic": claude, "openai-codex": codex}
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d)
            with host_delegation(depth=2), \
                 patch("model_router._load_config", return_value=cfg), \
                 patch("model_router._log_decision"), \
                 patch("model_router._hermes_delegation_target_names", return_value=("sonnet5", "opus5")), \
                 patch.object(usage_guard, "peek", side_effect=lambda account, _cfg: readings.get(account)), \
                 patch.object(claude_delegation, "_ACTIVE", True):
                routed = route_llm_request(
                    request=_anthropic_request(text, ["mcp__delegate_task", "mcp__tool_call"]),
                    provider="anthropic", model="claude-opus-5-5", api_call_count=1, turn_id="t1", platform="cli")
            events = [json.loads(line) for line in
                      Path(cfg["orchestration"]["path"]).read_text(encoding="utf-8").splitlines()]
        return routed["request"], events

    def test_a_busy_claude_turns_the_review_call_into_delegate_task_only(self):
        request, events = self._route(REVIEW, _reading(34, 86), _reading(1, 6))
        self.assertEqual([t["name"] for t in request["tools"]], ["mcp__delegate_task"])
        instruction = str(request["messages"][-1]["content"])
        self.assertIn("Balanced:", instruction)
        self.assertIn("Claude 5-hour 86%", instruction)
        forced = [e for e in events if e["event"] == "preflight_forced"][0]
        self.assertIn("Claude 5-hour 86%", forced["balanced"])

    def test_a_busy_codex_offers_claude_for_code(self):
        request, events = self._route(CODE, _reading(10, 12), _reading(20, 65))
        self.assertEqual([t["name"] for t in request["tools"]], ["mcp__delegate_task", "mcp__tool_call"])
        instruction = str(request["messages"][-1]["content"])
        self.assertIn('delegate_claude(tier="sonnet")', instruction)
        self.assertIn("Codex 5-hour 65%", instruction)

    def test_balanced_accounts_leave_the_call_as_configured(self):
        request, events = self._route(REVIEW, _reading(20, 25), _reading(10, 15))
        self.assertEqual([t["name"] for t in request["tools"]], ["mcp__delegate_task", "mcp__tool_call"])
        self.assertNotIn("Balanced:", str(request["messages"][-1]["content"]))
        forced = [e for e in events if e["event"] == "preflight_forced"][0]
        self.assertEqual(forced.get("balanced", ""), "")

    def test_conductor_contract_uses_the_balanced_root_order(self):
        request, _ = self._route(REVIEW, _reading(34, 86), _reading(1, 6))
        contract = request["tools"][0]["input_schema"]["properties"]["context"]["enum"][0]
        self.assertIn("review: terra > sonnet5 > opus5", contract)

    def test_local_tiers_survive_and_conductor_advice_refreshes(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d)
            cfg["preferences"] = {"explore": ["luna", "haiku"]}
            request = {"tools": [{"name": "delegate_task", "parameters": {}}]}
            with patch.object(claude_delegation, "_ACTIVE", True), \
                 patch.object(model_router, "_delegation_target_names", return_value=("haiku",)), \
                 patch.object(usage_guard, "peek", return_value=_reading(0, 0)):
                self.assertIn("explore: luna > haiku", model_router._worker_order_note(request, cfg))
            with patch.object(claude_delegation, "_ACTIVE", True), \
                 patch.object(model_router, "_delegation_target_names", return_value=("haiku",)), \
                 patch.object(usage_guard, "peek", side_effect=lambda a, c: _reading(0, 65 if a == "openai-codex" else 0)):
                self.assertIn("explore: haiku > luna", model_router._worker_order_note(request, cfg))


class WorkerOrderNoteTests(unittest.TestCase):
    _BRIDGE = ("tool_search", "tool_describe", "tool_call")

    def _route_leaf(self, *, model, provider, request):
        with tempfile.TemporaryDirectory() as directory:
            cfg = _cfg(directory)
            with patch("model_router._load_config", return_value=cfg), \
                 patch("model_router._log_decision"), \
                 patch("model_router._hermes_delegation_target_names", return_value=("sonnet5", "terra")), \
                 patch.object(usage_guard, "read", return_value=None), \
                 patch.object(usage_guard, "peek", return_value=None), \
                 patch.object(claude_delegation, "_ACTIVE", True):
                routed = route_llm_request(
                    request=request, provider=provider, model=model,
                    api_call_count=1, turn_id="parent:sa-1", platform="subagent",
                )
        routed_request = routed["request"] if routed is not None else request
        return str(routed_request["messages"][-1]["content"])

    def test_a_leaf_with_only_the_tool_search_bridge_gets_no_worker_order_note(self):
        cases = (
            ("claude-sonnet-5-5", "anthropic", _anthropic_request("Inspect the parser.", self._BRIDGE)),
            ("gpt-5.6-terra", "openai-codex", _openai_request("Inspect the parser.", self._BRIDGE)),
        )
        for model, provider, request in cases:
            with self.subTest(model=model):
                instruction = self._route_leaf(model=model, provider=provider, request=request)
                self.assertNotIn("[ROUTER] Current worker order", instruction)

    def test_a_leaf_with_a_direct_delegation_tool_still_gets_the_worker_order_note(self):
        for tools in (("delegate_task",), ("delegate_claude",), ("mcp__delegate_claude",)):
            with self.subTest(tools=tools):
                request = _openai_request("Inspect the parser.", tools)
                instruction = self._route_leaf(model="gpt-5.6-terra", provider="openai-codex", request=request)
                self.assertIn("[ROUTER] Current worker order", instruction)


if __name__ == "__main__":
    unittest.main()
