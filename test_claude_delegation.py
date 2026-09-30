"""The Claude wing: Claude workers next to the Codex workforce.

Hermes's delegate_task has one route per process, pinned to Codex on this host.
delegate_claude reaches Claude by calling the same delegate_task with a per-call
route pinned to the anthropic provider. These tests cover the wing without any
network or model call.
"""

import ast
import inspect
import json
import sys
import tempfile
import textwrap
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from model_router import claude_delegation
from model_router.host_delegation_fixtures import host_delegation
from model_router.claude_delegation import (
    TARGET_FOR_TIER,
    TIER_FOR_TARGET,
    availability_block,
    target_for_model,
    target_names,
    tier_model,
    delegation_config,
)

CLAUDE_DELEGATION = {
    "enabled": True,
    "tiers": {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"},
    "default_tier": "sonnet",
}


def _cfg(**overrides):
    cfg = {
        "models": {"luna": "gpt-luna", "spark": "gpt-spark", "terra": "gpt-terra", "sol": "gpt-sol"},
        "callable": {"luna": True, "spark": True, "terra": True, "sol": True,
                     "opus5": True, "sonnet5": True, "haiku": True},
        "tier_providers": {"opus5": "anthropic", "sonnet5": "anthropic", "haiku": "anthropic",
                          "terra": "openai-codex", "sol": "openai-codex",
                          "spark": "openai-codex", "luna": "openai-codex"},
        "fallbacks": {"opus5": "sol"},
        "peer_groups": {"heavy": ["terra", "opus5", "sonnet5"], "light": ["luna", "spark", "haiku"]},
        "default_model": "terra",
        "claude_delegation": json.loads(json.dumps(CLAUDE_DELEGATION)),
        "usage_guard": {"cache_seconds": 300, "accounts": {
            "anthropic": {"soft_percent": 70, "hard_percent": 90, "step_down": {"opus5": "sonnet5"}},
        }},
    }
    cfg.update(overrides)
    return cfg


class DelegationConfigTests(unittest.TestCase):
    def test_the_block_no_longer_carries_an_on_off_flag(self):
        """The Claude switches in `callable` decide availability; `enabled` is retired."""
        self.assertNotIn("enabled", delegation_config({}))
        self.assertNotIn("enabled", delegation_config(None))
        self.assertNotIn("enabled", delegation_config({"claude_delegation": {"enabled": True}}))

    def test_a_partial_block_keeps_the_other_defaults(self):
        settings = delegation_config({"claude_delegation": {"enabled": True, "default_tier": "haiku"}})
        self.assertEqual(settings["default_tier"], "haiku")
        self.assertEqual(settings["tiers"]["opus"], "claude-opus-5-5")
        self.assertNotIn("usage_guard", settings)

    def test_names_map_both_ways(self):
        self.assertEqual(TARGET_FOR_TIER, {"haiku": "haiku", "sonnet": "sonnet5", "opus": "opus5"})
        self.assertEqual(TIER_FOR_TARGET["opus5"], "opus")

    def test_tier_model(self):
        self.assertEqual(tier_model("haiku", _cfg()), "claude-haiku-4-5-20251001")
        self.assertEqual(tier_model("gpt", _cfg()), "")

    def test_target_names_skip_a_tier_without_a_model(self):
        cfg = _cfg()
        self.assertEqual(target_names(cfg), ("haiku", "opus5", "sonnet5"))
        cfg["claude_delegation"]["tiers"]["haiku"] = ""
        self.assertEqual(target_names(cfg), ("opus5", "sonnet5"))

    def test_a_model_maps_back_to_its_target(self):
        self.assertEqual(target_for_model("claude-haiku-4-5-20251001", _cfg()), "haiku")
        self.assertIsNone(target_for_model("gpt-terra", _cfg()))
        self.assertIsNone(target_for_model("", _cfg()))


class AvailabilityBlockTests(unittest.TestCase):
    def test_a_stale_enabled_flag_changes_nothing(self):
        cfg = _cfg()
        cfg["claude_delegation"]["enabled"] = False
        self.assertEqual(availability_block(cfg), "")

    def test_every_claude_target_switched_off_is_not_offered(self):
        cfg = _cfg()
        for target in ("haiku", "sonnet5", "opus5"):
            cfg["callable"][target] = False
        self.assertIn("switched off", availability_block(cfg))

    def test_an_enabled_wing_is_offered(self):
        self.assertEqual(availability_block(_cfg()), "")


from model_router import usage_guard  # noqa: E402
from model_router.usage_guard import GuardOutcome, Reading  # noqa: E402


def _reading(weekly, session=10.0, fetched_at=None):
    return Reading(weekly, session, None, None, time.time() if fetched_at is None else fetched_at)


from model_router.claude_delegation import build_schema, handle_delegate_claude  # noqa: E402


def _fake_host(parent, result=None):
    """A recording delegate_task and a fixed active parent."""
    calls = []

    def delegate_task(**kwargs):
        calls.append(kwargs)
        return json.dumps(result if result is not None else {"status": "dispatched", "delegation_id": "d1"})

    return calls, (lambda: (delegate_task, lambda: parent))


class SchemaTests(unittest.TestCase):
    def test_the_schema_mirrors_delegate_task_plus_a_tier(self):
        with patch.object(claude_delegation, "_independent_completions", return_value=False):
            schema = build_schema(_cfg())
        self.assertEqual(schema["name"], "delegate_claude")
        properties = schema["parameters"]["properties"]
        self.assertEqual(properties["tier"]["enum"], ["haiku", "sonnet", "opus"])
        self.assertIn("goal", properties["tasks"]["items"]["properties"])
        self.assertEqual(properties["tasks"]["items"]["required"], ["goal"])
        self.assertNotIn("group", properties["tasks"]["items"]["properties"])
        self.assertNotIn("action", properties)
        self.assertIn("delegate_task", schema["description"])


class HandlerTests(unittest.TestCase):
    def setUp(self):
        usage_guard._reset_cache()
        self.addCleanup(usage_guard._reset_cache)

    def _call(self, args, *, cfg=None, parent=None, usage=40.0, result=None):
        parent = parent if parent is not None else SimpleNamespace(_delegate_depth=0)
        calls, host = _fake_host(parent, result)
        reading = None if usage is None else _reading(usage)
        with patch("model_router._load_config", return_value=cfg or _cfg()), \
             patch.object(claude_delegation, "_host", host), \
             patch.object(usage_guard, "read", return_value=reading):
            raw = handle_delegate_claude(args)
        return json.loads(raw), calls

    def test_a_call_runs_on_a_pinned_anthropic_route(self):
        payload, calls = self._call({"tasks": [{"goal": "g", "acp_command": "x"}], "tier": "haiku"})
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["credentials_cfg"], {"provider": "anthropic",
                                                   "model": "claude-haiku-4-5-20251001",
                                                   "fallback_providers": []})
        self.assertTrue(call["background"])
        self.assertEqual(call["tasks"], [{"goal": "g"}])
        self.assertEqual(payload["claude_tier"], "haiku")
        self.assertEqual(payload["delegation_id"], "d1")

    def test_the_default_tier_applies_when_none_is_given(self):
        _payload, calls = self._call({"tasks": [{"goal": "g"}]})
        self.assertEqual(calls[0]["credentials_cfg"]["model"], "claude-sonnet-5-5")

    def test_an_orchestrator_child_waits_for_its_workers(self):
        """Same rule as Hermes: a child at depth > 0 needs results within its turn."""
        _payload, calls = self._call({"tasks": [{"goal": "g"}]}, parent=SimpleNamespace(_delegate_depth=1))
        self.assertFalse(calls[0]["background"])

    def test_the_caller_is_the_parent(self):
        parent = SimpleNamespace(_delegate_depth=0)
        _payload, calls = self._call({"tasks": [{"goal": "g"}]}, parent=parent)
        self.assertIs(calls[0]["parent_agent"], parent)

    def test_an_unknown_tier_is_refused(self):
        payload, calls = self._call({"tasks": [{"goal": "g"}], "tier": "gpt"})
        self.assertIn("Unknown tier", payload["error"])
        self.assertEqual(calls, [])

    def test_no_active_parent_is_refused(self):
        calls, _host = _fake_host(None)
        with patch("model_router._load_config", return_value=_cfg()), \
             patch.object(claude_delegation, "_host", lambda: (lambda **k: calls.append(k), lambda: None)):
            payload = json.loads(handle_delegate_claude({"tasks": [{"goal": "g"}]}))
        self.assertIn("agent turn", payload["error"])
        self.assertEqual(calls, [])

    def test_a_switched_off_tier_is_refused_with_a_pointer(self):
        cfg = _cfg()
        cfg["callable"]["haiku"] = False
        payload, calls = self._call({"tasks": [{"goal": "g"}], "tier": "haiku"}, cfg=cfg)
        self.assertIn("switched off", payload["error"])
        self.assertIn("delegate_task with a goal prefixed [luna]", payload["error"])
        self.assertEqual(calls, [])

    def test_the_soft_limit_lowers_opus_and_says_so(self):
        payload, calls = self._call({"tasks": [{"goal": "g"}], "tier": "opus"}, usage=75.0)
        self.assertEqual(calls[0]["credentials_cfg"]["model"], "claude-sonnet-5-5")
        self.assertEqual(payload["claude_tier"], "sonnet")
        self.assertEqual(payload["tier_adjusted"], "opus→sonnet (weekly usage 75%)")

    def test_the_hard_limit_refuses_and_names_the_codex_call(self):
        payload, calls = self._call({"tasks": [{"goal": "g"}], "tier": "opus"}, usage=95.0)
        self.assertIn("Claude delegation closed", payload["error"])
        self.assertIn("[sol]", payload["error"])
        self.assertEqual(calls, [])

    def test_an_unknown_usage_is_marked(self):
        payload, _calls = self._call({"tasks": [{"goal": "g"}]}, usage=None)
        self.assertEqual(payload["usage"], "unknown")

    def test_a_host_error_becomes_a_tool_error(self):
        def boom(**_kwargs):
            raise ValueError("Cannot resolve delegation provider 'anthropic'")

        with patch("model_router._load_config", return_value=_cfg()), \
             patch.object(claude_delegation, "_host", lambda: (boom, lambda: SimpleNamespace(_delegate_depth=0))), \
             patch.object(usage_guard, "read", return_value=_reading(10)):
            payload = json.loads(handle_delegate_claude({"tasks": [{"goal": "g"}]}))
        self.assertIn("Cannot resolve delegation provider", payload["error"])

    def test_a_configured_audit_log_gets_one_line_per_call(self):
        self.addCleanup(claude_delegation._reset_reasoning_bridge_for_tests)

        def fake_resolver(*, parent_agent, model, override_provider, **_kwargs):
            return {"provider": override_provider, "model": model}

        fake_module = types.ModuleType("tools.delegate_tool")
        fake_module._resolve_child_runtime = fake_resolver
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(sys.modules, {"tools.delegate_tool": fake_module,
                                      "tools.delegate_tool_config": fake_module}):
            ok, reason = claude_delegation.install_reasoning_bridge()
            self.assertTrue(ok, reason)
            cfg = _cfg()
            log = Path(directory) / "claude-delegation.jsonl"
            cfg["claude_delegation"]["log_path"] = str(log)
            self._call_with_resolver_capture(
                {"tasks": [{"goal": "g"}], "tier": "opus"}, cfg=cfg, usage=75.0,
            )
            entry = json.loads(log.read_text(encoding="utf-8").strip())
        self.assertEqual(entry["event"], "delegate_claude")
        self.assertEqual((entry["tier_requested"], entry["tier_used"], entry["outcome"]),
                         ("opus", "sonnet", "lowered"))
        self.assertNotIn("tier", entry)  # keeps it out of the router's per-account load

    def test_a_delegate_task_error_is_audited_as_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = _cfg()
            log = Path(directory) / "claude-delegation.jsonl"
            cfg["claude_delegation"]["log_path"] = str(log)
            self._call({"tasks": [{"goal": "g"}], "tier": "opus"}, cfg=cfg,
                       result={"error": "Delegation depth limit reached"})
            entry = json.loads(log.read_text(encoding="utf-8").strip())
        self.assertEqual(entry["outcome"], "error")
        self.assertEqual(entry["message"], "Delegation depth limit reached")

    def test_audit_lines_name_the_calling_session_and_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = _cfg()
            log = Path(directory) / "claude-delegation.jsonl"
            cfg["claude_delegation"]["log_path"] = str(log)
            parent = SimpleNamespace(_delegate_depth=0, session_id="sess-1", _current_turn_id="turn-9")
            self._call({"tasks": [{"goal": "g"}]}, cfg=cfg, parent=parent)
            entry = json.loads(log.read_text(encoding="utf-8").strip())
        self.assertEqual((entry["session_id"], entry["turn_id"]), ("sess-1", "turn-9"))

    def test_switching_every_claude_model_off_refuses_on_the_next_call(self):
        cfg = _cfg()
        for target in ("haiku", "sonnet5", "opus5"):
            cfg["callable"][target] = False
        payload, calls = self._call({"tasks": [{"goal": "g"}]}, cfg=cfg)
        self.assertEqual(payload["error"], "Claude delegation is off: every Claude model is switched off in "
                                           "Settings. Use delegate_task, which runs on the Codex route.")
        self.assertEqual(calls, [])

    def test_switching_every_claude_model_off_is_audited(self):
        """B2-N3: this refusal predates the availability_block branch and wrote no audit."""
        cfg = _cfg()
        for target in ("haiku", "sonnet5", "opus5"):
            cfg["callable"][target] = False
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "claude-delegation.jsonl"
            cfg["claude_delegation"]["log_path"] = str(log)
            self._call({"tasks": [{"goal": "g"}]}, cfg=cfg)
            entry = json.loads(log.read_text(encoding="utf-8").strip())
        self.assertEqual(entry["outcome"], "refused")
        self.assertIn("every Claude model is switched off", entry["message"])

    def test_a_tier_with_no_configured_model_is_audited(self):
        """B2-N3: the 'tier has no model' refusal wrote no audit either."""
        cfg = _cfg()
        cfg["claude_delegation"]["tiers"]["sonnet"] = ""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "claude-delegation.jsonl"
            cfg["claude_delegation"]["log_path"] = str(log)
            payload, calls = self._call({"tasks": [{"goal": "g"}], "tier": "sonnet"}, cfg=cfg)
            entry = json.loads(log.read_text(encoding="utf-8").strip())
        self.assertIn("has no model", payload["error"])
        self.assertEqual(calls, [])
        self.assertEqual(entry["outcome"], "refused")
        self.assertIn("has no model", entry["message"])

    def _call_with_resolver_capture(self, args, *, cfg=None, parent=None, usage=40.0, result=None):
        """Like _call, but the fake delegate_task also calls the installed resolver wrapper
        (tools.delegate_tool._resolve_child_runtime) with realistic kwargs, so tests can observe
        whether/how the reasoning-effort bridge substituted reasoning_config -- without any
        production test seam."""
        import sys
        parent = parent if parent is not None else SimpleNamespace(_delegate_depth=0)
        resolver_calls = []

        def delegate_task(**kwargs):
            resolver = sys.modules["tools.delegate_tool"]._resolve_child_runtime
            credentials_cfg = kwargs.get("credentials_cfg") or {}
            resolved = resolver(
                parent_agent=kwargs.get("parent_agent"),
                delegation_cfg={},
                parent_api_key=None,
                model=credentials_cfg.get("model"),
                override_provider=credentials_cfg.get("provider"),
                override_base_url=None,
                override_api_key=None,
                override_api_mode=None,
                override_acp_command=None,
                override_acp_args=None,
            )
            resolver_calls.append(resolved)
            return json.dumps(result if result is not None else {"status": "dispatched", "delegation_id": "d1"})

        reading = None if usage is None else _reading(usage)
        with patch("model_router._load_config", return_value=cfg or _cfg()), \
             patch.object(claude_delegation, "_host", lambda: (delegate_task, lambda: parent)), \
             patch.object(usage_guard, "read", return_value=reading):
            raw = handle_delegate_claude(args)
        return json.loads(raw), resolver_calls

    def test_an_opus_request_lowered_to_sonnet_scopes_sonnet_effort(self):
        self.addCleanup(claude_delegation._reset_reasoning_bridge_for_tests)

        def fake_resolver(*, parent_agent, model, override_provider, **_kwargs):
            return {"provider": override_provider, "model": model, "base_url": "", "requested_provider": override_provider}

        import sys
        import types
        fake_module = types.ModuleType("tools.delegate_tool")
        fake_module._resolve_child_runtime = fake_resolver
        cfg = _cfg()
        cfg["claude_delegation"]["reasoning_effort"] = {"sonnet": "high", "opus": "low"}
        with patch.dict(sys.modules, {"tools.delegate_tool": fake_module,
                                       "tools.delegate_tool_config": fake_module}):
            ok, reason = claude_delegation.install_reasoning_bridge()
            self.assertTrue(ok, reason)
            payload, resolver_results = self._call_with_resolver_capture(
                {"tasks": [{"goal": "g"}], "tier": "opus"}, cfg=cfg, usage=75.0,
            )
        self.assertEqual(payload["claude_tier"], "sonnet")
        self.assertEqual(len(resolver_results), 1)
        self.assertEqual(resolver_results[0]["reasoning_config"], {"enabled": True, "effort": "high"})

    def test_a_haiku_call_never_sets_a_scope_and_bypasses_bridge_unavailability(self):
        """Haiku must delegate normally regardless of bridge state -- extended thinking is
        unsupported for Haiku, so the bridge is simply irrelevant to it."""
        self.addCleanup(claude_delegation._reset_reasoning_bridge_for_tests)
        with patch.object(claude_delegation, "install_reasoning_bridge", return_value=(False, "boom")):
            payload, calls = self._call({"tasks": [{"goal": "g"}], "tier": "haiku"})
        self.assertEqual(payload["claude_tier"], "haiku")
        self.assertEqual(len(calls), 1)

    def test_a_sonnet_call_audits_when_the_host_never_applies_its_reasoning_effort(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = _cfg()
            log = Path(directory) / "claude-delegation.jsonl"
            cfg["claude_delegation"]["log_path"] = str(log)
            with patch.object(claude_delegation, "install_reasoning_bridge", return_value=(True, "")):
                payload, calls = self._call({"tasks": [{"goal": "g"}], "tier": "sonnet"}, cfg=cfg)
            entry = json.loads(log.read_text(encoding="utf-8").strip())
        self.assertEqual(len(calls), 1)
        self.assertEqual(payload["reasoning_effort"], "not applied")
        self.assertEqual(entry["outcome"], "error")
        self.assertEqual(entry["message"], "reasoning effort not applied: host seam did not see the child")

    def test_a_haiku_call_is_not_audited_for_missing_reasoning_effort(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = _cfg()
            log = Path(directory) / "claude-delegation.jsonl"
            cfg["claude_delegation"]["log_path"] = str(log)
            payload, calls = self._call({"tasks": [{"goal": "g"}], "tier": "haiku"}, cfg=cfg)
            entry = json.loads(log.read_text(encoding="utf-8").strip())
        self.assertEqual(len(calls), 1)
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(entry["outcome"], "ran")

    def test_a_non_haiku_call_is_refused_when_the_bridge_is_unavailable(self):
        self.addCleanup(claude_delegation._reset_reasoning_bridge_for_tests)
        with tempfile.TemporaryDirectory() as directory:
            cfg = _cfg()
            log = Path(directory) / "claude-delegation.jsonl"
            cfg["claude_delegation"]["log_path"] = str(log)
            with patch.object(claude_delegation, "install_reasoning_bridge", return_value=(False, "host seam moved")):
                payload, calls = self._call({"tasks": [{"goal": "g"}], "tier": "sonnet"}, cfg=cfg)
            entry = json.loads(log.read_text(encoding="utf-8").strip())
        self.assertIn("Claude reasoning effort is unavailable", payload["error"])
        self.assertIn("host seam moved", payload["error"])
        self.assertEqual(calls, [])
        self.assertEqual(entry["outcome"], "refused")
        self.assertIn("Claude reasoning effort is unavailable", entry["message"])
        self.assertIn("host seam moved", entry["message"])

    def test_a_tier_without_a_reasoning_level_is_refused_and_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = _cfg()
            log = Path(directory) / "claude-delegation.jsonl"
            cfg["claude_delegation"]["log_path"] = str(log)
            with patch.object(claude_delegation, "install_reasoning_bridge", return_value=(True, "")), \
                 patch.object(claude_delegation, "reasoning_effort_config", return_value={}):
                payload, calls = self._call({"tasks": [{"goal": "g"}], "tier": "sonnet"}, cfg=cfg)
            entry = json.loads(log.read_text(encoding="utf-8").strip())
        self.assertEqual(payload["error"], "Claude reasoning effort for sonnet is not configured")
        self.assertEqual(calls, [])
        self.assertEqual(entry["outcome"], "refused")
        self.assertEqual(entry["message"], "Claude reasoning effort for sonnet is not configured")

    def test_an_unparseable_reasoning_level_is_refused_as_invalid_and_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = _cfg()
            log = Path(directory) / "claude-delegation.jsonl"
            cfg["claude_delegation"]["log_path"] = str(log)
            with patch.object(claude_delegation, "install_reasoning_bridge", return_value=(True, "")), \
                 patch.object(claude_delegation, "reasoning_effort_config", return_value={"sonnet": "bogus"}):
                payload, calls = self._call({"tasks": [{"goal": "g"}], "tier": "sonnet"}, cfg=cfg)
            entry = json.loads(log.read_text(encoding="utf-8").strip())
        self.assertEqual(payload["error"], "Claude reasoning effort for sonnet is invalid")
        self.assertEqual(calls, [])
        self.assertEqual(entry["outcome"], "refused")
        self.assertEqual(entry["message"], "Claude reasoning effort for sonnet is invalid")


class ClaudeReasoningConfigTests(unittest.TestCase):
    def test_defaults_expose_only_editable_sonnet_and_opus_levels(self):
        self.assertEqual(
            claude_delegation.reasoning_effort_config({"claude_delegation": {}}),
            {"sonnet": "medium", "opus": "medium"},
        )

    def test_invalid_or_haiku_config_values_fall_back_to_the_safe_default(self):
        config = {"claude_delegation": {"reasoning_effort": {
            "sonnet": " HIGH ", "opus": "external", "haiku": "xhigh",
        }}}
        self.assertEqual(
            claude_delegation.reasoning_effort_config(config),
            {"sonnet": "high", "opus": "medium"},
        )


class ReasoningBridgeTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(claude_delegation._reset_reasoning_bridge_for_tests)

    def test_a_missing_runtime_resolver_is_reported_as_unavailable(self):
        fake = types.ModuleType("tools.delegate_tool")
        fake_config = types.ModuleType("tools.delegate_tool_config")
        with patch.dict(sys.modules, {"tools.delegate_tool": fake, "tools.delegate_tool_config": fake_config}):
            ok, reason = claude_delegation.install_reasoning_bridge()
        self.assertFalse(ok)
        self.assertIn("_resolve_child_runtime", reason)

    def test_a_non_callable_runtime_resolver_is_reported_as_unavailable(self):
        fake = types.ModuleType("tools.delegate_tool")
        fake._resolve_child_runtime = object()
        fake_config = types.ModuleType("tools.delegate_tool_config")
        fake_config._resolve_child_runtime = fake._resolve_child_runtime
        with patch.dict(sys.modules, {"tools.delegate_tool": fake, "tools.delegate_tool_config": fake_config}):
            ok, reason = claude_delegation.install_reasoning_bridge()
        self.assertFalse(ok)
        self.assertIn("not callable", reason)

    def test_the_bridge_install_is_idempotent_and_records_availability(self):
        ok, reason = claude_delegation.install_reasoning_bridge()
        self.assertTrue(ok, reason)
        self.assertEqual(claude_delegation.reasoning_bridge_status(), (True, ""))
        # Re-install must be a no-op that still reports available (idempotent).
        ok2, reason2 = claude_delegation.install_reasoning_bridge()
        self.assertTrue(ok2, reason2)
        self.assertEqual(claude_delegation.reasoning_bridge_status(), (True, ""))

    def test_the_wrapper_counts_a_matching_reasoning_substitution(self):
        def resolver(*, parent_agent, model, override_provider):
            return {"provider": override_provider, "model": model}

        parent = SimpleNamespace()
        scope = claude_delegation._ReasoningScope(
            parent, "sonnet", "claude-sonnet-5-5", {"enabled": True, "effort": "high"},
        )
        token = claude_delegation._REASONING_SCOPE.set(scope)
        try:
            result = claude_delegation._wrap_resolve_child_runtime(resolver)(
                parent_agent=parent, model="claude-sonnet-5-5", override_provider="anthropic",
            )
        finally:
            claude_delegation._REASONING_SCOPE.reset(token)
        self.assertEqual(result["reasoning_config"], {"enabled": True, "effort": "high"})
        self.assertEqual(scope.applied[0], 1)

    def test_an_unchanged_installed_seam_skips_full_validation(self):
        def resolver(*, parent_agent, model, override_provider):
            return {"model": model}

        fake = types.ModuleType("tools.delegate_tool")
        fake._resolve_child_runtime = resolver
        fake_config = types.ModuleType("tools.delegate_tool_config")
        fake_config._resolve_child_runtime = resolver
        with patch.dict(sys.modules, {"tools.delegate_tool": fake, "tools.delegate_tool_config": fake_config}):
            ok, reason = claude_delegation.install_reasoning_bridge()
            self.assertTrue(ok, reason)
            wrapper = fake._resolve_child_runtime
            with patch.object(claude_delegation, "_validate_reasoning_bridge_seam",
                              side_effect=AssertionError("full validation should not run")):
                self.assertEqual(claude_delegation.install_reasoning_bridge(), (True, ""))
        self.assertIs(fake._resolve_child_runtime, wrapper)

    def test_a_replaced_installed_seam_is_revalidated(self):
        def resolver(*, parent_agent, model, override_provider):
            return {"model": model}

        fake = types.ModuleType("tools.delegate_tool")
        fake._resolve_child_runtime = resolver
        fake_config = types.ModuleType("tools.delegate_tool_config")
        fake_config._resolve_child_runtime = resolver
        with patch.dict(sys.modules, {"tools.delegate_tool": fake, "tools.delegate_tool_config": fake_config}):
            ok, reason = claude_delegation.install_reasoning_bridge()
            self.assertTrue(ok, reason)
            fake._resolve_child_runtime = resolver
            with patch.object(claude_delegation, "_validate_reasoning_bridge_seam",
                              return_value=(False, "host seam moved", fake, resolver)) as validate:
                self.assertEqual(claude_delegation.install_reasoning_bridge(), (False, "host seam moved"))
            validate.assert_called_once_with()

    def test_install_survives_a_second_module_copy_wrapping_the_seam_first(self):
        """A second import of this module (plugin reload, or importing it both as
        ``claude_delegation`` and ``model_router.claude_delegation``) must not
        permanently disable the bridge for either copy.

        Simulated here by hand-installing a foreign wrapper that carries
        ``__model_router_original__`` -- exactly what this module's own wrapper
        looks like from a second copy's point of view -- directly onto the real
        host seam, bypassing this module's own bookkeeping.
        """
        import tools.delegate_tool as delegate_tool

        real_original = delegate_tool._resolve_child_runtime

        def foreign_wrapper(*args, **kwargs):
            return real_original(*args, **kwargs)

        foreign_wrapper.__model_router_original__ = real_original
        delegate_tool._resolve_child_runtime = foreign_wrapper
        try:
            ok, reason = claude_delegation.install_reasoning_bridge()
            self.assertTrue(ok, reason)
            self.assertEqual(reason, "")
            installed = delegate_tool._resolve_child_runtime
            self.assertIs(installed.__model_router_original__, real_original)
        finally:
            claude_delegation._reset_reasoning_bridge_for_tests()
            delegate_tool._resolve_child_runtime = real_original
        self.assertIs(delegate_tool._resolve_child_runtime, real_original)

    def test_two_module_copies_share_scope_and_lock_for_the_installed_wrapper(self):
        """A scope from copy A must configure the wrapper installed by copy B."""
        import importlib.util

        source = Path(claude_delegation.__file__)
        copy_names = ("model_router._claude_reasoning_copy_a", "model_router._claude_reasoning_copy_b")
        for name in copy_names:
            sys.modules.pop(name, None)
            self.addCleanup(sys.modules.pop, name, None)

        def resolver(*, parent_agent, model, override_provider, **_kwargs):
            return {"provider": override_provider, "model": model}

        delegate_tool = types.ModuleType("tools.delegate_tool")
        delegate_tool._resolve_child_runtime = resolver
        delegate_tool_config = types.ModuleType("tools.delegate_tool_config")
        delegate_tool_config._resolve_child_runtime = resolver
        with patch.dict(sys.modules, {
            "tools.delegate_tool": delegate_tool,
            "tools.delegate_tool_config": delegate_tool_config,
        }):
            copies = []
            for name in copy_names:
                spec = importlib.util.spec_from_file_location(name, source)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                copies.append(module)
            copy_a, copy_b = copies
            self.addCleanup(copy_a._reset_reasoning_bridge_for_tests)
            self.addCleanup(copy_b._reset_reasoning_bridge_for_tests)

            self.assertIs(copy_a._REASONING_BRIDGE_LOCK, copy_b._REASONING_BRIDGE_LOCK)
            self.assertIs(copy_a._REASONING_SCOPE, copy_b._REASONING_SCOPE)
            self.assertEqual(copy_b.install_reasoning_bridge(), (True, ""))

            parent = SimpleNamespace()
            with copy_a.reasoning_scope(parent, "sonnet", "claude-sonnet-5-5", {"enabled": True, "effort": "high"}):
                result = delegate_tool._resolve_child_runtime(
                    parent_agent=parent, model="claude-sonnet-5-5", override_provider="anthropic",
                )
        self.assertEqual(result["reasoning_config"], {"enabled": True, "effort": "high"})

    def test_a_newer_copy_backfills_an_older_shared_holder(self):
        """A copy can start when an existing holder predates newer state fields."""
        import importlib.util

        source = Path(claude_delegation.__file__)
        copy_name = "model_router._claude_reasoning_older_holder_copy"
        sys.modules.pop(copy_name, None)
        self.addCleanup(sys.modules.pop, copy_name, None)
        older_scope = claude_delegation.ContextVar("older_claude_delegation_reasoning_scope", default=None)
        older_lock = claude_delegation.threading.Lock()
        older_holder = SimpleNamespace(
            scope=older_scope,
            lock=older_lock,
            installed=True,
            reason="installed by an older copy",
        )
        with patch.dict(sys.modules, {claude_delegation._REASONING_BRIDGE_STATE_KEY: older_holder}):
            spec = importlib.util.spec_from_file_location(copy_name, source)
            copy = importlib.util.module_from_spec(spec)
            sys.modules[copy_name] = copy
            spec.loader.exec_module(copy)

            self.assertIs(copy._REASONING_SCOPE, older_scope)
            self.assertIs(copy._REASONING_BRIDGE_LOCK, older_lock)
            self.assertTrue(older_holder.installed)
            self.assertEqual(older_holder.reason, "installed by an older copy")
            self.assertIsNone(older_holder.original)
            self.assertIsNone(older_holder.wrapper)


class ReasoningBridgeCompatibilityTests(unittest.TestCase):
    """The side-effect-free probe a separate dashboard process can call safely."""

    def setUp(self):
        self.addCleanup(claude_delegation._reset_reasoning_bridge_for_tests)

    def test_the_probe_reports_compatible_on_the_real_host_without_installing(self):
        import tools.delegate_tool as delegate_tool

        before = delegate_tool._resolve_child_runtime
        ok, reason = claude_delegation.reasoning_bridge_compatibility()
        self.assertEqual((ok, reason), (True, ""))
        # Never installs: the real host's resolver is untouched and the module's
        # own bridge-installed state stays False, unlike install_reasoning_bridge().
        self.assertIs(delegate_tool._resolve_child_runtime, before)
        self.assertFalse(claude_delegation._REASONING_BRIDGE_STATE.installed)

    def test_the_probe_reports_incompatible_for_a_fake_module_missing_the_resolver(self):
        fake = types.ModuleType("tools.delegate_tool")
        with patch.dict(sys.modules, {"tools.delegate_tool": fake}):
            ok, reason = claude_delegation.reasoning_bridge_compatibility()
        self.assertFalse(ok)
        self.assertIn("_resolve_child_runtime", reason)

    def test_status_reports_available_on_the_real_host_before_any_install(self):
        # A fresh, uninstalled state (e.g. a standalone dashboard process that
        # never calls install_reasoning_bridge()) must still see the seam as
        # available whenever it is compatible: "available" means "the host
        # seam is compatible", not "this process installed the wrapper".
        # Force a clean start: another test elsewhere in the suite may have
        # installed the real bridge without resetting it afterward, and this
        # test's whole point is to observe the state BEFORE any install.
        claude_delegation._reset_reasoning_bridge_for_tests()
        self.assertFalse(claude_delegation._REASONING_BRIDGE_STATE.installed)
        self.assertEqual(claude_delegation.reasoning_bridge_status(), (True, ""))


class ReasoningBridgeStateHolderTests(unittest.TestCase):
    """B-M3: the shared per-process holder must be a real module object, not a
    SimpleNamespace, so code that walks sys.modules expecting modules (reload
    tooling, some warning/pickle helpers, inspect.getmodule) does not trip on it."""

    def test_the_state_holder_is_a_real_module_registered_under_its_key(self):
        self.assertIsInstance(claude_delegation._REASONING_BRIDGE_STATE, types.ModuleType)
        self.assertIs(
            sys.modules[claude_delegation._REASONING_BRIDGE_STATE_KEY],
            claude_delegation._REASONING_BRIDGE_STATE,
        )


class ReasoningBridgeStatusAfterReplacementTests(unittest.TestCase):
    """B-M4: status must notice the host function was swapped out from under an
    installed bridge, not just trust the ``installed`` flag forever."""

    def setUp(self):
        self.addCleanup(claude_delegation._reset_reasoning_bridge_for_tests)

    def test_status_reports_unavailable_once_the_host_seam_is_replaced(self):
        ok, reason = claude_delegation.install_reasoning_bridge()
        self.assertTrue(ok, reason)
        self.assertEqual(claude_delegation.reasoning_bridge_status(), (True, ""))

        import tools.delegate_tool as delegate_tool

        wrapper = claude_delegation._REASONING_BRIDGE_STATE.wrapper
        replacement = lambda **kwargs: None
        delegate_tool._resolve_child_runtime = replacement
        try:
            ok2, reason2 = claude_delegation.reasoning_bridge_status()
            self.assertFalse(ok2)
            self.assertNotEqual(reason2, "")
            self.assertIn("_resolve_child_runtime", reason2)
        finally:
            delegate_tool._resolve_child_runtime = wrapper


class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, claude_delegation, "_ACTIVE", False)

    def test_an_enabled_wing_registers_in_the_delegation_toolset(self):
        ctx = MagicMock()
        with patch.object(claude_delegation, "host_check", return_value=(True, "")), \
             patch.object(claude_delegation, "_independent_completions", return_value=False), \
             patch.object(claude_delegation, "_exempt_from_sequential_deadline", return_value=True):
            self.assertTrue(claude_delegation.register(ctx, _cfg()))
        kwargs = ctx.register_tool.call_args.kwargs
        self.assertEqual((kwargs["name"], kwargs["toolset"]), ("delegate_claude", "delegation"))
        self.assertIs(kwargs["handler"], handle_delegate_claude)
        self.assertTrue(claude_delegation.is_active())

    def test_a_host_without_the_api_registers_nothing(self):
        ctx = MagicMock()
        with patch.object(claude_delegation, "host_check", return_value=(False, "delegate_task lacks credentials_cfg")):
            self.assertFalse(claude_delegation.register(ctx, _cfg()))
        ctx.register_tool.assert_not_called()
        self.assertFalse(claude_delegation.is_active())

    def test_register_tool_returning_none_means_not_registered(self):
        ctx = MagicMock()
        ctx.register_tool.return_value = None
        with patch.object(claude_delegation, "host_check", return_value=(True, "")), \
             patch.object(claude_delegation, "_independent_completions", return_value=False), \
             patch.object(claude_delegation, "_exempt_from_sequential_deadline", return_value=True):
            self.assertFalse(claude_delegation.register(ctx, _cfg()))
        self.assertFalse(claude_delegation.is_active())

    def test_register_calls_the_deadline_exemption_only_after_success(self):
        ctx = MagicMock()
        exempt = MagicMock(return_value=True)
        with patch.object(claude_delegation, "host_check", return_value=(True, "")), \
             patch.object(claude_delegation, "_independent_completions", return_value=False), \
             patch.object(claude_delegation, "_exempt_from_sequential_deadline", exempt):
            self.assertTrue(claude_delegation.register(ctx, _cfg()))
        exempt.assert_called_once_with()

    def test_register_does_not_block_when_the_exemption_fails(self):
        ctx = MagicMock()
        with patch.object(claude_delegation, "host_check", return_value=(True, "")), \
             patch.object(claude_delegation, "_independent_completions", return_value=False), \
             patch.object(claude_delegation, "_exempt_from_sequential_deadline", return_value=False):
            self.assertTrue(claude_delegation.register(ctx, _cfg()))
        self.assertTrue(claude_delegation.is_active())

    def test_registration_is_logged_either_way(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "claude-delegation.jsonl"
            cfg = _cfg()
            cfg["claude_delegation"]["log_path"] = str(log)
            with patch.object(claude_delegation, "host_check", return_value=(False, "delegate_task lacks credentials_cfg")):
                claude_delegation.register(MagicMock(), cfg)
            with patch.object(claude_delegation, "host_check", return_value=(True, "")), \
                 patch.object(claude_delegation, "_independent_completions", return_value=False), \
                 patch.object(claude_delegation, "_exempt_from_sequential_deadline", return_value=True):
                cfg["callable"].update(haiku=False, sonnet5=False, opus5=False)
                claude_delegation.register(MagicMock(), cfg)
            lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(l["event"], l["registered"]) for l in lines],
                         [("registration", False), ("registration", True)])
        self.assertIn("credentials_cfg", lines[0]["reason"])
        self.assertFalse(lines[1]["available"], "registered, but unavailable until a Claude model is switched on")


class SequentialDeadlineExemptionTests(unittest.TestCase):
    def test_a_fake_module_gains_delegate_claude_and_keeps_existing_members(self):
        fake = types.ModuleType("agent.tool_executor")
        fake._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS = frozenset({"delegate_task", "manage_connections"})
        with patch.dict(sys.modules, {"agent.tool_executor": fake}):
            self.assertTrue(claude_delegation._exempt_from_sequential_deadline())
        self.assertEqual(
            fake._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS,
            frozenset({"delegate_task", "manage_connections", "delegate_claude"}),
        )

    def test_a_missing_attribute_returns_false_without_raising(self):
        fake = types.ModuleType("agent.tool_executor")
        with patch.dict(sys.modules, {"agent.tool_executor": fake}):
            self.assertFalse(claude_delegation._exempt_from_sequential_deadline())

    def test_an_unimportable_module_returns_false_without_raising(self):
        with patch.dict(sys.modules, {"agent.tool_executor": None}):
            self.assertFalse(claude_delegation._exempt_from_sequential_deadline())


def _hermes_importable():
    try:
        import tools.delegate_tool  # noqa: F401
        return True
    except Exception:
        return False


class _StaleInstallMapFinder:
    """Fails the named Hermes imports the way an outdated editable-install map does.

    A Hermes update can add a top-level module (``hermes_yaml``) that the venv's
    install map does not list yet; the standalone dashboard, which has no
    PYTHONPATH, then cannot import ``tools.*`` / ``agent.*`` until the Hermes
    checkout is put on ``sys.path``. The test harness always puts it there, so
    this finder stands in for the stale map: it refuses the names until
    ``checkout`` is on ``sys.path``, then steps aside for the normal finders.
    """

    def __init__(self, names, checkout):
        self.names = set(names)
        self.checkout = checkout

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.names and self.checkout not in sys.path:
            raise ModuleNotFoundError("No module named 'hermes_yaml'", name="hermes_yaml")
        return None


@unittest.skipUnless(_hermes_importable(), "Hermes is not importable in this interpreter")
class StaleInstallMapTests(unittest.TestCase):
    """The dashboard must see the Claude seam without a usage Refresh having run first."""

    NAMES = ("tools.delegate_tool", "tools.delegate_tool_config", "agent.subagent_lifecycle")

    def setUp(self):
        import importlib
        import agent
        import tools

        importlib.import_module("tools.delegate_tool")
        importlib.import_module("agent.subagent_lifecycle")
        checkout = tempfile.mkdtemp()
        # Put the real modules and their package attributes back afterwards: the
        # retried import re-executes them under fresh module objects.
        saved_attrs = [(tools, "delegate_tool", tools.delegate_tool),
                       (tools, "delegate_tool_config", tools.delegate_tool_config),
                       (agent, "subagent_lifecycle", agent.subagent_lifecycle)]
        modules = patch.dict(sys.modules)
        modules.start()
        self.addCleanup(modules.stop)
        for owner, attr, value in saved_attrs:
            self.addCleanup(setattr, owner, attr, value)
        for name in self.NAMES:
            sys.modules.pop(name, None)
        finder = _StaleInstallMapFinder(self.NAMES, checkout)
        sys.meta_path.insert(0, finder)
        self.addCleanup(sys.meta_path.remove, finder)
        self.addCleanup(lambda: sys.path.remove(checkout) if checkout in sys.path else None)
        hermes_path = patch.object(usage_guard, "hermes_path", return_value=Path(checkout))
        hermes_path.start()
        self.addCleanup(hermes_path.stop)
        self.addCleanup(claude_delegation._reset_reasoning_bridge_for_tests)

    def test_the_effort_seam_is_found_through_the_hermes_checkout(self):
        self.assertEqual(claude_delegation.reasoning_bridge_compatibility(), (True, ""))

    def test_the_host_check_finds_the_delegation_api_through_the_hermes_checkout(self):
        self.assertEqual(claude_delegation.host_check(), (True, ""))


@unittest.skipUnless(_hermes_importable(), "Hermes is not importable in this interpreter")
class RealHostTests(unittest.TestCase):
    """Against the installed Hermes: the guarantees the wing leans on."""

    def test_the_installed_hermes_passes_the_host_check(self):
        self.assertEqual(claude_delegation.host_check(), (True, ""))

    def test_a_leaf_loses_the_delegation_toolset(self):
        from tools.delegate_tool_toolsets import _strip_blocked_tools
        self.assertNotIn("delegation", _strip_blocked_tools(["delegation", "file"]))

    def test_the_exemption_helper_adds_delegate_claude_on_the_real_host(self):
        import agent.tool_executor as tool_executor

        original = tool_executor._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS
        self.addCleanup(setattr, tool_executor, "_SEQUENTIAL_DEADLINE_EXEMPT_TOOLS", original)
        self.assertTrue(claude_delegation._exempt_from_sequential_deadline())
        self.assertIn("delegate_task", tool_executor._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS)
        self.assertIn("delegate_claude", tool_executor._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS)

    def test_the_depth_limit_holds_for_delegate_claude(self):
        """Nothing spawns from an agent at max_spawn_depth, whichever tool asked."""
        self.addCleanup(claude_delegation._reset_reasoning_bridge_for_tests)
        parent = SimpleNamespace(_delegate_depth=99)
        with patch("model_router._load_config", return_value=_cfg()), \
             patch.object(claude_delegation, "_host", lambda: (claude_delegation._host_delegate_task(), lambda: parent)), \
             patch.object(usage_guard, "read", return_value=_reading(10)):
            payload = json.loads(handle_delegate_claude({"tasks": [{"goal": "g"}]}))
        self.assertIn("depth limit", payload["error"].lower())

    def test_the_child_builder_forwards_the_resolved_runtime_to_aiagent(self):
        import tools.delegate_tool as delegate_tool

        tree = ast.parse(textwrap.dedent(inspect.getsource(delegate_tool._build_child_agent)))
        resolver_assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_resolve_child_runtime"
        ]
        self.assertTrue(resolver_assignments, "_build_child_agent must call _resolve_child_runtime(...)")
        resolved_names = {
            target.id
            for node in resolver_assignments
            for target in ((node.targets if isinstance(node, ast.Assign) else [node.target]))
            if isinstance(target, ast.Name)
        }
        self.assertTrue(resolved_names,
                        "_build_child_agent must assign the resolved runtime to a plain name")
        forwards_resolved_runtime = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "AIAgent"
            and any(keyword.arg is None and isinstance(keyword.value, ast.Name)
                    and keyword.value.id in resolved_names for keyword in node.keywords)
            for node in ast.walk(tree)
        )
        self.assertTrue(forwards_resolved_runtime,
                        "_build_child_agent must pass its resolved runtime to AIAgent as **<resolved name>")

    def test_a_scoped_child_construction_receives_the_scoped_reasoning_config(self):
        """Proves the bridge on the real host seam, called the way real construction calls it.

        The brief's preferred shape is a real ``tools.delegate_tool._build_child_agent`` call
        with ``run_agent.AIAgent`` patched to a recorder. In this interpreter that path is
        impractical: ``_build_child_agent`` calls the real ``_load_config()`` from
        ``tools.delegate_tool_config``, which (independent of anything this plugin does) can kick
        off a real, network-bound `hermes update` dependency/build sync outside the sandbox
        (reproduced with a standalone probe: the hang starts inside ``_build_child_agent``, not at
        import time, and is unrelated to reasoning_config). So this proves the same guarantee one
        layer down: calling the real, installed ``tools.delegate_tool._resolve_child_runtime`` --
        already wrapped by ``install_reasoning_bridge()`` -- with the exact kwargs
        ``_build_child_agent`` passes it (see tools/delegate_tool.py:219-225), while the scope is
        active for that exact child. This is the same call ``ReasoningScopeIsolationTests`` below
        exercises repeatedly; kept here too since it is this test class's natural home per the
        brief's Step 5/6.
        """
        import sys
        import tools.delegate_tool as real_delegate_tool

        self.addCleanup(claude_delegation._reset_reasoning_bridge_for_tests)
        ok, reason = claude_delegation.install_reasoning_bridge()
        self.assertTrue(ok, reason)

        parent = SimpleNamespace(
            _delegate_depth=0, model="claude-opus-5-5", provider="anthropic", base_url="https://api.anthropic.com",
            api_key="k", request_overrides={}, session_id="sess-real",
        )
        resolver = sys.modules["tools.delegate_tool"]._resolve_child_runtime
        with claude_delegation.reasoning_scope(
            parent, "sonnet", "claude-sonnet-5-5", {"enabled": True, "effort": "high"},
        ):
            # Same call shape as tools/delegate_tool.py's _build_child_agent, ~219-225.
            rt = resolver(
                parent, delegation_cfg={}, parent_api_key="k", model="claude-sonnet-5-5",
                override_provider="anthropic", override_base_url=None, override_api_key=None,
                override_api_mode=None, override_acp_command=None, override_acp_args=None,
                routing_cfg=None,
            )
        self.assertEqual(rt.get("reasoning_config"), {"enabled": True, "effort": "high"})

    def test_the_scoped_config_produces_the_real_anthropic_wire_shape(self):
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            "claude-sonnet-5-5", [], [], 4096, {"enabled": True, "effort": "high"},
        )
        self.assertEqual(kwargs["thinking"]["type"], "adaptive")
        self.assertEqual(kwargs["output_config"]["effort"], "high")

        haiku = build_anthropic_kwargs(
            "claude-haiku-4-5-20251001", [], [], 4096, {"enabled": True, "effort": "high"},
        )
        self.assertNotIn("thinking", haiku)
        self.assertNotIn("output_config", haiku)

    def test_a_disabled_bridge_fixture_refuses_a_non_haiku_call_on_the_real_host(self):
        """The compatibility refusal path, proven against the real host_check-passing
        interpreter with a deliberately disabled bridge fixture."""
        self.addCleanup(claude_delegation._reset_reasoning_bridge_for_tests)
        with patch.object(claude_delegation, "install_reasoning_bridge", return_value=(False, "disabled for this test")):
            with patch("model_router._load_config", return_value=_cfg()), \
                 patch.object(claude_delegation, "_host",
                              lambda: (claude_delegation._host_delegate_task(), lambda: SimpleNamespace(_delegate_depth=0))), \
                 patch.object(usage_guard, "read", return_value=_reading(10)):
                payload = json.loads(handle_delegate_claude({"tasks": [{"goal": "g"}], "tier": "sonnet"}))
        self.assertIn("Claude reasoning effort is unavailable", payload["error"])
        self.assertIn("disabled for this test", payload["error"])


def _parent_stub():
    """A parent agent with the attributes the host resolver reads directly
    (``model``, ``base_url``); everything else it reads through getattr."""
    return SimpleNamespace(model=None, base_url=None)


class ReasoningScopeIsolationTests(unittest.TestCase):
    """Deferred Task 1 finding: the wrapper's substitution/non-substitution logic, covered
    directly against the real installed host seam."""

    def setUp(self):
        self.addCleanup(claude_delegation._reset_reasoning_bridge_for_tests)
        ok, reason = claude_delegation.install_reasoning_bridge()
        self.assertTrue(ok, reason)
        import sys
        self.resolver = sys.modules["tools.delegate_tool"]._resolve_child_runtime

    def _resolve(self, *, parent_agent, model, override_provider="anthropic"):
        return self.resolver(
            parent_agent=parent_agent, delegation_cfg={}, parent_api_key=None, model=model,
            override_provider=override_provider, override_base_url=None, override_api_key=None,
            override_api_mode=None, override_acp_command=None, override_acp_args=None,
        )

    def test_a_matching_scope_substitutes_the_reasoning_config(self):
        parent = _parent_stub()
        with claude_delegation.reasoning_scope(parent, "sonnet", "claude-sonnet-5-5", {"enabled": True, "effort": "high"}):
            result = self._resolve(parent_agent=parent, model="claude-sonnet-5-5")
        self.assertEqual(result["reasoning_config"], {"enabled": True, "effort": "high"})

    def _stub_original_resolver(self, sentinel):
        """Re-point the *real* host's ``_resolve_child_runtime`` at a stub returning
        ``sentinel``, then reinstall the bridge so the wrapper's ``original`` closure
        captures that stub instead of the real Hermes resolver -- restored on cleanup.

        ``install_reasoning_bridge()``'s wrapper closes over its ``original`` argument
        directly (see ``_wrap_resolve_child_runtime``); it does not re-read
        ``_REASONING_BRIDGE_STATE.original`` on every call. So proving identity passthrough
        requires the stub to be the thing the wrapper actually calls, not just a
        bookkeeping global -- otherwise this test would pass against a real resolver's
        freshly built dict and never catch a wrapper bug that returns a copy.
        """
        import sys
        delegate_tool = sys.modules["tools.delegate_tool"]
        delegate_tool_config = sys.modules["tools.delegate_tool_config"]
        real_original = claude_delegation._REASONING_BRIDGE_STATE.original
        self.assertIsNotNone(real_original, "bridge must already be installed")

        def stub(*, parent_agent=None, delegation_cfg=None, parent_api_key=None, model=None,
                 override_provider=None, override_base_url=None, override_api_key=None,
                 override_api_mode=None, override_acp_command=None, override_acp_args=None,
                 routing_cfg=None):
            return sentinel

        # Uninstall first so install_reasoning_bridge() sees an unwrapped resolver and
        # is willing to wrap again (it treats an already-wrapped current as a no-op).
        claude_delegation._reset_reasoning_bridge_for_tests()
        delegate_tool._resolve_child_runtime = stub
        delegate_tool_config._resolve_child_runtime = stub
        ok, reason = claude_delegation.install_reasoning_bridge()
        self.assertTrue(ok, reason)
        self.resolver = sys.modules["tools.delegate_tool"]._resolve_child_runtime

        def _restore():
            claude_delegation._reset_reasoning_bridge_for_tests()
            delegate_tool._resolve_child_runtime = real_original
            delegate_tool_config._resolve_child_runtime = real_original
            ok2, reason2 = claude_delegation.install_reasoning_bridge()
            self.assertTrue(ok2, reason2)
            self.resolver = sys.modules["tools.delegate_tool"]._resolve_child_runtime

        self.addCleanup(_restore)

    def test_an_unscoped_call_returns_the_original_result_object(self):
        sentinel = {"marker": object()}
        self._stub_original_resolver(sentinel)
        parent = _parent_stub()
        # No scope active: the wrapper must pass the host's object straight through,
        # unchanged, not a copy of it.
        returned = self._resolve(parent_agent=parent, model="claude-sonnet-5-5")
        self.assertIs(returned, sentinel)

    def test_a_different_parent_object_is_not_substituted(self):
        sentinel = {"marker": object()}
        self._stub_original_resolver(sentinel)
        scoped_parent, other_parent = _parent_stub(), _parent_stub()
        with claude_delegation.reasoning_scope(scoped_parent, "sonnet", "claude-sonnet-5-5",
                                                {"enabled": True, "effort": "high"}):
            returned = self._resolve(parent_agent=other_parent, model="claude-sonnet-5-5")
        self.assertIs(returned, sentinel)

    def test_a_non_anthropic_override_provider_is_not_substituted(self):
        sentinel = {"marker": object()}
        self._stub_original_resolver(sentinel)
        parent = _parent_stub()
        with claude_delegation.reasoning_scope(parent, "sonnet", "claude-sonnet-5-5",
                                                {"enabled": True, "effort": "high"}):
            returned = self._resolve(parent_agent=parent, model="claude-sonnet-5-5", override_provider="openai-codex")
        self.assertIs(returned, sentinel)

    def test_a_different_model_is_not_substituted(self):
        sentinel = {"marker": object()}
        self._stub_original_resolver(sentinel)
        parent = _parent_stub()
        with claude_delegation.reasoning_scope(parent, "sonnet", "claude-sonnet-5-5",
                                                {"enabled": True, "effort": "high"}):
            returned = self._resolve(parent_agent=parent, model="claude-opus-5-5")
        self.assertIs(returned, sentinel)

    def test_two_threads_each_get_their_own_scoped_effort(self):
        import threading

        results = {}
        errors = []

        def worker(name, parent, model, effort):
            try:
                with claude_delegation.reasoning_scope(parent, "sonnet", model, {"enabled": True, "effort": effort}):
                    results[name] = self._resolve(parent_agent=parent, model=model)
            except Exception as exc:  # pragma: no cover - surfaced via errors list
                errors.append((name, exc))

        parent_a, parent_b = _parent_stub(), _parent_stub()
        t1 = threading.Thread(target=worker, args=("a", parent_a, "claude-sonnet-5-5", "high"))
        t2 = threading.Thread(target=worker, args=("b", parent_b, "claude-opus-5-5", "low"))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
        self.assertEqual(errors, [])
        self.assertFalse(t1.is_alive())
        self.assertFalse(t2.is_alive())
        self.assertEqual(results["a"]["reasoning_config"], {"enabled": True, "effort": "high"})
        self.assertEqual(results["b"]["reasoning_config"], {"enabled": True, "effort": "low"})


import model_router  # noqa: E402

try:
    import yaml  # noqa: E402
except ImportError:  # pragma: no cover
    yaml = None


class OfferedNamesTests(unittest.TestCase):
    def _hermes_config(self, directory):
        path = Path(directory) / "config.yaml"
        path.write_text(
            "delegation:\n  targets:\n"
            "    opus5: {provider: anthropic, model: claude-opus-5-5}\n"
            "    sonnet5: {provider: anthropic, model: claude-sonnet-5-5}\n",
            encoding="utf-8",
        )
        return path

    def test_an_inactive_wing_observes_models_without_offering_them(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(model_router, "_HERMES_CONFIG_PATH", self._hermes_config(directory)), \
             patch.object(claude_delegation, "_ACTIVE", False), \
             patch("model_router._load_config", return_value=_cfg()):
            self.assertEqual(model_router._delegation_target_names(), ("opus5", "sonnet5"))
            self.assertEqual(model_router._external_target_for_model("claude-haiku-4-5-20251001"), "haiku")

    def test_an_active_wing_offers_and_counts_haiku(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(model_router, "_HERMES_CONFIG_PATH", self._hermes_config(directory)), \
             patch.object(claude_delegation, "_ACTIVE", True), \
             patch("model_router._load_config", return_value=_cfg()):
            self.assertEqual(model_router._delegation_target_names(), ("haiku", "opus5", "sonnet5"))
            self.assertEqual(model_router._external_target_for_model("claude-haiku-4-5-20251001"), "haiku")


@unittest.skipIf(yaml is None, "PyYAML missing")
class ShippedConfigTests(unittest.TestCase):
    def setUp(self):
        path = Path(model_router.__file__).resolve().parent / "router_config.yaml"
        self.cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_the_wing_ships_with_its_files_and_without_an_on_off_switch(self):
        # Availability is the Claude switches in `callable`, which ship off.
        settings = self.cfg["claude_delegation"]
        self.assertNotIn("workflow", self.cfg)
        self.assertNotIn("enabled", settings)
        self.assertEqual(settings["tiers"], CLAUDE_DELEGATION["tiers"])
        self.assertEqual(settings["default_tier"], "sonnet")
        self.assertNotIn("usage_guard", settings)
        self.assertEqual(settings["log_path"], "~/.hermes/logs/claude-delegation.jsonl")

    def test_the_usage_guard_covers_both_accounts(self):
        guard = self.cfg["usage_guard"]
        self.assertEqual(guard["state_path"], "~/.hermes/state/model-router-usage.json")
        self.assertEqual(guard["accounts"]["anthropic"]["step_down"], {"opus5": "sonnet5"})
        self.assertEqual(guard["accounts"]["openai-codex"]["step_down"], {"sol": "terra"})
        # The limits are the operator's to tune from the dashboard (the live file is
        # this one), so pin their shape, not the shipped 70/90.
        for account in ("anthropic", "openai-codex"):
            soft = guard["accounts"][account]["soft_percent"]
            hard = guard["accounts"][account]["hard_percent"]
            self.assertTrue(0 < soft < hard <= 100, (account, soft, hard))

    def test_haiku_is_a_known_claude_target(self):
        # Shipped off like every Claude model: it needs a Claude login first.
        self.assertIs(self.cfg["callable"]["haiku"], False)
        self.assertEqual(self.cfg["tier_providers"]["haiku"], "anthropic")
        self.assertIn("haiku", self.cfg["peer_groups"]["light"])

    def test_the_starting_preferences(self):
        # Shipped as master had them: the built-in routes. An operator's chains live
        # in router_config.local.yaml; the shipped file carries them as a comment.
        self.assertEqual(self.cfg["preferences"], {})


from model_router import (  # noqa: E402
    _claude_target_sentence,
    _dispatch_phrase,
    _host_delegate_has_model,
    _model_param_contract,
    _preference_sentence,
    _prepare_orchestration_delegation,
    _quota_redispatch_instruction,
    _without_router_contract,
)
from model_router.test_external_orchestrator import _delegating_request  # noqa: E402
from model_router.test_quota_redispatch import CFG as REDISPATCH_CFG  # noqa: E402
from model_router.test_quota_redispatch import TARGETS as REDISPATCH_TARGETS  # noqa: E402
from model_router.test_quota_redispatch import envelope, request_for  # noqa: E402

CONTRACT_TARGETS = ("haiku", "luna", "opus5", "sol", "sonnet5", "terra")


def _contract(cfg, *, active, model_param=True):
    with patch.object(claude_delegation, "_ACTIVE", active), \
         patch("model_router._delegation_target_names", return_value=CONTRACT_TARGETS), \
         patch("model_router._tier_cooldown_remaining", return_value=0.0), \
         patch("model_router._recent_account_load", return_value={}):
        return _model_param_contract("terra", cfg, model_param=model_param)


class ContractTextTests(unittest.TestCase):
    def test_the_host_schema_decides_whether_a_model_parameter_exists(self):
        request = _delegating_request()
        self.assertFalse(_host_delegate_has_model(request))
        request["tools"][0]["parameters"]["properties"]["model"] = {"type": "string"}
        self.assertTrue(_host_delegate_has_model(request))

    def test_no_model_parameter_means_no_instruction_to_set_one(self):
        contract = _contract(_cfg(), active=True, model_param=False)
        self.assertNotIn("Set the delegate_task 'model' parameter", contract)
        self.assertTrue(contract.startswith("Route choice for delegated workers"))

    def test_an_inactive_wing_respects_the_absent_model_parameter(self):
        contract = _contract(_cfg(), active=False, model_param=False)
        self.assertTrue(contract.startswith("Route choice for delegated workers"))
        self.assertNotIn("Set the delegate_task 'model' parameter", contract)

    def test_the_dispatch_phrase_names_the_goal_prefix_route_while_active(self):
        with patch.object(claude_delegation, "_ACTIVE", True):
            self.assertEqual(_dispatch_phrase("terra"), "delegate_task (goal prefix [terra])")
        with patch.object(claude_delegation, "_ACTIVE", False):
            self.assertEqual(_dispatch_phrase("terra"), "model:terra")
            self.assertEqual(_dispatch_phrase("opus5"), "model:opus5")

    def test_the_default_keeps_todays_opening(self):
        self.assertTrue(_contract(_cfg(), active=False).startswith("Set the delegate_task 'model' parameter"))

    def test_the_new_opening_and_the_note_header_are_stripped_from_a_leaf(self):
        goal = "[spark] Read-only discovery of the booking list."
        for marker in ("Route choice for delegated workers", "[ROUTER] This turn classifies as"):
            with self.subTest(marker=marker):
                self.assertEqual(_without_router_contract(f"{goal}\n\n{marker} rest"), goal)

    def test_an_active_wing_names_delegate_claude(self):
        contract = _contract(_cfg(), active=True)
        self.assertIn("[opus5] and [sonnet5] are not labels", contract)
        self.assertIn("stopped at its first call", contract)
        self.assertIn('delegate_claude with tier "haiku", "sonnet" or "opus"', contract)
        self.assertNotIn("Name those targets only in the model parameter", contract)

    def test_an_inactive_wing_keeps_todays_text(self):
        contract = _contract(_cfg(), active=False)
        self.assertIn("Name those targets only in the model parameter", contract)
        self.assertNotIn("delegate_claude", contract)

    def test_the_active_contract_names_the_deferred_tool_hint(self):
        contract = _contract(_cfg(), active=True)
        self.assertIn(
            'If delegate_claude is not in your tool list it is a deferred tool: load it once with '
            'tool_describe, then call it through tool_call with name "delegate_claude".',
            contract,
        )

    def test_an_inactive_contract_has_no_deferred_tool_hint(self):
        contract = _contract(_cfg(), active=False)
        self.assertNotIn("deferred tool", contract)

    def test_the_preference_order_names_the_call(self):
        cfg = _cfg(preferences={"code": ["terra", "sonnet5"]})
        with patch.object(claude_delegation, "_ACTIVE", True):
            active = _preference_sentence(["terra", "sonnet5"], cfg)
        with patch.object(claude_delegation, "_ACTIVE", False):
            inactive = _preference_sentence(["terra", "sonnet5"], cfg)
        self.assertIn("code: terra > sonnet5", active)
        self.assertIn("delegate_claude", active)
        self.assertNotIn("model: parameter", active)
        self.assertIn("model: parameter", inactive)

    def test_the_claude_sentence_covers_haiku_and_the_tool(self):
        with patch.object(claude_delegation, "_ACTIVE", True):
            sentence = _claude_target_sentence(["haiku", "opus5", "sonnet5"], _cfg())
        self.assertIn("haiku", sentence)
        self.assertIn('delegate_claude(tier="haiku"|"sonnet"|"opus")', sentence)
        self.assertIn("Use sonnet5 by default", sentence)

    def test_the_redispatch_notice_names_delegate_claude(self):
        with patch.object(claude_delegation, "_ACTIVE", True), \
             patch("model_router._tier_cooldown_remaining", return_value=0.0), \
             patch("model_router._delegation_target_names", return_value=REDISPATCH_TARGETS):
            instruction = _quota_redispatch_instruction(request_for(envelope()), REDISPATCH_CFG)
        self.assertIn('delegate_claude(tier="opus")', instruction)
        self.assertNotIn("model:opus5", instruction)
        self.assertIn("with the call named above", instruction)

    def test_the_preflight_prefers_a_claude_worker_through_the_tool(self):
        cfg = _cfg(orchestration={"enabled": True, "max_tasks": 2})
        with patch.object(claude_delegation, "_ACTIVE", True), \
             patch("model_router._delegation_target_names", return_value=CONTRACT_TARGETS), \
             patch("model_router._recent_account_load", return_value={}):
            routed = _prepare_orchestration_delegation(_delegating_request(), "plan-x", 2, cfg=cfg)
        text = routed["messages"][-1]["content"]
        self.assertIn("prefer a native Claude worker through delegate_claude", text)
        self.assertNotIn("model:opus5 / model:sonnet5", text)
        self.assertNotIn("Set the delegate_task 'model' parameter", text)
        self.assertIn(
            'If delegate_claude is not in your tool list it is a deferred tool: load it once with '
            'tool_describe, then call it through tool_call with name "delegate_claude".',
            text,
        )

    def test_the_preflight_only_advertises_reachable_routes_while_the_wing_is_off(self):
        cfg = _cfg(orchestration={"enabled": True, "max_tasks": 2})
        with patch.object(claude_delegation, "_ACTIVE", False), \
             patch("model_router._delegation_target_names", return_value=CONTRACT_TARGETS), \
             patch("model_router._recent_account_load", return_value={}):
            routed = _prepare_orchestration_delegation(_delegating_request(), "plan-x", 2, cfg=cfg)
        text = routed["messages"][-1]["content"]
        self.assertIn("Cross-account targets cannot be reached", text)
        self.assertNotIn("Set the delegate_task 'model' parameter", text)
        self.assertNotIn("model:opus5", text)
        self.assertNotIn("delegate_claude", text)


from model_router import _routing_note, route_llm_request  # noqa: E402
from model_router.test_external_orchestrator import ACTIONABLE  # noqa: E402
from model_router.test_model_router import chat_request  # noqa: E402

PREFS = {
    "design": ["sol", "opus5"], "code": ["terra", "sonnet5"], "explore": ["spark", "luna", "haiku"],
    "review": ["sonnet5", "opus5", "terra"],
}


def _parent_request(tools=("delegate_task", "delegate_claude")):
    # The same actionable prompt the external-orchestrator tests use, so the
    # forced-preflight case below is not skipped by an unrelated gate.
    request = chat_request(ACTIONABLE)
    request["model"] = "claude-opus-5-5"
    request["tools"] = [{"type": "function", "name": name, "parameters": {"type": "object", "properties": {}}}
                        for name in tools]
    return request


def _note(kind="code", weekly=40.0, codex=None, cfg=None, active=True, request=None, **kwargs):
    reading = None if weekly is None else _reading(weekly)
    kwargs = {"api_call_count": 1, "turn_id": "t1", "platform": "cli", **kwargs}
    if cfg is None:
        cfg = _cfg(preferences=PREFS)
        cfg["usage_guard"]["accounts"]["openai-codex"] = {
            "soft_percent": 70, "hard_percent": 90, "step_down": {"sol": "terra"},
        }
    with patch.object(claude_delegation, "_ACTIVE", active), \
         patch.object(usage_guard, "peek", side_effect=lambda account, cfg: (
             reading if account == "anthropic" else (
                 _reading(codex) if codex is not None and account == "openai-codex" else None))), \
         patch("model_router.classify_request", return_value=SimpleNamespace(kind=kind)), \
         patch("model_router._delegation_target_names", return_value=("haiku", "opus5", "sonnet5")), \
         patch("model_router._tier_cooldown_remaining", return_value=0.0):
        return _routing_note(request or _parent_request(), kwargs, cfg)


class RoutingNoteTests(unittest.TestCase):
    def test_the_note_recommends_the_kinds_chain_as_calls(self):
        note = _note()
        self.assertIn("[ROUTER] This turn classifies as: code.", note)
        self.assertIn('If you delegate code work: terra → delegate_task (goal prefix [terra]) > '
                      'sonnet5 → delegate_claude(tier="sonnet").', note)
        self.assertIn("Other kinds:", note)
        self.assertIn("review: sonnet5 > opus5 > terra", note)
        self.assertIn("Usage: Claude weekly 40% (soft 70%, hard 90%); "
                     "Codex weekly unknown (soft 70%, hard 90%).", note)
        self.assertIn("Advisory: if you route differently, say why in one line.", note)

    def test_the_note_names_the_deferred_tool_hint_before_the_advisory_line(self):
        note = _note()
        hint = (
            'If delegate_claude is not in your tool list it is a deferred tool: load it once with '
            'tool_describe, then call it through tool_call with name "delegate_claude".'
        )
        self.assertIn(hint, note)
        self.assertLess(note.index(hint), note.index("Advisory:"))

    def test_the_soft_limit_moves_claude_behind_codex(self):
        note = _note(kind="review", weekly=75.0)
        self.assertIn("If you delegate review work: terra → delegate_task", note)
        self.assertIn('sonnet5 → delegate_claude(tier="sonnet") [Claude soft limit]', note)

    def test_the_hard_limit_marks_claude_closed(self):
        self.assertIn("[Claude closed]", _note(kind="review", weekly=95.0))

    def test_an_unknown_reading_says_so(self):
        self.assertIn("Claude weekly unknown", _note(weekly=None))

    def test_a_codex_soft_limit_moves_codex_behind_claude(self):
        note = _note(kind="code", weekly=40.0, codex=72.0)
        self.assertIn('If you delegate code work: sonnet5 → delegate_claude(tier="sonnet") > '
                      'terra → delegate_task (goal prefix [terra]) [Codex soft limit].', note)

    def test_no_claude_preference_still_announces_the_wing(self):
        note = _note(cfg=_cfg(preferences={"code": ["terra"]}))
        self.assertIn("Claude delegation is available through delegate_claude", note)

    def test_a_kind_without_a_preference_keeps_the_built_in_route(self):
        self.assertIn("delegate_task keeps its built-in route", _note(kind="chat"))

    def test_the_note_is_only_for_a_root_parents_first_call(self):
        cases = {
            "inactive delegation": dict(active=False),
            "subagent": dict(platform="subagent"),
            "mid-loop": dict(api_call_count=2),
            "no delegation tool": dict(request=_parent_request(tools=("read_file",))),
        }
        for name, overrides in cases.items():
            with self.subTest(case=name):
                self.assertEqual(_note(**overrides), "")

    def test_the_claude_code_prefix_still_counts_as_a_delegation_tool(self):
        self.assertIn("[ROUTER]", _note(request=_parent_request(tools=("mcp__delegate_claude",))))

    def test_a_completion_envelope_gets_no_note(self):
        with patch("model_router._is_delegation_outcome_text", return_value=True):
            self.assertEqual(_note(), "")


class RoutingNoteMiddlewareTests(unittest.TestCase):
    def _route(self, orchestration):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cfg = _cfg(preferences=PREFS, enabled=True, provider="openai-codex",
                   orchestration={"enabled": orchestration, "max_tasks": 2,
                                  "path": str(Path(directory.name) / "orchestration.jsonl")},
                   logging={"enabled": False}, shadow={"enabled": False})
        with host_delegation(depth=2), \
             patch.object(claude_delegation, "_ACTIVE", True), \
             patch.object(usage_guard, "peek", side_effect=lambda account, cfg: _reading(40) if account == "anthropic" else None), \
             patch("model_router._load_config", return_value=cfg), \
             patch("model_router._log_decision"), \
             patch("model_router._orchestration_event"), \
             patch("model_router._delegation_target_names", return_value=("haiku", "opus5", "sonnet5")):
            return route_llm_request(request=_parent_request(), provider="anthropic", model="claude-opus-5-5",
                                     api_call_count=1, turn_id="root-turn", platform="cli")

    def test_a_claude_parent_gets_the_note_and_keeps_its_model(self):
        routed = self._route(orchestration=False)
        self.assertIsNotNone(routed)
        self.assertIn("[ROUTER] This turn classifies as", routed["request"]["messages"][-1]["content"])
        self.assertEqual(routed["request"]["model"], "claude-opus-5-5")

    def test_a_forced_preflight_carries_the_contract_instead(self):
        routed = self._route(orchestration=True)
        self.assertNotIn("[ROUTER] This turn classifies as", json.dumps(routed["request"]))

    def test_a_broken_routing_note_does_not_drop_the_redispatch_notice(self):
        """The routing note must never abort the whole response: a raise there
        must not cost the parent its redispatch notice, or return None entirely."""
        with patch("model_router._routing_note", side_effect=RuntimeError("boom")):
            routed = self._route(orchestration=False)
        self.assertIsNone(routed)


class UsageStepDownIntegrationTests(unittest.TestCase):
    """The step-down must land on the *dispatched* tier: the orchestration
    gates (forced preflight, forced shadow) have to see the request's original
    classification, not a tier the usage guard already moved it off of."""

    def test_a_codex_root_request_at_the_soft_limit_retains_its_route(self):
        cfg = _cfg(preferences=PREFS, provider="openai-codex",
                   orchestration={"enabled": False}, logging={"enabled": False},
                   shadow={"enabled": False})
        cfg["usage_guard"]["accounts"]["openai-codex"] = {
            "soft_percent": 70, "hard_percent": 90, "step_down": {"sol": "terra"},
        }
        # The same "long request" fixture test_model_router.py uses to prove a
        # long prompt classifies to sol -- non-mandatory, so it is eligible for
        # the step-down.
        request = chat_request("Elemezd részletesen. " + "x" * 4200)
        preflight = MagicMock(wraps=model_router._force_terra_supervisor_preflight)
        log_decision = MagicMock()
        with patch.object(usage_guard, "peek", side_effect=lambda account, cfg: (
                 _reading(72.0) if account == "openai-codex" else None)), \
             patch("model_router._load_config", return_value=cfg), \
             patch("model_router._log_decision", log_decision), \
             patch("model_router._force_terra_supervisor_preflight", preflight):
            routed = route_llm_request(request=request, provider="openai-codex", model="gpt-terra",
                                       api_call_count=1, turn_id="root-turn", platform="cli")
        self.assertIsNotNone(routed)
        self.assertEqual(routed["request"]["model"], "gpt-sol")
        self.assertNotIn("usage soft limit", routed["reason"])
        logged = log_decision.call_args.args[0]
        self.assertNotIn("usage soft limit", logged.reason)
        preflight_decision = preflight.call_args.args[2]
        self.assertEqual(preflight_decision.tier, "sol")


if __name__ == "__main__":
    unittest.main()
