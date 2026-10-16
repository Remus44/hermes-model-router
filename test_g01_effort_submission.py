"""Registered-chain regressions for G01 submitted reasoning-effort authority."""

from __future__ import annotations

import copy
import os
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

import model_router as router
from model_router import triage, usage_guard, worker_admission


ROOT = Path(__file__).resolve().parent


def clone(value):
    return copy.deepcopy(value)


def effort_of(request):
    return (request.get("reasoning") or {}).get("effort") if isinstance(request, dict) else None


def shipped_config():
    cfg = yaml.safe_load((ROOT / "router_config.yaml").read_text(encoding="utf-8"))
    cfg.update({
        "enabled": True,
        "provider": "openai-codex",
        "models": {
            "luna": "gpt-6-luna",
            "spark": "gpt-5.3-codex-spark",
            "terra": "gpt-5.6-terra",
            "sol": "gpt-6.1-sol",
        },
        "tier_providers": {tier: "openai-codex" for tier in ("luna", "spark", "terra", "sol")},
        "callable": {
            "luna": True, "spark": True, "terra": True, "sol": True,
            "haiku": False, "sonnet5": False, "opus5": False, "qwen": False, "grok": False,
        },
        "preferences": {},
        "session_policy": {"pin_root_parent": False},
        "triage": {"enabled": False, "max_workers_per_turn": 4, "max_children_per_worker": 1},
        "orchestration": {"enabled": False},
        "shadow": {"enabled": False},
        "cooldown": {"enabled": False},
        "logging": {"enabled": False},
        "coding_agent": {"enabled": False, "delegated_review": {"enabled": False}},
    })
    return cfg


def responses_request(text):
    return {
        "model": "gpt-5.6-terra",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": text}]}],
        "tools": [{"type": "function", "name": "g01_probe", "parameters": {"type": "object"}}],
        "temperature": 0.2,
    }


def advisory_request():
    return {
        "model": "claude-opus-5-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Implement the parser fix."}]}],
        "tools": [
            {"name": "mcp__delegate_task", "input_schema": {"type": "object", "properties": {}}},
            {"name": "mcp__tool_call", "input_schema": {"type": "object", "properties": {}}},
        ],
        "tool_choice": {"type": "tool", "name": "mcp__web_search"},
    }


def rewriter(mutate):
    def callback(**kwargs):
        rewritten = clone(kwargs["request"])
        mutate(rewritten)
        return {"request": rewritten, "source": "g01-after", "reason": "G01 test rewriter"}
    return callback


def set_effort(value):
    def mutate(request):
        reasoning = dict(request.get("reasoning") or {})
        reasoning["effort"] = value
        request["reasoning"] = reasoning
    return mutate


def erase_label_and_set_effort(value):
    def mutate(request):
        request["input"][-1]["content"][-1]["text"] = "Inspect and report."
        set_effort(value)(request)
    return mutate


class G01EffortSubmissionTests(unittest.TestCase):
    def setUp(self):
        prior_efforts = dict(router._CODEX_TIER_EFFORTS_CACHE)
        prior_warning = router._CODEX_TIER_EFFORTS_WARNING_EMITTED

        def restore_effort_globals():
            router._CODEX_TIER_EFFORTS_CACHE.clear()
            router._CODEX_TIER_EFFORTS_CACHE.update(prior_efforts)
            router._CODEX_TIER_EFFORTS_WARNING_EMITTED = prior_warning

        router._CODEX_TIER_EFFORTS_CACHE.clear()
        router._CODEX_TIER_EFFORTS_WARNING_EMITTED = False
        self.addCleanup(restore_effort_globals)
        self.addCleanup(triage.reset_state)

    def run_registered(self, *, text="[sol:high] Inspect and report.", before_mutate=None, mutate=None, worker=False,
                       advisory=False, enabled=True, provider=None, api_mode=None, configure=None,
                       initial_model=None, bridge_result=None, inject_swallowed_usage_warning=False):
        from hermes_cli import plugins
        from hermes_cli.middleware import apply_llm_request_middleware, run_llm_execution_middleware
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        cfg = shipped_config()
        cfg["enabled"] = enabled
        if configure is not None:
            configure(cfg)
        if worker:
            cfg["usage_guard"] = {
                "cache_seconds": 300,
                "state_path": "",
                "accounts": {"openai-codex": {"soft_percent": 70, "hard_percent": 90,
                                                   "step_down": {"sol": "terra"}}},
            }
        if advisory:
            cfg.update({
                "provider": "anthropic",
                "models": {"terra": "claude-opus-5-5"},
                "tier_providers": {"terra": "anthropic"},
                "callable": {"terra": True, "opus5": False, "sonnet5": False},
                "triage": {"enabled": True, "max_workers_per_turn": 4, "max_children_per_worker": 1},
            })
        request = advisory_request() if advisory else responses_request(text)
        if initial_model is not None:
            request["model"] = initial_model
        caller_before = clone(request)
        manager = PluginManager()
        manager._discovered = True
        ctx = PluginContext(PluginManifest(name="g01-effort", source="test", key="g01-effort"), manager)
        terminal_payloads = []
        bridge = Mock(
            side_effect=AssertionError("G01 bridge must not be reached")
            if bridge_result is None else None,
            return_value=bridge_result,
        )
        admission = []
        warnings = []
        hook_failures = []
        usage_attempts = []
        route_observations = []
        real_guarded = usage_guard.guarded
        real_route_llm_request = router.route_llm_request

        def terminal(payload):
            terminal_payloads.append(clone(payload))
            return {"terminal": True, "model": payload.get("model"), "effort": effort_of(payload)}

        def guarded(account, guarded_cfg):
            return real_guarded(account, guarded_cfg)

        def peek(account, guard_cfg):
            if inject_swallowed_usage_warning:
                usage_attempts.append("read")
                raise AssertionError("deliberately blocked usage read")
            return usage_guard.Reading(45.0, 54.0, None, None, time.time())

        def blocked_usage(operation):
            def callback(*_args, **_kwargs):
                usage_attempts.append(operation)
                return usage_guard.Reading(45.0, 54.0, None, None, time.time())
            return callback

        def open_admission(account, model, admission_cfg, *, blocking=False):
            admission.append((account, model, blocking))
            return ""

        def observe_registered_route(**route_kwargs):
            observation = {
                "input_model": route_kwargs.get("model"),
                "input_effort": effort_of(route_kwargs.get("request")),
            }
            route_observations.append(observation)
            routed = real_route_llm_request(**route_kwargs)
            routed_request = routed.get("request") if isinstance(routed, dict) else None
            observation.update({
                "routed_model": routed_request.get("model") if isinstance(routed_request, dict) else None,
                "routed_effort": effort_of(routed_request),
            })
            return routed

        original_active = router.claude_delegation._ACTIVE
        original_available = router.claude_delegation._LAST_AVAILABLE
        try:
            with ExitStack() as stack:
                manager._report_hook_failure = Mock(side_effect=lambda *args, **kwargs: hook_failures.append((args, kwargs)))
                stack.enter_context(patch.object(plugins, "_plugin_manager", manager))
                stack.enter_context(patch("hermes_cli.plugins._delivery_manager", return_value=manager))
                stack.enter_context(patch.object(router, "_load_config", return_value=cfg))
                stack.enter_context(patch.object(router.claude_delegation, "availability_block", return_value="offline fixture"))
                stack.enter_context(patch.object(router.claude_delegation, "note_availability", return_value=None))
                stack.enter_context(patch.object(router, "_run_opus5_bridge", bridge))
                stack.enter_context(patch.object(router._logger, "warning", side_effect=lambda *args, **kwargs: warnings.append((args, kwargs))))
                if worker:
                    stack.enter_context(patch.object(usage_guard, "guarded", side_effect=guarded))
                    stack.enter_context(patch.object(usage_guard, "peek", side_effect=peek))
                    stack.enter_context(patch.object(usage_guard, "read", side_effect=blocked_usage("read")))
                    stack.enter_context(patch.object(usage_guard, "_start_refresh", side_effect=blocked_usage("refresh")))
                    stack.enter_context(patch.object(worker_admission, "refusal", side_effect=open_admission))
                if before_mutate is not None:
                    ctx.register_middleware("llm_request", rewriter(before_mutate))
                # Observe the one callback installed by the real router registration.
                stack.enter_context(patch.object(router, "route_llm_request", observe_registered_route))
                router.register(ctx)
                self.assertEqual(manager._middleware["llm_request"].count(observe_registered_route), 1)
                if mutate is not None:
                    ctx.register_middleware("llm_request", rewriter(mutate))
                actual_provider = provider or ("anthropic" if advisory else "openai-codex")
                actual_api_mode = api_mode or ("anthropic_messages" if advisory else "codex_responses")
                platform = "cli" if advisory else ("subagent" if worker else "cli")
                turn_id = "g01:sa-submission" if worker else "g01:submission"
                composed = apply_llm_request_middleware(
                    request, model=request["model"], provider=actual_provider, api_mode=actual_api_mode,
                    api_call_count=1, platform=platform, turn_id=turn_id,
                )
                result = run_llm_execution_middleware(
                    composed.payload, terminal, original_request=composed.original_payload,
                    model=request["model"], provider=actual_provider, api_mode=actual_api_mode,
                    api_call_count=1, platform=platform, turn_id=turn_id,
                )
        finally:
            for registration in reversed(getattr(manager, "_registration_order", [])):
                registration.dispose()
            router.claude_delegation._ACTIVE = original_active
            router.claude_delegation._LAST_AVAILABLE = original_available
        self.assertEqual(request, caller_before, "G01 caller request mutated by registered chain")
        self.assertEqual(composed.original_payload, caller_before, "G01 host original payload did not preserve caller authority")
        self.assertEqual(len(route_observations), 1, "G01 registered router callback must run exactly once")
        if worker:
            self.assertEqual(usage_attempts, [], "G01 blocked authority attempted usage read/refresh")
        self.assertEqual(warnings, [], "G01 fixture observed an unexpected warning/setup failure")
        self.assertEqual(hook_failures, [], "G01 fixture observed an unexpected host hook failure")
        return composed, result, terminal_payloads, bridge, {
            "warnings": warnings, "hook_failures": hook_failures, "usage_attempts": usage_attempts,
            "admission": admission, "route_observations": route_observations,
        }

    def assert_route_observation(self, observations, model, effort):
        observed = observations["route_observations"]
        self.assertEqual(len(observed), 1)
        self.assertEqual((observed[0]["routed_model"], observed[0]["routed_effort"]), (model, effort))

    def assert_public_stop(self, result, terminal_payloads, bridge):
        self.assertIn("ROUTER WORKER STOPPED", getattr(result, "output_text", ""))
        self.assertEqual(terminal_payloads, [])
        bridge.assert_not_called()
        self.assertEqual(getattr(getattr(result, "usage", None), "total_tokens", None), 0)

    def test_registered_chain_after_router_effort_overwrite_stops_before_terminal(self):
        for name, text, replacement in (
            ("high-to-medium", "[sol:high] Inspect and report.", "medium"),
            ("xhigh-to-medium", "[sol:xhigh] Inspect and report.", "medium"),
            ("high-to-xhigh", "[sol:high] Inspect and report.", "xhigh"),
            ("xhigh-to-high", "[sol:xhigh] Inspect and report.", "high"),
            ("worker-high-to-medium", "[sol:high] Inspect and report.", "medium"),
            ("worker-xhigh-to-medium", "[sol:xhigh] Inspect and report.", "medium"),
        ):
            with self.subTest(cell=name):
                _composed, result, terminal, bridge, observations = self.run_registered(
                    text=text, mutate=set_effort(replacement), worker=name.startswith("worker-"),
                )
                self.assert_route_observation(
                    observations, "gpt-6.1-sol", "xhigh" if "xhigh" in text else "high",
                )
                self.assert_public_stop(result, terminal, bridge)

    def test_registered_chain_rejects_unsupported_missing_and_malformed_final_effort(self):
        def unsupported(request):
            request["model"] = "gpt-5.3-codex-spark"
            set_effort("max")(request)

        def missing(request):
            request["reasoning"] = {}

        def malformed(request):
            request["reasoning"] = {"effort": ["medium"]}

        for name, text, mutate in (
            ("unsupported-destination-level", "[sol:max] Inspect and report.", unsupported),
            ("missing-final-effort", "[sol:high] Inspect and report.", missing),
            ("malformed-final-effort", "[sol:xhigh] Inspect and report.", malformed),
            ("removed-label-still-authoritative", "[sol:high] Inspect and report.", erase_label_and_set_effort("medium")),
        ):
            with self.subTest(cell=name):
                _composed, result, terminal, bridge, _observations = self.run_registered(text=text, mutate=mutate)
                self.assert_public_stop(result, terminal, bridge)

    def test_registered_chain_preserves_supported_authority_and_permits_defaults(self):
        def to_terra_high(request):
            request["model"] = "gpt-5.6-terra"
            set_effort("high")(request)

        def harmless(request):
            request["max_output_tokens"] = 256

        cases = (
            ("sole-high", "[sol:high] Inspect and report.", None, "gpt-6.1-sol", "high", "gpt-6.1-sol", "high"),
            ("sole-xhigh", "[sol:xhigh] Inspect and report.", None, "gpt-6.1-sol", "xhigh", "gpt-6.1-sol", "xhigh"),
            ("harmless-after", "[sol:high] Inspect and report.", harmless, "gpt-6.1-sol", "high", "gpt-6.1-sol", "high"),
            ("same-effort-target", "[sol:high] Inspect and report.", to_terra_high, "gpt-5.6-terra", "high", "gpt-6.1-sol", "high"),
            ("bare-default-not-mandatory", "[sol] Inspect and report.", set_effort("high"), "gpt-6.1-sol", "high", "gpt-6.1-sol", "medium"),
        )
        for name, text, mutate, model, effort, routed_model, routed_effort in cases:
            with self.subTest(cell=name):
                _composed, result, terminal, bridge, observations = self.run_registered(text=text, mutate=mutate)
                self.assertNotIn("ROUTER WORKER STOPPED", getattr(result, "output_text", ""))
                bridge.assert_not_called()
                self.assert_route_observation(observations, routed_model, routed_effort)
                self.assertEqual(len(terminal), 1)
                self.assertEqual((terminal[0].get("model"), effort_of(terminal[0])), (model, effort))

    def test_registered_chain_pre_router_restoration_remains(self):
        _composed, result, terminal, bridge, _observations = self.run_registered(
            text="[sol:high] Inspect and report.", before_mutate=set_effort("medium"),
        )
        self.assertEqual(len(terminal), 1)
        self.assertEqual(effort_of(terminal[0]), "high")
        bridge.assert_not_called()

    def test_registered_chain_existing_named_refusal_remains(self):
        _composed, stopped, advisory_terminal, advisory_bridge, _observations = self.run_registered(advisory=True, mutate=set_effort("medium"))
        self.assert_public_stop(stopped, advisory_terminal, advisory_bridge)

    def test_registered_chain_does_not_apply_authority_guard_when_disabled_or_unrelated(self):
        for name, enabled, provider in (
            ("router-disabled", False, "openai-codex"),
            ("unrelated-provider", True, "unrelated-provider"),
        ):
            with self.subTest(cell=name):
                _composed, result, terminal, bridge, _observations = self.run_registered(
                    text="[sol:high] Inspect and report.", mutate=set_effort("medium"), enabled=enabled, provider=provider,
                )
                self.assertNotIn("ROUTER WORKER STOPPED", getattr(result, "output_text", ""))
                bridge.assert_not_called()
                self.assertEqual(len(terminal), 1)
                self.assertEqual(effort_of(terminal[0]), "medium")

    def test_r1_bridge_eligible_terra_contradiction_stops_before_bridge(self):
        cfg = shipped_config()
        cfg["coding_agent"] = {"enabled": True, "default_repo": str(ROOT), "delegated_review": {"enabled": False}}
        cfg["callable"]["opus5"] = True
        original = responses_request("[terra:high] Fix the parser and add tests.")

        def run_path(final_effort):
            request = clone(original)
            request["reasoning"] = {"effort": final_effort}
            terminal = Mock(return_value={"terminal": True})
            bridge = Mock(return_value={
                "result": "offline bridge fixture", "effective_model": "claude-opus-5-5", "usage": {},
            })
            with ExitStack() as stack:
                stack.enter_context(patch.object(router, "_load_config", return_value=cfg))
                stack.enter_context(patch.object(router, "_run_opus5_bridge", bridge))
                stack.enter_context(patch.object(router.claude_delegation, "availability_block",
                                                 side_effect=AssertionError("real availability check")))
                guarded = stack.enter_context(patch.object(usage_guard, "guarded", return_value=False))
                read = stack.enter_context(patch.object(usage_guard, "read",
                                                        side_effect=AssertionError("real usage read")))
                peek = stack.enter_context(patch.object(usage_guard, "peek",
                                                        side_effect=AssertionError("real usage peek")))
                refresh = stack.enter_context(patch.object(usage_guard, "_start_refresh",
                                                           side_effect=AssertionError("real usage refresh")))
                fetch = Mock(side_effect=AssertionError("real Anthropic usage fetch"))
                stack.enter_context(patch.dict(usage_guard.FETCHERS, {"anthropic": fetch}))
                result = router.run_llm_with_transient_failover(
                    request=request, next_call=terminal, original_request=original,
                    provider="openai-codex", api_mode="codex_responses", api_call_count=1,
                )
            return result, terminal, bridge, guarded, read, peek, refresh, fetch

        result, terminal, bridge, guarded, read, peek, refresh, fetch = run_path("medium")
        self.assertIn("ROUTER WORKER STOPPED", getattr(result, "output_text", ""))
        bridge.assert_not_called()
        terminal.assert_not_called()
        guarded.assert_not_called()
        read.assert_not_called()
        peek.assert_not_called()
        refresh.assert_not_called()
        fetch.assert_not_called()

        result, terminal, bridge, guarded, read, peek, refresh, fetch = run_path("high")
        self.assertNotIn("ROUTER WORKER STOPPED", getattr(result, "output_text", ""))
        bridge.assert_called_once()
        terminal.assert_not_called()
        guarded.assert_called_once_with("anthropic", cfg)
        read.assert_not_called()
        peek.assert_not_called()
        refresh.assert_not_called()
        fetch.assert_not_called()

    def test_r2_compatible_retry_keeps_immutable_original_effort_after_rewrite(self):
        cfg = shipped_config()
        cfg["callable"]["spark"] = True
        cfg["effort"]["spark"] = "medium"
        for cell, original_level, final_text in (
            ("label-erased-high", "high", "Inspect and report."),
            ("label-changed-xhigh", "xhigh", "[sol:medium] Inspect and report."),
        ):
            with self.subTest(cell=cell):
                original = responses_request(f"[sol:{original_level}] Inspect and report.")
                request = responses_request(final_text)
                request["model"] = cfg["models"]["sol"]
                request["reasoning"] = {"effort": original_level}
                submitted = []

                def retry_call(payload):
                    submitted.append(clone(payload))
                    return "recovered"

                with patch.object(router, "_load_config", return_value=cfg):
                    recovered = router.run_llm_with_transient_failover(
                        request=request,
                        original_request=original,
                        next_call=Mock(side_effect=RuntimeError("HTTP 503: upstream connect error")),
                        retry_call=retry_call,
                        provider="openai-codex",
                        api_mode="codex_responses",
                    )
                self.assertEqual(recovered, "recovered")
                self.assertEqual(submitted[0]["model"], cfg["models"]["spark"])
                self.assertEqual(submitted[0]["reasoning"]["effort"], original_level)

    def test_r2_compatible_retry_without_callback_propagates_original_exception(self):
        cfg = shipped_config()
        cfg["callable"]["spark"] = True
        original = responses_request("[sol:high] Inspect and report.")
        request = responses_request("Inspect and report.")
        request["model"] = cfg["models"]["sol"]
        request["reasoning"] = {"effort": "high"}
        failure = RuntimeError("HTTP 503: upstream connect error")
        with patch.object(router, "_load_config", return_value=cfg), \
             self.assertRaisesRegex(RuntimeError, "upstream connect error") as caught:
            router.run_llm_with_transient_failover(
                request=request, original_request=original, next_call=Mock(side_effect=failure),
                provider="openai-codex", api_mode="codex_responses",
            )
        self.assertIs(caught.exception, failure)

    def test_f1_nonstated_native_none_retry_reaches_callback(self):
        cfg = shipped_config()
        cfg["callable"]["spark"] = True
        cfg["effort"]["spark"] = "none"
        original = responses_request("Inspect and report.")
        request = clone(original)
        request["model"] = cfg["models"]["sol"]
        submitted = []

        def retry_call(payload):
            submitted.append(clone(payload))
            return "recovered"

        with patch.object(router, "_load_config", return_value=cfg):
            recovered = router.run_llm_with_transient_failover(
                request=request, original_request=original,
                next_call=Mock(side_effect=RuntimeError("HTTP 503: upstream connect error")),
                retry_call=retry_call, provider="openai-codex", api_mode="codex_responses",
            )
        self.assertEqual(recovered, "recovered")
        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0]["model"], cfg["models"]["spark"])
        self.assertEqual(submitted[0]["reasoning"]["effort"], "none")

    def test_r2_compatible_retry_refuses_incompatible_substitute(self):
        cfg = shipped_config()
        original = responses_request("[sol:high] Inspect and report.")
        request = clone(original)
        request["model"] = cfg["models"]["sol"]
        request["reasoning"] = {"effort": "high"}
        retry = Mock(return_value="must not run")
        with patch.object(router, "_load_config", return_value=cfg), \
             patch.object(router, "_transient_fallback_model", return_value="not-a-configured-model"):
            result = router.run_llm_with_transient_failover(
                request=request, original_request=original,
                next_call=Mock(side_effect=RuntimeError("HTTP 503: upstream connect error")),
                retry_call=retry, provider="openai-codex", api_mode="codex_responses",
            )
        self.assertIn("ROUTER WORKER STOPPED", getattr(result, "output_text", ""))
        retry.assert_not_called()

    def test_r3_final_effort_is_literal_canonical_value(self):
        for cell, text, replacement in (
            ("uppercase-high", "[SOL:High] Inspect and report.", "HIGH"),
            ("mixed-xhigh", "[sol:XHIGH] Inspect and report.", "XhIgH"),
        ):
            with self.subTest(cell=cell):
                _composed, result, terminal, bridge, _observations = self.run_registered(
                    text=text, mutate=set_effort(replacement),
                )
                self.assert_public_stop(result, terminal, bridge)

    def test_r4_environment_disable_and_matching_qwen_sanitization_bypass_codex_guard(self):
        with self.subTest(cell="environment-disable"):
            with patch.dict(os.environ, {"HERMES_MODEL_ROUTER_DISABLE": "1"}, clear=False):
                _composed, result, terminal, bridge, _observations = self.run_registered(
                    text="[sol:high] Inspect and report.", mutate=set_effort("medium"),
                )
            self.assertNotIn("ROUTER WORKER STOPPED", getattr(result, "output_text", ""))
            bridge.assert_not_called()
            self.assertEqual(effort_of(terminal[0]), "medium")

        def qwen_config(cfg):
            cfg["provider"] = "qwen-token"
            cfg["models"]["terra"] = "qwen3.7-plus"
            cfg["tier_providers"]["terra"] = "qwen-token"

        with self.subTest(cell="matching-qwen-sanitization"):
            _composed, result, terminal, bridge, _observations = self.run_registered(
                text="[terra:high] Inspect and report.", configure=qwen_config,
                initial_model="qwen3.7-plus", provider="qwen-token", api_mode="anthropic_messages",
            )
            self.assertNotIn("ROUTER WORKER STOPPED", getattr(result, "output_text", ""))
            bridge.assert_not_called()
            self.assertEqual(len(terminal), 1)
            self.assertNotIn("reasoning", terminal[0])

    def test_r5_positive_worker_control_observes_open_admission(self):
        _composed, result, terminal, bridge, observations = self.run_registered(
            text="[sol:high] Inspect and report.", worker=True,
        )
        self.assertNotIn("ROUTER WORKER STOPPED", getattr(result, "output_text", ""))
        self.assertEqual(len(terminal), 1)
        bridge.assert_not_called()
        self.assert_route_observation(observations, "gpt-6.1-sol", "high")
        self.assertEqual(len(observations["admission"]), 1)

    def test_r5_fixture_oracle_rejects_swallowed_blocked_usage_warning(self):
        with self.assertRaisesRegex(AssertionError, "blocked authority attempted usage read/refresh"):
            self.run_registered(worker=True, inject_swallowed_usage_warning=True)


if __name__ == "__main__":
    unittest.main()
