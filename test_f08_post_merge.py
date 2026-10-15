"""F08 post-merge evidence against the installed Hermes host.

Defects reproduced here remain expected failures pending an owner-scoped repair.
"""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent import relay_tools, tool_executor
from agent.tool_call_batches import expand_local_tool_batches
from jsonschema import Draft202012Validator
import model_tools
import model_router as router
from model_router import execution_adapters, runtime_capabilities as runtime, triage, usage_guard, worker_admission
from model_router.host_delegation_fixtures import delegate_task_request, host_delegation
import tools.tool_search as tool_search
from tools.tool_search import bridge_tool_schemas, resolve_underlying_call
from tools.tool_search_validation import normalize_tool_call_entries


TASK = "Implement the parser fix, add a regression test, and verify the result."


def _host_bridge():
    return next(tool for tool in bridge_tool_schemas(1)
                if tool["function"]["name"] == "tool_call")


def _cfg(path, *, triage_enabled=True):
    return {
        "enabled": True,
        "models": {"terra": "gpt-5.6-terra", "sol": "gpt-6.1-sol"},
        "tier_providers": {"terra": "openai-codex", "sol": "openai-codex"},
        "callable": {"terra": True, "sol": True,
                     "haiku": True, "sonnet5": True, "opus5": True},
        "default_model": "terra",
        "triage": {
            "enabled": triage_enabled,
            "max_workers_per_turn": 4,
            "max_children_per_worker": 1,
            "path": str(Path(path) / "triage.jsonl"),
        },
        "orchestration": {"enabled": False},
        "shadow": {"enabled": False},
    }


def _openai_request(text=TASK):
    return {
        "model": "gpt-5.6-terra",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": text}]}],
        "tools": [_host_bridge()],
    }


def _anthropic_request(model, *, thinking=False):
    request = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": TASK}]}],
        "tools": [{"name": "mcp__delegate_task", "input_schema": {"type": "object", "properties": {}}},
                  {"name": "mcp__tool_call", "input_schema": _host_bridge()["function"]["parameters"]}],
    }
    if thinking:
        request["thinking"] = {"type": "adaptive"}
    return request


class F08PostMergeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cfg = _cfg(self.directory.name)
        triage.reset_state()
        self.addCleanup(triage.reset_state)
        runtime.clear_cache()
        self.addCleanup(runtime.clear_cache)

    def _events(self, cfg=None):
        path = Path((cfg or self.cfg)["triage"]["path"])
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def _assert_p01_advertised_calls_envelope(self):
        """Forced bridge payloads must only dispatch the triage tool through calls[]."""
        routed = triage.force(_openai_request(), self.cfg, anthropic=False)
        parameters = routed["tools"][0]["function"]["parameters"]
        calls = parameters["properties"].get("calls")
        self.assertIsInstance(calls, dict, "P01: triage replaced the advertised calls[] bridge envelope")
        self.assertEqual(parameters["required"], ["calls"],
                         "P01: triage replaced the advertised calls[] bridge envelope")
        self.assertLessEqual(calls.get("maxItems", 1), 1,
                             "P01: forced bridge must permit at most one calls[] entry")
        item_properties = calls["items"]["properties"]
        name_schema = item_properties["name"]
        single_value_pin = (
            name_schema.get("const") == triage.TOOL_NAME
            or (isinstance(name_schema.get("enum"), list) and name_schema["enum"] == [triage.TOOL_NAME])
        )
        self.assertTrue(single_value_pin,
                        "P01: calls[*].name must be pinned to triage_task")
        self.assertIn("arguments", item_properties)

        canonical = {"calls": [{"name": triage.TOOL_NAME,
                                 "arguments": {"decision": "solo", "rationale": "small task"}}]}
        validator = Draft202012Validator(parameters)
        self.assertIsNone(next(validator.iter_errors(canonical), None),
                          "P01: canonical calls[] triage payload must validate")
        entries, normalization_error = normalize_tool_call_entries(canonical)
        self.assertIsNone(normalization_error)
        self.assertEqual(entries, canonical["calls"])
        with patch.object(tool_search, "is_deferrable_tool_name",
                          side_effect=lambda name, *_args: name == triage.TOOL_NAME):
            resolved_name, resolved_args, resolve_error = resolve_underlying_call(canonical)
        self.assertIsNone(resolve_error)
        self.assertEqual(resolved_name, triage.TOOL_NAME)
        self.assertEqual(resolved_args, canonical["calls"][0]["arguments"])

        non_triage_payloads = (
            {"calls": [{"name": "delegate_task", "arguments": {"goal": "not triage"}}]},
            {"name": triage.TOOL_NAME, "arguments": {"decision": "solo"},
             "calls": [{"name": "delegate_task", "arguments": {"goal": "not triage"}}]},
            {"calls": [{"name": "mcp__delegate_task", "arguments": {"goal": "prefixed"}}]},
            {"calls": [{"arguments": {}}]},
            {"calls": "delegate_task"},
            {"calls": {"name": "delegate_task", "arguments": {"goal": "dict"}}},
            {"calls": [
                {"name": triage.TOOL_NAME, "arguments": {"decision": "solo"}},
                {"name": triage.TOOL_NAME, "arguments": {"decision": "delegate"}},
            ]},
            {"name": "delegate_task", "arguments": {"goal": "not triage"}},
        )
        for payload in non_triage_payloads:
            with self.subTest(payload=payload):
                self.assertIsNotNone(next(validator.iter_errors(payload), None),
                                     "P01: forced bridge legalized an invalid or non-triage payload")

    def test_p01_advertised_calls_envelope_remains_pinned(self):
        """Forced calls[] accepts triage only and resolves to triage_task."""
        self._assert_p01_advertised_calls_envelope()

    def test_p01_host_validation_distinguishes_conflicting_prefixed_and_malformed_calls(self):
        conflicting, conflicting_error = normalize_tool_call_entries({
            "name": triage.TOOL_NAME,
            "arguments": {"decision": "solo"},
            "calls": [{"name": "delegate_task", "arguments": {"goal": "not triage"}}],
        })
        prefixed, prefixed_error = normalize_tool_call_entries({
            "calls": [{"name": "mcp__delegate_task", "arguments": {"goal": "prefixed"}}],
        })
        malformed, malformed_error = normalize_tool_call_entries({"calls": [{"arguments": {}}]})
        self.assertIsNone(conflicting_error)
        self.assertEqual(conflicting, [{"name": "delegate_task", "arguments": {"goal": "not triage"}}])
        self.assertIsNone(prefixed_error)
        self.assertEqual(prefixed, [{"name": "mcp__delegate_task", "arguments": {"goal": "prefixed"}}])
        self.assertEqual(malformed, [])
        self.assertIn("calls[0] requires a 'name'", malformed_error)

    def test_p02_retained_replay_uses_host_entrypoint_orderings_and_real_admission_counts(self):
        """The host wrappers, not a test-owned counter, expose the two replay orderings."""
        root = SimpleNamespace(session_id="p02-root", _delegate_depth=0)
        raw_result = json.dumps({"delegation_id": "child-1"})

        def dispatch_router_pre_hook(name, args, **hook):
            decision = router.on_pre_tool_call(name, args, **hook)
            return (decision or {}).get("message"), None

        def run_normal_executor():
            book, raw_calls = execution_adapters.ReservationBook(), []
            agent = SimpleNamespace(
                session_id="p02-normal",
                _current_turn_id="p02:normal",
                _current_api_request_id="",
                _tool_guardrails=SimpleNamespace(
                    before_call=lambda *_args: SimpleNamespace(allows_execution=True)),
            )

            def downstream(sent):
                outcome = tool_executor._run_agent_tool_execution_middleware(
                    agent,
                    function_name="delegate_task",
                    function_args=sent,
                    effective_task_id="p02-normal",
                    tool_call_id="same",
                    execute=lambda final: raw_calls.append(final) or raw_result,
                )
                return outcome.result

            with patch.object(execution_adapters, "RESERVATIONS", book), \
                 patch.object(router, "_load_config", return_value=self.cfg), \
                 patch.object(router, "_delegation_targets_detail", return_value={}), \
                 patch.object(worker_admission, "delegate_task_route", return_value=("openai-codex", "gpt-5.6-terra")), \
                 patch.object(triage, "_caller", return_value=root), \
                 patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", side_effect=dispatch_router_pre_hook), \
                 patch.object(relay_tools, "execute", side_effect=lambda _name, args, callback, **_kw: (callback(args), args)), \
                 patch("hermes_cli.middleware.apply_tool_request_middleware", side_effect=lambda _name, args, **_kw: SimpleNamespace(payload=args, trace=[])), \
                 patch("hermes_cli.middleware.run_tool_execution_middleware", side_effect=lambda _name, args, callback, **_kw: callback(args)), \
                 patch.object(tool_executor, "_begin_tool_execution"), \
                 patch.object(tool_executor, "_run_with_activity_heartbeat", side_effect=lambda _agent, _name, call: call()):
                first = execution_adapters.guard_legacy_tool_execution(
                    tool_name="delegate_task", args={"goal": "normal"}, next_call=downstream,
                    session_id="p02-normal", turn_id="p02:normal", tool_call_id="same",
                )
                replay = execution_adapters.guard_legacy_tool_execution(
                    tool_name="delegate_task", args={"goal": "normal"}, next_call=downstream,
                    session_id="p02-normal", turn_id="p02:normal", tool_call_id="same",
                )
            return first, replay, raw_calls

        def run_direct_registry():
            book, raw_calls = execution_adapters.ReservationBook(), []

            def guarded_registry_execute(function_name, function_args, _original_args, ids, **_kwargs):
                return execution_adapters.guard_legacy_tool_execution(
                    tool_name=function_name,
                    args=function_args,
                    next_call=lambda final: raw_calls.append(final) or raw_result,
                    session_id=ids.session_id,
                    turn_id=ids.turn_id,
                    tool_call_id=ids.tool_call_id,
                )

            with patch.object(execution_adapters, "RESERVATIONS", book), \
                 patch.object(router, "_load_config", return_value=self.cfg), \
                 patch.object(router, "_delegation_targets_detail", return_value={}), \
                 patch.object(worker_admission, "delegate_task_route", return_value=("openai-codex", "gpt-5.6-terra")), \
                 patch.object(triage, "_caller", return_value=root), \
                 patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", side_effect=dispatch_router_pre_hook), \
                 patch.object(model_tools, "_execute_tool", side_effect=guarded_registry_execute):
                first = model_tools.handle_function_call(
                    "delegate_claude", {"goal": "registry", "tier": "haiku"}, tool_call_id="same",
                    session_id="p02-registry", turn_id="p02:registry",
                    skip_tool_request_middleware=True, skip_tool_execution_middleware=True,
                )
                replay = model_tools.handle_function_call(
                    "delegate_claude", {"goal": "registry", "tier": "haiku"}, tool_call_id="same",
                    session_id="p02-registry", turn_id="p02:registry",
                    skip_tool_request_middleware=True, skip_tool_execution_middleware=True,
                )
            return first, replay, raw_calls

        before_normal = dict(triage._TURN_WORKERS)
        normal = run_normal_executor()
        normal_delta = triage._TURN_WORKERS.get("p02:normal", 0) - before_normal.get("p02:normal", 0)
        before_registry = dict(triage._TURN_WORKERS)
        registry = run_direct_registry()
        registry_delta = triage._TURN_WORKERS.get("p02:registry", 0) - before_registry.get("p02:registry", 0)
        self.assertEqual(normal[:2], (raw_result, raw_result))
        self.assertEqual(registry[:2], (raw_result, raw_result))
        self.assertEqual(len(normal[2]), 1, "P02 normal executor replay reached raw dispatch twice")
        self.assertEqual(len(registry[2]), 1, "P02 registry replay reached raw dispatch twice")
        self.assertEqual(normal_delta, 1, "P02 normal executor charged a sealed replay")
        self.assertEqual(registry_delta, 1, "R12-6 supported Claude registry replay charged a sealed response")

    def test_r12_6_supported_claude_registry_replay_is_retained_across_exhaustion(self):
        root = SimpleNamespace(session_id="r12-root", _delegate_depth=0)
        book, raw_calls = execution_adapters.ReservationBook(), []
        raw_result = json.dumps({"delegation_id": "r12-child"})

        def dispatch_router_pre_hook(name, args, **hook):
            decision = router.on_pre_tool_call(name, args, **hook)
            return (decision or {}).get("message"), None

        def guarded_registry_execute(function_name, function_args, _original_args, ids, **_kwargs):
            return execution_adapters.guard_legacy_tool_execution(
                tool_name=function_name, args=function_args,
                next_call=lambda final: raw_calls.append(final) or raw_result,
                session_id=ids.session_id, turn_id=ids.turn_id, tool_call_id=ids.tool_call_id,
            )

        def deliver(call_id, args=None):
            return model_tools.handle_function_call(
                "delegate_claude", args or {"goal": "registry", "tier": "haiku"}, tool_call_id=call_id,
                session_id="r12-registry", turn_id="r12:turn",
                skip_tool_request_middleware=True, skip_tool_execution_middleware=True,
            )

        with patch.object(execution_adapters, "RESERVATIONS", book), \
             patch.object(router, "_load_config", return_value=self.cfg), \
             patch.object(router, "_delegation_targets_detail", return_value={}), \
             patch.object(worker_admission, "delegate_task_route", return_value=("openai-codex", "gpt-5.6-terra")), \
             patch.object(triage, "_caller", return_value=root), \
             patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", side_effect=dispatch_router_pre_hook), \
             patch.object(model_tools, "_execute_tool", side_effect=guarded_registry_execute):
            replies = [deliver("same") for _ in range(5)]
            changed = deliver("same", {"goal": "changed", "tier": "haiku"})
            fresh = deliver("fresh")
            self.assertEqual(deliver("third"), raw_result)
            self.assertEqual(deliver("fourth"), raw_result)
            exhausted_replay = deliver("same")
            blocked = deliver("fifth")
            changed_transport = router.on_pre_tool_call(
                "delegate_task", {"goal": "registry"}, session_id="r12-registry", turn_id="r12:turn",
                tool_call_id="same",
            )

        self.assertEqual(replies, [raw_result] * 5)
        self.assertEqual(fresh, raw_result, "a distinct valid call must fit after retained replays")
        self.assertEqual(exhausted_replay, raw_result)
        self.assertIn("error", json.loads(changed))
        self.assertIn("error", json.loads(blocked))
        self.assertEqual(changed_transport["action"], "block")
        self.assertIn("identity", changed_transport["message"])
        self.assertEqual(len(raw_calls), 4)
        self.assertEqual(triage._TURN_WORKERS.get("r12:turn"), 4)

    def test_r12_6_registry_concurrency_reserves_one_admitted_attempt(self):
        root = SimpleNamespace(session_id="r12-concurrent-root", _delegate_depth=0)
        book, raw_calls, results = execution_adapters.ReservationBook(), [], []
        started, release = threading.Event(), threading.Event()
        raw_result = json.dumps({"delegation_id": "r12-concurrent-child"})

        def dispatch_router_pre_hook(name, args, **hook):
            decision = router.on_pre_tool_call(name, args, **hook)
            return (decision or {}).get("message"), None

        def guarded_registry_execute(function_name, function_args, _original_args, ids, **_kwargs):
            def raw(final):
                raw_calls.append(final)
                started.set()
                self.assertTrue(release.wait(5))
                return raw_result
            return execution_adapters.guard_legacy_tool_execution(
                tool_name=function_name, args=function_args, next_call=raw,
                session_id=ids.session_id, turn_id=ids.turn_id, tool_call_id=ids.tool_call_id,
            )

        def deliver():
            return model_tools.handle_function_call(
                "delegate_claude", {"goal": "registry", "tier": "haiku"}, tool_call_id="same",
                session_id="r12-concurrent", turn_id="r12:concurrent",
                skip_tool_request_middleware=True, skip_tool_execution_middleware=True,
            )

        with patch.object(execution_adapters, "RESERVATIONS", book), \
             patch.object(router, "_load_config", return_value=self.cfg), \
             patch.object(router, "_delegation_targets_detail", return_value={}), \
             patch.object(worker_admission, "delegate_task_route", return_value=("openai-codex", "gpt-5.6-terra")), \
             patch.object(triage, "_caller", return_value=root), \
             patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", side_effect=dispatch_router_pre_hook), \
             patch.object(model_tools, "_execute_tool", side_effect=guarded_registry_execute):
            thread = threading.Thread(target=lambda: results.append(deliver()))
            thread.start()
            self.assertTrue(started.wait(5))
            try:
                with patch.object(execution_adapters, "_RECORD_TTL_SECONDS", 0):
                    duplicate = deliver()
            finally:
                release.set()
                thread.join(5)
            replay = deliver()

        self.assertEqual(results, [raw_result])
        self.assertIn("error", json.loads(duplicate))
        self.assertEqual(replay, raw_result)
        self.assertEqual(len(raw_calls), 1)
        self.assertEqual(triage._TURN_WORKERS.get("r12:concurrent"), 1)

    def test_f08_triage_switch_replaces_preflight_shadow_and_budget_but_keeps_the_note(self):
        decision = router.RouteDecision("terra", "gpt-5.6-terra", "test")
        kwargs = {"request": _openai_request(), "model": "gpt-5.6-terra", "turn_id": "f08:i1", "api_call_count": 1}
        triaged = _openai_request()
        with patch.object(router, "_load_config", return_value=self.cfg), \
             patch.object(router, "classify_request", return_value=decision), \
             patch.object(router, "_require_callable", side_effect=lambda value, _cfg: value), \
             patch.object(router, "_triage_request", return_value=triaged) as triage_request, \
             patch.object(router, "_force_terra_supervisor_preflight") as preflight, \
             patch.object(router, "_force_shadow_delegation_if_eligible") as shadow, \
             patch.object(router, "_routing_note", return_value="\\nROUTING NOTE") as note:
            routed = router._route_llm_request(**kwargs)
        triage_request.assert_called_once()
        preflight.assert_not_called()
        shadow.assert_not_called()
        note.assert_called_once()
        self.assertIn("ROUTING NOTE", str(routed["request"]))

        off = _cfg(self.directory.name, triage_enabled=False)
        preflight_request = _openai_request()
        with patch.object(router, "_load_config", return_value=off), \
             patch.object(router, "classify_request", return_value=decision), \
             patch.object(router, "_require_callable", side_effect=lambda value, _cfg: value), \
             patch.object(router, "_triage_request", return_value=None), \
             patch.object(router, "_force_terra_supervisor_preflight", return_value=preflight_request) as preflight, \
             patch.object(router, "_force_shadow_delegation_if_eligible") as shadow:
            routed = router._route_llm_request(**kwargs)
        preflight.assert_called_once()
        shadow.assert_not_called()
        self.assertEqual(routed["request"]["model"], "gpt-5.6-terra")

        with patch.object(router, "_load_config", return_value=self.cfg), \
             patch.object(triage, "spawn_block", return_value="") as spawn, \
             patch.object(router, "_bounded_dispatch_block", return_value="") as bounded:
            self.assertIsNone(router.on_pre_tool_call("delegate_task", {"goal": "x"}, turn_id="f08:on"))
        spawn.assert_called_once()
        bounded.assert_not_called()
        with patch.object(router, "_load_config", return_value=off), \
             patch.object(triage, "spawn_block", return_value="") as spawn, \
             patch.object(router, "_bounded_dispatch_block", return_value="") as bounded:
            self.assertIsNone(router.on_pre_tool_call("delegate_task", {"goal": "x"}, turn_id="f08:off"))
        spawn.assert_not_called()
        bounded.assert_called_once()

    def test_f08_mixed_local_batch_expansion_has_stable_ids_and_charges_each_delegate_entry(self):
        mixed_entries = [
            {"name": "delegate_task", "arguments": {"goal": "first"}},
            {"name": "triage_task", "arguments": {"decision": "solo"}},
            {"name": "web_search", "arguments": {"query": "non-local"}},
        ]
        singleton_entry = {"name": "delegate_task", "arguments": {"goal": "singleton"}}
        connector_entries = [
            {"name": "connectors__drive__list", "arguments": {}},
            {"name": "connectors__drive__search", "arguments": {"query": "x"}},
        ]
        def bridge_call(call_id, entries):
            return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(
                name="tool_call", arguments=json.dumps({"calls": entries})))

        original = [bridge_call("mixed", mixed_entries), bridge_call("singleton", [singleton_entry]),
                    bridge_call("connector-only", connector_entries)]
        expanded = expand_local_tool_batches(original)
        rerun_ids = [call.id for call in expand_local_tool_batches(original)]
        expanded_entries = [json.loads(call.function.arguments)["calls"] for call in expanded]
        self.assertEqual(expanded_entries[:3], [[entry] for entry in mixed_entries])
        self.assertEqual(expanded_entries[3:], [[singleton_entry], connector_entries])
        self.assertEqual(len(expanded), 5)
        self.assertEqual([call.id for call in expanded], rerun_ids)
        self.assertEqual(expanded[0].id, "mixed")
        self.assertEqual(len({call.id for call in expanded}), len(expanded))

        root = SimpleNamespace(session_id="p02-batch-root", _delegate_depth=0)
        with patch.object(router, "_load_config", return_value=self.cfg), \
             patch.object(router, "_delegation_targets_detail", return_value={}), \
             patch.object(worker_admission, "delegate_task_route", return_value=("openai-codex", "gpt-5.6-terra")), \
             patch.object(triage, "_caller", return_value=root):
            for call in expanded:
                entries, error = normalize_tool_call_entries(json.loads(call.function.arguments))
                self.assertIsNone(error)
                for entry in entries:
                    self.assertIsNone(router.on_pre_tool_call(
                        entry["name"], entry["arguments"], session_id="p02-batch-root", turn_id="p02:batch"))
        self.assertEqual(triage._TURN_WORKERS, {"p02:batch": 2})

    def test_p03_router_refusals_do_not_consume_worker_admission(self):
        cfg = _cfg(self.directory.name)
        cfg["callable"].update({"sol": False, "haiku": False, "sonnet5": False, "opus5": False})
        cfg["orchestration"] = {
            "enabled": True,
            "min_goal_chars": 20,
            "path": str(Path(self.directory.name) / "orchestration.jsonl"),
        }
        root = SimpleNamespace(session_id="p03-root", _delegate_depth=0)
        conductor = {"tasks": [{
            "goal": "tiny",
            "context": "You are the terra planning conductor.",
        }]}
        with patch.object(router, "_load_config", return_value=cfg), \
             patch.object(router, "_delegation_targets_detail", return_value={}), \
             patch.object(triage, "_caller", return_value=root):
            for _ in range(4):
                refusal = router.on_pre_tool_call(
                    "delegate_task", {"model": "sol", "goal": "switched off"},
                    session_id="p03-root", turn_id="p03:turn")
                self.assertEqual(refusal["action"], "block")
                self.assertIn('"sol" is switched off', refusal["message"])
            self.assertEqual(triage._TURN_WORKERS, {},
                             "P03: router target refusals consumed worker admission")
            self.assertIsNone(router.on_pre_tool_call(
                "delegate_task", {"model": "terra", "goal": "valid"},
                session_id="p03-root", turn_id="p03:turn"))
            self.assertEqual(triage._TURN_WORKERS, {"p03:turn": 1})
            before_conductor = dict(triage._TURN_WORKERS)
            declined = router.on_pre_tool_call(
                "delegate_task", conductor, session_id="p03-root", turn_id="p03:turn")
            self.assertEqual(declined["action"], "block")
            self.assertIn("No conductor for this turn", declined["message"])
            self.assertEqual(triage._TURN_WORKERS, before_conductor,
                             "P03: declined conductor consumed worker admission")
            triage.reset_state()
            claude = router.on_pre_tool_call(
                "delegate_claude", {"tier": "sonnet", "goal": "switched off"},
                session_id="p03-root", turn_id="p03:claude")
        self.assertEqual(claude["action"], "block")
        self.assertIn("Claude delegation is off", claude["message"])
        self.assertEqual(triage._TURN_WORKERS, {},
                         "P03: delegate_claude router refusal consumed worker admission")

    def test_p03_host_later_refusal_retains_the_admitted_allowance(self):
        """A host rejection after the real pre-tool hook has no refund seam (S13)."""
        root = SimpleNamespace(session_id="p03-host", _delegate_depth=0)

        def dispatch_router_pre_hook(name, args, **hook):
            decision = router.on_pre_tool_call(name, args, **hook)
            return (decision or {}).get("message"), None

        host_refusal = json.dumps({"error": "host rejected after pre-tool admission"})
        with patch.object(router, "_load_config", return_value=self.cfg), \
             patch.object(router, "_delegation_targets_detail", return_value={}), \
             patch.object(worker_admission, "delegate_task_route", return_value=("openai-codex", "gpt-5.6-terra")), \
             patch.object(triage, "_caller", return_value=root), \
             patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", side_effect=dispatch_router_pre_hook), \
             patch.object(model_tools, "_AGENT_LOOP_TOOLS", model_tools._AGENT_LOOP_TOOLS - {"delegate_task"}), \
             patch.object(model_tools, "_execute_tool", return_value=host_refusal):
            result = model_tools.handle_function_call(
                "delegate_task", {"goal": "valid"}, session_id="p03-host", turn_id="p03:host",
                skip_tool_request_middleware=True, skip_tool_execution_middleware=True,
            )
        self.assertEqual(result, host_refusal)
        self.assertEqual(self._events()[-1]["event"], "spawn_admitted")
        self.assertEqual(triage._TURN_WORKERS, {"p03:host": 1},
                         "P03: host-later refusal unexpectedly restored the admitted allowance")

    def test_b_i3_pre_admission_accepts_handler_authorized_effective_claude_tier_only(self):
        cfg = _cfg(self.directory.name)
        cfg["usage_guard"] = {"accounts": {
            "anthropic": {"soft_percent": 70, "hard_percent": 90, "step_down": {"opus5": "sonnet5"}},
        }}
        fresh_soft = usage_guard.Reading(75.0, 0.0, None, None, time.time())
        fresh_hard = usage_guard.Reading(95.0, 0.0, None, None, time.time())

        def preflight(label, *, callable_updates, reading=fresh_soft, cooldown=0.0):
            triage.reset_state()
            scenario = json.loads(json.dumps(cfg))
            scenario["callable"].update(callable_updates)
            with patch.object(router, "_load_config", return_value=scenario), \
                 patch.object(usage_guard, "read", return_value=reading), \
                 patch.object(router, "_tier_cooldown_remaining",
                              side_effect=lambda tier, _cfg: cooldown if tier == "opus5" else 0.0):
                result = router.on_pre_tool_call(
                    "delegate_claude", {"tier": "opus", "tasks": [{"goal": "inspect"}]},
                    session_id="b-i3", turn_id=f"b-i3:{label}", tool_call_id=label,
                )
            return result

        for label, callable_updates, cooldown in (
            ("switched-off", {"opus5": False, "sonnet5": True}, 0.0),
            ("cooling", {"opus5": True, "sonnet5": True}, 60.0),
        ):
            with self.subTest(viable=label):
                self.assertIsNone(preflight(label, callable_updates=callable_updates, cooldown=cooldown))
                self.assertEqual(triage._TURN_WORKERS, {f"b-i3:{label}": 1})

        for label, callable_updates, reading in (
            ("no-step-down", {"opus5": False, "sonnet5": True}, fresh_soft),
            ("effective-unavailable", {"opus5": False, "sonnet5": False, "haiku": True}, fresh_soft),
            ("globally-disabled", {"opus5": False, "sonnet5": False, "haiku": False}, fresh_soft),
            ("hard-closed", {"opus5": True, "sonnet5": True}, fresh_hard),
        ):
            with self.subTest(refused=label):
                if label == "no-step-down":
                    cfg["usage_guard"]["accounts"]["anthropic"]["step_down"] = {}
                result = preflight(label, callable_updates=callable_updates, reading=reading)
                self.assertEqual((result or {}).get("action"), "block")
                self.assertEqual(triage._TURN_WORKERS, {})
                cfg["usage_guard"]["accounts"]["anthropic"]["step_down"] = {"opus5": "sonnet5"}

        triage.reset_state()
        with patch.object(router, "_load_config", return_value=cfg), \
             patch.object(usage_guard, "read", return_value=fresh_soft):
            unknown = router.on_pre_tool_call(
                "delegate_claude", {"tier": "unknown", "tasks": [{"goal": "inspect"}]},
                session_id="b-i3", turn_id="b-i3:unknown", tool_call_id="unknown",
            )
        self.assertEqual((unknown or {}).get("action"), "block")
        self.assertEqual(triage._TURN_WORKERS, {})

    def test_b_i4_supported_delegate_task_controls_bypass_ignored_spawn_fields_and_admission(self):
        stale = {
            "subagent_id": "sa-existing", "message": "change course", "model": "sol",
            "tasks": [{"goal": "[sol:bogus] Implement an ignored task."}],
        }
        for triage_enabled in (True, False):
            with self.subTest(triage_enabled=triage_enabled):
                triage.reset_state()
                cfg = _cfg(self.directory.name, triage_enabled=triage_enabled)
                cfg["callable"]["sol"] = False
                if triage_enabled:
                    cfg["triage"]["max_workers_per_turn"] = 1
                    with patch.object(router, "_load_config", return_value=cfg), \
                         patch.object(router, "_delegation_targets_detail", return_value={}):
                        self.assertIsNone(router.on_pre_tool_call(
                            "delegate_task", {"goal": "ordinary spawn"}, turn_id="b-i4:triage",
                        ))
                        before = dict(triage._TURN_WORKERS)
                        for action in ("list", "steer", "stop"):
                            result = router.on_pre_tool_call(
                                "delegate_task", {**stale, "action": action}, turn_id="b-i4:triage",
                            )
                            self.assertIsNone(result, action)
                            self.assertEqual(triage._TURN_WORKERS, before, action)
                else:
                    budget = Path(self.directory.name) / "b-i4-bounded.jsonl"
                    cfg["task_budget"] = {"enabled": True, "path": str(budget), "max_routing_decisions": 1}
                    with patch.object(router, "_load_config", return_value=cfg), \
                         patch.object(router, "_delegation_targets_detail", return_value={}):
                        self.assertIsNone(router.on_pre_tool_call(
                            "delegate_task", {"goal": "ordinary spawn"}, turn_id="b-i4:bounded",
                        ))
                        before = budget.read_text(encoding="utf-8")
                        for action in ("list", "steer", "stop"):
                            result = router.on_pre_tool_call(
                                "delegate_task", {**stale, "action": action}, turn_id="b-i4:bounded",
                            )
                            self.assertIsNone(result, action)
                            self.assertEqual(budget.read_text(encoding="utf-8"), before, action)

    def _assert_p04_advisory_is_not_forced(self):
        decision = router.RouteDecision("terra", "gpt-5.6-terra", "test")
        kwargs = {"request": _anthropic_request("claude-opus-5-5"), "api_call_count": 1,
                  "turn_id": "f08-p04", "platform": "cli", "model": "claude-opus-5-5"}
        with patch.object(router, "classify_request", return_value=decision) as classified:
            routed = router._triage_request(kwargs, self.cfg, decision)
        classified.assert_called_once_with(kwargs["request"], api_call_count=1, config=self.cfg)
        self.assertIsNotNone(routed)
        self.assertNotIn("tool_choice", routed,
                         "P04 setup must exercise the advisory Claude 5 path")
        self.assertEqual(self._events()[-1]["event"], "triage_advisory",
                         "P04: advisory rewrite was logged as triage_forced")

    def test_p04_advisory_claude_five_rewrite_is_not_telemetry_forced(self):
        self._assert_p04_advisory_is_not_forced()

    # --- R12-7: advisory tool-choice compatibility --------------------------------

    @staticmethod
    def _r12_7_direct_triage_request(model, *, thinking):
        request = _anthropic_request(model, thinking=thinking)
        request["tools"] = [
            {"name": triage.TOOL_NAME, "input_schema": triage.SCHEMA_PARAMETERS},
            *request["tools"],
        ]
        return request

    def test_r12_7_triage_advisory_downgrades_generic_any_without_mutating_or_pinning(self):
        advisory_models = (
            ("claude-opus-5-5", False),
            ("claude-opus-5-5", True),
            ("claude-sonnet-4-6", True),
        )
        for shape, build in (("bridge", _anthropic_request), ("direct", self._r12_7_direct_triage_request)):
            for model, thinking in advisory_models:
                with self.subTest(shape=shape, model=model, thinking=thinking):
                    request = build(model, thinking=thinking)
                    request["tool_choice"] = {"type": "any"}
                    original = json.loads(json.dumps(request))
                    routed = triage.force(request, self.cfg, anthropic=True)
                    self.assertEqual(request, original, "R12-7: triage mutated the caller request")
                    self.assertEqual(routed["tool_choice"], {"type": "auto"})
                    self.assertEqual(len(routed["tools"]), len(request["tools"]),
                                     "R12-7: advisory triage narrowed the toolset")
                    self.assertIn("[ROUTER TRIAGE]", routed["messages"][-1]["content"][-1]["text"])
                    if shape == "bridge":
                        schema = next(tool for tool in routed["tools"] if tool["name"] == "mcp__tool_call")["input_schema"]
                        self.assertEqual(schema["required"], ["name", "arguments"],
                                         "R12-7: advisory bridge pinning changed")
                        name = schema["properties"]["calls"]["items"]["properties"]["name"]
                        self.assertNotEqual(name.get("enum"), [triage.TOOL_NAME])
                        self.assertNotIn("const", name)

    def test_r12_7_triage_advisory_preserves_auto_and_forces_legal_older_control(self):
        advisory = _anthropic_request("claude-opus-5-5", thinking=False)
        advisory["tool_choice"] = {"type": "auto"}
        original = json.loads(json.dumps(advisory))
        routed = triage.force(advisory, self.cfg, anthropic=True)
        self.assertEqual(advisory, original)
        self.assertEqual(routed["tool_choice"], {"type": "auto"})

        forced = _anthropic_request("claude-haiku-4-5", thinking=False)
        forced["tool_choice"] = {"type": "any"}
        forced_original = json.loads(json.dumps(forced))
        legal = triage.force(forced, self.cfg, anthropic=True)
        self.assertEqual(forced, forced_original)
        self.assertEqual(legal["tool_choice"], {"type": "tool", "name": "mcp__tool_call"})
        self.assertEqual([tool["name"] for tool in legal["tools"]], ["mcp__tool_call"])

    def test_r12_7_triage_advisory_refuses_named_authority_in_each_supported_shape(self):
        advisory_models = (
            ("claude-opus-5-5", False),
            ("claude-opus-5-5", True),
            ("claude-sonnet-4-6", True),
        )
        for shape, build in (("bridge", _anthropic_request), ("direct", self._r12_7_direct_triage_request)):
            for model, thinking in advisory_models:
                with self.subTest(shape=shape, model=model, thinking=thinking):
                    request = build(model, thinking=thinking)
                    request["tool_choice"] = {"type": "tool", "name": "mcp__web_search"}
                    original = json.loads(json.dumps(request))
                    refused = triage.force(request, self.cfg, anthropic=True)
                    self.assertEqual(request, original, "R12-7: triage mutated an exact choice request")
                    self.assertEqual(refused["_model_router_refusal"],
                                     "Cannot run advisory triage because exact tool 'mcp__web_search' is required.")
                    self.assertEqual(refused["tools"], request["tools"])
                    self.assertNotIn("[ROUTER TRIAGE]", json.dumps(refused))

    def test_r12_7_named_advisory_choice_stops_through_registered_host_chain(self):
        from hermes_cli import plugins
        from hermes_cli.middleware import apply_llm_request_middleware, run_llm_execution_middleware
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        cfg = _cfg(self.directory.name)
        cfg.update(provider="anthropic", models={"terra": "claude-opus-5-5"},
                   tier_providers={"terra": "anthropic"})
        request = _anthropic_request("claude-opus-5-5", thinking=False)
        request["tool_choice"] = {"type": "tool", "name": "mcp__web_search"}
        original = json.loads(json.dumps(request))
        manager = PluginManager()
        manager._discovered = True
        ctx = PluginContext(PluginManifest(name="r12-7", source="test", key="r12-7"), manager)
        terminal = Mock(return_value="ORDINARY PROVIDER EXECUTED")
        active_before = router.claude_delegation._ACTIVE
        availability_before = router.claude_delegation._LAST_AVAILABLE
        self.addCleanup(setattr, router.claude_delegation, "_ACTIVE", active_before)
        self.addCleanup(setattr, router.claude_delegation, "_LAST_AVAILABLE", availability_before)
        with patch.object(plugins, "_plugin_manager", manager), \
             patch("hermes_cli.plugins._delivery_manager", return_value=manager), \
             patch.object(router, "_load_config", return_value=cfg), \
             patch.object(router, "_run_opus5_bridge") as bridge:
            router.register(ctx)
            self.addCleanup(lambda: [registration.dispose() for registration in reversed(manager._registration_order)])
            composed = apply_llm_request_middleware(
                request, model="claude-opus-5-5", provider="anthropic", api_mode="anthropic_messages",
                api_call_count=1, platform="cli", turn_id="r12-7:exact",
            )
            stopped = run_llm_execution_middleware(
                composed.payload, terminal, provider="anthropic", api_mode="anthropic_messages",
                api_call_count=1, platform="cli", turn_id="r12-7:exact",
            )
        self.assertEqual(request, original, "R12-7: host request composition mutated input")
        self.assertEqual(composed.payload["_model_router_refusal"],
                         "Cannot run advisory triage because exact tool 'mcp__web_search' is required.")
        self.assertEqual(len(composed.payload["tools"]), len(request["tools"]))
        self.assertEqual(self._events(cfg)[-1]["event"], "triage_skipped")
        self.assertEqual(self._events(cfg)[-1]["reason"], "authoritative_tool_choice")
        terminal.assert_not_called()
        bridge.assert_not_called()
        self.assertEqual(stopped.usage.total_tokens, 0)
        self.assertIn("ROUTER WORKER STOPPED", stopped.output_text)
        self.assertIn("exact tool 'mcp__web_search'", stopped.output_text)

    def test_r12_7_conductor_advisory_reconciles_or_refuses_through_preflight(self):
        cfg = _cfg(self.directory.name, triage_enabled=False)
        cfg["orchestration"] = {
            "enabled": True, "min_chars": 1, "max_tasks": 2,
            "path": str(Path(self.directory.name) / "orchestration.jsonl"),
        }
        decision = router.RouteDecision("terra", "gpt-5.6-terra", "test")
        advisory_models = (
            ("claude-opus-5-5", False),
            ("claude-opus-5-5", True),
            ("claude-sonnet-4-6", True),
        )
        for model, thinking in advisory_models:
            for choice, expected in (
                ({"type": "any"}, {"type": "auto"}),
                ({"type": "tool", "name": "mcp__web_search"}, None),
            ):
                with self.subTest(model=model, thinking=thinking, choice=choice), host_delegation(depth=2):
                    request = _anthropic_request(model, thinking=thinking)
                    request["tool_choice"] = choice
                    original = json.loads(json.dumps(request))
                    routed = router._force_terra_supervisor_preflight(
                        {"request": request, "model": "gpt-5.6-terra", "provider": "anthropic",
                         "api_call_count": 1, "turn_id": f"r12-7:conductor:{model}:{thinking}:{choice['type']}", "platform": "cli"},
                        cfg, decision,
                    )
                    self.assertEqual(request, original, "R12-7: conductor mutated the caller request")
                    self.assertIsNotNone(routed)
                    if expected is None:
                        self.assertEqual(routed["_model_router_refusal"],
                                         "Cannot run advisory preflight because exact tool 'mcp__web_search' is required.")
                        self.assertEqual(routed["tools"], request["tools"])
                    else:
                        self.assertEqual(routed["tool_choice"], expected)
                        self.assertEqual(len(routed["tools"]), len(request["tools"]),
                                         "R12-7: advisory conductor narrowed the toolset")

    # --- P05: a late child is charged to the root turn it came from -------------------------
    # The host reports the originating turn to the router in the subagent_start hook; the
    # tests drive the real hook (router.on_subagent_start) and the real gate (spawn_block).

    def _p05_root(self, session="root"):
        return SimpleNamespace(session_id=session, _delegate_depth=0)

    def _p05_child(self, session):
        return SimpleNamespace(session_id=session, _delegate_depth=1, _parent_session_id="root")

    def _p05_spawn(self, agent, turn_id, *, tool="delegate_task", tasks=1, cfg=None):
        args = {"tasks": [{"goal": f"g{i}"} for i in range(tasks)]} if tasks > 1 else {"goal": "g"}
        with patch.object(triage, "_caller", return_value=agent):
            return triage.spawn_block(tool, args, cfg or self.cfg, {"turn_id": turn_id})

    def _p05_start(self, child_session, parent_turn_id, *, parent_session="root", cfg=None):
        with patch.object(router, "_load_config", return_value=cfg or self.cfg):
            router.on_subagent_start(parent_session_id=parent_session, parent_turn_id=parent_turn_id,
                                     child_session_id=child_session, child_subagent_id=f"sa-{child_session}",
                                     child_role="leaf", child_goal="late work")

    def test_p05_late_child_keeps_its_originating_root_turn(self):
        """Old turn at its ceiling, newer turn admitted, then the old turn's child asks to spawn."""
        root = self._p05_root()
        self.assertEqual(self._p05_spawn(root, "root:old", tasks=4), "")
        self.assertEqual(self._p05_spawn(root, "root:new"), "")
        self._p05_start("child", "root:old")
        self._p05_spawn(self._p05_child("child"), "child:turn")
        self.assertEqual(self._events()[-1]["reason"], "turn_worker_limit",
                         "P05: late child was charged to the newest root turn")
        self.assertEqual(triage._TURN_WORKERS, {"root:old": 4, "root:new": 1})
        self.assertEqual(self._events()[-1].get("origin_source"), "child_link")
        self.assertEqual(self._events()[-1].get("root_turn"), "root:old")

    def test_p05_older_child_is_charged_to_its_origin_not_the_full_newer_turn(self):
        """Reverse case: the newer turn is full, the older turn has headroom."""
        root = self._p05_root()
        self.assertEqual(self._p05_spawn(root, "root:old"), "")
        self.assertEqual(self._p05_spawn(root, "root:new", tasks=4), "")
        self._p05_start("child", "root:old")
        self.assertEqual(self._p05_spawn(self._p05_child("child"), "child:turn"), "",
                         "P05: older turn's child was refused because a newer turn is full")
        self.assertEqual(triage._TURN_WORKERS, {"root:old": 2, "root:new": 4})
        self.assertEqual(self._events()[-1].get("root_turn"), "root:old")

    def test_p05_several_children_of_two_turns_each_use_their_own_allowance(self):
        root = self._p05_root()
        self.assertEqual(self._p05_spawn(root, "root:old", tasks=2), "")
        self.assertEqual(self._p05_spawn(root, "root:new"), "")
        for child, origin in (("a", "root:old"), ("b", "root:old"), ("c", "root:old"), ("n", "root:new")):
            self._p05_start(child, origin)
        self.assertEqual(self._p05_spawn(self._p05_child("a"), "a:turn"), "")
        self.assertEqual(self._p05_spawn(self._p05_child("b"), "b:turn"), "")
        refused = self._p05_spawn(self._p05_child("c"), "c:turn")
        self.assertIn("4 of 4", refused, "P05: third child of the old turn was not held to its ceiling")
        self.assertEqual(self._events()[-1]["reason"], "turn_worker_limit")
        self.assertEqual(self._events()[-1].get("root_turn"), "root:old")
        self.assertEqual(self._p05_spawn(self._p05_child("n"), "n:turn"), "")
        self.assertEqual(triage._TURN_WORKERS, {"root:old": 4, "root:new": 2})

    def test_p05_both_spawn_transports_resolve_the_origin(self):
        for tool in ("delegate_task", "delegate_claude"):
            with self.subTest(tool=tool):
                triage.reset_state()
                root = self._p05_root()
                self.assertEqual(self._p05_spawn(root, "root:old", tasks=4, tool=tool), "")
                self.assertEqual(self._p05_spawn(root, "root:new", tool=tool), "")
                self._p05_start("child", "root:old")
                refused = self._p05_spawn(self._p05_child("child"), "child:turn", tool=tool)
                self.assertIn("4 of 4", refused, f"P05: {tool} child charged to the newest turn")
                self.assertEqual(self._events()[-1]["tool"], tool)
                self.assertEqual(self._events()[-1].get("root_turn"), "root:old")
                self.assertEqual(triage._TURN_WORKERS, {"root:old": 4, "root:new": 1})

    def test_p05_origin_is_honoured_through_the_pre_tool_call_hook(self):
        for tool in ("delegate_task", "delegate_claude"):
            with self.subTest(tool=tool):
                triage.reset_state()
                root = self._p05_root()
                self.assertEqual(self._p05_spawn(root, "root:old", tasks=4, tool=tool), "")
                self.assertEqual(self._p05_spawn(root, "root:new", tool=tool), "")
                self._p05_start("child", "root:old")
                with patch.object(router, "_load_config", return_value=self.cfg), \
                     patch.object(router, "_delegation_targets_detail", return_value={}), \
                     patch.object(triage, "_caller", return_value=self._p05_child("child")):
                    refusal = router.on_pre_tool_call(tool, {"goal": "late"}, session_id="child",
                                                      turn_id="child:turn")
                self.assertEqual((refusal or {}).get("action"), "block")
                self.assertIn("4 of 4", (refusal or {}).get("message", ""))
                self.assertEqual(self._events()[-1].get("root_turn"), "root:old")

    def test_p05_unknown_child_falls_back_and_the_event_says_so(self):
        root = self._p05_root()
        self.assertEqual(self._p05_spawn(root, "root:old", tasks=4), "")
        self.assertEqual(self._p05_spawn(root, "root:new"), "")
        self.assertEqual(self._p05_spawn(self._p05_child("stranger"), "stranger:turn"), "")
        event = self._events()[-1]
        self.assertEqual(event.get("origin_source"), "latest_session_pointer")
        self.assertEqual(event.get("root_turn"), "root:new")
        self.assertEqual(triage._TURN_WORKERS, {"root:old": 4, "root:new": 2})

    def test_p05_a_link_is_not_recorded_without_an_unambiguous_origin(self):
        off_cfg = _cfg(self.directory.name, triage_enabled=False)
        cases = {
            "triage off": ("off", "root:old", off_cfg),
            "empty parent turn": ("empty", "", None),
            "sub-agent parent turn": ("sub", "w1:sa-0-x:turn", None),
        }
        for label, (child, parent_turn, cfg) in cases.items():
            with self.subTest(label):
                triage.reset_state()
                root = self._p05_root()
                self.assertEqual(self._p05_spawn(root, "root:old", tasks=4), "")
                self.assertEqual(self._p05_spawn(root, "root:new"), "")
                self._p05_start(child, parent_turn, cfg=cfg)
                self.assertEqual(self._p05_spawn(self._p05_child(child), f"{child}:turn"), "")
                self.assertEqual(self._events()[-1].get("origin_source"), "latest_session_pointer")
                self.assertEqual(self._events()[-1].get("root_turn"), "root:new")

    def test_p05_conflicting_origins_are_refused_not_guessed(self):
        root = self._p05_root()
        self.assertEqual(self._p05_spawn(root, "root:old"), "")
        self.assertEqual(self._p05_spawn(root, "root:new"), "")
        self._p05_start("child", "root:old")
        self._p05_start("child", "root:new")
        refused = self._p05_spawn(self._p05_child("child"), "child:turn")
        self.assertIn("ambiguous", refused)
        self.assertEqual(self._events()[-1]["reason"], "turn_origin_ambiguous")
        self.assertEqual(triage._TURN_WORKERS, {"root:old": 1, "root:new": 1})

    def test_p05_repeated_start_for_the_same_origin_is_not_ambiguous(self):
        root = self._p05_root()
        self.assertEqual(self._p05_spawn(root, "root:old"), "")
        self.assertEqual(self._p05_spawn(root, "root:new"), "")
        self._p05_start("child", "root:old")
        self._p05_start("child", "root:old")
        self.assertEqual(self._p05_spawn(self._p05_child("child"), "child:turn"), "")
        self.assertEqual(self._events()[-1].get("root_turn"), "root:old")

    def test_p05_origin_map_is_capped_oldest_first(self):
        root = self._p05_root()
        self.assertEqual(self._p05_spawn(root, "root:old", tasks=4), "")
        self.assertEqual(self._p05_spawn(root, "root:new"), "")
        with patch.object(triage, "_ORIGIN_MAX", 2, create=True):
            for child in ("c1", "c2", "c3"):
                self._p05_start(child, "root:old")
        self.assertEqual(len(getattr(triage, "_ORIGIN_BY_CHILD", {})), 2)
        for child in ("c2", "c3"):
            self.assertIn("4 of 4", self._p05_spawn(self._p05_child(child), f"{child}:turn"),
                          f"P05: surviving link for {child} was ignored")
            self.assertEqual(self._events()[-1].get("origin_source"), "child_link")
        self.assertEqual(self._p05_spawn(self._p05_child("c1"), "c1:turn"), "")
        self.assertEqual(self._events()[-1].get("origin_source"), "latest_session_pointer")

    def test_p05_origin_links_expire(self):
        clock = [1000.0]
        root = self._p05_root()
        self.assertEqual(self._p05_spawn(root, "root:old", tasks=4), "")
        self.assertEqual(self._p05_spawn(root, "root:new"), "")
        with patch.object(triage, "_monotonic", lambda: clock[0], create=True), \
             patch.object(triage, "_ORIGIN_TTL_SECONDS", 60, create=True):
            self._p05_start("c1", "root:old")
            clock[0] = 1030.0
            self._p05_start("c2", "root:old")
            clock[0] = 1061.0  # c1 is 61s old, c2 31s
            self.assertEqual(self._p05_spawn(self._p05_child("c1"), "c1:turn"), "")
            self.assertEqual(self._events()[-1].get("origin_source"), "latest_session_pointer")
            self.assertIn("4 of 4", self._p05_spawn(self._p05_child("c2"), "c2:turn"))
            self.assertEqual(self._events()[-1].get("origin_source"), "child_link")
            self._p05_start("c3", "root:old")  # recording purges the expired c1
            self.assertEqual(len(getattr(triage, "_ORIGIN_BY_CHILD", {})), 2)

    def test_p05_reset_state_clears_the_origin_links(self):
        root = self._p05_root()
        self.assertEqual(self._p05_spawn(root, "root:old", tasks=4), "")
        self.assertEqual(self._p05_spawn(root, "root:new"), "")
        self._p05_start("child", "root:old")
        self.assertIn("4 of 4", self._p05_spawn(self._p05_child("child"), "child:turn"))
        triage.reset_state()
        self.assertEqual(self._p05_spawn(root, "root:old", tasks=4), "")
        self.assertEqual(self._p05_spawn(root, "root:new"), "")
        self.assertEqual(self._p05_spawn(self._p05_child("child"), "child:turn"), "")
        self.assertEqual(self._events()[-1].get("origin_source"), "latest_session_pointer")

    def test_p06_triage_guidance_is_not_structured_execution_authority(self):
        scratch_cfg = _cfg(self.directory.name)
        with patch.object(router, "_load_config", return_value=scratch_cfg):
            response = json.loads(triage.handle_triage({"decision": "solo", "rationale": "small"}))
        adapters = execution_adapters.legacy_adapters(delegate_task=lambda *_: None,
                                                       delegate_claude=lambda *_: None,
                                                       claude_bridge=lambda *_: None)
        capabilities = adapters["hermes_codex"].capabilities(object())
        self.assertIn("Do the work yourself", response["next"])
        self.assertEqual(capabilities.capabilities, ())
        self.assertTrue(Path(scratch_cfg["triage"]["path"]).exists())

    def test_p07_code_defaults_retire_explicit_effort_keys(self):
        self.assertEqual(router._DEFAULT_CONFIG["effort"]["sol"], "high")
        self.assertNotIn("explicit_sol", router._DEFAULT_CONFIG["effort"])

    def test_p07_shipped_config_keeps_bare_and_stated_sol_effort_distinct(self):
        shipped = router.yaml.safe_load(Path(__file__).with_name("router_config.yaml").read_text(encoding="utf-8"))
        bare = router._classify_request(_openai_request("[sol] Inspect and report."), config=shipped)
        stated = router._classify_request(_openai_request("[sol:high] Inspect and report."), config=shipped)
        self.assertEqual((bare.tier, bare.effort), ("sol", "medium"))
        self.assertEqual((stated.tier, stated.effort), ("sol", "high"))

    def test_f08_host_tool_call_schema_accepts_advertised_and_legacy_shapes(self):
        advertised, advertised_error = normalize_tool_call_entries(
            {"calls": [{"name": "delegate_task", "arguments": {"goal": "x"}}]})
        legacy, legacy_error = normalize_tool_call_entries(
            {"name": "delegate_task", "arguments": {"goal": "x"}})
        self.assertIsNone(advertised_error)
        self.assertIsNone(legacy_error)
        self.assertEqual(advertised, legacy)

    def test_f08_triage_thinking_and_nonthinking_paths_are_discriminating(self):
        advisory_cases = (
            ("claude-opus-5-5", True),
            ("claude-opus-5-5", False),
            ("claude-sonnet-5-5", True),
            ("claude-sonnet-5-5", False),
            ("claude-haiku-4-5", True),
        )
        for model, thinking in advisory_cases:
            with self.subTest(model=model, thinking=thinking):
                routed = triage.force(_anthropic_request(model, thinking=thinking), self.cfg, anthropic=True)
                self.assertNotIn("tool_choice", routed)
                self.assertEqual(len(routed["tools"]), 2)
                bridge = next(tool for tool in routed["tools"] if tool["name"] == "mcp__tool_call")
                parameters = bridge["input_schema"]
                self.assertEqual(parameters["required"], ["name", "arguments"],
                                 "P01 advisory bridge behavior must remain unchanged")
                advisory_name = parameters["properties"]["calls"]["items"]["properties"]["name"]
                self.assertNotIn("const", advisory_name)
                self.assertFalse(isinstance(advisory_name.get("enum"), list)
                                 and len(advisory_name["enum"]) == 1)
                self.assertIn("[ROUTER TRIAGE]", routed["messages"][-1]["content"][-1]["text"])
        forced = triage.force(_anthropic_request("claude-haiku-4-5"), self.cfg, anthropic=True)
        self.assertEqual(forced["tool_choice"], {"type": "tool", "name": "mcp__tool_call"})
        self.assertEqual(len(forced["tools"]), 1)
        haiku_bridge = forced["tools"][0]["input_schema"]
        haiku_name = haiku_bridge["properties"]["calls"]["items"]["properties"]["name"]
        self.assertTrue(
            haiku_name.get("const") == triage.TOOL_NAME
            or haiku_name.get("enum") == [triage.TOOL_NAME],
            "P01: forced Anthropic haiku bridge must pin calls[*].name to triage_task",
        )
        codex = triage.force(_openai_request(), self.cfg, anthropic=False)
        self.assertEqual(codex["tool_choice"], "required")

    def test_f08_conductor_preflight_reaches_the_claude_five_force_choice_decision(self):
        cfg = _cfg(self.directory.name)
        cfg["orchestration"] = {"enabled": True, "max_tasks": 3}
        decision = router.RouteDecision("terra", "gpt-5.6-terra", "test")
        preflight = _anthropic_request("claude-sonnet-5-5")
        kwargs = {"request": preflight, "turn_id": "f08-preflight", "api_call_count": 1}
        sentinel = {"preflight": "routed"}
        with patch.object(router, "_orchestration_skip_reason", return_value=None), \
             patch.object(router, "_orchestration_forced_event", return_value=False), \
             patch.object(router, "_orchestration_event"), \
             patch.object(router, "_claude_first_choice", return_value=("", "", "")), \
             patch.object(router, "_prepare_orchestration_delegation", return_value=sentinel) as prepare:
            routed = router._force_terra_supervisor_preflight(kwargs, cfg, decision)
        self.assertIs(routed, sentinel)
        self.assertFalse(prepare.call_args.kwargs["force_tools"])
        self.assertEqual(prepare.call_args.args[0]["model"], "claude-sonnet-5-5")

    def test_f08_triage_on_off_and_host_depth_are_distinct(self):
        root = SimpleNamespace(session_id="root", _delegate_depth=0)
        off = _cfg(self.directory.name, triage_enabled=False)
        with patch.object(triage, "_caller", return_value=root):
            self.assertEqual(triage.spawn_block("delegate_task", {"tasks": [{"goal": "x"}] * 5}, off,
                                                {"turn_id": "root:off"}), "")
            self.assertIn("asks for 5 more", triage.spawn_block(
                "delegate_task", {"tasks": [{"goal": "x"}] * 5}, self.cfg, {"turn_id": "root:on"},
            ))
        cfg = {"orchestration": {"enabled": True, "conductor": "terra"},
               "callable": {"terra": True}, "claude_delegation": {"enabled": False}}
        for depth, expected in ((1, "unsupported"), (2, "supported")):
            with self.subTest(depth=depth), host_delegation(depth=depth):
                runtime.clear_cache()
                snap = runtime.snapshot(delegate_task_request(), cfg)
                self.assertEqual(snap.max_spawn_depth.value, depth)
                self.assertEqual(runtime.resolve_topology(snap, "nested_conductor").status, expected)


if __name__ == "__main__":
    unittest.main()
