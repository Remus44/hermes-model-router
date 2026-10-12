"""F08 post-merge evidence against the installed Hermes host.

Defects reproduced here remain expected failures pending an owner-scoped repair.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import model_router as router
from model_router import execution_adapters, runtime_capabilities as runtime, triage, worker_admission
from model_router.host_delegation_fixtures import delegate_task_request, host_delegation
from tools.tool_search import bridge_tool_schemas
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
        "callable": {"terra": True, "sol": True},
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
        routed = triage.force(_openai_request(), self.cfg, anthropic=False)
        parameters = routed["tools"][0]["function"]["parameters"]
        calls = parameters["properties"].get("calls")
        self.assertIsInstance(calls, dict, "P01: triage replaced the advertised calls[] bridge envelope")
        self.assertEqual(parameters["required"], ["calls"],
                         "P01: triage replaced the advertised calls[] bridge envelope")
        self.assertEqual(calls["items"]["properties"]["name"].get("enum"), [triage.TOOL_NAME],
                         "P01: calls[*].name is not pinned to triage_task")

    @unittest.expectedFailure  # F08 P01: owner decision pending
    def test_p01_advertised_calls_envelope_remains_pinned(self):
        """The installed bridge must pin each advertised calls[].name to triage_task."""
        self._assert_p01_advertised_calls_envelope()

    def test_p01_red_evidence_is_an_assertion(self):
        with self.assertRaisesRegex(AssertionError, "P01: (triage replaced|calls\\[\\*\\].name)"):
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

    def test_p02_retained_replay_differs_between_normal_executor_and_registry_entrypoints(self):
        """F05's outer claim replays before normal pre-hooks, but not registry pre-hooks."""
        root = SimpleNamespace(session_id="p02-root", _delegate_depth=0)
        raw_result = json.dumps({"delegation_id": "child-1"})

        def run_normal_executor():
            book, raw_calls, pre_hooks = execution_adapters.ReservationBook(), [], []

            def downstream(sent):
                pre_hooks.append(router.on_pre_tool_call(
                    "delegate_task", sent, session_id="p02-normal", turn_id="p02:normal",
                ))
                self.assertIsNone(pre_hooks[-1])
                raw_calls.append(sent)
                return raw_result

            with patch.object(execution_adapters, "RESERVATIONS", book), \
                 patch.object(router, "_load_config", return_value=self.cfg), \
                 patch.object(router, "_delegation_targets_detail", return_value={}), \
                 patch.object(worker_admission, "delegate_task_route", return_value=("openai-codex", "gpt-5.6-terra")), \
                 patch.object(triage, "_caller", return_value=root):
                first = execution_adapters.guard_legacy_tool_execution(
                    tool_name="delegate_task", args={"goal": "normal"}, next_call=downstream,
                    session_id="p02-normal", turn_id="p02:normal", tool_call_id="same",
                )
                replay = execution_adapters.guard_legacy_tool_execution(
                    tool_name="delegate_task", args={"goal": "normal"}, next_call=downstream,
                    session_id="p02-normal", turn_id="p02:normal", tool_call_id="same",
                )
            return first, replay, raw_calls, pre_hooks

        def run_direct_registry():
            book, raw_calls, pre_hooks = execution_adapters.ReservationBook(), [], []

            def registry_call():
                blocked = router.on_pre_tool_call(
                    "delegate_task", {"goal": "registry"}, session_id="p02-registry", turn_id="p02:registry",
                )
                pre_hooks.append(blocked)
                self.assertIsNone(blocked)
                return execution_adapters.guard_legacy_tool_execution(
                    tool_name="delegate_task", args={"goal": "registry"},
                    next_call=lambda sent: raw_calls.append(sent) or raw_result,
                    session_id="p02-registry", turn_id="p02:registry", tool_call_id="same",
                )

            with patch.object(execution_adapters, "RESERVATIONS", book), \
                 patch.object(router, "_load_config", return_value=self.cfg), \
                 patch.object(router, "_delegation_targets_detail", return_value={}), \
                 patch.object(worker_admission, "delegate_task_route", return_value=("openai-codex", "gpt-5.6-terra")), \
                 patch.object(triage, "_caller", return_value=root):
                first, replay = registry_call(), registry_call()
            return first, replay, raw_calls, pre_hooks

        normal = run_normal_executor()
        registry = run_direct_registry()
        self.assertEqual(normal[:2], (raw_result, raw_result))
        self.assertEqual(registry[:2], (raw_result, raw_result))
        self.assertEqual(len(normal[2]), 1, "P02 normal executor replay reached raw dispatch twice")
        self.assertEqual(len(registry[2]), 1, "P02 registry replay reached raw dispatch twice")
        self.assertEqual(len(normal[3]), 1, "P02 normal executor charged a sealed replay")
        self.assertEqual(len(registry[3]), 2, "P02 registry path did not charge before sealed replay")

    def test_p03_downstream_conductor_refusal_retains_the_admitted_worker_allowance(self):
        cfg = _cfg(self.directory.name)
        cfg["orchestration"] = {
            "enabled": True,
            "min_goal_chars": 20,
            "path": str(Path(self.directory.name) / "orchestration.jsonl"),
        }
        root = SimpleNamespace(session_id="p03-root", _delegate_depth=0)
        args = {"tasks": [{
            "goal": "tiny",
            "context": "You are the terra planning conductor.",
        }]}
        with patch.object(router, "_load_config", return_value=cfg), \
             patch.object(router, "_delegation_targets_detail", return_value={}), \
             patch.object(triage, "_caller", return_value=root):
            refusal = router.on_pre_tool_call("delegate_task", args, session_id="p03-root", turn_id="p03:turn")
        self.assertEqual(refusal["action"], "block")
        self.assertIn("No conductor for this turn", refusal["message"])
        events = self._events(cfg)
        self.assertEqual(events[-1]["event"], "spawn_admitted")
        self.assertEqual(triage._TURN_WORKERS, {"p03:turn": 1},
                         "P03: downstream refusal unexpectedly restored the admitted allowance")

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

    @unittest.expectedFailure  # F08 P04: owner decision pending
    def test_p04_advisory_claude_five_rewrite_is_not_telemetry_forced(self):
        self._assert_p04_advisory_is_not_forced()

    def test_p04_red_evidence_is_an_assertion(self):
        with self.assertRaisesRegex(AssertionError, "P04: advisory rewrite"):
            self._assert_p04_advisory_is_not_forced()

    def _assert_p05_late_child_keeps_origin(self):
        root = SimpleNamespace(session_id="root", _delegate_depth=0)
        child = SimpleNamespace(session_id="child", _delegate_depth=1, _parent_session_id="root")
        with patch.object(triage, "_caller", return_value=root):
            self.assertEqual(triage.spawn_block("delegate_task", {"tasks": [{"goal": "a"}] * 4}, self.cfg,
                                                {"turn_id": "root:old"}), "")
            self.assertEqual(triage.spawn_block("delegate_task", {"goal": "new"}, self.cfg,
                                                {"turn_id": "root:new"}), "")
        with patch.object(triage, "_caller", return_value=child):
            triage.spawn_block("delegate_task", {"goal": "late-child"}, self.cfg, {"turn_id": "child:turn"})
        self.assertEqual(self._events()[-1]["reason"], "turn_worker_limit",
                         "P05: late child was charged to the newest root turn")

    @unittest.expectedFailure  # F08 P05: owner decision pending
    def test_p05_late_child_keeps_its_originating_root_turn(self):
        self._assert_p05_late_child_keeps_origin()

    def test_p05_red_evidence_is_an_assertion(self):
        with self.assertRaisesRegex(AssertionError, "P05: late child"):
            self._assert_p05_late_child_keeps_origin()

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
            ("claude-haiku-4-5", True),
        )
        for model, thinking in advisory_cases:
            with self.subTest(model=model, thinking=thinking):
                routed = triage.force(_anthropic_request(model, thinking=thinking), self.cfg, anthropic=True)
                self.assertNotIn("tool_choice", routed)
                self.assertEqual(len(routed["tools"]), 2)
                self.assertIn("[ROUTER TRIAGE]", routed["messages"][-1]["content"][-1]["text"])
        forced = triage.force(_anthropic_request("claude-haiku-4-5"), self.cfg, anthropic=True)
        self.assertEqual(forced["tool_choice"], {"type": "tool", "name": "mcp__tool_call"})
        self.assertEqual(len(forced["tools"]), 1)
        codex = triage.force(_openai_request(), self.cfg, anthropic=False)
        self.assertEqual(codex["tool_choice"], "required")

    def test_f08_conductor_preflight_keeps_claude_five_toolset_advisory(self):
        routed = router._prepare_orchestration_delegation(
            _anthropic_request("claude-opus-5-5"), "f08-plan", 1, cfg=self.cfg,
            force_tools=False, parent_tier="terra",
        )
        self.assertNotIn("tool_choice", routed)
        self.assertEqual(len(routed["tools"]), 2)
        self.assertIn("[INTERNAL ORCHESTRATOR PREFLIGHT]", routed["messages"][-1]["content"][-1]["text"])

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
