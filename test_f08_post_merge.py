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
from model_router import execution_adapters, triage
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
        "orchestration": {"enabled": True},
        "shadow": {"enabled": False},
    }


def _openai_request():
    return {
        "model": "gpt-5.6-terra",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": TASK}]}],
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

    def _events(self):
        path = Path(self.cfg["triage"]["path"])
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def _assert_p01_advertised_calls_envelope(self):
        routed = triage.force(_openai_request(), self.cfg, anthropic=False)
        parameters = routed["tools"][0]["function"]["parameters"]
        self.assertIn("calls", parameters["properties"],
                      "P01: triage replaced the advertised calls[] bridge envelope")
        self.assertEqual(parameters["required"], ["calls"],
                         "P01: triage replaced the advertised calls[] bridge envelope")

    @unittest.expectedFailure  # F08 P01: owner decision pending
    def test_p01_advertised_calls_envelope_remains_pinned(self):
        """The installed bridge advertises calls[]; triage must not replace it with legacy fields."""
        self._assert_p01_advertised_calls_envelope()

    def test_p01_red_evidence_is_an_assertion(self):
        with self.assertRaisesRegex(AssertionError, "P01: triage replaced"):
            self._assert_p01_advertised_calls_envelope()

    def test_p02_retained_attempt_replays_without_a_second_raw_dispatch(self):
        calls = []
        book = execution_adapters.ReservationBook()
        key = ("session", 0, "turn", "call")

        def raw_dispatch():
            calls.append("raw")
            return {"delegation_id": "child-1"}

        first = execution_adapters.dispatch_legacy(raw_dispatch, transport="hermes_codex",
                                                   scope="f08", attempt_key=key,
                                                   reservations=book, fingerprint="same")
        replay = execution_adapters.dispatch_legacy(raw_dispatch, transport="hermes_codex",
                                                    scope="f08", attempt_key=key,
                                                    reservations=book, fingerprint="same")
        self.assertEqual(first, replay)
        self.assertEqual(calls, ["raw"], "P02: replay reached raw dispatch more than once")

    def test_p03_triage_admission_records_an_attempt_not_a_launch(self):
        root = SimpleNamespace(session_id="root", _delegate_depth=0)
        with patch.object(triage, "_caller", return_value=root):
            refusal = triage.spawn_block("delegate_task", {"goal": "later-refused"}, self.cfg,
                                         {"turn_id": "root:turn"})
        self.assertEqual(refusal, "")
        event = self._events()[-1]
        self.assertEqual(event["event"], "spawn_admitted")
        self.assertNotIn("launched", event, "P03: admission is not a launch metric")

    def _assert_p04_advisory_is_not_forced(self):
        decision = router.RouteDecision("terra", "gpt-5.6-terra", "test")
        kwargs = {"request": _anthropic_request("claude-opus-5-5"), "api_call_count": 1,
                  "turn_id": "f08-p04", "platform": "cli", "model": "claude-opus-5-5"}
        with patch.object(router, "classify_request", return_value=decision):
            routed = router._triage_request(kwargs, self.cfg, decision)
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
            refusal = triage.spawn_block("delegate_task", {"goal": "late-child"}, self.cfg,
                                         {"turn_id": "child:turn"})
        self.assertIn("4 of 4", refusal,
                      "P05: late child was charged to the newest root turn")

    @unittest.expectedFailure  # F08 P05: owner decision pending
    def test_p05_late_child_keeps_its_originating_root_turn(self):
        self._assert_p05_late_child_keeps_origin()

    def test_p05_red_evidence_is_an_assertion(self):
        with self.assertRaisesRegex(AssertionError, "P05: late child"):
            self._assert_p05_late_child_keeps_origin()

    def test_p06_triage_guidance_is_not_structured_execution_authority(self):
        response = json.loads(triage.handle_triage({"decision": "solo", "rationale": "small"}))
        adapters = execution_adapters.legacy_adapters(delegate_task=lambda *_: None,
                                                       delegate_claude=lambda *_: None,
                                                       claude_bridge=lambda *_: None)
        capabilities = adapters["hermes_codex"].capabilities(object())
        self.assertIn("Do the work yourself", response["next"])
        self.assertEqual(capabilities.capabilities, ())

    def test_p07_code_defaults_retire_explicit_effort_keys(self):
        self.assertNotIn("explicit_sol", router._DEFAULT_CONFIG["effort"])
        self.assertIn("sol", router._DEFAULT_CONFIG["effort"])

    def test_f08_host_tool_call_schema_accepts_advertised_and_legacy_shapes(self):
        advertised, advertised_error = normalize_tool_call_entries(
            {"calls": [{"name": "delegate_task", "arguments": {"goal": "x"}}]})
        legacy, legacy_error = normalize_tool_call_entries(
            {"name": "delegate_task", "arguments": {"goal": "x"}})
        self.assertIsNone(advertised_error)
        self.assertIsNone(legacy_error)
        self.assertEqual(advertised, legacy)

    def test_f08_triage_thinking_and_nonthinking_paths(self):
        for model, thinking, forced in (("claude-opus-5-5", True, False),
                                        ("claude-sonnet-5-5", False, False),
                                        ("claude-haiku-4-5", False, True)):
            with self.subTest(model=model, thinking=thinking):
                routed = triage.force(_anthropic_request(model, thinking=thinking), self.cfg, anthropic=True)
                self.assertEqual("tool_choice" in routed, forced)
        codex = triage.force(_openai_request(), self.cfg, anthropic=False)
        self.assertEqual(codex["tool_choice"], "required")

    def test_f08_triage_on_and_off_keep_distinct_budget_behavior(self):
        root = SimpleNamespace(session_id="root", _delegate_depth=0)
        off = _cfg(self.directory.name, triage_enabled=False)
        with patch.object(triage, "_caller", return_value=root):
            self.assertEqual(triage.spawn_block("delegate_task", {"goal": "x"}, off,
                                                {"turn_id": "root:off"}), "")
            self.assertEqual(triage.spawn_block("delegate_task", {"tasks": [{"goal": "x"}] * 5}, self.cfg,
                                                {"turn_id": "root:on"}),
                             "Worker budget for this user turn: 0 of 4 workers already started (sub-workers included); this call asks for 5 more. Do the remaining work yourself or wait for the running workers. Nothing was spawned.")


if __name__ == "__main__":
    unittest.main()
