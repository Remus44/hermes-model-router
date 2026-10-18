"""The parent triages its own user turn; the router enforces the worker budget.

Observed 2026-10-05: with the forced conductor switched off (2026-09-30, after a
pricing-copy fix fanned out into a planner plus two workers) the parent
delegated almost nothing -- a fix, a dev release and a test plan ran as 39
Terra calls. Triage makes the parent decide on its first call, with the whole
conversation in view, and caps what follows: four workers per user turn,
sub-workers included, one sub-worker per worker, none below that.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import model_router as router
from model_router import execution_adapters, on_pre_tool_call, route_llm_request, triage

MODELS = {"luna": "gpt-6-luna", "terra": "gpt-5.6-terra", "sol": "gpt-6-sol", "grok": "grok-4.7"}
PROVIDERS = {"luna": "openai-codex", "terra": "openai-codex", "sol": "openai-codex", "grok": "xai-oauth"}
TARGETS = {"grok": {"provider": "xai-oauth", "model": "grok-4.7"},
           "terra": {"provider": "openai-codex", "model": "gpt-5.6-terra"}}

TASK = "javitsd ki a foglalasi naptar dupla felugro ablakat, rakd ki developmentre es adj tesztelesi lepessort"

DELEGATE = {"type": "function", "name": "delegate_task",
            "parameters": {"type": "object", "properties": {"goal": {"type": "string"}}}}
BRIDGE = {"type": "function", "name": "tool_call",
          "parameters": {"type": "object", "properties": {
              "name": {"type": "string"}, "arguments": {"type": "object"}},
              "required": ["name", "arguments"]}}
TERMINAL = {"type": "function", "name": "terminal",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}


def _cfg(temp_dir, triage_on=True):
    return {
        "enabled": True, "provider": "openai-codex",
        "models": MODELS, "tier_providers": PROVIDERS,
        "callable": {**{tier: True for tier in MODELS},
                     "haiku": True, "sonnet5": True, "opus5": True},
        "default_model": "terra",
        "effort": {"terra": "medium", "sol": "medium", "luna": "low", "grok": "medium"},
        "session_policy": {"pin_root_parent": True},
        "orchestration": {"enabled": True, "min_chars": 10, "max_tasks": 2,
                          "path": str(Path(temp_dir) / "orchestration.jsonl")},
        "triage": {"enabled": triage_on, "max_workers_per_turn": 4, "max_children_per_worker": 1,
                   "path": str(Path(temp_dir) / "triage.jsonl")},
        "shadow": {"enabled": False},
        "fallbacks": {"grok": "terra", "sol": "terra"},
    }


def _request(model, text, tools=(DELEGATE, TERMINAL, BRIDGE), prior=False):
    items = []
    if prior:
        items += [{"role": "user", "content": [{"type": "input_text", "text": "mi a hiba a naptarban?"}]},
                  {"role": "assistant", "content": [{"type": "output_text", "text": "A dupla submit."}]}]
    items.append({"role": "user", "content": [{"type": "input_text", "text": text}]})
    return {"model": model, "input": items, "tools": [dict(tool) for tool in tools]}


class _Patched(unittest.TestCase):
    triage_on = True

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.cfg = _cfg(self._dir.name, self.triage_on)
        for target, value in (
            ("_delegation_targets_detail", TARGETS),
            ("_hermes_delegation_target_names", tuple(sorted(TARGETS))),
            ("_host_delegation_limits", {"conductor_available": True}),
        ):
            patcher = patch.object(router, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for target in ("_load_config",):
            patcher = patch.object(router, target, side_effect=lambda: self.cfg)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(router, "_log_decision")
        patcher.start()
        self.addCleanup(patcher.stop)
        triage.reset_state()
        self.addCleanup(triage.reset_state)

    def route(self, text=TASK, tier="terra", api_call_count=1, platform="cli", **request_kw):
        model = MODELS[tier]
        return route_llm_request(
            request=_request(model, text, **request_kw), model=model, provider=PROVIDERS[tier],
            api_call_count=api_call_count, turn_id="sess:sess:turn1", platform=platform)

    def events(self):
        path = Path(self.cfg["triage"]["path"])
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TriageIsForcedTests(_Patched):
    def test_an_actionable_turn_must_triage_through_the_bridge(self):
        request = self.route()["request"]
        self.assertEqual([t["name"] for t in request["tools"]], ["tool_call"])
        self.assertEqual(request["tool_choice"], "required")
        parameters = request["tools"][0]["parameters"]
        self.assertEqual(parameters["required"], ["calls"])
        calls = parameters["properties"]["calls"]
        self.assertEqual((calls["minItems"], calls["maxItems"]), (1, 1))
        entry = calls["items"]
        self.assertEqual(entry["required"], ["name", "arguments"])
        self.assertEqual(entry["properties"]["name"]["enum"], ["triage_task"])
        self.assertEqual(entry["properties"]["arguments"]["properties"]["decision"]["enum"], ["solo", "delegate"])
        text = request["input"][-1]["content"][-1]["text"]
        self.assertIn("[ROUTER TRIAGE]", text)
        self.assertIn("at most 4 workers", text)

    def test_a_directly_listed_triage_tool_is_forced_by_itself(self):
        direct = {"type": "function", "name": "triage_task", "parameters": triage.SCHEMA_PARAMETERS}
        request = self.route(tools=(DELEGATE, TERMINAL, direct))["request"]
        self.assertEqual([t["name"] for t in request["tools"]], ["triage_task"])

    def test_triage_replaces_the_forced_conductor(self):
        request = self.route()["request"]
        self.assertNotIn("role", json.dumps(request["tools"]))
        self.assertFalse(Path(self.cfg["orchestration"]["path"]).exists())

    def test_a_short_go_ahead_after_a_discussion_is_triaged(self):
        """'csinald meg' is exactly the prompt that opened the 39-call turn."""
        request = self.route(text="rendben csinald meg !", prior=True)["request"]
        self.assertEqual([t["name"] for t in request["tools"]], ["tool_call"])

    def test_an_imperative_labelled_chat_is_still_triaged(self):
        """Live 2026-10-05: the classifier called this prompt brief conversation."""
        text = "rendben inditsd el ennek a javitasat majd rakjad ki developmentre"
        with patch.object(router, "classify_request",
                          return_value=router.RouteDecision("luna", "gpt-6-luna", "brief", kind="chat")):
            routed = router._triage_request(
                {"request": _request(MODELS["terra"], text), "api_call_count": 1,
                 "turn_id": "sess:sess:turn1", "platform": "cli", "provider": "openai-codex"},
                self.cfg, router.RouteDecision("terra", MODELS["terra"], "pinned"))
        self.assertEqual([t["name"] for t in routed["tools"]], ["tool_call"])

    def test_a_chat_label_without_an_imperative_skips(self):
        with patch.object(router, "classify_request",
                          return_value=router.RouteDecision("luna", "gpt-6-luna", "brief", kind="chat")):
            routed = router._triage_request(
                {"request": _request(MODELS["terra"], "koszonom szepen, ez nagyon jo lett igy most mar"),
                 "api_call_count": 1, "turn_id": "sess:sess:turn1", "platform": "cli",
                 "provider": "openai-codex"},
                self.cfg, router.RouteDecision("terra", MODELS["terra"], "pinned"))
        self.assertIsNone(routed)
        self.assertEqual(self.events()[-1]["reason"], "chat")

    def test_the_decision_is_logged(self):
        self.route()
        forced = [e for e in self.events() if e["event"] == "triage_forced"]
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0]["parent_model"], "terra")

    def test_a_grok_parent_triages_too(self):
        request = self.route(tier="grok")["request"]
        self.assertEqual(request["model"], "grok-4.7")
        self.assertEqual([t["name"] for t in request["tools"]], ["tool_call"])
    def test_bridge_instruction_names_calls_not_the_legacy_singular_shape(self):
        self.assertIn("`calls`", triage.instruction(self.cfg, via_bridge=True))
        self.assertNotIn("with name \"triage_task\" and the triage as `arguments`", triage.instruction(self.cfg, via_bridge=True))


class TriageIsSkippedTests(_Patched):
    def assert_not_forced(self, routed):
        request = (routed or {}).get("request") or {}
        names = [t["name"] for t in request.get("tools") or []]
        self.assertNotEqual(names, ["tool_call"])
        self.assertNotIn("[ROUTER TRIAGE]", json.dumps(request))

    def test_a_question_is_not_triaged(self):
        self.assert_not_forced(self.route(text="mi a kovetkezo kritikus ?"))
        self.assertEqual(self.events()[-1]["reason"], "conversational")

    def test_a_short_remark_is_not_triaged(self):
        self.assert_not_forced(self.route(text="ment a level"))

    def test_a_later_call_of_the_turn_is_not_triaged(self):
        self.assert_not_forced(self.route(api_call_count=2))
        self.assertEqual(self.events(), [])

    def test_a_worker_is_not_triaged(self):
        self.assert_not_forced(self.route(platform="subagent"))

    def test_a_memory_review_is_not_triaged(self):
        self.assert_not_forced(self.route(
            text="Review the conversation above and consider saving to memory if appropriate. Focus on facts"))

    def test_a_session_without_delegate_task_is_not_triaged(self):
        self.assert_not_forced(self.route(tools=(TERMINAL, BRIDGE)))

    def test_an_anthropic_parent_with_thinking_is_asked_not_forced(self):
        tools = [{"name": "mcp__delegate_task", "input_schema": {"type": "object", "properties": {}}},
                 {"name": "mcp__tool_call", "input_schema": {"type": "object", "properties": {
                     "name": {"type": "string"}, "arguments": {"type": "object"}}}}]
        request = {"model": "claude-opus-5-5", "thinking": {"type": "adaptive"}, "tools": tools,
                   "messages": [{"role": "user", "content": [{"type": "text", "text": TASK}]}]}
        routed = triage.force(request, self.cfg, anthropic=True)
        self.assertNotIn("tool_choice", routed)
        self.assertEqual(len(routed["tools"]), 2)
        self.assertEqual(routed["tools"][1]["input_schema"]["properties"]["name"]["enum"], ["triage_task"])
        self.assertIn("[ROUTER TRIAGE]", routed["messages"][-1]["content"][-1]["text"])

    def test_a_claude_five_parent_without_thinking_is_asked_not_forced(self):
        """claude-opus-5-5 / claude-sonnet-5-5 400 a forced tool_choice even without
        thinking ("not supported for this model"); Hermes then fell back to Terra."""
        for model in ("claude-opus-5-5", "claude-sonnet-5-5"):
            with self.subTest(model=model):
                tools = [{"name": "mcp__delegate_task", "input_schema": {"type": "object", "properties": {}}},
                         {"name": "mcp__tool_call", "input_schema": {"type": "object", "properties": {}}}]
                request = {"model": model, "tools": tools,
                           "messages": [{"role": "user", "content": [{"type": "text", "text": TASK}]}]}
                routed = triage.force(request, self.cfg, anthropic=True)
                self.assertNotIn("tool_choice", routed)
                self.assertEqual(len(routed["tools"]), 2)
                self.assertIn("[ROUTER TRIAGE]", routed["messages"][-1]["content"][-1]["text"])

    def test_an_anthropic_parent_without_thinking_is_forced_by_name(self):
        tools = [{"name": "mcp__tool_call", "input_schema": {"type": "object", "properties": {}}}]
        request = {"model": "claude-haiku-4-5", "tools": tools,
                   "messages": [{"role": "user", "content": [{"type": "text", "text": TASK}]}]}
        routed = triage.force(request, self.cfg, anthropic=True)
        self.assertEqual(routed["tool_choice"], {"type": "tool", "name": "mcp__tool_call"})


class TriageOffKeepsTheConductorTests(_Patched):
    triage_on = False

    def test_the_forced_conductor_still_runs_when_triage_is_off(self):
        request = self.route()["request"]
        self.assertEqual([t["name"] for t in request["tools"]], ["delegate_task"])
        self.assertIn("orchestrator", json.dumps(request["tools"]))


class TriageToolTests(_Patched):
    def test_delegate_tells_the_parent_to_dispatch(self):
        with patch.object(triage, "_caller", return_value=SimpleNamespace(session_id="s1", _relay_pending_turn_id="t1")):
            result = json.loads(triage.handle_triage({
                "decision": "delegate", "rationale": "two independent parts",
                "subtasks": [{"goal": "fix", "route": "grok"}, {"goal": "map", "route": "sonnet5"}]}))
        self.assertEqual(result["recorded"], "delegate")
        self.assertIn("Dispatch the 2 subtask(s)", result["next"])
        logged = [e for e in self.events() if e["event"] == "triage_decision"][0]
        self.assertEqual(logged["turn_id"], "t1")
        self.assertEqual([s["route"] for s in logged["subtasks"]], ["grok", "sonnet5"])

    def test_solo_tells_the_parent_to_work(self):
        result = json.loads(triage.handle_triage({"decision": "solo", "rationale": "one file"}))
        self.assertEqual(result["recorded"], "solo")
        self.assertIn("Do the work yourself", result["next"])

    def test_garbage_never_raises(self):
        self.assertEqual(json.loads(triage.handle_triage(None))["recorded"], "solo")


class WorkerBudgetTests(_Patched):
    root = SimpleNamespace(session_id="root", _delegate_depth=0)

    def spawn(self, agent, turn_id="root:root:turn1", tasks=1, tool="delegate_task"):
        args = {"tasks": [{"goal": f"g{i}"} for i in range(tasks)]} if tasks > 1 else {"goal": "g"}
        with patch.object(triage, "_caller", return_value=agent):
            return on_pre_tool_call(tool_name=tool, args=args, turn_id=turn_id, session_id=agent.session_id)

    def worker(self, name):
        return SimpleNamespace(session_id=name, _delegate_depth=1, _parent_session_id="root")

    def test_the_root_may_start_four_workers_per_turn(self):
        self.assertIsNone(self.spawn(self.root, tasks=2))
        self.assertIsNone(self.spawn(self.root))
        self.assertIsNone(self.spawn(self.root, tool="delegate_claude"))
        blocked = self.spawn(self.root)
        self.assertEqual(blocked["action"], "block")
        self.assertIn("4 of 4 admitted spawn attempts", blocked["message"])

    def test_a_batch_over_the_budget_is_refused_whole(self):
        self.assertIsNone(self.spawn(self.root, tasks=3))
        self.assertIn("asks for 2 more", self.spawn(self.root, tasks=2)["message"])
        self.assertIsNone(self.spawn(self.root))

    def test_the_next_user_turn_has_a_fresh_budget(self):
        self.spawn(self.root, tasks=4)
        self.assertIsNone(self.spawn(self.root, turn_id="root:root:turn2"))

    def test_a_worker_may_start_one_sub_worker(self):
        self.spawn(self.root)
        worker = self.worker("w1")
        self.assertIsNone(self.spawn(worker, turn_id="w1:sa-0-x:t"))
        blocked = self.spawn(worker, turn_id="w1:sa-0-x:t")
        self.assertIn("at most 1 sub-worker", blocked["message"])

    def test_a_worker_may_not_start_two_at_once(self):
        self.spawn(self.root)
        self.assertIn("at most 1 sub-worker", self.spawn(self.worker("w1"), tasks=2)["message"])

    def test_sub_workers_count_toward_the_turn(self):
        self.spawn(self.root, tasks=3)
        self.assertIsNone(self.spawn(self.worker("w1")))
        self.assertIn("4 of 4", self.spawn(self.worker("w2"))["message"])

    def start_worker(self, name, turn_id):
        router.on_subagent_start(parent_session_id="root", parent_turn_id=turn_id,
                                 child_session_id=name, child_subagent_id=f"sa-{name}",
                                 child_role="leaf", child_goal="late")

    def test_a_late_worker_is_charged_to_the_turn_it_started_from(self):
        self.spawn(self.root, tasks=4)
        self.spawn(self.root, turn_id="root:root:turn2")
        self.start_worker("w1", "root:root:turn1")
        blocked = self.spawn(self.worker("w1"), turn_id="w1:sa-0-x:t")
        self.assertIn("4 of 4", blocked["message"])
        event = self.events()[-1]
        self.assertEqual((event["origin_source"], event["root_turn"]), ("child_link", "root:root:turn1"))

    def test_a_worker_of_the_newest_turn_keeps_its_own_allowance(self):
        self.spawn(self.root, tasks=4)
        self.spawn(self.root, turn_id="root:root:turn2")
        self.start_worker("w2", "root:root:turn2")
        self.assertIsNone(self.spawn(self.worker("w2"), turn_id="w2:sa-0-x:t"))
        self.assertEqual(triage._TURN_WORKERS, {"root:root:turn1": 4, "root:root:turn2": 2})

    def test_an_unlinked_worker_falls_back_to_the_latest_turn_and_says_so(self):
        self.spawn(self.root)
        self.assertIsNone(self.spawn(self.worker("w1"), turn_id="w1:sa-0-x:t"))
        event = self.events()[-1]
        self.assertEqual((event["origin_source"], event["root_turn"]),
                         ("latest_session_pointer", "root:root:turn1"))

    def test_a_sub_worker_never_delegates(self):
        leaf = SimpleNamespace(session_id="leaf", _delegate_depth=2, _parent_session_id="w1")
        self.assertIn("cannot delegate further", self.spawn(leaf)["message"])

    def test_list_and_stop_are_not_spawns(self):
        self.spawn(self.root, tasks=4)
        with patch.object(triage, "_caller", return_value=self.root):
            self.assertIsNone(on_pre_tool_call(tool_name="delegate_task", args={"action": "list"},
                                               turn_id="root:root:turn1"))

    def test_spawns_are_logged(self):
        self.spawn(self.root, tasks=4)
        self.spawn(self.root)
        events = [e["event"] for e in self.events()]
        self.assertEqual(events, ["spawn_admitted", "spawn_blocked"])


class RetainedAdmissionTests(_Patched):
    root = SimpleNamespace(session_id="retained-root", _delegate_depth=0)

    def spawn(self, tool, args, *, tool_call_id=None):
        with patch.object(triage, "_caller", return_value=self.root):
            return on_pre_tool_call(tool_name=tool, args=args, session_id="retained-root",
                                    turn_id="retained:turn", tool_call_id=tool_call_id)

    def test_stable_ids_are_one_admitted_attempt_but_missing_ids_are_distinct(self):
        args = {"goal": "g", "tier": "haiku"}
        self.assertIsNone(self.spawn("delegate_claude", args, tool_call_id="same"))
        self.assertIsNone(self.spawn("delegate_claude", args, tool_call_id="same"))
        self.assertEqual(triage._TURN_WORKERS, {"retained:turn": 1})
        changed = self.spawn("delegate_claude", {"goal": "changed", "tier": "haiku"}, tool_call_id="same")
        transport = self.spawn("delegate_task", {"goal": "g"}, tool_call_id="same")
        self.assertEqual(changed["action"], "block")
        self.assertEqual(transport["action"], "block")
        self.assertIn("identity", changed["message"])
        self.assertIn("identity", transport["message"])

        triage.reset_state()
        self.assertIsNone(self.spawn("delegate_claude", args))
        self.assertIsNone(self.spawn("delegate_claude", args))
        self.assertEqual(triage._TURN_WORKERS, {"retained:turn": 2})

    def test_replay_exemption_expires_with_its_f05_generation(self):
        args = {"goal": "g", "tier": "haiku"}
        book = execution_adapters.ReservationBook()
        invocation = execution_adapters.stable_legacy_invocation(
            "delegate_claude", args, session_id="retained-root", turn_id="retained:turn", tool_call_id="expiry",
        )
        with patch.object(execution_adapters, "RESERVATIONS", book):
            self.assertIsNone(self.spawn("delegate_claude", args, tool_call_id="expiry"))
            fresh, claim = book._begin(invocation.scope, invocation.key, None, 1, invocation.transport,
                                       invocation.fingerprint, protect=True)
            self.assertTrue(fresh)
            self.assertTrue(book._seal_tool_response(invocation.key, claim, json.dumps({"delegation_id": "one"})))
            book._release(invocation.key, claim)
            self.assertTrue(triage.bind_retained_admission(invocation, claim.generation))
            self.assertIsNone(self.spawn("delegate_claude", args, tool_call_id="expiry"))
            self.assertEqual(triage._TURN_WORKERS, {"retained:turn": 1})
            with patch.object(execution_adapters, "_RECORD_TTL_SECONDS", 0):
                self.assertIsNone(self.spawn("delegate_claude", args, tool_call_id="expiry"))
        self.assertEqual(triage._TURN_WORKERS, {"retained:turn": 2})

    def test_replay_exemption_is_removed_when_f05_lru_evicts_it(self):
        args = {"goal": "g", "tier": "haiku"}
        book = execution_adapters.ReservationBook()
        invocation = execution_adapters.stable_legacy_invocation(
            "delegate_claude", args, session_id="retained-root", turn_id="retained:turn", tool_call_id="lru",
        )
        other = execution_adapters.stable_legacy_invocation(
            "delegate_claude", args, session_id="other-session", turn_id="other:turn", tool_call_id="other",
        )
        with patch.object(execution_adapters, "RESERVATIONS", book), \
             patch.object(execution_adapters, "_MAX_RECORDS", 1):
            self.assertIsNone(self.spawn("delegate_claude", args, tool_call_id="lru"))
            fresh, claim = book._begin(invocation.scope, invocation.key, None, 1, invocation.transport,
                                       invocation.fingerprint, protect=True)
            self.assertTrue(fresh)
            self.assertTrue(book._seal_tool_response(invocation.key, claim, json.dumps({"delegation_id": "one"})))
            book._release(invocation.key, claim)
            self.assertTrue(triage.bind_retained_admission(invocation, claim.generation))
            fresh, replacement = book._begin(other.scope, other.key, None, 1, other.transport,
                                             other.fingerprint, protect=True)
            self.assertTrue(fresh)
            self.assertTrue(book._seal_tool_response(other.key, replacement, json.dumps({"delegation_id": "two"})))
            book._release(other.key, replacement)
            self.assertIsNone(self.spawn("delegate_claude", args, tool_call_id="lru"))
        self.assertEqual(triage._TURN_WORKERS, {"retained:turn": 2})


class WorkerBudgetOffTests(_Patched):
    triage_on = False

    def test_without_triage_the_budget_is_not_enforced_here(self):
        self.assertEqual(triage.spawn_block("delegate_task", {"goal": "g"}, self.cfg, {}), "")


class ConversationalTests(unittest.TestCase):
    def test_imperatives_are_actionable(self):
        for text in ("rendben csinald meg !", "mehet prodra is", "rakjad ki dev re", "javitsd meg a hibat",
                     "fix the parser", "nezd meg a logot"):
            self.assertFalse(triage.is_conversational(text), text)

    def test_questions_and_remarks_are_conversation(self):
        for text in ("mi a kovetkezo kritikus ?", "jo lett", "ment a level", "miert ment ez ilyen nehezen !?"):
            self.assertTrue(triage.is_conversational(text), text)


if __name__ == "__main__":
    unittest.main()
