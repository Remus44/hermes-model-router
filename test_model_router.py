import json
from datetime import datetime, timedelta, timezone
import tempfile

import unittest
from pathlib import Path
from unittest.mock import patch

from model_router.host_delegation_fixtures import host_delegation
from model_router import (
    RouteDecision,
    _is_callable_tier,
    _is_spark_read_only_work,
    _log_decision,
    _account_load_sentence,
    _model_param_contract,
    _recent_account_load,
    _record_tier_failure,
    _require_callable,
    _target_availability,
    _tier_cooldown_remaining,
    _lifecycle_event_kind,
    _prompt_preview,
    classify_request,
    on_post_llm_call,
    on_subagent_start,
    on_subagent_stop,
    route_llm_request,
    run_llm_with_transient_failover,
)


MODELS = {
    "luna": "gpt-6-luna",
    "spark": "gpt-5.3-codex-spark",
    "terra": "gpt-5.6-terra",
    "sol": "gpt-6-sol",
}
CALLABLE = {**{tier: True for tier in MODELS}, "opus5": True, "qwen": True}


def default_test_config():
    return {
        "enabled": True,
        "provider": "openai-codex",
        "models": MODELS,
        "callable": CALLABLE,
        "effort": {"luna": "low", "spark": "low", "terra": "medium", "sol": "medium"},
        "quota_fallbacks": {"spark": {"model": "luna", "effort": "medium"}},
    }

# Deliberately carries none of the old multi-step marker words: the planner must
# be forced by the turn being actionable, not by keyword matching. Long enough to
# clear orchestration.min_chars, which gates fan-out on decomposable work only.
ACTIONABLE_TERRA_PROMPT = (
    "Nézd meg, mi a baj. Tegnap óta más az eredmény, és nem tudom eldönteni, hogy a bemenet "
    "változott-e meg vagy a feldolgozás. Nézd át a vonatkozó részeket, és mondd meg, mit találsz, "
    "mielőtt bármit módosítanánk rajta."
)


def chat_request(text, *, with_tool_result=False):
    messages = [{"role": "user", "content": text}]
    if with_tool_result:
        messages.extend([
            {"role": "assistant", "tool_calls": [{"id": "1", "type": "function"}]},
            {"role": "tool", "tool_call_id": "1", "content": "result"},
        ])
    return {"model": MODELS["terra"], "messages": messages, "temperature": 0.2}


def responses_request(text):
    return {
        "model": MODELS["terra"],
        "input": [{
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        }],
    }


def chat_request_with_image(text, *, image_type="image_url"):
    request = chat_request(text)
    request["messages"][0]["content"] = [
        {"type": "text", "text": text},
        {"type": image_type, "image_url": "https://example.test/ui.png"},
    ]
    return request


def chat_request_with_historical_image(current_text):
    """A plain current turn after a previous visual turn in the same session."""
    return {
        "model": MODELS["terra"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Korábbi képes kérés."},
                    {"type": "image_url", "image_url": "https://example.test/old-ui.png"},
                ],
            },
            {"role": "assistant", "content": "A képet már elemeztem."},
            {"role": "user", "content": current_text},
        ],
    }


class ModelRouterTests(unittest.TestCase):
    def setUp(self):
        # Router tests construct realistic requests, including preflight fixtures.
        # Never let those fixtures append to the operator's active JSONL route log.
        # Direct _log_decision tests retain their own explicit temporary paths.
        self._route_log_patch = patch("model_router._log_decision")
        self._route_log_patch.start()
        self._config_patch = patch("model_router._load_config", side_effect=default_test_config)
        self._config_patch.start()
        self._host_delegation = host_delegation(depth=2)
        self._host_delegation.__enter__()

    def tearDown(self):
        self._host_delegation.__exit__(None, None, None)
        self._config_patch.stop()
        self._route_log_patch.stop()

    def test_synthetic_completion_log_is_typed_without_copying_its_result_into_provenance(self):
        request = chat_request("[ASYNC DELEGATION COMPLETE — deleg_test] token=not-for-provenance")
        self.assertEqual(_lifecycle_event_kind(request), "async_delegation_completion")

    def test_synthetic_completion_log_keeps_only_a_stable_id_not_raw_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.jsonl"
            request = chat_request("[ASYNC DELEGATION COMPLETE — deleg_test] token=not-for-log")
            _log_decision(
                RouteDecision("terra", MODELS["terra"], "test"),
                {"turn_id": "completion-turn", "request": request},
                {"logging": {"enabled": True, "path": str(path)}},
            )
            record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["event_kind"], "async_delegation_completion")
        self.assertEqual(record["delegation_id"], "deleg_test")
        self.assertEqual(record["prompt_preview"], "Delegált feladat befejezési eseménye")
        self.assertNotIn("not-for-log", json.dumps(record))

    def test_spark_eligible_request_records_the_tier_it_lost_out_on(self):
        """A read-only first call qualifies for Spark; nothing routes it there."""
        decision = classify_request(
            chat_request("Olvasd el a config fájlt és mondd meg, mi van benne. Ne módosíts semmit."),
            api_call_count=1,
        )
        self.assertEqual(decision.tier, "terra")
        self.assertIn("spark", decision.vetoed_by)

    def test_chosen_tier_is_never_listed_as_vetoed(self):
        decision = classify_request(chat_request("Készíts CSS elrendezést a kártyához."), 1)
        self.assertEqual(decision.tier, "sol")
        self.assertNotIn("sol", decision.vetoed_by)

    def test_terra_is_never_listed_as_vetoed_because_it_is_the_default(self):
        decision = classify_request(chat_request("Szia!"), 1)
        self.assertEqual(decision.tier, "luna")
        self.assertNotIn("terra", decision.vetoed_by)

    def test_vetoed_tiers_reach_the_route_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.jsonl"
            _log_decision(
                RouteDecision("terra", MODELS["terra"], "test", "medium", ("spark",)),
                {"turn_id": "veto-turn", "request": chat_request("bármi")},
                {"logging": {"enabled": True, "path": str(path)}},
            )
            record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["vetoed_by"], ["spark"])

    def test_route_log_omits_vetoed_by_when_nothing_was_preempted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.jsonl"
            _log_decision(
                RouteDecision("terra", MODELS["terra"], "test"),
                {"turn_id": "no-veto-turn", "request": chat_request("bármi")},
                {"logging": {"enabled": True, "path": str(path)}},
            )
            record = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("vetoed_by", record)

    def test_terra_is_the_default_for_normal_work(self):
        decision = classify_request(chat_request("Hasonlítsd össze ezt a két megoldást."), api_call_count=1)
        self.assertEqual(decision.tier, "terra")
        self.assertEqual(decision.model, MODELS["terra"])

    def test_benchmark_force_spark_rejects_mutating_work(self):
        with patch.dict("os.environ", {"MODEL_ROUTER_BENCHMARK_FORCE_MODEL": "spark"}):
            decision = classify_request(
                chat_request("Implement a complex regression fix.", with_tool_result=True),
                api_call_count=99,
            )

        self.assertEqual(decision.tier, "terra")
        self.assertIn("read-only", decision.reason)

    def test_luna_handles_clearly_simple_low_risk_requests(self):
        self.assertEqual(classify_request(chat_request("Szia!"), 1).tier, "luna")
        self.assertEqual(
            classify_request(chat_request("Fordítsd angolra: Jó reggelt!"), 1).tier,
            "luna",
        )

    def test_historical_image_does_not_lock_a_new_plain_turn_to_terra(self):
        decision = classify_request(chat_request_with_historical_image("Szia!"), 1)
        self.assertEqual(decision.tier, "luna")

    @patch("model_router._log_decision")
    def test_historical_image_design_request_routes_to_sol(self, mocked_log):
        result = route_llm_request(
            request=chat_request_with_historical_image("Írj egy CSS példát egy reszponzív kártyához."),
            provider="openai-codex",
            model=MODELS["terra"],
            api_call_count=1,
            turn_id="history-image-sol-turn",
        )
        self.assertEqual(result["metadata"]["tier"], "sol")
        payload = json.dumps(result["request"])
        self.assertIn("old-ui.png", payload)
        self.assertIn('"image_url"', payload)

    def test_sol_design_preflight_is_sol_owned_and_contractually_opus5(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {"enabled": True, "max_tasks": 3, "path": str(Path(temp_dir) / "orchestration.jsonl")},
                "sol_opus5_preflight": {"enabled": True, "owner": "sol", "bridge_model": "claude-opus-5-5", "require_successful_auth_probe": True},
                "shadow": {"enabled": False},
            }
            request = chat_request("Készíts UX/UI preflight értékelést egy bejelentkezési oldal vizuális elrendezéséről.")
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                result = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="sol-opus5-preflight-turn",
                )
        self.assertEqual(result["metadata"]["tier"], "sol")
        preflight = json.dumps(result["request"], ensure_ascii=False)
        self.assertIn("INTERNAL SOL + CLAUDE OPUS 5 PREFLIGHT", preflight)
        self.assertIn("claude-opus-5-5", preflight)
        self.assertIn("goal beginning [sol]", preflight)
        self.assertNotIn("[spark]", preflight)
        self.assertNotIn("[terra]", preflight)

    @patch("model_router._log_decision")
    def test_complex_current_image_task_can_start_terra_supervisor_preflight(self, mocked_log):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {"enabled": True, "min_chars": 40, "max_tasks": 3, "path": str(Path(temp_dir) / "orchestration.jsonl")},
                "shadow": {"enabled": False},
            }
            request = chat_request_with_image(
                "Elemezd a képet, majd bontsd részfeladatokra a javítást, közben ellenőrizd a komponenst és végül tervezz teszteket."
            )
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                result = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="current-image-supervisor-turn",
                )
        self.assertEqual(result["metadata"]["tier"], "terra")
        self.assertEqual(result["request"]["tool_choice"], "required")
        preflight = json.dumps(result["request"], ensure_ascii=False)
        self.assertIn("inspect any current image itself", preflight)
        self.assertIn("prefix a consequential worker goal with [sol]", preflight)

    def test_brief_non_actionable_chat_uses_luna_with_low_effort(self):
        decision = classify_request(chat_request("Ez szomorú."), api_call_count=1)
        self.assertEqual(decision.tier, "luna")
        self.assertEqual(decision.effort, "low")

    def test_design_praise_without_a_new_request_uses_luna_with_low_effort(self):
        decision = classify_request(
            chat_request("A UI design nagyon jó lett, köszönöm!"),
            api_call_count=1,
        )
        self.assertEqual(decision.tier, "luna")
        self.assertEqual(decision.effort, "low")

    def test_approval_only_follow_up_uses_luna_but_approval_to_execute_remains_actionable(self):
        acknowledgements = (
            "Nagyon jó lett, jóváhagyom.",
            "A UI design rendben van, elfogadom.",
            "Jóváhagyom, köszönöm!",
        )
        for prompt in acknowledgements:
            with self.subTest(prompt=prompt):
                decision = classify_request(chat_request(prompt), api_call_count=1)
                self.assertEqual(decision.tier, "luna")
                self.assertEqual(decision.effort, "low")

        actionable = classify_request(chat_request("Jóváhagyom, mehet prodra."), api_call_count=1)
        self.assertEqual(actionable.tier, "sol")

    def test_screenshot_backed_ui_action_is_sol_only(self):
        decision = classify_request(
            chat_request("Ez kerüljön át a név mögé. [Image attached at: /tmp/ui.png]"),
            api_call_count=1,
        )
        self.assertEqual(decision.tier, "sol")
        self.assertIn("Sol-only", decision.reason)

    def test_repeated_simple_turn_call_is_promoted_out_of_luna_without_tool_marker(self):
        decision = classify_request(chat_request("Ez szomorú."), api_call_count=2)
        self.assertEqual(decision.tier, "terra")
        self.assertIn("repeat", decision.reason.lower())

    def test_short_explanation_question_uses_luna_with_low_effort(self):
        decision = classify_request(chat_request("Miért fordítva van a használati sáv?"), api_call_count=1)
        self.assertEqual(decision.tier, "luna")
        self.assertEqual(decision.effort, "low")

    def test_short_css_design_implementation_is_sol_only(self):
        decision = classify_request(chat_request("Írj egy CSS példát egy reszponzív kártyához."), 1)
        self.assertEqual(decision.tier, "sol")
        self.assertEqual(decision.model, MODELS["sol"])
        self.assertEqual(decision.effort, "medium")

    def test_unlabelled_coding_tool_loop_stays_with_terra(self):
        decision = classify_request(
            chat_request("Írj egy SQL példát a legutóbbi 10 rendeléshez.", with_tool_result=True),
            2,
        )
        self.assertEqual(decision.tier, "terra")
        self.assertIn("tool", decision.reason)

    def test_small_local_router_dashboard_changes_start_with_terra_planner(self):
        decision = classify_request(
            chat_request("A Model Router dashboardhoz add hozzá a Spark számláló kártyát."),
            1,
        )
        self.assertEqual(decision.tier, "terra")
        self.assertIn("default", decision.reason)

    def test_screenshot_backed_router_design_change_is_sol_only(self):
        decision = classify_request(
            chat_request("A Model Router dashboardhoz add hozzá a Spark kártyát. [Image attached at: /tmp/ui.png]"),
            1,
        )
        self.assertEqual(decision.tier, "sol")
        self.assertIn("Sol-only", decision.reason)

    def test_structured_chat_image_never_uses_spark_even_with_explicit_override(self):
        decision = classify_request(chat_request_with_image("[spark] Add hozzá a Spark kártyát."), 1)
        self.assertEqual(decision.tier, "terra")
        self.assertIn("image", decision.reason)

    def test_responses_input_image_never_uses_spark(self):
        request = responses_request("[spark] Nézd meg a képet és igazítsd a kártyát.")
        request["input"][0]["content"].append({"type": "input_image", "image_url": "https://example.test/ui.png"})
        decision = classify_request(request, 1)
        self.assertEqual(decision.tier, "terra")
        self.assertIn("image", decision.reason)

    def test_benchmark_spark_force_cannot_override_image_safety_guard(self):
        with patch.dict("os.environ", {"MODEL_ROUTER_BENCHMARK_FORCE_MODEL": "spark"}):
            decision = classify_request(chat_request_with_image("Rövid CSS kérés."), 1)
        self.assertEqual(decision.tier, "sol")
        self.assertIn("Sol-only", decision.reason)

    def test_benchmark_spark_force_cannot_override_text_design_boundary(self):
        with patch.dict("os.environ", {"MODEL_ROUTER_BENCHMARK_FORCE_MODEL": "spark"}):
            decision = classify_request(chat_request("Implement a responsive CSS card."), 1)
        self.assertEqual(decision.tier, "sol")
        self.assertIn("Sol-only", decision.reason)

    @patch("model_router._log_decision")
    def test_root_spark_prefix_cannot_bypass_sol_only_design_policy(self, mocked_log):
        result = route_llm_request(
            request=chat_request("[spark] Írj egy CSS példát egy reszponzív kártyához."),
            provider="openai-codex", model=MODELS["terra"], api_call_count=1,
            turn_id="root-manual-spark-override",
        )
        self.assertEqual(result["metadata"]["tier"], "sol")
        self.assertIn("Sol-only", result["reason"])

    def test_explicit_spark_implementation_override_is_promoted_to_terra(self):
        decision = classify_request(
            chat_request("[spark] Add hozzá a Spark kártyát.", with_tool_result=True),
            2,
        )
        self.assertEqual(decision.tier, "terra")
        self.assertIn("read-only", decision.reason)

    @patch("model_router._log_decision")
    def test_delegated_spark_worker_stays_spark_for_a_bounded_tool_loop(self, mocked_log):
        result = route_llm_request(
            request=chat_request("Inspect the local parser configuration and report the affected setting.", with_tool_result=True),
            provider="openai-codex",
            model=MODELS["spark"],
            platform="subagent",
            api_call_count=2,
            turn_id="spark-child",
        )
        self.assertEqual(result["metadata"]["tier"], "spark")
        self.assertEqual(result["request"]["model"], MODELS["spark"])
        self.assertEqual(result["reason"], "eligible delegated Spark subtask")

    @patch("model_router._log_decision")
    def test_delegated_read_only_worker_is_not_escalated_by_negated_safety_constraints(self, mocked_log):
        result = route_llm_request(
            request=chat_request(
                "Read-only configuration audit: inspect the local router config and report the relevant settings. "
                "Do not edit files, restart services, or access credentials."
            ),
            provider="openai-codex",
            model=MODELS["spark"],
            platform="subagent",
            api_call_count=2,
            turn_id="spark-child-safe-constraints",
        )
        self.assertEqual(result["metadata"]["tier"], "spark")
        self.assertEqual(result["reason"], "eligible delegated Spark subtask")

    @patch("model_router._log_decision")
    def test_delegated_spark_worker_escalates_hard_safety_signal_to_sol(self, mocked_log):
        result = route_llm_request(
            request=chat_request("Debugold a production szerver konfigurációját."),
            provider="openai-codex",
            model=MODELS["spark"],
            platform="subagent",
            api_call_count=1,
            turn_id="spark-child-risky",
        )
        self.assertEqual(result["metadata"]["tier"], "sol")

    def test_terra_owns_normal_repo_implementation_while_sol_keeps_consequential_actions(self):
        risky = classify_request(
            chat_request("SSH-n lépj be a production szerverre, módosítsd a konfigurációt és indítsd újra."),
            1,
        )
        debug = classify_request(
            chat_request("Debugold ezt a hibát, javítsd a kódot, majd futtasd a teszteket."),
            1,
        )
        continuation = classify_request(
            chat_request("Csináld meg a repo módosítást a megbeszéltek szerint."),
            1,
        )
        self.assertEqual(risky.tier, "sol")
        self.assertEqual(debug.tier, "terra")
        self.assertEqual(continuation.tier, "terra")
        self.assertEqual(debug.effort, "medium")

    def test_sol_keeps_security_database_migration_and_production_deploy_work(self):
        prompts = [
            "Javítsd a payment webhook hibáját.",
            "Javítsd az auth jogosultsági regressziót.",
            "Migráld az adatbázist az új sémára.",
            "Deployold productionre és indítsd újra a szervert.",
        ]
        self.assertTrue(all(classify_request(chat_request(prompt), 1).tier == "sol" for prompt in prompts))

    def test_generic_implementation_question_stays_on_terra(self):
        decision = classify_request(chat_request("Hogyan implementáljam ezt az űrlapot?"), 1)
        self.assertEqual(decision.tier, "terra")
        self.assertEqual(decision.effort, "medium")

    def test_explicit_sol_override_is_limited_to_medium_effort(self):
        decision = classify_request(chat_request("[sol] Oldd meg ezt a kritikus production hibát"), 1)
        self.assertEqual(decision.tier, "sol")
        self.assertEqual(decision.effort, "high")

    def test_xhigh_override_is_still_capped_at_medium(self):
        decision = classify_request(chat_request("[sol:xhigh] Készíts részletes implementációs tervet."), 1)
        self.assertEqual(decision.tier, "sol")
        self.assertEqual(decision.effort, "high")

    def test_completed_terra_supervisor_review_stays_with_terra_despite_long_result(self):
        result = "[ASYNC DELEGATION BATCH COMPLETE — deleg_test] Role: orchestrator [terra] " + ("reviewed evidence " * 500)
        decision = classify_request(chat_request(result))
        self.assertEqual(decision.tier, "terra")
        self.assertEqual(decision.reason, "completed Terra supervisor review")

    def test_long_requests_are_routed_to_sol(self):
        decision = classify_request(chat_request("Elemezd részletesen. " + "x" * 4200), 1)
        self.assertEqual(decision.tier, "sol")
        self.assertEqual(decision.effort, "high")

    def test_merely_medium_length_request_stays_on_terra(self):
        decision = classify_request(chat_request("Elemezd ezt. " + "x" * 2200), 1)
        self.assertEqual(decision.tier, "terra")

    def test_normal_tool_loop_follow_up_stays_on_terra(self):
        decision = classify_request(chat_request("Nézd meg ezt.", with_tool_result=True), 2)
        self.assertEqual(decision.tier, "terra")
        self.assertIn("tool", decision.reason.lower())

    def test_normal_repo_tool_loop_follow_up_stays_on_terra(self):
        decision = classify_request(
            chat_request(
                "Debugold ezt a hibát, javítsd a kódot, majd futtasd a teszteket.",
                with_tool_result=True,
            ),
            2,
        )
        self.assertEqual(decision.tier, "terra")
        self.assertIn("tool", decision.reason.lower())

    def test_explicit_prefix_overrides_heuristics(self):
        self.assertEqual(classify_request(chat_request("[luna] Elemezd részletesen a szervert"), 1).tier, "luna")
        self.assertEqual(classify_request(chat_request("[spark] Oldd meg ezt a kritikus production hibát"), 1).tier, "sol")
        self.assertEqual(classify_request(chat_request("[terra] SSH szerver konfigurálása"), 1).tier, "terra")
        self.assertEqual(classify_request(chat_request("[sol] Szia"), 1).tier, "sol")

    def test_codex_responses_input_is_supported(self):
        decision = classify_request(responses_request("Fordítsd magyarra: good morning"), 1)
        self.assertEqual(decision.tier, "luna")

    @patch("model_router._log_decision")
    def test_middleware_rewrites_only_model_and_returns_trace_metadata(self, mocked_log):
        request = chat_request("Szia!")
        result = route_llm_request(
            request=request,
            provider="openai-codex",
            model=MODELS["terra"],
            api_call_count=1,
            turn_id="turn-1",
        )
        self.assertEqual(result["request"]["model"], MODELS["luna"])
        self.assertEqual(result["request"]["temperature"], 0.2)
        self.assertEqual(result["request"]["reasoning"]["effort"], "low")
        self.assertEqual(result["source"], "model-router")
        self.assertTrue(result["reason"])

    def test_shadow_lifecycle_logs_a_deterministic_parent_child_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shadow.jsonl"
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "shadow": {"enabled": True, "limit": 10, "path": str(path)},
            }
            request = chat_request("Hasonlítsd össze ezt a két megoldást.")
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="benchmark-turn-1",
                )
                on_subagent_start(
                    parent_turn_id="benchmark-turn-1",
                    child_session_id="spark-child-session",
                    child_subagent_id="sa-1",
                    child_role="leaf",
                    child_goal="Read-only comparison.",
                )
                on_subagent_stop(
                    parent_turn_id="benchmark-turn-1",
                    child_session_id="spark-child-session",
                    child_status="completed",
                    child_model=MODELS["spark"],
                    child_api_calls=3,
                    input_tokens=100,
                    output_tokens=25,
                    cost_usd=0.01,
                    exit_reason="completed",
                    duration_ms=1250,
                    child_summary="Evidence-based Spark result.",
                )
                on_post_llm_call(
                    turn_id="benchmark-turn-1",
                    model=MODELS["terra"],
                    assistant_response="Evidence-based Terra answer.",
                )
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                [event["event"] for event in events],
                ["delegation_forced", "child_started", "child_completed", "parent_completed"],
            )
            self.assertEqual(len({event["benchmark_id"] for event in events}), 1)
            self.assertEqual(events[-2]["child_model"], MODELS["spark"])
            self.assertEqual(events[-2]["child_api_calls"], 3)
            self.assertNotIn("summary_preview", events[-2])
            self.assertIn("summary_sha256", events[-2])
            self.assertEqual(events[-1]["parent_model"], MODELS["terra"])
            self.assertNotIn("parent_response_preview", events[-1])
            self.assertIn("parent_response_sha256", events[-1])

    def test_terra_supervisor_preflight_dispatches_real_bounded_spark_work_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "orchestration.jsonl"
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {"enabled": True, "min_chars": 40, "max_tasks": 3, "path": str(path)},
                "shadow": {"enabled": False},
            }
            request = chat_request(
                "Vizsgáld meg a futási naplókat, utána ellenőrizd a komponens állapotát, "
                "végül tervezz regressziós teszteket."
            )
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                first = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="supervisor-turn-1",
                )
                second = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="supervisor-turn-1",
                )
            self.assertEqual(first["request"]["model"], MODELS["terra"])
            self.assertEqual(first["request"]["tool_choice"], "required")
            self.assertEqual(len(first["request"]["tools"]), 1)
            self.assertIn("INTERNAL ORCHESTRATOR PREFLIGHT", first["request"]["messages"][-1]["content"])
            self.assertIn("SUPERVISOR DECISION: ACCEPT or REJECT", first["request"]["messages"][-1]["content"])
            self.assertNotIn("tool_choice", second["request"])
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["event"] for event in events], ["preflight_forced"])


    def test_compaction_envelope_routes_the_actual_current_user_request(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "orchestration.jsonl"
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {"enabled": True, "min_chars": 40, "max_tasks": 3, "path": str(path)},
                "shadow": {"enabled": False},
            }
            request = chat_request(
                "[CONTEXT COMPACTION — REFERENCE ONLY] historical context\n"
                "[END OF CONTEXT SUMMARY — respond to the message below]\n"
                "Csináld újra a videót: először válaszd ki a szolgáltatást, aztán zoomolj rá, "
                "közben lassan scrollozz a következő vezérlőre, végül ellenőrizd a regressziót."
            )
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                routed = route_llm_request(
                    request=request, provider="openai-codex", model=MODELS["terra"],
                    api_call_count=1, turn_id="compacted-complex-turn",
                )
            self.assertEqual(routed["request"]["tool_choice"], "required")
            self.assertTrue(path.exists())

    def test_long_terra_tool_loop_is_rescued_once_with_real_supervisor_dispatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "orchestration.jsonl"
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {
                    "enabled": True, "min_chars": 40, "max_tasks": 3,
                    "rescue_min_calls": 6, "path": str(path),
                },
                "shadow": {"enabled": False},
            }
            request = chat_request(
                "Készítsd el az animációt: majd zoomolj a kártyára, utána lassan scrollozz, "
                "közben ellenőrizd a futási naplókat és komponens állapotát, végül tervezz regressziós teszteket.",
                with_tool_result=True,
            )
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                rescued = route_llm_request(
                    request=request, provider="openai-codex", model=MODELS["terra"],
                    api_call_count=6, turn_id="terra-loop-without-preflight",
                )
                later = route_llm_request(
                    request=request, provider="openai-codex", model=MODELS["terra"],
                    api_call_count=7, turn_id="terra-loop-without-preflight",
                )
            self.assertEqual(rescued["request"]["tool_choice"], "required")
            self.assertNotIn("tool_choice", later["request"])
            event = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(event["event"], "preflight_forced")
            self.assertEqual(event["phase"], "rescue")
            self.assertEqual(event["api_call_count"], 6)

    def test_hungarian_aztan_starts_preflight_but_internal_memory_prompt_does_not(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True, "provider": "openai-codex", "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {"enabled": True, "min_chars": 40, "max_tasks": 3, "path": str(Path(temp_dir) / "orchestration.jsonl")},
                "shadow": {"enabled": False},
            }
            video = ("Csináld újra a videót úgy, hogy minden látszódjon: először nyisd ki a blokkokat, "
                     "aztán görgess, közben ellenőrizd a zoomot, végül készíts regressziós ellenőrzést.")
            memory = ("Review the conversation above and consider saving to memory if appropriate. "
                      "Focus on the user preferences and work style, then save them.")
            video_request, memory_request = chat_request(video), chat_request(memory)
            video_request["tools"] = memory_request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                video_result = route_llm_request(request=video_request, provider="openai-codex", model=MODELS["terra"], api_call_count=1, turn_id="video-turn")
                memory_result = route_llm_request(request=memory_request, provider="openai-codex", model=MODELS["terra"], api_call_count=1, turn_id="memory-turn")
            self.assertEqual(video_result["request"].get("tool_choice"), "required")
            self.assertNotIn("tool_choice", memory_result["request"])

    def test_completed_supervisor_handback_is_not_delegated_again(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {"enabled": True, "min_chars": 40, "max_tasks": 3, "path": str(Path(temp_dir) / "orchestration.jsonl")},
                "shadow": {"enabled": False},
            }
            handback = "[ASYNC DELEGATION BATCH COMPLETE — deleg_test] Role: orchestrator [terra] " + ("SUPERVISOR DECISION: ACCEPT reviewed evidence " * 100)
            request = chat_request(handback)
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                routed = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="supervisor-handback",
                )
            self.assertEqual(routed["request"]["model"], MODELS["terra"])
            self.assertNotIn("tool_choice", routed["request"])

    def test_eligible_terra_turn_forces_one_read_only_spark_shadow_delegation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "shadow": {"enabled": True, "limit": 10, "path": str(Path(temp_dir) / "shadow.jsonl")},
            }
            request = chat_request("Hasonlítsd össze ezt a két megoldást.")
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                result = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="benchmark-turn-1",
                )
            self.assertEqual(result["request"]["model"], MODELS["terra"])
            self.assertEqual(result["request"]["tool_choice"], "required")
            self.assertEqual(len(result["request"]["tools"]), 1)
            self.assertEqual(result["request"]["tools"][0]["name"], "delegate_task")
            self.assertIn("INTERNAL SPARK MEDIUM SHADOW BENCHMARK", result["request"]["messages"][-1]["content"])

    def test_incomplete_forced_shadows_do_not_consume_the_ten_completed_lifecycle_quota(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shadow.jsonl"
            events = []
            for index in range(10):
                benchmark_id = f"shadow-{index}"
                events.append({"event": "delegation_forced", "turn_id": f"old-turn-{index}", "benchmark_id": benchmark_id})
                if index < 8:
                    events.extend([
                        {"event": "child_started", "turn_id": f"old-turn-{index}", "benchmark_id": benchmark_id},
                        {"event": "child_completed", "turn_id": f"old-turn-{index}", "benchmark_id": benchmark_id},
                        {"event": "parent_completed", "turn_id": f"old-turn-{index}", "benchmark_id": benchmark_id},
                    ])
            path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "shadow": {"enabled": True, "limit": 10, "path": str(path)},
            }
            request = chat_request("Hasonlítsd össze ezt a két megoldást.")
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                result = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="replacement-turn",
                )
            self.assertEqual(result["request"]["tool_choice"], "required")
            updated_events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(event.get("event") == "delegation_forced" for event in updated_events), 11)

    def test_new_shadow_cycle_ignores_completed_lifecycles_from_an_earlier_cycle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shadow.jsonl"
            events = []
            for index in range(10):
                benchmark_id = f"old-shadow-{index}"
                for event_name in ("delegation_forced", "child_started", "child_completed", "parent_completed"):
                    events.append({
                        "event": event_name,
                        "turn_id": f"old-turn-{index}",
                        "benchmark_id": benchmark_id,
                        "cycle_id": "old",
                        "child_session_id": f"old-child-{index}",
                    })
            path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "shadow": {"enabled": True, "limit": 10, "cycle_id": "new", "path": str(path)},
            }
            request = chat_request("Hasonlítsd össze ezt a két megoldást.")
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                result = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="new-cycle-turn",
                )
            self.assertEqual(result["request"]["tool_choice"], "required")
            updated_events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(updated_events[-1]["event"], "delegation_forced")
            self.assertEqual(updated_events[-1]["cycle_id"], "new")

    def test_sol_promoted_children_do_not_consume_the_actual_spark_benchmark_quota(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shadow_path = Path(temp_dir) / "shadow.jsonl"
            route_path = Path(temp_dir) / "router.jsonl"
            events = []
            routes = []
            for index in range(10):
                benchmark_id = f"shadow-{index}"
                child_session_id = f"child-{index}"
                events.extend([
                    {"event": "delegation_forced", "turn_id": f"old-turn-{index}", "benchmark_id": benchmark_id},
                    {"event": "child_started", "turn_id": f"old-turn-{index}", "benchmark_id": benchmark_id, "child_session_id": child_session_id},
                    {"event": "child_completed", "turn_id": f"old-turn-{index}", "benchmark_id": benchmark_id, "child_session_id": child_session_id},
                    {"event": "parent_completed", "turn_id": f"old-turn-{index}", "benchmark_id": benchmark_id},
                ])
                routes.append({
                    "turn_id": f"parent:{child_session_id}:worker",
                    "model": MODELS["spark"] if index < 8 else MODELS["sol"],
                })
            shadow_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
            route_path.write_text("\n".join(json.dumps(route) for route in routes) + "\n", encoding="utf-8")
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "logging": {"path": str(route_path)},
                "shadow": {"enabled": True, "limit": 10, "path": str(shadow_path)},
            }
            request = chat_request("Hasonlítsd össze ezt a két megoldást.")
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                result = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="verified-spark-replacement-turn",
                )
            self.assertEqual(result["request"]["tool_choice"], "required")

    @patch("model_router._log_decision")
    def test_turn_id_marked_terra_planner_stays_terra_until_it_labels_a_leaf(self, mocked_log):
        result = route_llm_request(
            request=chat_request("Read only inspect this local file and report its fields."),
            provider="openai-codex",
            model=MODELS["terra"],
            api_call_count=2,
            turn_id="parent-session:sa-0-child-session:worker-turn",
        )
        self.assertEqual(result["request"]["model"], MODELS["terra"])
        self.assertEqual(result["request"]["reasoning"]["effort"], "medium")
        self.assertEqual(result["reason"], "Terra planner or integration subagent")

    @patch("model_router._log_decision")
    def test_spark_subagent_with_design_image_is_rerouted_to_sol(self, mocked_log):
        result = route_llm_request(
            request=chat_request_with_image("Inspect this UI screenshot and report the affected selector."),
            provider="openai-codex",
            model=MODELS["spark"],
            platform="subagent",
            api_call_count=2,
            turn_id="spark-child-with-image",
        )
        self.assertEqual(result["metadata"]["tier"], "sol")
        self.assertEqual(result["request"]["model"], MODELS["sol"])

    @patch("model_router._log_decision")
    def test_conductor_label_survives_a_design_flavoured_objective(self, mocked_log):
        """The conductor coordinates design work; it does not perform it. Judging
        it by keywords put the planner on Sol whenever the objective touched UI,
        and Sol then owned both the conducting and the [sol] leaf it should have
        delegated -- 15 of 16 routing decisions on one real turn."""
        result = route_llm_request(
            request=chat_request(
                "[terra] Fix the daily-calendar card layout in /home/deepwell/booking-saas; "
                "trace the computed layout/markup before implementing."
            ),
            provider="openai-codex", model=MODELS["terra"], platform="subagent",
            api_call_count=1, turn_id="conductor-design-objective",
        )
        self.assertEqual(result["metadata"]["tier"], "terra")

    @patch("model_router._log_decision")
    def test_root_design_turn_still_ignores_a_typed_label(self, mocked_log):
        """The exemption is for plan labels only. A user typing [terra] on a root
        turn must not be able to route design work away from Sol."""
        decision = classify_request(
            chat_request("[terra] Design a responsive CSS card layout."),
            api_call_count=1,
        )
        self.assertEqual(decision.tier, "sol")
        self.assertIn("Sol-only", decision.reason)

    def _usage_log(self, directory, entries):
        path = Path(directory) / "route.jsonl"
        now = datetime.now(timezone.utc)
        lines = []
        for tier, age_seconds in entries:
            stamp = (now - timedelta(seconds=age_seconds)).replace(microsecond=0).isoformat()
            lines.append(json.dumps({"timestamp": stamp, "tier": tier, "turn_id": "s:s:1"}))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _usage_cfg(self, log_path, cooldown_path):
        return {
            "enabled": True, "provider": "openai-codex", "models": MODELS, "callable": CALLABLE,
            "tier_providers": {"luna": "openai-codex", "spark": "openai-codex",
                               "terra": "openai-codex", "sol": "openai-codex",
                               "qwen": "qwen-token"},
            "logging": {"enabled": True, "path": str(log_path)},
            "usage_report": {"enabled": True, "window_seconds": 3600},
            "cooldown": {"enabled": True, "path": str(cooldown_path)},
        }

    def test_account_load_counts_only_the_recent_window(self):
        with tempfile.TemporaryDirectory() as directory:
            log = self._usage_log(directory, [
                ("terra", 60), ("sol", 120), ("qwen", 180),
                ("terra", 7200),  # older than the window
            ])
            cfg = self._usage_cfg(log, Path(directory) / "cool.json")
            self.assertEqual(
                _recent_account_load(cfg, 3600), {"openai-codex": 2, "qwen-token": 1}
            )

    def test_the_contract_names_an_account_that_has_taken_nothing(self):
        """Spreading work was an instruction with nothing behind it: the conductor
        was told to use separate accounts but could not see that one had taken
        every call for an hour and another had taken none."""
        with tempfile.TemporaryDirectory() as directory:
            log = self._usage_log(directory, [("terra", 30), ("sol", 60)])
            cfg = self._usage_cfg(log, Path(directory) / "cool.json")
            sentence = _account_load_sentence(cfg)
            self.assertIn("openai-codex 2", sentence)
            self.assertIn("qwen-token has taken none", sentence)
            # Call counts are what the log supports; a percentage would be invented.
            self.assertIn("not quota readings", sentence)

    def test_a_cooling_target_is_annotated_rather_than_hidden(self):
        """LiteLLM drops a deployment over its limit, but its deployments are
        interchangeable and ours are not: hiding a cooling Sol would invite the
        planner to send design work somewhere the classifier then refuses."""
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._usage_cfg(Path(directory) / "route.jsonl", Path(directory) / "cool.json")
            _record_tier_failure("sol", cfg, quota=True)
            notes = _target_availability(["luna", "sol", "opus5"], cfg)
            self.assertIn("unavailable for another", notes["sol"])
            # Luna shares Sol's account, so a usage quota takes it down too — but it
            # is still listed, annotated, for the same reason Sol is.
            self.assertIn("unavailable for another", notes["luna"])
            # A target on a different account must stay clean, or benching Codex
            # would quietly remove the alternative the planner is meant to reach for.
            self.assertEqual(notes["opus5"], "")

    def test_a_config_without_a_log_path_writes_nothing(self):
        """Test-suite entries appended to the real audit log then read back as
        real traffic -- 56 of them, which is exactly what the usage report
        attributed to an account that had taken no calls at all."""
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "route.jsonl"
            cfg = {"enabled": True, "models": MODELS, "callable": CALLABLE,
                   "logging": {"enabled": True}}
            _log_decision(RouteDecision("terra", MODELS["terra"], "any"), {"request": {}}, cfg)
            self.assertFalse(marker.exists())

    def _cooldown_cfg(self, path, **overrides):
        policy = {
            "enabled": True, "path": str(path), "quota_seconds": 900,
            "allowed_fails": 3, "failure_window_seconds": 60, "failure_seconds": 60,
        }
        policy.update(overrides)
        return {
            "enabled": True, "provider": "openai-codex", "models": MODELS,
            "callable": CALLABLE, "fallbacks": {"sol": "terra", "spark": "luna"},
            "thresholds": {"sol_min_chars": 3500, "luna_max_chars": 700},
            "cooldown": policy,
        }

    def test_a_quota_rejection_is_remembered_even_with_no_fallback(self):
        """The gap this closes: a tier with no configured quota fallback simply
        re-raised and left nothing behind, so the next call walked straight back
        into the same exhausted account. The two Hermes processes do not share
        memory either, which is why the note goes to a file."""
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cooldown_cfg(Path(directory) / "cooldowns.json")

            def exhausted(_request):
                raise RuntimeError("HTTP 429: usage limit reached")

            with patch("model_router._load_config", return_value=cfg):
                self.assertTrue(_is_callable_tier("sol", cfg))
                with self.assertRaises(RuntimeError):
                    run_llm_with_transient_failover(
                        request={**chat_request("Anything."), "model": MODELS["sol"]},
                        next_call=exhausted, provider="openai-codex", turn_id="quota-turn",
                    )
                self.assertFalse(_is_callable_tier("sol", cfg))
                self.assertGreater(_tier_cooldown_remaining("sol", cfg), 600)

    def test_a_failure_on_another_provider_is_still_recorded(self):
        """The provider guard exists because this middleware rewrites
        request["model"] within one provider. Noticing that an account just
        refused a call needs none of that, and skipping it meant a Qwen weekly
        quota 429 left no cooldown -- while the load report kept describing that
        account as the one with no traffic."""
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cooldown_cfg(Path(directory) / "cooldowns.json")

            def exhausted(_request):
                raise RuntimeError(
                    "HTTP 429: Your token-plan 1-week quota has been exhausted."
                )

            cfg["models"] = {**MODELS, "qwen": "qwen3.7-plus"}
            with patch("model_router._load_config", return_value=cfg):
                with self.assertRaises(RuntimeError):
                    run_llm_with_transient_failover(
                        request={**chat_request("Anything."), "model": "qwen3.7-plus"},
                        next_call=exhausted,
                        provider="qwen-token",           # not the configured provider
                        api_mode="anthropic_messages",
                        turn_id="off-provider-turn",
                    )
                self.assertGreater(_tier_cooldown_remaining("qwen", cfg), 600)

    def _peer_cfg(self, path):
        cfg = self._cooldown_cfg(path)
        cfg["models"] = {**MODELS, "qwen": "qwen3.7-plus"}
        cfg["callable"] = {**CALLABLE, "qwen": True, "opus5": True, "sonnet5": True}
        cfg["peer_groups"] = {"heavy": ["terra", "opus5", "qwen"],
                              "light": ["luna", "sonnet5", "spark"]}
        return cfg

    def test_a_cooling_target_stays_visible_and_names_its_replacement(self):
        """Dropping it told the planner only that it was gone. Knowing what
        replaces it is what turns one account's exhaustion into work continuing
        somewhere else rather than queueing."""
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._peer_cfg(Path(directory) / "cooldowns.json")
            _record_tier_failure("opus5", cfg, quota=True)
            with patch("model_router._delegation_target_names",
                       return_value=("luna", "opus5", "qwen", "sol", "sonnet5")):
                contract = _model_param_contract("terra", cfg)
        self.assertIn("opus5 [unavailable for another", contract)
        self.assertIn("use qwen instead", contract)

    def test_the_substitution_groups_are_stated(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._peer_cfg(Path(directory) / "cooldowns.json")
            with patch("model_router._delegation_target_names",
                       return_value=("luna", "opus5", "qwen", "sol", "sonnet5")):
                contract = _model_param_contract("terra", cfg)
        self.assertIn("opus5 / qwen", contract)
        self.assertIn("luna / sonnet5", contract)
        # Comparable strength is not comparable permission.
        self.assertIn("Substituting is for capacity only", contract)

    def test_the_goal_must_carry_what_the_worker_cannot_see(self):
        """A worker starts at history=0 on every target. One Opus leaf spent all
        sixteen iterations and twenty tool calls rediscovering a repository its
        goal never described, and made no edit; the leaf whose goal carried its
        own state finished in nine with one write."""
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._peer_cfg(Path(directory) / "cooldowns.json")
            with patch("model_router._delegation_target_names",
                       return_value=("luna", "opus5", "qwen", "sol", "sonnet5")):
                contract = _model_param_contract("terra", cfg)
        self.assertIn("does not share this conversation", contract)
        self.assertIn("absolute worktree path", contract)
        self.assertIn("one finishable artefact", contract)

    def test_the_reading_is_moved_off_the_expensive_account(self):
        """The Opus leaf of 2026-09-17 had delegate_task and the depth to use it,
        and still spent all sixteen iterations reading: twenty tool calls, no edit,
        exit_reason=max_iterations. It cannot fix that from inside -- by the time it
        knows enough to delegate the reading it has already paid for it -- so the
        order goes to the conductor, before the leaf exists."""
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._peer_cfg(Path(directory) / "cooldowns.json")
            cfg["preferences"] = {"explore": ["luna", "spark", "sonnet5"]}
            with patch("model_router._delegation_target_names",
                       return_value=("luna", "opus5", "qwen", "sol", "sonnet5")):
                contract = _model_param_contract("terra", cfg)
        self.assertIn("read-only recon leaf on luna", contract)
        self.assertIn("must not spend a budget on orientation", contract)
        # The recon leaf is dispatched to hand its findings over, not to file a report.
        self.assertIn("`context`", contract)

    def test_the_conductors_own_tier_is_still_covered_by_the_rule(self):
        """`names` drops the conductor's own tier because it answers "other targets
        to spread across". This rule answers a different question -- which account
        pays for the reading -- and with the operator's code chain making opus5 the
        conductor, taking that list dropped opus5 out of its own rule."""
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._peer_cfg(Path(directory) / "cooldowns.json")
            cfg["preferences"] = {"explore": ["luna"], "code": ["opus5", "terra"]}
            with patch("model_router._delegation_target_names",
                       return_value=("luna", "opus5", "qwen", "sol", "sonnet5")):
                contract = _model_param_contract("opus5", cfg)
        self.assertNotIn("targets: opus5", contract)      # still not a load-spreading option
        self.assertIn("opus5 and sonnet5 must not spend a budget", contract)

    def test_the_recon_target_comes_from_the_operators_explore_order(self):
        """"Cheap" is an account fact, not a property of a name: hardcoding a tier
        here would keep sending recon to a target the operator had switched off."""
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._peer_cfg(Path(directory) / "cooldowns.json")
            cfg["preferences"] = {"explore": ["qwen", "luna"]}
            with patch("model_router._delegation_target_names",
                       return_value=("luna", "opus5", "qwen", "sol", "sonnet5")):
                contract = _model_param_contract("terra", cfg)
        self.assertIn("recon leaf on qwen or luna", contract)

    def test_with_no_explore_order_the_light_peer_group_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._peer_cfg(Path(directory) / "cooldowns.json")
            cfg["peer_groups"] = {"heavy": ["terra", "opus5"], "light": ["luna", "qwen"]}
            with patch("model_router._delegation_target_names",
                       return_value=("luna", "opus5", "qwen", "sol", "sonnet5")):
                contract = _model_param_contract("terra", cfg)
        self.assertIn("recon leaf on luna or qwen", contract)

    def test_no_cheap_target_means_no_recon_order(self):
        """An instruction to move the reading somewhere that does not exist costs
        the leaf a refused dispatch and buys nothing."""
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._peer_cfg(Path(directory) / "cooldowns.json")
            cfg["peer_groups"] = {}
            with patch("model_router._delegation_target_names",
                       return_value=("opus5", "sonnet5")):
                contract = _model_param_contract("terra", cfg)
        self.assertNotIn("recon leaf", contract)

    def test_a_fleet_without_a_claude_target_is_told_nothing_about_recon(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._peer_cfg(Path(directory) / "cooldowns.json")
            with patch("model_router._delegation_target_names",
                       return_value=("luna", "sol")):
                contract = _model_param_contract("terra", cfg)
        self.assertNotIn("recon leaf", contract)

    def test_a_switched_off_tier_is_not_offered_as_a_target(self):
        """The cross-provider guard raises for a disabled tier mid-session, so
        offering it produces a leaf that never runs: exactly what happened when
        a review leaf was sent to Qwen while qwen was callable: false."""
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cooldown_cfg(Path(directory) / "cooldowns.json")
            cfg["callable"] = {**CALLABLE, "qwen": False}
            cfg["models"] = {**MODELS, "qwen": "qwen3.7-plus"}
            with patch("model_router._delegation_target_names",
                       return_value=("luna", "qwen", "sol")):
                contract = _model_param_contract("terra", cfg)
        self.assertIn("luna", contract)
        self.assertNotIn("qwen", contract)

    def test_a_cooling_tier_falls_back_but_a_policy_route_still_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cooldown_cfg(Path(directory) / "cooldowns.json")
            _record_tier_failure("sol", cfg, quota=True)

            long_request = classify_request(chat_request("Analyse this. " + "x" * 3600), config=cfg)
            self.assertEqual(_require_callable(long_request, cfg).tier, "terra")

            design = classify_request(chat_request("Design a responsive CSS card layout."), config=cfg)
            with self.assertRaises(RuntimeError) as raised:
                _require_callable(design, cfg)
            self.assertIn("cooling down", str(raised.exception))

    def test_repeated_failures_cool_a_tier_down_only_at_the_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cooldown_cfg(Path(directory) / "cooldowns.json", allowed_fails=3)
            for _ in range(2):
                _record_tier_failure("luna", cfg, quota=False)
                self.assertTrue(_is_callable_tier("luna", cfg))
            _record_tier_failure("luna", cfg, quota=False)
            self.assertFalse(_is_callable_tier("luna", cfg))

    def test_a_config_without_a_cooldown_path_writes_nothing(self):
        """A component that writes to a shared location takes that location from
        the config it was handed. Inventing a default means any caller with a
        partial config silently writes to the production file."""
        cfg = {"enabled": True, "provider": "openai-codex", "models": MODELS, "callable": CALLABLE}
        _record_tier_failure("sol", cfg, quota=True)
        self.assertTrue(_is_callable_tier("sol", cfg))

    def test_a_policy_route_is_not_laundered_by_the_fallback_chain(self):
        """Design work reaches Sol because only Sol may do it. Answering "Sol is
        unavailable" with Terra performs the work on the tier the rule exists to
        keep it away from -- and does so exactly when Sol has run out of quota,
        which is when the rule matters most."""
        cfg = {
            "enabled": True, "provider": "openai-codex", "models": MODELS,
            "callable": {**CALLABLE, "sol": False},
            "fallbacks": {"sol": "terra"},
        }
        decision = classify_request(
            chat_request("Design a responsive CSS card layout."), config=cfg
        )
        self.assertEqual(decision.tier, "sol")
        self.assertTrue(decision.mandatory)
        with self.assertRaises(RuntimeError) as raised:
            _require_callable(decision, cfg)
        self.assertIn("no fallback may take its place", str(raised.exception))

    def test_a_preference_route_still_falls_back(self):
        """The rule is about policy, not about every Sol route: a long request
        prefers Sol for capacity, and demoting that is a quality trade, not a
        boundary violation."""
        cfg = {
            "enabled": True, "provider": "openai-codex", "models": MODELS,
            "callable": {**CALLABLE, "sol": False},
            "fallbacks": {"sol": "terra"},
            "thresholds": {"sol_min_chars": 3500, "luna_max_chars": 700},
        }
        decision = classify_request(chat_request("Analyse this. " + "x" * 3600), config=cfg)
        self.assertEqual(decision.tier, "sol")
        self.assertFalse(decision.mandatory)
        self.assertEqual(_require_callable(decision, cfg).tier, "terra")

    @patch("model_router._log_decision")
    def test_a_labelled_leaf_survives_its_own_callable_fallback(self, mocked_log):
        """The planner-tier default asked "was this labelled?" by string-matching
        the reason, which the callable fallback rewrites: a [spark] leaf becomes
        "fallback from disabled spark" the moment Spark is not callable, so every
        legitimate Spark leaf was demoted to the planner tier and lost its Luna
        fallback -- the separate model the leaf was meant to run on."""
        cfg = {
            "enabled": True, "provider": "openai-codex", "models": MODELS,
            "callable": {**CALLABLE, "spark": False, "luna": True},
            "fallbacks": {"spark": "luna"},
            "default_model": "terra",
        }
        with patch("model_router._load_config", return_value=cfg):
            result = route_llm_request(
                request=chat_request("[spark] Read-only inventory. Identify the build commands."),
                provider="openai-codex", model=MODELS["terra"], platform="subagent",
                api_call_count=1, turn_id="session:sa-1-leaf:turn",
            )
        self.assertEqual(result["metadata"]["tier"], "luna")

    @patch("model_router._log_decision")
    def test_a_rejected_leaf_keeps_its_escalation(self, mocked_log):
        """Demoting a rejected [spark] leaf to the planner tier turned a
        deliberate escalation into the very tier it existed to avoid."""
        with patch("model_router._load_config", return_value={
            "enabled": True, "provider": "openai-codex", "models": MODELS,
            "callable": CALLABLE, "default_model": "terra",
        }):
            result = route_llm_request(
                request=chat_request("[spark] Review the deploy scripts in production."),
                provider="openai-codex", model=MODELS["terra"], platform="subagent",
                api_call_count=1, turn_id="session:sa-1-risky:turn",
            )
        self.assertEqual(result["metadata"]["tier"], "sol")
        self.assertIn("consequential", result["reason"])

    @patch("model_router._log_decision")
    def test_unlabelled_child_work_still_defaults_to_the_planner_tier(self, mocked_log):
        with patch("model_router._load_config", return_value={
            "enabled": True, "provider": "openai-codex", "models": MODELS,
            "callable": CALLABLE, "default_model": "terra",
        }):
            result = route_llm_request(
                request=chat_request("Continue the integration work for the calendar fix."),
                provider="openai-codex", model=MODELS["terra"], platform="subagent",
                api_call_count=1, turn_id="session:sa-1-plain:turn",
            )
        self.assertEqual(result["metadata"]["tier"], "terra")
        self.assertIn("planner or integration subagent", result["reason"])

    @patch("model_router._log_decision")
    def test_a_leaf_need_not_prove_it_is_read_only_in_a_known_phrasing(self, mocked_log):
        """The conductor already declared the leaf read-only by labelling it, so
        demanding a second positive signal lets the router overrule that claim
        whenever the wording falls outside a hand-written verb list. A Hungarian
        goal did exactly that on its first outing: "Tárd fel..." reads nothing
        but says so with a verb the list never had, and eight calls of pure
        discovery ran on the planner tier."""
        cfg = {
            "enabled": True, "provider": "openai-codex", "models": MODELS,
            "callable": {**CALLABLE, "spark": False, "luna": True},
            "fallbacks": {"spark": "luna"}, "default_model": "terra",
        }
        with patch("model_router._load_config", return_value=cfg):
            result = route_llm_request(
                request=chat_request(
                    "[spark] Tárd fel tényalapon a napi időrács renderelési útvonalát "
                    "és a rendelkezésre álló teszteket."
                ),
                provider="openai-codex", model=MODELS["terra"], platform="subagent",
                api_call_count=1, turn_id="session:sa-1-discovery:turn",
            )
        self.assertEqual(result["metadata"]["tier"], "luna")

    def test_a_root_spark_label_must_still_show_its_read_only_intent(self):
        """No conductor vouched for a label someone typed, so the stricter form
        stays: absence of a write verb is not a declaration of intent."""
        decision = classify_request(
            chat_request("[spark] Oldd meg ezt a kritikus production hibát"), 1
        )
        self.assertEqual(decision.tier, "sol")

    @patch("model_router._log_decision")
    def test_read_only_discovery_leaf_keeps_its_spark_label(self, mocked_log):
        """"layout" is an ordinary noun in frontend source discovery. Judging the
        leaf by that word sent every such leaf to Sol -- the planner had already
        decided, with the screenshot and the repo in hand, that this was bounded
        read-only evidence work."""
        result = route_llm_request(
            request=chat_request(
                "[spark] Perform read-only source discovery in /home/deepwell/booking-saas. "
                "Identify the exact daily calendar booking-card renderer, duration layout "
                "branches, and the editor overlay state owner."
            ),
            provider="openai-codex", model=MODELS["terra"], platform="subagent",
            api_call_count=1, turn_id="spark-discovery-leaf",
        )
        self.assertEqual(result["metadata"]["tier"], "spark")

    @patch("model_router._log_decision")
    def test_a_spark_label_still_has_to_be_true(self, mocked_log):
        """The label is trusted, not obeyed: a leaf that writes is not read-only
        evidence work whatever it is labelled, and the design test survives to
        place the rejected leaf rather than to preempt it."""
        decision = classify_request(
            chat_request("[spark] Implement a responsive CSS card."),
            api_call_count=1,
            allow_plan_label_over_design=True,
        )
        self.assertEqual(decision.tier, "sol")

    @patch("model_router._log_decision")
    def test_text_only_design_spark_subagent_is_rerouted_to_sol(self, mocked_log):
        result = route_llm_request(
            request=chat_request("[spark] Implement a responsive CSS card."),
            provider="openai-codex", model=MODELS["spark"], platform="subagent",
            api_call_count=1, turn_id="spark-design-child",
        )
        self.assertEqual(result["metadata"]["tier"], "sol")
        self.assertIn("Sol-only", result["reason"])

    @patch("model_router._log_decision")
    def test_text_only_design_terra_subagent_is_rerouted_to_sol(self, mocked_log):
        result = route_llm_request(
            request=chat_request("Review and implement the UI layout CSS."),
            provider="openai-codex", model=MODELS["terra"], platform="subagent",
            api_call_count=1, turn_id="terra-design-child",
        )
        self.assertEqual(result["metadata"]["tier"], "sol")
        self.assertIn("Sol-only", result["reason"])

    @patch("model_router._log_decision")
    def test_spark_subagent_non_read_only_implementation_is_promoted_to_terra(self, mocked_log):
        result = route_llm_request(
            request=chat_request("Modify the parser implementation and write the patch."),
            provider="openai-codex", model=MODELS["spark"], platform="subagent",
            api_call_count=1, turn_id="spark-implementation-child",
        )
        self.assertEqual(result["metadata"]["tier"], "terra")
        self.assertIn("read-only", result["reason"])

    def test_spark_quota_429_uses_retry_call_when_provided(self):
        next_calls = []
        retry_calls = []

        def next_call(request):
            next_calls.append(request["model"])
            raise RuntimeError("HTTP 429: weekly quota exhausted for gpt-5.3-codex-spark")

        def retry_call(request):
            retry_calls.append(request["model"])
            self.assertEqual(request["reasoning"]["effort"], "medium")
            return "recovered by Luna"

        result = run_llm_with_transient_failover(
            request={**chat_request("[spark] Bounded read-only source inspection."), "model": MODELS["spark"]},
            next_call=next_call,
            retry_call=retry_call,
            provider="openai-codex",
        )

        self.assertEqual(result, "recovered by Luna")
        self.assertEqual(next_calls, [MODELS["spark"]])
        self.assertEqual(retry_calls, [MODELS["luna"]])

    def test_spark_quota_429_routes_to_luna_at_medium_effort(self):
        calls = []

        def next_call(request):
            calls.append(request["model"])
            if len(calls) == 1:
                raise RuntimeError("HTTP 429: weekly quota exhausted for gpt-5.3-codex-spark")
            self.assertEqual(request["reasoning"]["effort"], "medium")
            return "recovered by Luna"

        result = run_llm_with_transient_failover(
            request={**chat_request("[spark] Bounded read-only source inspection."), "model": MODELS["spark"]},
            next_call=next_call,
            provider="openai-codex",
        )

        self.assertEqual(result, "recovered by Luna")
        self.assertEqual(calls, [MODELS["spark"], MODELS["luna"]])

    def test_weekly_spark_quota_exhaustion_sticks_for_later_calls_in_same_turn(self):
        turn_id = "spark-weekly-quota-sticky-turn"

        def exhausted(request):
            raise RuntimeError("HTTP 429: weekly quota exhausted for gpt-5.3-codex-spark")

        run_llm_with_transient_failover(
            request={**chat_request("[spark] Bounded read-only source inspection."), "model": MODELS["spark"]},
            next_call=exhausted,
            retry_call=lambda request: "recovered",
            provider="openai-codex",
            turn_id=turn_id,
        )

        routed = route_llm_request(
            request=chat_request("[spark] Continue the same bounded read-only inspection."),
            provider="openai-codex",
            model=MODELS["spark"],
            platform="subagent",
            api_call_count=2,
            turn_id=turn_id,
        )

        self.assertEqual(routed["metadata"]["tier"], "luna")
        self.assertEqual(routed["request"]["model"], MODELS["luna"])
        self.assertEqual(routed["request"]["reasoning"]["effort"], "medium")
        self.assertEqual(routed["reason"], "Spark quota already exhausted for this turn")

    def test_generic_429_rate_limit_does_not_change_transient_failover_behavior(self):
        calls = []

        def next_call(request):
            calls.append(request["model"])
            raise RuntimeError("HTTP 429: rate limit exceeded; retry after 30 seconds")

        with self.assertRaisesRegex(RuntimeError, "rate limit"):
            run_llm_with_transient_failover(
                request={**chat_request("[spark] Bounded read-only source inspection."), "model": MODELS["spark"]},
                next_call=next_call,
                provider="openai-codex",
            )
        self.assertEqual(calls, [MODELS["spark"]])

    def test_spark_weekly_quota_429_fails_over_once_to_luna(self):
        calls = []

        def next_call(request):
            calls.append(request["model"])
            if len(calls) == 1:
                raise RuntimeError("HTTP 429: weekly limit reached for this account")
            self.assertEqual(request["reasoning"]["effort"], "medium")
            return "luna-recovered"

        result = run_llm_with_transient_failover(
            request={**chat_request("[spark] Bounded read-only source inspection."), "model": MODELS["spark"]},
            next_call=next_call,
            provider="openai-codex",
        )

        self.assertEqual(result, "luna-recovered")
        self.assertEqual(calls, [MODELS["spark"], MODELS["luna"]])

    def test_non_spark_quota_429_does_not_trigger_luna_failover(self):
        calls = []

        def next_call(request):
            calls.append(request["model"])
            raise RuntimeError("HTTP 429: weekly quota exhausted")

        with self.assertRaisesRegex(RuntimeError, "weekly quota"):
            run_llm_with_transient_failover(
                request={**chat_request("[terra] Continue."), "model": MODELS["terra"]},
                next_call=next_call,
                provider="openai-codex",
            )
        self.assertEqual(calls, [MODELS["terra"]])

    def test_transient_fallback_from_sol_with_an_image_skips_spark(self):
        calls = []

        def next_call(request):
            calls.append(request["model"])
            if len(calls) == 1:
                raise RuntimeError("HTTP 503: upstream connect error")
            return "recovered"

        result = run_llm_with_transient_failover(
            request={**chat_request_with_image("Inspect this screenshot."), "model": MODELS["sol"]},
            next_call=next_call,
            provider="openai-codex",
        )
        self.assertEqual(result, "recovered")
        self.assertEqual(calls, [MODELS["sol"], MODELS["terra"]])

    def test_middleware_bypasses_other_providers_and_unrelated_models(self):
        request = chat_request("Szia!")
        self.assertIsNone(route_llm_request(
            request=request,
            provider="anthropic",
            model="claude-sonnet-4-6",
            api_call_count=1,
        ))
        self.assertIsNone(route_llm_request(
            request=request,
            provider="openai-codex",
            model="gpt-5.4",
            api_call_count=1,
        ))

    def test_transient_503_immediately_retries_once_with_a_different_model(self):
        calls = []

        def next_call(request):
            calls.append(request["model"])
            if len(calls) == 1:
                raise RuntimeError("HTTP 503: upstream connect error or disconnect/reset before headers")
            return "recovered"

        result = run_llm_with_transient_failover(
            request={**chat_request("[sol] Folytasd a fejlesztést."), "model": MODELS["sol"]},
            next_call=next_call,
            provider="openai-codex",
        )

        self.assertEqual(result, "recovered")
        self.assertEqual(calls, [MODELS["sol"], MODELS["spark"]])

    def test_sol_design_transient_failure_never_falls_back_to_another_tier(self):
        calls = []

        def next_call(request):
            calls.append(request["model"])
            raise RuntimeError("HTTP 503: upstream connect error")

        with self.assertRaisesRegex(RuntimeError, "503"):
            run_llm_with_transient_failover(
                request={**chat_request("Implement the responsive CSS card design."), "model": MODELS["sol"]},
                next_call=next_call,
                provider="openai-codex",
            )
        self.assertEqual(calls, [MODELS["sol"]])


    def test_non_transient_error_does_not_retry_with_another_model(self):
        calls = []

        def next_call(request):
            calls.append(request["model"])
            raise RuntimeError("HTTP 401: invalid authentication")

        with self.assertRaisesRegex(RuntimeError, "401"):
            run_llm_with_transient_failover(
                request={**chat_request("[sol] Folytasd a fejlesztést."), "model": MODELS["sol"]},
                next_call=next_call,
                provider="openai-codex",
            )
        self.assertEqual(calls, [MODELS["sol"]])

    def test_credential_and_password_actions_route_to_sol_in_hungarian_inflections(self):
        prompts = (
            "Állítsd be a belépési adatokat és jelszót.",
            "Add meg a demo account jelszavát.",
        )
        self.assertTrue(all(classify_request(chat_request(prompt), 1).tier == "sol" for prompt in prompts))

    @patch("model_router._log_decision")
    def test_forced_planner_schema_requires_an_orchestrator_child(self, mocked_log):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True, "provider": "openai-codex", "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {"enabled": True, "max_tasks": 3, "path": str(Path(temp_dir) / "orchestration.jsonl")},
                "shadow": {"enabled": False},
            }
            request = chat_request(ACTIONABLE_TERRA_PROMPT)
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {"type": "object", "properties": {"goal": {"type": "string"}, "role": {"type": "string"}}}}]
            with patch("model_router._load_config", return_value=cfg):
                routed = route_llm_request(request=request, provider="openai-codex", model=MODELS["terra"], api_call_count=1, turn_id="planner-schema-turn")
        schema = routed["request"]["tools"][0]["parameters"]
        self.assertEqual(schema["properties"]["role"]["enum"], ["orchestrator"])
        self.assertIn("role", schema["required"])
        self.assertIn("context", schema["required"])
        self.assertEqual(len(schema["properties"]["context"]["enum"]), 1)
        contract = schema["properties"]["context"]["enum"][0]
        # "terra": the conductor tier follows default_model, which this config omits.
        # The fallback used to differ between the contract builder ("qwen") and the
        # orchestration gate ("terra") — one key, two answers. Now there is one.
        self.assertIn("terra planning conductor", contract)
        self.assertIn("Do not perform design analysis or design implementation", contract)
        self.assertIn("Sol", contract)
        self.assertIn("non-design read-only", contract)

    @patch("model_router._log_decision")
    def test_terra_orchestrator_subagent_is_not_rewritten_to_spark(self, mocked_log):
        result = route_llm_request(
            request=chat_request("Assess the task and delegate only useful bounded workers."),
            provider="openai-codex", model=MODELS["terra"], platform="subagent",
            api_call_count=1, turn_id="terra-planner:sa-0-child:turn",
        )
        self.assertEqual(result["metadata"]["tier"], "terra")

    @patch("model_router._log_decision")
    def test_every_actionable_terra_turn_forces_a_planner_without_regex_keywords(self, mocked_log):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True, "provider": "openai-codex", "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {"enabled": True, "max_tasks": 3, "path": str(Path(temp_dir) / "orchestration.jsonl")},
                "shadow": {"enabled": False},
            }
            request = chat_request(ACTIONABLE_TERRA_PROMPT)
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                routed = route_llm_request(
                    request=request, provider="openai-codex", model=MODELS["terra"],
                    api_call_count=1, turn_id="short-actionable-turn",
                )
        self.assertEqual(routed["request"]["tool_choice"], "required")
        preflight = routed["request"]["messages"][-1]["content"]
        self.assertIn("Create a structured dispatch plan", preflight)
        self.assertIn("zero to 3", preflight)

    @patch("model_router._log_decision")
    def test_terra_turn_below_min_chars_is_not_fanned_out(self, mocked_log):
        """A turn too small to decompose must not pay for a planner plus workers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True, "provider": "openai-codex", "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {
                    "enabled": True, "max_tasks": 3, "min_chars": 180,
                    "path": str(Path(temp_dir) / "orchestration.jsonl"),
                },
                "shadow": {"enabled": False},
            }
            request = chat_request("Nézd meg, mi a baj.")
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                routed = route_llm_request(
                    request=request, provider="openai-codex", model=MODELS["terra"],
                    api_call_count=1, turn_id="below-min-chars-turn",
                )
            orchestration_log = Path(temp_dir) / "orchestration.jsonl"
            events = [
                json.loads(line)
                for line in orchestration_log.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(routed["metadata"]["tier"], "terra")
        self.assertNotIn("tool_choice", routed["request"])
        # The declined turn is now logged with its reason; only a forced
        # dispatch would mean the gate leaked.
        self.assertFalse(
            [event for event in events if event["event"] == "preflight_forced"],
            "a sub-min_chars turn must not emit a preflight dispatch",
        )
        self.assertEqual(
            [event["skip_reason"].split(":")[0] for event in events],
            ["prompt_shorter_than_min_chars"],
        )

    @patch("model_router._log_decision")
    def test_sol_preflight_ignores_min_chars_because_it_is_not_a_fan_out(self, mocked_log):
        """min_chars gates worker fan-out; the Sol design review is not fan-out."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True, "provider": "openai-codex", "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {
                    "enabled": True, "max_tasks": 3, "min_chars": 180,
                    "path": str(Path(temp_dir) / "orchestration.jsonl"),
                },
                "sol_opus5_preflight": {
                    "enabled": True, "owner": "sol", "bridge_model": "claude-opus-5-5",
                    "require_successful_auth_probe": True,
                },
                "shadow": {"enabled": False},
            }
            prompt = "Készíts UX/UI preflight értékelést egy bejelentkezési oldal vizuális elrendezéséről."
            self.assertLess(len(prompt), 180, "fixture must sit below min_chars to be meaningful")
            request = chat_request(prompt)
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                result = route_llm_request(
                    request=request, provider="openai-codex", model=MODELS["terra"],
                    api_call_count=1, turn_id="sol-short-preflight-turn",
                )
        self.assertEqual(result["metadata"]["tier"], "sol")
        self.assertIn("claude-opus-5-5", json.dumps(result["request"], ensure_ascii=False))

    @patch("model_router._log_decision")
    def test_long_terra_loop_is_rescued_even_when_original_prompt_has_no_regex_markers(self, mocked_log):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True, "provider": "openai-codex", "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {
                    "enabled": True, "max_tasks": 3, "rescue_min_calls": 6,
                    "path": str(Path(temp_dir) / "orchestration.jsonl"),
                },
                "shadow": {"enabled": False},
            }
            request = chat_request(ACTIONABLE_TERRA_PROMPT, with_tool_result=True)
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                routed = route_llm_request(
                    request=request, provider="openai-codex", model=MODELS["terra"],
                    api_call_count=6, turn_id="generic-long-loop-turn",
                )
        self.assertEqual(routed["request"]["tool_choice"], "required")

    def test_first_non_design_coding_call_executes_opus5_instead_of_openai(self):
        with tempfile.TemporaryDirectory() as repo:
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "high", "luna": "low"},
                "coding_agent": {
                    "enabled": True,
                    "tier": "opus5",
                    "model": "claude-opus-5-5",
                    "default_repo": repo,
                    "max_turns": 8,
                    "max_budget_usd": 5.0,
                },
            }
            result = {
                "result": "OPUS IMPLEMENTATION COMPLETE",
                "effective_model": "claude-opus-5-5",
                "usage": {"input_tokens": 17, "output_tokens": 9},
            }
            with patch("model_router._load_config", return_value=cfg), patch(
                "model_router._run_opus5_bridge", return_value=result
            ) as bridge:
                response = run_llm_with_transient_failover(
                    request=responses_request("Implement the parser fix and add tests."),
                    next_call=lambda _request: self.fail("OpenAI downstream must not run"),
                    provider="openai-codex",
                    api_mode="codex_responses",
                    api_call_count=1,
                    turn_id="coding-turn",
                )

        bridge.assert_called_once()
        self.assertEqual(response.model, "claude-opus-5-5")
        self.assertEqual(response.output[0].content[0].text, "OPUS IMPLEMENTATION COMPLETE")

    def test_explicit_opus_review_executes_read_only_for_non_coding_review(self):
        with tempfile.TemporaryDirectory() as repo:
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "coding_agent": {
                    "enabled": True,
                    "canonical_model": "claude-opus-5-5",
                    "default_repo": repo,
                    "reviewer": {"enabled": True, "max_chars": 8000},
                },
            }
            result = {"result": "OPUS REVIEW COMPLETE", "effective_model": "claude-opus-5-5"}
            prompt = "[opus-review] Review the access-control proposal for missing risks. Do not modify files."
            with patch("model_router._load_config", return_value=cfg), patch(
                "model_router.shutil.which", return_value="/usr/bin/claude"
            ), patch("model_router._recent_verified_opus5_route", return_value=True), patch(
                "model_router._run_opus5_bridge", return_value=result
            ) as bridge:
                response = run_llm_with_transient_failover(
                    request=responses_request(prompt),
                    original_request=responses_request(prompt),
                    next_call=lambda _request: self.fail("OpenAI downstream must not run"),
                    provider="openai-codex",
                    api_mode="codex_responses",
                    api_call_count=1,
                    turn_id="opus-review-turn",
                )

        bridge.assert_called_once()
        self.assertFalse(bridge.call_args.kwargs["write"])
        self.assertTrue(bridge.call_args.kwargs["review"])
        self.assertEqual(response.model, "claude-opus-5-5")
        self.assertEqual(response.output[0].content[0].text, "OPUS REVIEW COMPLETE")

    def _delegated_review_cfg(self, repo, **overrides):
        policy = {"enabled": True, "max_chars": 8000, "models": ["opus", "sonnet"]}
        policy.update(overrides)
        return {
            "enabled": True,
            "provider": "openai-codex",
            "models": MODELS, "callable": {**CALLABLE, "sonnet5": True},
            "coding_agent": {
                # Deliberately off: the delegated path must not depend on the
                # switch that also arms the label-free coding classifier.
                "enabled": False,
                "canonical_model": "claude-opus-5-5",
                "default_repo": repo,
                "delegated_review": policy,
            },
        }

    def test_delegated_sonnet_review_leaf_runs_on_claude(self):
        """A read-only review leaf is the one job that can leave the Codex
        account entirely: Claude is reachable only through its own CLI, so the
        leaf's single call becomes a bridge subprocess and its verdict becomes
        the leaf's answer."""
        with tempfile.TemporaryDirectory() as repo:
            result = {"result": "SONNET REVIEW COMPLETE", "effective_model": "claude-sonnet-5-5"}
            prompt = "[sonnet-review] Review the pending calendar diff for regressions. Report only."
            with patch("model_router._load_config", return_value=self._delegated_review_cfg(repo)), patch(
                "model_router.shutil.which", return_value="/usr/bin/claude"
            ), patch("model_router._run_opus5_bridge", return_value=result) as bridge:
                response = run_llm_with_transient_failover(
                    request=responses_request(prompt),
                    original_request=responses_request(prompt),
                    next_call=lambda _request: self.fail("the Codex downstream must not run"),
                    provider="openai-codex",
                    api_mode="codex_responses",
                    api_call_count=1,
                    platform="subagent",
                    turn_id="session:sa-1-child:turn",
                )

        bridge.assert_called_once()
        self.assertEqual(bridge.call_args.kwargs["model"], "sonnet")
        self.assertTrue(bridge.call_args.kwargs["review"])
        self.assertFalse(bridge.call_args.kwargs["write"])
        self.assertEqual(response.model, "claude-sonnet-5-5")

    def test_a_root_turn_is_never_diverted_into_the_bridge(self):
        """The documented hazard of this bridge is that it captures the first
        call of a turn. That only matters for the parent that still has to plan,
        so the delegated path admits delegated workers and nothing else."""
        with tempfile.TemporaryDirectory() as repo:
            prompt = "[opus-review] Review the access-control proposal. Report only."
            sentinel = object()
            with patch("model_router._load_config", return_value=self._delegated_review_cfg(repo)), patch(
                "model_router.shutil.which", return_value="/usr/bin/claude"
            ), patch("model_router._run_opus5_bridge") as bridge:
                response = run_llm_with_transient_failover(
                    request=responses_request(prompt),
                    original_request=responses_request(prompt),
                    next_call=lambda _request: sentinel,
                    provider="openai-codex",
                    api_mode="codex_responses",
                    api_call_count=1,
                    turn_id="plain-root-turn",
                )

        bridge.assert_not_called()
        self.assertIs(response, sentinel)

    def test_a_claude_tier_left_out_of_the_config_is_not_reachable(self):
        with tempfile.TemporaryDirectory() as repo:
            cfg = self._delegated_review_cfg(repo, models=["opus"])
            prompt = "[sonnet-review] Review the pending calendar diff. Report only."
            sentinel = object()
            with patch("model_router._load_config", return_value=cfg), patch(
                "model_router.shutil.which", return_value="/usr/bin/claude"
            ), patch("model_router._run_opus5_bridge") as bridge:
                response = run_llm_with_transient_failover(
                    request=responses_request(prompt),
                    original_request=responses_request(prompt),
                    next_call=lambda _request: sentinel,
                    provider="openai-codex",
                    api_mode="codex_responses",
                    api_call_count=1,
                    platform="subagent",
                    turn_id="session:sa-1-child:turn",
                )

        bridge.assert_not_called()
        self.assertIs(response, sentinel)

    def test_runtime_opus_bridge_uses_the_real_dispatch_entrypoint(self):
        cfg = {"coding_agent": {"timeout_seconds": 300, "max_turns": 12, "max_budget_usd": 2.5}}
        expected = {"result": "ok", "model": "claude-opus-5-5"}
        with tempfile.TemporaryDirectory() as repo, patch(
            "model_router.claude_opus_bridge.dispatch", return_value=expected
        ) as dispatch:
            from model_router import _run_opus5_bridge

            result = _run_opus5_bridge(
                repo=repo,
                task="Inspect the parser.",
                write=False,
                cfg=cfg,
                turn_id="parent:entrypoint-test",
            )

        dispatch.assert_called_once()
        self.assertEqual(result, expected)
        self.assertEqual(dispatch.call_args.kwargs["parent_session_id"], "parent")
        self.assertEqual(dispatch.call_args.kwargs["parent_turn_id"], "parent:entrypoint-test")
        self.assertEqual(dispatch.call_args.kwargs["max_turns"], 12)
        self.assertEqual(dispatch.call_args.kwargs["max_budget_usd"], 2.5)

    def test_design_coding_call_stays_on_existing_sol_route(self):
        cfg = {
            "enabled": True,
            "provider": "openai-codex",
            "models": MODELS, "callable": CALLABLE,
            "coding_agent": {"enabled": True, "tier": "opus5", "model": "claude-opus-5-5"},
        }
        sentinel = object()
        with patch("model_router._load_config", return_value=cfg), patch(
            "model_router._run_opus5_bridge"
        ) as bridge:
            response = run_llm_with_transient_failover(
                request=responses_request("Implement the CSS layout from this design."),
                next_call=lambda _request: sentinel,
                provider="openai-codex",
                api_mode="codex_responses",
                api_call_count=1,
                turn_id="design-turn",
            )

        bridge.assert_not_called()
        self.assertIs(response, sentinel)

    @patch("model_router._log_decision")
    def test_explicit_bounded_opus_ui_request_skips_planner_preflight(self, mocked_log):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "coding_agent": {"enabled": True, "explicit_ui": {"enabled": True, "max_chars": 1200}},
                "orchestration": {
                    "enabled": True,
                    "max_tasks": 3,
                    "path": str(Path(temp_dir) / "orchestration.jsonl"),
                },
                "sol_opus5_preflight": {
                    "enabled": True,
                    "owner": "sol",
                    "bridge_model": "claude-opus-5-5",
                    "require_successful_auth_probe": True,
                },
                "shadow": {"enabled": False},
            }
            request = chat_request("Ha lehet, az Opus dolgozzon ezen a kis UI címke javításon.")
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                result = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="direct-explicit-opus-ui",
                )

        self.assertEqual(result["metadata"]["tier"], "sol")
        self.assertNotIn("tool_choice", result["request"])
        self.assertNotIn("INTERNAL SOL + CLAUDE OPUS 5 PREFLIGHT", json.dumps(result["request"]))

    def test_explicit_bounded_opus_ui_request_executes_one_verified_bridge_call(self):
        with tempfile.TemporaryDirectory() as repo, tempfile.TemporaryDirectory() as temp_dir:
            route_log = Path(temp_dir) / "router.jsonl"
            route_log.write_text(json.dumps({
                "timestamp": "2099-01-01T00:00:00+00:00",
                "tier": "opus5",
                "model": "claude-opus-5-5",
                "reason": "verified probe",
            }) + "\n", encoding="utf-8")
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "logging": {"enabled": True, "path": str(route_log)},
                "coding_agent": {
                    "enabled": True,
                    "canonical_model": "claude-opus-5-5",
                    "default_repo": repo,
                    "timeout_seconds": 300,
                    "explicit_ui": {
                        "enabled": True,
                        "max_chars": 1200,
                        "require_recent_verified_probe_seconds": 86400,
                    },
                },
            }
            result = {
                "result": "OPUS UI FIX COMPLETE",
                "effective_model": "claude-opus-5-5",
                "usage": {"input_tokens": 11, "output_tokens": 7},
            }
            with patch("model_router._load_config", return_value=cfg), patch(
                "model_router.shutil.which", return_value="/usr/bin/claude"
            ), patch("model_router._run_opus5_bridge", return_value=result) as bridge:
                response = run_llm_with_transient_failover(
                    request=responses_request("Ha lehet, az Opus dolgozzon ezen a kis UI címke javításon."),
                    original_request=responses_request("Ha lehet, az Opus dolgozzon ezen a kis UI címke javításon."),
                    next_call=lambda _request: self.fail("Sol/OpenAI downstream must not run"),
                    provider="openai-codex",
                    api_mode="codex_responses",
                    api_call_count=1,
                    turn_id="direct-explicit-opus-ui",
                )

        bridge.assert_called_once()
        self.assertTrue(bridge.call_args.kwargs["task"].startswith("[opus5]"))
        self.assertEqual(response.model, "claude-opus-5-5")

    def test_explicit_ui_opus_request_without_verified_bridge_stays_on_sol(self):
        cfg = {
            "enabled": True,
            "provider": "openai-codex",
            "models": MODELS, "callable": CALLABLE,
            "logging": {"enabled": True, "path": "/missing/router.jsonl"},
            "coding_agent": {
                "enabled": True,
                "default_repo": "/missing/repo",
                "explicit_ui": {"enabled": True, "max_chars": 1200},
            },
        }
        sentinel = object()
        with patch("model_router._load_config", return_value=cfg), patch(
            "model_router._run_opus5_bridge"
        ) as bridge:
            response = run_llm_with_transient_failover(
                request=responses_request("Let Opus work on this small CSS label fix."),
                original_request=responses_request("Let Opus work on this small CSS label fix."),
                next_call=lambda _request: sentinel,
                provider="openai-codex",
                api_mode="codex_responses",
                api_call_count=1,
                turn_id="unavailable-explicit-opus-ui",
            )

        bridge.assert_not_called()
        self.assertIs(response, sentinel)

    def test_later_agent_loop_call_does_not_launch_another_opus_process(self):
        cfg = {
            "enabled": True,
            "provider": "openai-codex",
            "models": MODELS, "callable": CALLABLE,
            "coding_agent": {"enabled": True, "tier": "opus5", "model": "claude-opus-5-5"},
        }
        sentinel = object()
        with patch("model_router._load_config", return_value=cfg), patch(
            "model_router._run_opus5_bridge"
        ) as bridge:
            response = run_llm_with_transient_failover(
                request=responses_request("Implement the parser fix and add tests."),
                next_call=lambda _request: sentinel,
                provider="openai-codex",
                api_mode="codex_responses",
                api_call_count=2,
                turn_id="coding-turn",
            )

        bridge.assert_not_called()
        self.assertIs(response, sentinel)

    def test_log_timestamp_has_whole_second_precision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "router.jsonl"
            cfg = {"logging": {"enabled": True, "path": str(path)}}
            with patch("model_router.datetime") as mocked_datetime:
                mocked_datetime.now.return_value = __import__("datetime").datetime(
                    2026,
                    7,
                    14,
                    7,
                    1,
                    2,
                    987654,
                    tzinfo=__import__("datetime").timezone.utc,
                )
                _log_decision(
                    RouteDecision("terra", MODELS["terra"], "test"),
                    {
                        "turn_id": "turn-1",
                        "api_call_count": 1,
                        "request": chat_request("Ez egy teszt prompt."),
                    },
                    cfg,
                )
            entry = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(entry["timestamp"], "2026-07-14T07:01:02+00:00")
            self.assertEqual(entry["prompt_preview"], "Ez egy teszt prompt.")

    def test_prompt_preview_is_single_line_and_not_truncated(self):
        text = "Első sor\n  második sor " + "á" * 60
        preview = _prompt_preview(responses_request(text))
        self.assertEqual(preview, "Első sor második sor " + "á" * 60)
        self.assertNotIn("\n", preview)


if __name__ == "__main__":
    unittest.main()


class GoalContractOnTheSchemaTests(unittest.TestCase):
    """The goal rules used to travel only inside the forced preflight.

    A root prompt under `orchestration.min_chars` creates no conductor, so the
    parent dispatched straight from its own toolset with nobody having been told
    what a goal must carry — one twelve-character prompt produced a whole-feature
    goal with no worktree, branch or base commit, and the leaf spent all sixteen
    iterations rediscovering them.
    """

    DELEGATE = {
        "name": "delegate_task",
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "What this subagent should accomplish."},
                        "context": {"type": "string", "description": "Background THIS child needs."},
                    },
                    "required": ["goal"],
                }},
            },
        },
    }

    def _request(self):
        return {
            "model": MODELS["terra"],
            "messages": [{"role": "user", "content": "inplementald"}],
            "tools": [{"name": "terminal"}, json.loads(json.dumps(self.DELEGATE))],
        }

    def _route(self, request):
        with patch("model_router._load_config", side_effect=default_test_config), \
             patch("model_router._log_decision"), \
             patch("model_router._force_terra_supervisor_preflight", return_value=None), \
             patch("model_router._force_shadow_delegation_if_eligible", return_value=None):
            return route_llm_request(
                request=request, provider="openai-codex", model=MODELS["terra"],
                api_call_count=1, turn_id="turn-goal-contract",
            )

    def _goal_description(self, request):
        tool = next(t for t in request["tools"] if t.get("name") == "delegate_task")
        properties = tool["parameters"]["properties"]["tasks"]["items"]["properties"]
        return properties["goal"]["description"]

    def test_the_requirements_reach_a_parent_that_got_no_preflight(self):
        description = self._goal_description(self._route(self._request())["request"])
        self.assertIn("absolute worktree path", description)
        self.assertIn("one finishable artefact", description)

    def test_the_recon_rule_survives_a_turn_with_no_conductor(self):
        """The turn that burned the Opus leaf had no conductor at all: a 41-char
        root prompt fell under orchestration.min_chars, so the parent dispatched
        straight from its own toolset. The schema is the only channel that reaches
        it, which is why the rule is on the goal description and not only in the
        preflight text."""
        description = self._goal_description(self._route(self._request())["request"])
        self.assertIn("read-only recon task first", description)
        self.assertIn("opus5 or sonnet5", description)

    def test_it_is_carried_on_the_schema_not_the_message(self):
        """A middleware edit does not persist into the conversation, and the
        parent may delegate on any call of the turn — an appended sentence would
        have to be repeated on every one of them."""
        routed = self._route(self._request())["request"]
        self.assertNotIn("absolute worktree path", routed["messages"][-1]["content"])

    def test_applying_it_twice_changes_nothing(self):
        from model_router import _with_goal_contract

        once = self._route(self._request())["request"]
        self.assertIsNone(_with_goal_contract(once))

    def test_the_callers_own_schema_is_not_mutated(self):
        request = self._request()
        self._route(request)
        self.assertNotIn("absolute worktree path", self._goal_description(request))

    def test_a_request_without_the_tool_is_untouched(self):
        from model_router import _with_goal_contract

        self.assertIsNone(_with_goal_contract({"tools": [{"name": "terminal"}]}))

    def _task_schema(self, request):
        tool = next(t for t in request["tools"] if t.get("name") == "delegate_task")
        return tool["parameters"]["properties"]["tasks"]["items"]

    def test_context_becomes_required(self):
        """Three descriptions have now lost to something: the built-in
        tie-breaker, the [spark]/[sol] vocabulary, and here to nothing at all.
        Requiring the field makes a context-free call invalid instead."""
        items = self._task_schema(self._route(self._request())["request"])
        self.assertEqual(items["required"], ["goal", "context"])

    def test_the_context_description_says_what_belongs_in_it(self):
        items = self._task_schema(self._route(self._request())["request"])
        description = items["properties"]["context"]["description"]
        self.assertIn("worktree it runs in", description)
        self.assertIn("branch and the commit it builds on", description)

    def test_requiring_it_twice_changes_nothing(self):
        from model_router import _with_goal_contract

        once = self._route(self._request())["request"]
        self.assertIsNone(_with_goal_contract(once))
        self.assertEqual(self._task_schema(once)["required"], ["goal", "context"])

    def test_the_callers_own_required_list_is_not_mutated(self):
        request = self._request()
        self._route(request)
        self.assertEqual(self._task_schema(request)["required"], ["goal"])


class CommitReferenceIsNotAWriteTests(unittest.TestCase):
    """`commit` as a noun, in the base commit the goal contract requires.

    Two changes in the same series collided: `commit` joined the write verbs, and
    the contract began requiring the goal to name the commit it builds on. The
    better the goal, the more certainly a read-only leaf read as mutating — a
    `[spark]` source map ending "at commit 7abc123" was escalated off Spark.
    """

    def test_a_named_base_commit_is_not_an_instruction_to_commit(self):
        for text in (
            "[spark] Map the role/scope logic in /home/x (branch feat/y, at commit 7abc123).",
            "[spark] Inspect /home/x on branch feat/y, base commit 9def456.",
            "[spark] Report the exports in /home/x at the commit it builds on.",
            "[spark] Read /home/x from commit hash abc1234 and list the migrations.",
        ):
            with self.subTest(text=text):
                self.assertTrue(_is_spark_read_only_work(text))

    def test_an_instruction_to_commit_still_is_one(self):
        for text in (
            "[luna] Finish and commit the already-started foundation.",
            "[spark] Fix the parser and commit the change.",
            "[luna] Implementáld és commitold a ledger integrációt.",
        ):
            with self.subTest(text=text):
                self.assertFalse(_is_spark_read_only_work(text))

    def test_a_well_formed_read_only_leaf_keeps_its_label(self):
        """It used to be escalated to Sol; the label survives the contract now."""
        goal = ("[spark] Produce a factual, read-only source map of customer-management "
                "role/scope logic in the Next.js repository at absolute path /home/x "
                "(git branch feat/tenant-wide-customer-access, at commit 7abc123).")
        decision = classify_request(chat_request(goal), 1, allow_plan_label_over_design=True)
        self.assertNotEqual(decision.reason, "consequential Spark task requires Sol")
