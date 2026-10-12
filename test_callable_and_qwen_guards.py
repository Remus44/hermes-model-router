import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_router import RouteDecision, _log_decision, route_llm_request, run_llm_with_transient_failover
from model_router.host_delegation_fixtures import host_delegation


MODELS = {
    "luna": "gpt-6-luna",
    "spark": "gpt-5.3-codex-spark",
    "terra": "gpt-5.6-terra",
    "sol": "gpt-6-sol",
    "qwen": "qwen3.7-plus",
}


def request(text, model="gpt-5.6-terra"):
    return {"model": model, "messages": [{"role": "user", "content": text}]}


def config(**overrides):
    result = {
        "enabled": True,
        "provider": "openai-codex",
        "models": MODELS,
        "default_model": "terra",
        "callable": {"luna": True, "spark": True, "terra": True, "sol": True, "opus5": True, "qwen": True},
        "effort": {"luna": "low", "spark": "low", "terra": "medium", "sol": "medium", "qwen": "medium"},
    }
    result.update(overrides)
    return result


class CallableGuardTests(unittest.TestCase):
    def setUp(self):
        # Stands in for pytest's tmp_path fixture.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)
        self._host_delegation = host_delegation(depth=2)
        self._host_delegation.__enter__()
        self.addCleanup(self._host_delegation.__exit__, None, None, None)

    def test_disabled_opus5_skips_bridge_before_import_or_auth(self):
        cfg = config(
            callable={**config()["callable"], "opus5": False},
            coding_agent={"enabled": True, "default_repo": "/tmp"},
        )
        with patch("model_router._load_config", return_value=cfg), patch("model_router._run_opus5_bridge") as bridge:
            response = run_llm_with_transient_failover(
                request=request("Implement the parser fix and add tests."),
                next_call=lambda value: value,
                provider="openai-codex",
                api_mode="codex_responses",
                api_call_count=1,
            )
        self.assertEqual(response["model"], "gpt-5.6-terra")
        bridge.assert_not_called()

    def test_disabled_spark_is_rerouted_to_a_live_fallback(self):
        cfg = config(
            callable={**config()["callable"], "spark": False},
            fallbacks={"spark": "terra"},
        )
        with patch("model_router._load_config", return_value=cfg):
            routed = route_llm_request(
                request=request("[spark] Inspect the parser configuration only.", MODELS["spark"]),
                provider="openai-codex",
                model=MODELS["spark"],
                platform="subagent",
                api_call_count=1,
            )
        self.assertEqual(routed["metadata"]["tier"], "terra")
        self.assertEqual(routed["request"]["model"], MODELS["terra"])

    def test_qwen_final_request_strips_unsupported_tool_controls(self):
        cfg = config(
            default_model="qwen",
            tier_providers={"qwen": "qwen-token", "terra": "openai-codex"},
        )
        qwen_request = request("Continue this conversation.", MODELS["qwen"])
        qwen_request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
        qwen_request["tool_choice"] = "required"
        qwen_request["parallel_tool_calls"] = False
        with patch("model_router._load_config", return_value=cfg):
            routed = route_llm_request(
                request=qwen_request,
                provider="openai-codex",
                model=MODELS["qwen"],
                api_call_count=2,
            )
        self.assertEqual(routed["metadata"]["tier"], "qwen")
        self.assertEqual(routed["request"]["tools"], qwen_request["tools"])
        self.assertNotIn("tool_choice", routed["request"])
        self.assertNotIn("parallel_tool_calls", routed["request"])

    def test_disabled_and_missing_callable_tiers_reroute_before_dispatch(self):
        for tier in ("spark", "luna", "sol"):
            live = {**config()["callable"], tier: False}
            cfg = config(callable=live, fallbacks={tier: "terra"})
            decision = RouteDecision(tier, MODELS[tier], "explicit test route", "medium")
            with (
                patch("model_router._load_config", return_value=cfg),
                patch("model_router.classify_request", return_value=decision),
            ):
                routed = route_llm_request(
                    request=request("Test disabled routing.", MODELS[tier]),
                    provider="openai-codex",
                    model=MODELS[tier],
                    api_call_count=1,
                )
            self.assertEqual(routed["metadata"]["tier"], "terra")
            self.assertEqual(routed["request"]["model"], MODELS["terra"])

        live = config()["callable"].copy()
        live.pop("luna")
        cfg = config(callable=live, fallbacks={"luna": "terra"})
        decision = RouteDecision("luna", MODELS["luna"], "missing callable entry", "low")
        with (
            patch("model_router._load_config", return_value=cfg),
            patch("model_router.classify_request", return_value=decision),
        ):
            routed = route_llm_request(
                request=request("Test missing routing entry.", MODELS["luna"]),
                provider="openai-codex",
                model=MODELS["luna"],
                api_call_count=1,
            )
        self.assertEqual(routed["metadata"]["tier"], "terra")

    def test_no_callable_fallback_fails_closed_before_dispatch(self):
        cfg = config(
            default_model="terra",
            callable={tier: False for tier in config()["callable"]},
            fallbacks={"qwen": "terra"},
        )
        decision = RouteDecision("qwen", MODELS["qwen"], "explicit test route", "medium")
        with (
            patch("model_router._load_config", return_value=cfg),
            patch("model_router.classify_request", return_value=decision),
            self.assertRaisesRegex(RuntimeError, "No enabled ModelRouter tier"),
        ):
            route_llm_request(
                request=request("Must fail closed.", MODELS["qwen"]),
                provider="qwen-token",
                model=MODELS["qwen"],
                api_call_count=1,
            )

    def test_qwen_sanitization_runs_after_forced_preflight_rewrite(self):
        cfg = config(
            default_model="terra",
            tier_providers={"qwen": "qwen-token", "terra": "openai-codex"},
        )
        qwen_request = request("Implement and verify this parser change.", MODELS["qwen"])
        qwen_request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
        forced = {
            **qwen_request,
            "model": MODELS["terra"],
            "tool_choice": "required",
            "parallel_tool_calls": True,
            "reasoning": {"effort": "medium"},
        }
        terra = RouteDecision("terra", MODELS["terra"], "forced supervisor", "medium")
        with (
            patch("model_router._load_config", return_value=cfg),
            patch("model_router.classify_request", return_value=terra),
            patch("model_router._force_terra_supervisor_preflight", return_value=forced) as preflight,
        ):
            routed = route_llm_request(
                request=qwen_request,
                provider="qwen-token",
                model=MODELS["qwen"],
                api_call_count=1,
            )

        preflight.assert_called_once()
        self.assertEqual(routed["metadata"]["tier"], "qwen")
        self.assertEqual(routed["request"]["model"], MODELS["qwen"])
        self.assertEqual(routed["request"]["tools"], qwen_request["tools"])
        self.assertNotIn("tool_choice", routed["request"])
        self.assertNotIn("parallel_tool_calls", routed["request"])
        self.assertNotIn("reasoning", routed["request"])

    def test_root_parent_is_pinned_when_classifier_wants_sol_worker(self):
        """A user-facing conversation must not silently become a cold Sol turn."""
        cfg = config(
            session_policy={"pin_root_parent": True},
            orchestration={"enabled": True, "min_chars": 1, "max_tasks": 3},
        )
        root_request = request("Készíts CSS elrendezést a kártyához.")
        root_request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
        with patch("model_router._load_config", return_value=cfg):
            routed = route_llm_request(
                request=root_request,
                provider="openai-codex",
                model=MODELS["terra"],
                api_call_count=1,
                turn_id="stable-parent-turn",
            )
        self.assertEqual(routed["metadata"]["tier"], "terra")
        self.assertEqual(routed["request"]["model"], MODELS["terra"])
        self.assertIn("pinned", routed["reason"])
        # A pinned root parent still gets the forced delegation preflight. This
        # assertion used to read assertNotIn, encoding the pin_root_parent gate
        # that c7d59eb (2026-09-07) deliberately removed from
        # _orchestration_eligible -- that gate was exactly what stopped a stable
        # parent from delegating at all. The pin decides *who answers the user*,
        # not whether the turn delegates, so tool_choice belongs here.
        self.assertEqual(routed["request"]["tool_choice"], "required")

    def test_route_log_uses_bounded_redacted_preview_when_policy_enabled(self):
        path = self.tmp_path / "router.jsonl"
        cfg = {
            "logging": {
                "enabled": True,
                "path": str(path),
                "prompt_preview_chars": 24,
                "redact_prompt_preview": True,
            }
        }
        _log_decision(
            RouteDecision("terra", MODELS["terra"], "test"),
            {"turn_id": "safe-log-turn", "request": request("token=super-secret-value " + "x" * 80)},
            cfg,
        )
        record = json.loads(path.read_text())
        self.assertNotIn("super-secret-value", record["prompt_preview"])
        self.assertLessEqual(len(record["prompt_preview"]), 24)


# ── Qwen orchestrator delegation ─────────────────────────────────────────────
#
# Hermes converts to the provider wire format in ``build_api_kwargs`` *before*
# llm_request middleware runs, so an ``anthropic_messages`` route reaches the
# router already Anthropic-shaped: ``tools[].input_schema`` instead of
# ``function.parameters``, and ``text`` content blocks instead of
# ``input_text``.  The orchestration preflight was written for the Codex
# Responses shape only, which is why selecting Qwen as the orchestrator
# produced ``preflight_forced`` events with no delegated child.

ORCH_PROMPT = (
    "Nézd meg, mi a baj. Tegnap óta más az eredmény, és nem tudom eldönteni, hogy a bemenet "
    "változott-e meg vagy a feldolgozás. Nézd át a vonatkozó részeket, és mondd meg, mit találsz, "
    "mielőtt bármit módosítanánk rajta."
)

TOKENPLAN_URL = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/apps/anthropic"


def anthropic_request(text):
    """A request in the shape the middleware actually sees for a Qwen route."""
    return {
        "model": MODELS["qwen"],
        "max_tokens": 8192,
        "system": "You are Hermes.",
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
        "tools": [
            {
                "name": "delegate_task",
                "description": "Delegate a bounded task.",
                "input_schema": {
                    "type": "object",
                    "properties": {"goal": {"type": "string"}},
                    "required": ["goal"],
                },
            },
            {"name": "read_file", "description": "Read", "input_schema": {"type": "object", "properties": {}}},
            {"name": "bash", "description": "Run", "input_schema": {"type": "object", "properties": {}}},
        ],
    }


def orchestration_config(default_model, path):
    return config(
        default_model=default_model,
        tier_providers={
            "qwen": "qwen-token", "terra": "openai-codex", "sol": "openai-codex",
            "spark": "openai-codex", "luna": "openai-codex", "opus5": "openai-codex",
        },
        orchestration={"enabled": True, "min_chars": 40, "max_tasks": 1, "path": str(path)},
        shadow={"enabled": False},
        session_policy={"pin_root_parent": True},
    )


def route_qwen_preflight(tmp_path, turn_id):
    with patch("model_router._load_config", return_value=orchestration_config("qwen", tmp_path)):
        return route_llm_request(
            request=anthropic_request(ORCH_PROMPT),
            provider="qwen-token",
            model=MODELS["qwen"],
            api_mode="anthropic_messages",
            base_url=TOKENPLAN_URL,
            api_call_count=1,
            turn_id=turn_id,
        )


class QwenOrchestratorPreflightTests(unittest.TestCase):
    def setUp(self):
        # Stands in for pytest's tmp_path fixture.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)
        self._host_delegation = host_delegation(depth=2)
        self._host_delegation.__enter__()
        self.addCleanup(self._host_delegation.__exit__, None, None, None)

    def test_qwen_preflight_keeps_the_full_toolset(self):
        """TokenPlan rejects ``tool_choice`` entirely, so the preflight cannot be
        enforced there. Amputating the toolset to a single *optional* tool left the
        parent unable to act at all — it answered in prose and delegated nothing."""
        routed = route_qwen_preflight(self.tmp_path / "orch.jsonl", "qwen-preflight-toolset")["request"]
        self.assertEqual([tool["name"] for tool in routed["tools"]], ["delegate_task", "read_file", "bash"])
        self.assertNotIn("tool_choice", routed)
        self.assertNotIn("parallel_tool_calls", routed)

    def test_qwen_preflight_hardens_the_anthropic_input_schema(self):
        """The planner contract lives in ``input_schema`` on this wire shape; a
        ``parameters``-only lookup silently skipped it and produced a plain leaf."""
        routed = route_qwen_preflight(self.tmp_path / "orch.jsonl", "qwen-preflight-schema")["request"]
        delegate = next(tool for tool in routed["tools"] if tool["name"] == "delegate_task")
        schema = delegate["input_schema"]
        self.assertEqual(schema["properties"]["role"]["enum"], ["orchestrator"])
        self.assertIn("planning conductor", schema["properties"]["context"]["enum"][0])
        self.assertEqual(set(schema["required"]), {"goal", "role", "context"})

    def test_qwen_preflight_appends_a_valid_anthropic_content_block(self):
        """``input_text`` is a Responses-API part type. Sending it to an Anthropic
        endpoint either 400s the call or drops the block, so the orchestrator
        instruction never reached the model."""
        routed = route_qwen_preflight(self.tmp_path / "orch.jsonl", "qwen-preflight-block")["request"]
        blocks = routed["messages"][-1]["content"]
        self.assertEqual({block["type"] for block in blocks}, {"text"})
        self.assertIn("INTERNAL ORCHESTRATOR PREFLIGHT", blocks[-1]["text"])
        # This host schema has no model argument. Qwen is the external parent,
        # while the planner child follows the configured Codex delegate route.
        self.assertIn("[luna]", blocks[-1]["text"])

    def test_preflight_keeps_routing_policy_out_of_the_goal(self):
        """The conductor is re-classified from its own goal text, so routing policy
        repeated there is read as a description of the work. A goal whose only
        design token came from the boilerplate phrase "visual/ui analysis" pinned
        every [terra] orchestrator onto Sol, and Sol then owned both the conducting
        and the design leaf it was supposed to delegate."""
        routed = route_qwen_preflight(self.tmp_path / "orch.jsonl", "goal-content-contract")["request"]
        instruction = routed["messages"][-1]["content"][-1]["text"]
        self.assertIn("Do not restate this routing policy inside the goal", instruction)
        self.assertIn("objective and acceptance criteria only", instruction)
        # The policy must still reach the conductor -- just through the immutable
        # contract, which lands in its system prompt rather than its goal.
        delegate = next(tool for tool in routed["tools"] if tool["name"] == "delegate_task")
        contract = delegate["input_schema"]["properties"]["context"]["enum"][0]
        self.assertIn("planning conductor", contract)
        self.assertIn("[sol]", contract)

    def test_codex_preflight_still_forces_one_tool(self):
        """Regression guard: the Codex Responses route can force a tool call, and
        must keep doing so."""
        codex_request = {
            "model": MODELS["terra"],
            "input": [{"role": "user", "content": [{"type": "input_text", "text": ORCH_PROMPT}]}],
            "tools": [
                {"type": "function", "name": "delegate_task", "parameters": {"type": "object", "properties": {}}},
                {"type": "function", "name": "bash", "parameters": {"type": "object", "properties": {}}},
            ],
        }
        with patch("model_router._load_config",
                   return_value=orchestration_config("terra", self.tmp_path / "orch.jsonl")):
            routed = route_llm_request(
                request=codex_request, provider="openai-codex", model=MODELS["terra"],
                api_mode="codex_responses", api_call_count=1, turn_id="codex-preflight",
            )["request"]
        self.assertEqual(len(routed["tools"]), 1)
        self.assertEqual(routed["tool_choice"], "required")
        self.assertIs(routed["parallel_tool_calls"], False)
        self.assertEqual({block["type"] for block in routed["input"][-1]["content"]}, {"input_text"})

    def test_min_chars_gate_follows_the_configured_orchestrator(self):
        """The gate was pinned to the literal string "terra", so a Qwen orchestrator
        fanned out on every turn, including one-word replies."""
        path = self.tmp_path / "orch.jsonl"
        short_request = anthropic_request("Köszi!")
        with patch("model_router._load_config", return_value=orchestration_config("qwen", path)):
            routed = route_llm_request(
                request=short_request, provider="qwen-token", model=MODELS["qwen"],
                api_mode="anthropic_messages", base_url=TOKENPLAN_URL,
                api_call_count=1, turn_id="qwen-short-turn",
            )["request"]
        self.assertEqual(len(routed["tools"]), 3)
        # The gate now records why it declined, so assert the intent -- no dispatch --
        # rather than the absence of the log file the diagnostic also writes to.
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([event for event in events if event["event"] == "preflight_forced"], [])
        self.assertEqual([event["skip_reason"].split(":")[0] for event in events],
                         ["prompt_shorter_than_min_chars"])
