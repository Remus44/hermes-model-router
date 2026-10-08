"""A parent on a fallback account orchestrates like any other.

Observed 2026-09-09: with Codex exhausted, the orchestrator moved to Sonnet via the
Hermes fallback chain and then worked alone for 35 calls. `route_llm_request` returned
None for any model outside its own provider, so the parent never received the
delegation contract — and a second gate only recognised Sol and `default_model` as
orchestrators. Rewriting a model is provider-bound; orchestrating is not.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_router import _conductor_tier, claude_delegation, route_llm_request
from model_router.host_delegation_fixtures import host_delegation
from model_router.test_model_router import CALLABLE, MODELS, chat_request

ACTIONABLE = (
    "Javitsd meg a naptar komponens hibajat a repoban: a 15 perces racs akkor is latszik, "
    "amikor minden 15 perces szolgaltatas inaktiv. Irj ra regressziot is."
)


def _cfg(temp_dir, **overrides):
    cfg = {
        "enabled": True, "provider": "openai-codex", "models": MODELS, "callable": dict(CALLABLE),
        "default_model": "terra",
        "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
        "orchestration": {"enabled": True, "max_tasks": 3,
                          "path": str(Path(temp_dir) / "orchestration.jsonl")},
        "shadow": {"enabled": False},
        "fallbacks": {"terra": "luna", "luna": "sol"},
    }
    cfg.update(overrides)
    return cfg


def _delegating_request(model="claude-sonnet-5-5", tool_name="delegate_task", text=ACTIONABLE):
    request = chat_request(text)
    # The body carries the model actually in use, as it does in a real call.
    request["model"] = model
    request["tools"] = [{
        "type": "function", "name": tool_name,
        "parameters": {"type": "object", "properties": {"goal": {"type": "string"},
                                                        "role": {"type": "string"}}},
    }]
    return request


class ExternalParentOrchestrationTests(unittest.TestCase):
    def _route(self, model, cfg, tool_name="delegate_task"):
        with host_delegation(depth=2), \
             patch("model_router._load_config", return_value=cfg), \
             patch("model_router._log_decision"), \
             patch("model_router._delegation_target_names", return_value=("sonnet5", "opus5", "qwen")):
            return route_llm_request(
                request=_delegating_request(model, tool_name), provider="anthropic", model=model,
                api_call_count=1, turn_id="external-parent-turn")

    def test_the_claude_code_mcp_prefix_still_counts_as_the_delegate_tool(self):
        """Anthropic OAuth requests are normalised for Claude Code, which renames every
        tool to mcp__<name>. Matching the bare name told a Claude parent it had no
        delegate_task and skipped its preflight — measured live before this fix."""
        with tempfile.TemporaryDirectory() as d:
            routed = self._route("claude-sonnet-5-5", _cfg(d), tool_name="mcp__delegate_task")
        self.assertIsNotNone(routed, "the mcp__-prefixed tool was not recognised")
        schema = routed["request"]["tools"][0]["parameters"]
        self.assertEqual(schema["properties"]["role"]["enum"], ["orchestrator"])

    def test_the_forced_choice_names_the_tool_as_the_request_names_it(self):
        """Measured live on 2026-09-18: the preflight forced tool_choice
        {"name": "delegate_task"} on a Claude parent whose tool was mcp__delegate_task.
        Anthropic answered 400 "Tool 'delegate_task' not found in provided tools" and the
        turn fell back off Opus. The forced name must be the one actually offered."""
        # claude-haiku-4-5: Claude 5 bodies are never forced (test_claude_preflight).
        request = _delegating_request("claude-haiku-4-5", "mcp__delegate_task")
        request["tools"] = [{"name": "mcp__delegate_task",
                             "input_schema": request["tools"][0]["parameters"]}]
        with tempfile.TemporaryDirectory() as d, host_delegation(depth=2), \
             patch("model_router._load_config", return_value=_cfg(d)), \
             patch("model_router._log_decision"), \
             patch("model_router._delegation_target_names", return_value=("sonnet5", "opus5", "qwen")):
            routed = route_llm_request(request=request, provider="anthropic", model="claude-opus-5-5",
                                       api_call_count=1, turn_id="external-parent-turn")
        self.assertIsNotNone(routed)
        self.assertEqual(routed["request"]["tool_choice"], {"type": "tool", "name": "mcp__delegate_task"})
        self.assertEqual([tool["name"] for tool in routed["request"]["tools"]], ["mcp__delegate_task"])

    def test_a_request_with_no_delegate_tool_is_still_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(self._route("claude-sonnet-5-5", _cfg(d), tool_name="read_file"))

    def test_a_parent_on_another_account_still_gets_the_preflight(self):
        with tempfile.TemporaryDirectory() as d:
            routed = self._route("claude-sonnet-5-5", _cfg(d))
        self.assertIsNotNone(routed, "an external parent received no orchestration")
        schema = routed["request"]["tools"][0]["parameters"]
        self.assertEqual(schema["properties"]["role"]["enum"], ["orchestrator"])

    def test_the_external_parents_model_is_never_rewritten(self):
        """The router cannot switch providers; only the instructions are ours to add."""
        with tempfile.TemporaryDirectory() as d:
            routed = self._route("claude-sonnet-5-5", _cfg(d))
        self.assertEqual(routed["request"].get("model", "claude-sonnet-5-5"), "claude-sonnet-5-5")

    def test_an_unknown_model_is_still_left_alone(self):
        """Only a configured delegation target counts; anything else is not ours."""
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(self._route("some-other-vendor/model", _cfg(d)))

    def test_orchestration_disabled_still_means_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d, orchestration={"enabled": False, "path": str(Path(d) / "o.jsonl")})
            self.assertIsNone(self._route("claude-sonnet-5-5", cfg))


class ConductorTierTests(unittest.TestCase):
    """The forced conductor must not be pinned to an account that has run out."""

    def test_the_code_chain_no_longer_decides_where_planning_happens(self):
        """`code: [opus5, ...]` is an answer about who writes the code. Read as the
        conductor's route too, it silently moved every planner onto the Claude
        subscription: coordination paying external-account prices for a choice the
        operator made about leaves. One key cannot answer both questions."""
        cfg = {"models": MODELS, "callable": {**CALLABLE, "opus5": True},
               "default_model": "terra", "preferences": {"code": ["opus5", "terra"]}}
        self.assertEqual(_conductor_tier(cfg), "terra")

    def test_the_operator_can_still_pin_the_conductor_outright(self):
        """Moving the planner off a loaded account is the capability the `code`
        lookup was really serving; it keeps that, on a key that means only this."""
        cfg = {"models": MODELS, "callable": {**CALLABLE, "opus5": True},
               "default_model": "terra", "orchestration": {"conductor": "opus5"}}
        self.assertEqual(_conductor_tier(cfg), "opus5")

    def test_an_uncallable_pin_falls_through_rather_than_failing_the_preflight(self):
        cfg = {"models": MODELS, "callable": {**CALLABLE, "opus5": False},
               "default_model": "terra", "orchestration": {"conductor": "opus5"}}
        self.assertEqual(_conductor_tier(cfg), "terra")

    def test_an_unset_pin_is_not_a_pin(self):
        for empty in (None, "", "   "):
            with self.subTest(conductor=empty):
                cfg = {"models": MODELS, "callable": dict(CALLABLE), "default_model": "terra",
                       "orchestration": {"conductor": empty}}
                self.assertEqual(_conductor_tier(cfg), "terra")

    def test_without_a_pin_the_default_still_wins(self):
        cfg = {"models": MODELS, "callable": dict(CALLABLE), "default_model": "terra",
               "preferences": {"design": ["sol"]}}
        self.assertEqual(_conductor_tier(cfg), "terra")

    def test_the_configured_default_is_used_when_callable(self):
        self.assertEqual(_conductor_tier({"models": MODELS, "callable": dict(CALLABLE),
                                          "default_model": "terra"}), "terra")

    def test_an_exhausted_default_falls_through_its_chain(self):
        cfg = {"models": MODELS, "callable": {**CALLABLE, "terra": False},
               "default_model": "terra", "fallbacks": {"terra": "luna"}}
        self.assertEqual(_conductor_tier(cfg), "luna")

    def test_a_chain_of_exhausted_tiers_reaches_the_first_callable_one(self):
        cfg = {"models": MODELS, "callable": {**CALLABLE, "terra": False, "luna": False},
               "default_model": "terra", "fallbacks": {"terra": "luna", "luna": "sol"}}
        self.assertEqual(_conductor_tier(cfg), "sol")

    def test_with_no_chain_any_callable_tier_beats_a_dead_default(self):
        cfg = {"models": MODELS, "callable": {**{t: False for t in MODELS}, "sol": True},
               "default_model": "terra"}
        self.assertEqual(_conductor_tier(cfg), "sol")

    def test_nothing_callable_degrades_to_the_configured_default(self):
        cfg = {"models": MODELS, "callable": {t: False for t in MODELS}, "default_model": "terra"}
        self.assertEqual(_conductor_tier(cfg), "terra")


class ExplicitDelegationMentionTests(unittest.TestCase):
    """Operator ruling 2026-09-18: a user turn that explicitly names delegate_claude
    or delegate_task must not be forced through the delegate_task planning preflight
    -- the routing note (advisory only) still applies on that same call."""

    def _route(self, text, tool_name="delegate_task"):
        request = _delegating_request("claude-sonnet-5-5", tool_name, text=text)
        # A live session with Claude delegation carries delegate_claude next to
        # delegate_task, and the router offers it only then.
        request["tools"].append({"type": "function", "name": "delegate_claude",
                                 "parameters": {"type": "object", "properties": {}}})
        with tempfile.TemporaryDirectory() as d, host_delegation(depth=2), \
             patch("model_router._load_config", return_value=_cfg(
                 d, claude_delegation={"enabled": True}, callable={**CALLABLE, "haiku": True})), \
             patch("model_router._log_decision"), \
             patch("model_router._delegation_target_names", return_value=("sonnet5", "opus5", "qwen")), \
             patch.object(claude_delegation, "_ACTIVE", True):
            return route_llm_request(
                request=request, provider="anthropic", model="claude-sonnet-5-5",
                api_call_count=1, turn_id="external-parent-turn", platform="cli")

    def test_an_explicit_delegate_claude_mention_skips_the_forced_preflight(self):
        routed = self._route('Call delegate_claude with tier "haiku" to check the changelog.')
        self.assertIsNotNone(routed, "the routing note should still apply")
        schema = routed["request"]["tools"][0]["parameters"]
        self.assertNotIn("enum", schema["properties"].get("role", {}),
                         "the forced planner contract must not have been applied")

    def test_the_routing_note_still_applies_on_the_skipped_call(self):
        routed = self._route('Call delegate_claude with tier "haiku" to check the changelog.')
        self.assertIn("[ROUTER] This turn classifies as", routed["request"]["messages"][-1]["content"])

    def test_a_normal_actionable_message_still_gets_the_preflight(self):
        """Confirms the new gate is scoped to an explicit mention, not a blanket skip."""
        routed = ExternalParentOrchestrationTests()._route("claude-sonnet-5-5", _cfg(tempfile.mkdtemp()))
        self.assertIsNotNone(routed)
        schema = routed["request"]["tools"][0]["parameters"]
        self.assertEqual(schema["properties"]["role"]["enum"], ["orchestrator"])


class ContractTruthfulnessTests(unittest.TestCase):
    def test_the_conductor_is_not_told_claude_is_unavailable(self):
        """That claim was disproven on 2026-09-09; leaving it in suppressed the very
        delegation the plugin exists to produce."""
        with tempfile.TemporaryDirectory() as d:
            routed = ExternalParentOrchestrationTests()._route("claude-sonnet-5-5", _cfg(d))
        contract = routed["request"]["tools"][0]["parameters"]["properties"]["context"]["enum"][0]
        self.assertNotIn("does not fund third-party API access", contract)
        self.assertNotIn("Claude delegation target is switched off", contract)


if __name__ == "__main__":
    unittest.main()
