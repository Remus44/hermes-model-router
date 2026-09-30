"""The forced planning call offers Claude delegation next to delegate_task.

Measured 2026-09-19: under Claude delegation a review turn -- whose first choice is
sonnet5, was forced through a delegate_task-only preflight, so the parent could
not reach delegate_claude and the review ran on Terra. With Claude switched on the
forced call now offers both routes and still requires one of them; with every
Claude model switched off it keeps the original delegate_task-only call.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_router import _routing_note, claude_delegation, route_llm_request, usage_guard
from model_router.host_delegation_fixtures import host_delegation
from model_router.test_model_router import CALLABLE, MODELS, chat_request

REVIEW = "Review the last commit in ~/Repositories/hermes-model-router for correctness bugs and give me the top 3 findings."
# A router-owned parent is forced only from 180 characters on.
LONG_REVIEW = (REVIEW + " Read every changed function in full, check the fallback and default-tier paths, "
               "and rank the findings by severity.")
CODE = ("Javitsd meg a naptar komponens hibajat a repoban: a 15 perces racs akkor is latszik, "
        "amikor minden 15 perces szolgaltatas inaktiv. Irj ra regressziot is.")
SCHEMA = {"type": "object", "properties": {"goal": {"type": "string"}, "role": {"type": "string"}}}


def _cfg(temp_dir, claude=True):
    cfg = {
        "enabled": True, "provider": "openai-codex", "models": MODELS,
        "callable": {**CALLABLE, "opus5": claude, "sonnet5": claude, "haiku": claude},
        "tier_providers": {"opus5": "anthropic", "sonnet5": "anthropic", "haiku": "anthropic"},
        "default_model": "terra",
        "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
        "orchestration": {"enabled": True, "max_tasks": 3,
                          "path": str(Path(temp_dir) / "orchestration.jsonl")},
        "shadow": {"enabled": False},
        "fallbacks": {"terra": "luna", "luna": "sol"},
        "preferences": {"review": ["sonnet5", "opus5", "terra"], "code": ["terra", "sonnet5"]},
    }
    return cfg


def _anthropic_request(text, tools):
    request = chat_request(text)
    request["model"] = "claude-opus-5-5"
    request["tools"] = [{"name": name, "input_schema": dict(SCHEMA)} for name in tools]
    return request


def _openai_request(text, tools):
    request = chat_request(text)
    request["model"] = "gpt-5.6-terra"
    request["tools"] = [{"type": "function", "function": {"name": name, "parameters": dict(SCHEMA)}}
                        for name in tools]
    return request


def _names(request):
    return [tool.get("name") or tool["function"]["name"] for tool in request["tools"]]


class ForcedPreflightTests(unittest.TestCase):
    def _route(self, request, claude=True, model="claude-opus-5-5", provider="anthropic"):
        with tempfile.TemporaryDirectory() as d, host_delegation(depth=2), \
             patch("model_router._load_config", return_value=_cfg(d, claude)), \
             patch("model_router._log_decision"), \
             patch("model_router._hermes_delegation_target_names", return_value=("sonnet5", "opus5")), \
             patch.object(claude_delegation, "_ACTIVE", True):
            routed = route_llm_request(request=request, provider=provider, model=model,
                                       api_call_count=1, turn_id="parent-turn", platform="cli")
        self.assertIsNotNone(routed, "the forced preflight was not applied")
        return routed["request"]

    def _instruction(self, request):
        return str(request["messages"][-1]["content"])

    def test_a_deferred_claude_route_is_offered_through_the_tool_call_bridge(self):
        request = self._route(_anthropic_request(
            REVIEW, ["mcp__delegate_task", "mcp__tool_search", "mcp__tool_describe", "mcp__tool_call"]))
        self.assertEqual(_names(request), ["mcp__delegate_task", "mcp__tool_describe", "mcp__tool_call"])
        self.assertEqual(request["tool_choice"], {"type": "any"})
        instruction = self._instruction(request)
        self.assertIn('delegate_claude(tier="sonnet")', instruction)
        self.assertIn('tool_call with name "delegate_claude"', instruction)
        self.assertIn('load it once with tool_describe', instruction)
        self.assertIn("review", instruction)

    def test_a_directly_listed_delegate_claude_is_offered_as_itself(self):
        request = self._route(_anthropic_request(REVIEW, ["mcp__delegate_task", "mcp__delegate_claude"]))
        self.assertEqual(_names(request), ["mcp__delegate_task", "mcp__delegate_claude"])
        self.assertEqual(request["tool_choice"], {"type": "any"})
        self.assertNotIn("It is a deferred tool: call it through tool_call", self._instruction(request))

    def test_the_planner_schema_is_still_hardened(self):
        request = self._route(_anthropic_request(REVIEW, ["mcp__delegate_task", "mcp__tool_call"]))
        self.assertEqual(request["tools"][0]["input_schema"]["properties"]["role"]["enum"], ["orchestrator"])

    def test_an_openai_shaped_parent_is_required_to_make_one_of_the_two_calls(self):
        request = self._route(_openai_request(LONG_REVIEW, ["delegate_task", "tool_call"]),
                              model="gpt-5.6-terra", provider="openai-codex")
        self.assertEqual(_names(request), ["delegate_task", "tool_call"])
        self.assertEqual(request["tool_choice"], "required")
        self.assertFalse(request["parallel_tool_calls"])

    def test_claude_switched_off_keeps_the_original_delegate_task_only_call(self):
        request = self._route(_anthropic_request(REVIEW, ["mcp__delegate_task", "mcp__tool_call"]),
                              claude=False)
        self.assertEqual(_names(request), ["mcp__delegate_task"])
        self.assertEqual(request["tool_choice"], {"type": "tool", "name": "mcp__delegate_task"})
        self.assertNotIn("delegate_claude(", self._instruction(request))

    def test_a_kind_whose_first_choice_is_codex_keeps_the_delegate_task_only_call(self):
        request = self._route(_anthropic_request(CODE, ["mcp__delegate_task", "mcp__tool_call"]))
        self.assertEqual(_names(request), ["mcp__delegate_task"])
        self.assertEqual(request["tool_choice"], {"type": "tool", "name": "mcp__delegate_task"})

    def test_short_codex_parent_gets_claude_review_advice_without_preflight(self):
        request = self._route(_openai_request(REVIEW, ["delegate_task", "tool_call"]),
                              model="gpt-5.6-terra", provider="openai-codex")
        instruction = self._instruction(request)
        self.assertNotIn("[INTERNAL ORCHESTRATOR PREFLIGHT]", instruction)
        self.assertIn('sonnet5 → delegate_claude(tier="sonnet")', instruction)
        self.assertEqual(_names(request), ["delegate_task", "tool_call"])

    def test_a_session_with_no_route_to_claude_keeps_the_delegate_task_only_call(self):
        request = self._route(_anthropic_request(REVIEW, ["mcp__delegate_task", "mcp__read_file"]))
        self.assertEqual(_names(request), ["mcp__delegate_task"])


class OfferedTests(unittest.TestCase):
    def test_the_tool_search_bridge_counts_as_offering_delegate_claude(self):
        """Hermes defers plugin tools behind tool_search: live parent requests carry
        mcp__tool_call, never delegate_claude itself (terra-spark-orchestration.jsonl)."""
        self.assertTrue(claude_delegation.offered(["mcp__delegate_task", "mcp__tool_search", "mcp__tool_call"]))
        self.assertTrue(claude_delegation.offered(["tool_call"]))

    def test_a_request_with_neither_does_not(self):
        self.assertFalse(claude_delegation.offered(["mcp__delegate_task", "mcp__tool_search"]))

    def test_deferred_only_parent_receives_balanced_routing_note(self):
        with tempfile.TemporaryDirectory() as d, \
             patch.object(claude_delegation, "_ACTIVE", True), \
             patch("model_router._hermes_delegation_target_names", return_value=("sonnet5", "opus5")), \
             patch.object(usage_guard, "peek", return_value=None):
            request = _anthropic_request(REVIEW, ["mcp__tool_search", "mcp__tool_describe", "mcp__tool_call"])
            note = _routing_note(request, {"api_call_count": 1, "turn_id": "review-turn", "platform": "cli"},
                                 _cfg(d))
        self.assertIn('sonnet5 → delegate_claude(tier="sonnet")', note)


if __name__ == "__main__":
    unittest.main()
