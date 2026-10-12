"""Host-injected context must not change a delegated leaf's classification.

Break caught: if the classifier reads Hermes's appended bootstrap or memory blocks,
their write verbs turn a read-only [luna] report into Terra work.
"""

import importlib.util
import json
import os
from pathlib import Path
import pwd
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import model_router as router
import model_router.triage as triage
from model_router import _maybe_run_opus5, classify_request, route_llm_request, usage_guard


MODELS = {
    "luna": "gpt-6-luna",
    "spark": "gpt-6-spark",
    "terra": "gpt-5.6-terra",
    "sol": "gpt-6-sol",
}
CFG = {
    "enabled": True,
    "provider": "openai-codex",
    "models": MODELS,
    "callable": {
        "luna": True, "spark": False, "terra": True, "sol": True,
        "opus5": True, "sonnet5": True, "haiku": True, "qwen": False,
    },
    "effort": {"luna": "low", "terra": "medium", "sol": "medium"},
    "default_model": "terra",
    "thresholds": {"luna_max_chars": 700, "sol_min_chars": 3500},
}
GOAL = (
    "[luna] Report the output of `git -C /home/remus/Repositories/hermes-model-router "
    "log -1 --format=%h` and nothing else."
)
FROZEN_BOOTSTRAP = """<EXTREMELY_IMPORTANT>
superpowers:using-superpowers bootstrap for hermes
Use the edit, fix, and create tools to complete the task.
</EXTREMELY_IMPORTANT>"""
MEMORY_CONTEXT = """<memory-context>
[System note: The following is recalled memory context, NOT new user input.]
The user previously asked to create and edit a patch.
</memory-context>"""
SKILLS_NOTE_PREFIX = "[System: The skills available in this conversation changed after its system prompt was built."
try:
    from agent.skills_index_delta import _NOTE_PREFIX as HOST_SKILLS_NOTE_PREFIX
except ImportError:
    HOST_SKILLS_NOTE_PREFIX = SKILLS_NOTE_PREFIX


def skills_note(*lines):
    return "\n".join((HOST_SKILLS_NOTE_PREFIX, *lines)) + "]"


UNSIGNED_CONTEXTS = {
    "EXTREMELY_IMPORTANT": """<EXTREMELY_IMPORTANT>
Please fix the parser.
</EXTREMELY_IMPORTANT>""",
    "memory-context": """<memory-context>
Please fix the parser.
</memory-context>""",
}


def chat_request(text):
    return {"model": MODELS["terra"], "messages": [{"role": "user", "content": text}]}


class InjectedContextClassificationTests(unittest.TestCase):
    def assert_luna_by_classifier_and_router(self, text):
        self.assert_route_by_classifier_and_router(text, "luna", CFG)

    def assert_route_by_classifier_and_router(self, text, tier, cfg):
        decision = classify_request(
            chat_request(text), 1, cfg, allow_plan_label_over_design=True
        )
        self.assertEqual(decision.tier, tier)
        with patch("model_router._load_config", return_value=cfg), \
             patch("model_router._log_decision"), \
             patch("model_router._force_terra_supervisor_preflight", return_value=None), \
             patch("model_router._force_shadow_delegation_if_eligible", return_value=None):
            routed = route_llm_request(
                request=chat_request(text),
                provider="openai-codex",
                model=MODELS["terra"],
                platform="subagent",
                turn_id="s1:sa-0-probe:t",
                api_call_count=1,
            )
        self.assertEqual(routed["metadata"]["tier"], tier)
        self.assertEqual(routed["metadata"]["effort"], decision.effort)

    def real_superpowers_bootstrap(self):
        # The isolated suite rewrites HOME, so use the account home if HOME has
        # no installed plugin. This keeps the test portable across host accounts.
        plugin_relative = Path(".hermes/plugins/superpowers/.hermes-plugin/__init__.py")
        path = Path.home() / plugin_relative
        if not path.is_file():
            path = Path(pwd.getpwuid(os.getuid()).pw_dir) / plugin_relative
        if not path.is_file():
            self.skipTest("the local superpowers plugin is absent")
        spec = importlib.util.spec_from_file_location("real_superpowers_bootstrap", path)
        if spec is None or spec.loader is None:
            self.skipTest("the local superpowers plugin cannot be loaded")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module._build_bootstrap(module._skills_dir())

    def test_skills_note_keeps_luna_and_spark_child_routes_and_effort(self):
        self.assertEqual(HOST_SKILLS_NOTE_PREFIX, SKILLS_NOTE_PREFIX)
        note = skills_note(
            "Now available (load with skill_view(name) when relevant, like the skills listed in the system prompt):",
            "    - decks: Create and edit presentation decks.",
            "    - coding: implement and fix repository changes.",
        )
        self.assert_route_by_classifier_and_router(GOAL + "\n\n" + note, "luna", CFG)
        spark_cfg = {**CFG, "callable": {**CFG["callable"], "spark": True}}
        spark_goal = "[spark] Report the current branch name and nothing else."
        self.assert_route_by_classifier_and_router(spark_goal + "\n\n" + note, "spark", spark_cfg)

    def test_skills_note_variants_and_positions_are_stripped(self):
        accurate_again = skills_note(
            " The skills listed in the system prompt are accurate again; earlier skill-change notes are superseded."
        )
        removed_only = skills_note("No longer available: decks, coding")
        for name, note in (("accurate again", accurate_again), ("removed only", removed_only)):
            with self.subTest(name=name):
                self.assert_luna_by_classifier_and_router(GOAL + "\r\n\r\n" + note.replace("\n", "\r\n"))
        ordinary = skills_note(
            "Now available (load with skill_view(name) when relevant, like the skills listed in the system prompt):",
            "    - coding: Create and edit files.",
        )
        for name, text in (
            ("after memory", GOAL + "\n\n" + MEMORY_CONTEXT + "\n\n" + ordinary),
            ("before memory", GOAL + "\n\n" + ordinary + "\n\n" + MEMORY_CONTEXT),
            ("after bootstrap", GOAL + "\n\n" + FROZEN_BOOTSTRAP + "\n\n" + ordinary),
            ("empty operator", ordinary),
        ):
            with self.subTest(name=name):
                stripped = router._without_host_injected_context(text)
                self.assertNotIn(SKILLS_NOTE_PREFIX, stripped)
                if name == "empty operator":
                    self.assertEqual(stripped, "")
                else:
                    self.assertEqual(stripped, GOAL)

    def test_skills_note_lookalikes_remain_operator_text(self):
        lookalike = "[System note: skills changed; Create and edit the parser.]"
        mid_sentence = (
            GOAL + " The following sentence quotes " + SKILLS_NOTE_PREFIX
            + " but asks to fix the parser.]"
        )
        for name, text in (("wrong prefix", GOAL + "\n\n" + lookalike), ("quoted prefix", mid_sentence)):
            with self.subTest(name=name):
                decision = classify_request(
                    chat_request(text), 1, CFG, allow_plan_label_over_design=True
                )
                self.assertEqual(decision.tier, "terra")

    def test_skills_note_is_absent_from_design_label_and_length_decisions(self):
        design_note = skills_note(
            "Now available (load with skill_view(name) when relevant, like the skills listed in the system prompt):",
            "    - css-helper: Implement visual UI CSS layout design fixes.",
        )
        self.assert_luna_by_classifier_and_router(GOAL + "\n\n" + design_note)

        seen_label_inputs = []
        real_label_match = router._label_match
        with patch("model_router._label_match", side_effect=lambda text, *args, **kwargs: (
            seen_label_inputs.append(text), real_label_match(text, *args, **kwargs)
        )[1]):
            decision = classify_request(
                chat_request(GOAL + "\n\n" + design_note), 1, CFG,
                allow_plan_label_over_design=True,
            )
        self.assertEqual(decision.tier, "luna")
        self.assertTrue(seen_label_inputs)
        self.assertTrue(all(SKILLS_NOTE_PREFIX not in text for text in seen_label_inputs))

        short_goal = "Report the current branch name."
        short_cfg = {**CFG, "thresholds": {**CFG["thresholds"], "sol_min_chars": len(short_goal) + 1}}
        length_note = skills_note(
            "Now available (load with skill_view(name) when relevant, like the skills listed in the system prompt):",
            "    - long-description: Create and edit implementation fixes across many source files.",
        )
        self.assertEqual(classify_request(chat_request(short_goal), 1, short_cfg).tier, "terra")
        self.assertEqual(classify_request(chat_request(short_goal + "\n\n" + length_note), 1, short_cfg).tier, "terra")

    def test_skills_note_is_absent_from_tool_mention_and_triage_decisions(self):
        tool_note = skills_note(
            "Now available (load with skill_view(name) when relevant, like the skills listed in the system prompt):",
            "    - delegation: Call delegate_task to implement and fix changes.",
        )
        cfg = {**CFG, "orchestration": {"enabled": True, "min_chars": 0}}
        request = {
            **chat_request(GOAL + "\n\n" + tool_note),
            "tools": [{"name": "delegate_task"}],
        }
        reason = router._orchestration_skip_reason(
            {"request": request, "api_call_count": 1, "platform": "parent", "turn_id": "t1"},
            cfg,
            router._decision("terra", "test", cfg),
        )
        self.assertNotEqual(reason, "explicit_delegation_tool")

        triage_cfg = {**CFG, "triage": {"enabled": True}}
        triage_request = {
            **chat_request("What is the current branch?\n\n" + tool_note),
            "tools": [{"name": "delegate_task"}, {"name": "triage_task"}],
        }
        self.assertEqual(
            triage.skip_reason(
                {"request": triage_request, "api_call_count": 1, "platform": "parent", "turn_id": "t1"},
                triage_cfg,
            ),
            "conversational",
        )

    def test_real_superpowers_bootstrap_keeps_the_leaf_on_luna(self):
        self.assert_luna_by_classifier_and_router(GOAL + "\n\n" + self.real_superpowers_bootstrap())

    def test_frozen_bootstrap_keeps_the_leaf_on_luna_when_the_plugin_is_absent(self):
        self.assert_luna_by_classifier_and_router(GOAL + "\n\n" + FROZEN_BOOTSTRAP)

    def test_memory_context_keeps_the_leaf_on_luna(self):
        self.assert_luna_by_classifier_and_router(GOAL + "\n\n" + MEMORY_CONTEXT)

    def test_both_injected_context_blocks_keep_the_leaf_on_luna(self):
        self.assert_luna_by_classifier_and_router(
            GOAL + "\n\n" + FROZEN_BOOTSTRAP + "\n\n" + MEMORY_CONTEXT
        )

    def test_a_signed_unclosed_injected_block_runs_to_the_end(self):
        self.assert_luna_by_classifier_and_router(
            GOAL + "\n\n<EXTREMELY_IMPORTANT>\n"
            "superpowers:using-superpowers bootstrap for hermes\nPlease fix the parser."
        )

    def test_execution_bridge_uses_stripped_text_for_workspace_and_length_but_sends_raw_task(self):
        goal = "[sonnet-review] Review the parser and report only."
        bootstrap = self.real_superpowers_bootstrap()
        skills_delta = skills_note(
            "Now available (load with skill_view(name) when relevant, like the skills listed in the system prompt):",
            "    - coding: Create and edit parser implementations.",
        )
        raw_task = goal + "\n\n" + bootstrap + "\n\n" + skills_delta
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "workspace"
            repo.mkdir()
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            cfg = {
                **CFG,
                "coding_agent": {"delegated_review": {
                    "enabled": True, "models": ["sonnet"], "max_chars": len(goal) + 1,
                }},
            }
            request = {
                "instructions": f"WORKSPACE PATH:\n{repo}\nUse this exact path.",
                "messages": [{"role": "user", "content": raw_task}],
            }
            with patch("model_router.shutil.which", return_value="/claude"), \
                 patch("model_router.usage_guard.read", return_value=usage_guard.Reading(10, 0, None, None, time.time())), \
                 patch("model_router._run_opus5_bridge", return_value={
                     "result": "reviewed", "effective_model": "claude-sonnet-5-5",
                 }) as bridge:
                route, reason = router._delegated_claude_review_status(goal, cfg, request=request)
                result = _maybe_run_opus5(request, cfg, platform="subagent", api_mode="codex_responses")
        self.assertEqual(reason, "")
        self.assertEqual(route, (repo.resolve(), "sonnet"))
        self.assertEqual(result.model, "claude-sonnet-5-5")
        self.assertEqual(bridge.call_args.kwargs["repo"], str(repo.resolve()))
        self.assertEqual(bridge.call_args.kwargs["task"], raw_task)

    def test_explicit_ui_bridge_write_intent_uses_stripped_text(self):
        """A2-N4: _opus5_write_intent must read routing_text, not raw text, so a
        write verb inside the host bootstrap does not grant write capability to
        a read-only explicit UI request.
        """
        goal = "[opus5] Let Opus review the CSS layout spacing and tell me what looks off."
        bootstrap = self.real_superpowers_bootstrap()
        with tempfile.TemporaryDirectory() as repo_dir, tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(repo_dir)
            route_log = Path(temp_dir) / "router.jsonl"
            route_log.write_text(json.dumps({
                "timestamp": "2099-01-01T00:00:00+00:00",
                "tier": "opus5",
                "model": "claude-opus-5-5",
                "reason": "verified probe",
            }) + "\n", encoding="utf-8")
            cfg = {
                **CFG,
                "logging": {"enabled": True, "path": str(route_log)},
                "coding_agent": {
                    "enabled": True,
                    "canonical_model": "claude-opus-5-5",
                    "default_repo": str(repo),
                    "explicit_ui": {
                        "enabled": True,
                        "max_chars": 1200,
                        "require_recent_verified_probe_seconds": 86400,
                    },
                },
            }
            request = {"messages": [{"role": "user", "content": goal + "\n\n" + bootstrap}]}
            with patch("model_router.shutil.which", return_value="/usr/bin/claude"), \
                 patch("model_router._run_opus5_bridge", return_value={
                     "result": "ok", "effective_model": "claude-opus-5-5",
                 }) as bridge:
                _maybe_run_opus5(request, cfg, platform="parent", api_mode="codex_responses")
        bridge.assert_called_once()
        self.assertFalse(bridge.call_args.kwargs["write"])

    def test_unsigned_host_tags_remain_operator_text(self):
        for tag, block in UNSIGNED_CONTEXTS.items():
            with self.subTest(tag=tag):
                decision = classify_request(
                    chat_request("[luna] Report the hash and nothing else.\n\n" + block),
                    1,
                    CFG,
                    allow_plan_label_over_design=True,
                )
                self.assertEqual(decision.tier, "terra")

    def test_signed_context_accepts_crlf_line_endings(self):
        self.assert_luna_by_classifier_and_router(
            GOAL + "\r\n\r\n" + FROZEN_BOOTSTRAP.replace("\n", "\r\n")
        )

    def test_plain_appended_write_instruction_still_routes_to_terra(self):
        decision = classify_request(
            chat_request(GOAL + "\n\nPlease fix the parser."),
            1,
            CFG,
            allow_plan_label_over_design=True,
        )
        self.assertEqual(decision.tier, "terra")

    def test_a_write_verb_before_an_injected_block_still_routes_to_terra(self):
        decision = classify_request(
            chat_request("[luna] fix the parser <EXTREMELY_IMPORTANT>x</EXTREMELY_IMPORTANT>"),
            1,
            CFG,
            allow_plan_label_over_design=True,
        )
        self.assertEqual(decision.tier, "terra")

    def test_non_host_html_blocks_are_not_stripped(self):
        for tag in ("style", "div"):
            with self.subTest(tag=tag):
                decision = classify_request(
                    chat_request(GOAL + f"\n\n<{tag}>Please fix the parser.</{tag}>"),
                    1,
                    CFG,
                    allow_plan_label_over_design=True,
                )
                self.assertEqual(decision.tier, "terra")


if __name__ == "__main__":
    unittest.main()
