import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

import model_router as router
from model_router import worker_admission


ROOT = Path(__file__).resolve().parent
HOST_REASONING_EFFORT = Path("/home/remus/.hermes/hermes-agent/agent/reasoning_effort.py")


def host_effort_module():
    spec = importlib.util.spec_from_file_location("f09_host_reasoning_effort", HOST_REASONING_EFFORT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def host_tier_levels():
    host = host_effort_module()
    without_none = lambda levels: tuple(level for level in levels if level != "none")
    return {
        "luna": without_none(host.CODEX_GPT56_EFFORTS),
        "spark": without_none(host.CODEX_LEGACY_EFFORTS),
        "terra": without_none(host.CODEX_GPT56_EFFORTS),
        "sol": tuple(host.CODEX_ASTRA_EFFORTS),
    }


def shipped_config():
    cfg = yaml.safe_load((ROOT / "router_config.yaml").read_text(encoding="utf-8"))
    cfg["callable"].update({"luna": True, "spark": True, "terra": True, "sol": True})
    cfg["triage"] = {"enabled": False}
    cfg["orchestration"] = {"enabled": False}
    cfg["session_policy"] = {"pin_root_parent": False}
    cfg["shadow"] = {"enabled": False}
    cfg["logging"] = {"enabled": False}
    return cfg


def request(text, cfg):
    return {
        "model": cfg["models"]["terra"],
        "messages": [{"role": "user", "content": text}],
        "temperature": 0.2,
    }


class ExactEffortLabelTests(unittest.TestCase):
    def route(self, text, cfg):
        with patch.object(router, "_load_config", return_value=cfg), patch.object(router, "_log_decision"):
            return router.route_llm_request(
                request=request(text, cfg),
                provider="openai-codex",
                model=cfg["models"]["terra"],
                api_call_count=1,
                turn_id="f09-label-wire",
            )

    def test_plugin_level_table_tracks_the_host_vocabularies_without_none(self):
        self.assertEqual(getattr(router, "_CODEX_TIER_EFFORTS", {}), host_tier_levels())

    def test_stated_levels_reach_the_wire_and_route_record_exactly(self):
        cfg = shipped_config()
        cases = (
            ("[sol:high] Inspect the implementation and report.", "sol", "high"),
            ("[sol:xhigh] Inspect the implementation and report.", "sol", "xhigh"),
            ("[terra:xhigh] Implement the bounded repair.", "terra", "xhigh"),
            ("[luna:xhigh] Inspect the local configuration and report only.", "luna", "xhigh"),
        )
        for text, tier, expected in cases:
            with self.subTest(text=text):
                routed = self.route(text, cfg)
                self.assertEqual(routed["metadata"]["tier"], tier)
                self.assertEqual(routed["metadata"]["effort"], expected)
                self.assertEqual(routed["request"]["reasoning"]["effort"], expected)

    def test_bare_sol_uses_effort_sol_not_retired_explicit_sol_in_code_defaults(self):
        cfg = copy.deepcopy(router._DEFAULT_CONFIG)
        cfg["models"] = {"luna": "gpt-6-luna", "spark": "gpt-5.3-codex-spark", "terra": "gpt-5.6-terra", "sol": "gpt-6.1-sol"}
        decision = router.classify_request(
            {"model": cfg["models"]["terra"], "messages": [{"role": "user", "content": "[sol] Inspect and report."}]},
            config=cfg,
        )
        self.assertEqual(decision.effort, cfg["effort"]["sol"])

    def test_bare_sol_uses_effort_sol_when_a_local_overlay_keeps_retired_keys(self):
        cfg = shipped_config()
        cfg["effort"].update({"sol": "high", "explicit_sol": "low", "sol_long": "high", "terra": "high"})
        decision = router.classify_request(request("[sol] Inspect and report.", cfg), config=cfg)
        self.assertEqual(decision.effort, "high")

    def test_an_unsupported_delegation_goal_is_refused_before_dispatch(self):
        call = Mock(return_value="dispatched")
        cfg = shipped_config()
        with patch.object(router, "_load_config", return_value=cfg), \
             patch.object(worker_admission, "delegate_task_route", return_value=("openai-codex", cfg["models"]["terra"])), \
             patch.object(router, "_delegation_targets_detail", return_value={}), \
             patch.object(router, "_delegated_claude_review_status", return_value=(None, "not a review")), \
             patch.object(worker_admission, "refusal", return_value=""):
            result = worker_admission.guard_tool_execution(
                tool_name="delegate_task",
                args={"tasks": [{"goal": "[sol:bogus] Inspect and report.", "model": "sol"}]},
                next_call=call,
            )
        self.assertEqual(call.call_count, 0)
        self.assertIn("sol", result)
        self.assertIn("low, medium, high, xhigh, max", result)

    def test_an_unsupported_root_level_routes_as_unlabelled_text(self):
        cfg = shipped_config()
        unsupported = router.classify_request(request("[sol:bogus] Inspect and report.", cfg), config=cfg)
        unlabelled = router.classify_request(request("Inspect and report.", cfg), config=cfg)
        self.assertIsNone(router._label_match("[sol:bogus] Inspect and report."))
        self.assertEqual((unsupported.tier, unsupported.effort), (unlabelled.tier, unlabelled.effort))

    def test_route_log_effort_matches_the_stated_wire_effort(self):
        cfg = shipped_config()
        with tempfile.TemporaryDirectory() as directory:
            route_log = Path(directory) / "router.jsonl"
            cfg["logging"] = {"enabled": True, "path": str(route_log)}
            with patch.object(router, "_load_config", return_value=cfg):
                routed = router.route_llm_request(
                    request=request("[terra:xhigh] Implement the bounded repair.", cfg),
                    provider="openai-codex", model=cfg["models"]["terra"], api_call_count=1,
                    turn_id="f09-route-log",
                )
            record = json.loads(route_log.read_text(encoding="utf-8").strip())
        self.assertEqual(routed["request"]["reasoning"]["effort"], "xhigh")
        self.assertEqual(record["effort"], "xhigh")

    def test_local_overlay_explicit_keys_do_not_change_bare_sol_effort_or_write_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            shipped = Path(directory) / "router_config.yaml"
            local = Path(directory) / "router_config.local.yaml"
            shipped.write_text((ROOT / "router_config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
            local_text = "effort:\n  sol: high\n  explicit_sol: high\n  sol_long: high\n  terra: high\n"
            local.write_text(local_text, encoding="utf-8")
            with patch.object(router, "_CONFIG_PATH", shipped):
                cfg = router._load_config()
                decision = router.classify_request(request("[sol] Inspect and report.", cfg), config=cfg)
            self.assertEqual(local.read_text(encoding="utf-8"), local_text)
        self.assertEqual(decision.effort, "high")

    def test_host_skills_delta_note_currently_changes_luna_read_only_classification(self):
        cfg = shipped_config()
        goal = "[luna] Inspect the configuration and report only."
        skills_delta = (
            "[System: The skills available in this conversation changed after its system prompt was built.\n"
            "Now available (load with skill_view(name) when relevant, like the skills listed in the system prompt):\n"
            "    - creative: Create and edit artifacts.]"
        )
        plain = router.classify_request(request(goal, cfg), config=cfg)
        noted = router.classify_request(request(goal + "\n\n" + skills_delta, cfg), config=cfg)
        self.assertEqual((plain.tier, noted.tier), ("luna", "terra"))

    def test_typed_labels_keep_read_only_routing_and_admission_with_triage_on_or_off(self):
        from tools.delegate_tool import DELEGATE_TASK_SCHEMA

        tool_shapes = {
            "direct": [DELEGATE_TASK_SCHEMA],
            "deferred": [{
                "type": "function", "name": "tool_call",
                "parameters": {"type": "object", "properties": {"calls": {"type": "array"}}},
            }],
        }
        for triage_enabled in (False, True):
            for shape, tools in tool_shapes.items():
                with self.subTest(triage_enabled=triage_enabled, shape=shape):
                    cfg = shipped_config()
                    cfg["triage"] = {"enabled": triage_enabled}
                    routed_request = request("[luna:xhigh] Inspect the configuration and report only.", cfg)
                    routed_request["tools"] = tools
                    if shape == "deferred":
                        routed_request["deferred_tools"] = ["delegate_task"]
                    with patch.object(router, "_load_config", return_value=cfg), \
                         patch.object(router, "_log_decision"):
                        routed = router.route_llm_request(
                            request=routed_request, provider="openai-codex", model=cfg["models"]["terra"],
                            api_call_count=1, turn_id=f"f09-triage-{shape}-{triage_enabled}",
                        )
                    self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), ("luna", "xhigh"))
                    call = Mock(return_value="dispatched")
                    with patch.object(router, "_load_config", return_value=cfg), \
                         patch.object(worker_admission, "delegate_task_route", return_value=("openai-codex", cfg["models"]["terra"])), \
                         patch.object(router, "_delegation_targets_detail", return_value={}), \
                         patch.object(router, "_delegated_claude_review_status", return_value=(None, "not a review")), \
                         patch.object(worker_admission, "refusal", return_value=""):
                        worker_admission.guard_tool_execution(
                            tool_name="delegate_task", args={"tasks": [{"goal": "[luna:xhigh] Inspect the configuration and report only."}]},
                            next_call=call,
                        )
                    self.assertEqual(call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
