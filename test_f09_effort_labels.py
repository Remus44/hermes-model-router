import copy
import importlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

import model_router as router
from model_router import triage, worker_admission


ROOT = Path(__file__).resolve().parent
HOST_REASONING_EFFORT = Path("/home/remus/.hermes/hermes-agent/agent/reasoning_effort.py")

FALLBACK_TIER_LEVELS = {
    "luna": ("low", "medium", "high", "xhigh", "max"),
    "spark": ("low", "medium", "high", "xhigh"),
    "terra": ("low", "medium", "high", "xhigh", "max"),
    "sol": ("low", "medium", "high", "xhigh", "max"),
}


class _ReasoningEffortUnavailableFinder:
    """Simulate an installed Hermes whose effort module has disappeared."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "agent.reasoning_effort":
            raise ModuleNotFoundError("No module named 'agent.reasoning_effort'", name=fullname)
        return None


def load_plugin_without_reasoning_effort():
    package_name = "f09_no_reasoning_effort"
    module_name = f"{package_name}.model_router"
    package = types.ModuleType(package_name)
    package.__path__ = []
    finder = _ReasoningEffortUnavailableFinder()
    with patch.dict(sys.modules):
        sys.modules.pop("agent.reasoning_effort", None)
        sys.modules[package_name] = package
        sys.meta_path.insert(0, finder)
        try:
            spec = importlib.util.spec_from_file_location(
                module_name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
            )
            if spec is None or spec.loader is None:
                return None
            plugin = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = plugin
            try:
                spec.loader.exec_module(plugin)
            except ModuleNotFoundError:
                return None
            return plugin
        finally:
            sys.meta_path.remove(finder)


def host_effort_module():
    spec = importlib.util.spec_from_file_location("f09_host_reasoning_effort", HOST_REASONING_EFFORT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def host_tier_levels():
    host = host_effort_module()
    models = {
        "luna": "gpt-6-luna", "spark": "gpt-5.3-codex-spark",
        "terra": "gpt-5.6-terra", "sol": "gpt-6.1-sol",
    }
    return {
        tier: tuple(level for level in host.codex_supported_efforts(model) if level != "none")
        for tier, model in models.items()
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

    def test_plugin_import_and_label_validation_fall_back_when_host_effort_module_is_missing(self):
        plugin = load_plugin_without_reasoning_effort()
        self.assertIsNotNone(plugin, "the plugin must load without agent.reasoning_effort")
        assert plugin is not None
        self.assertEqual(plugin._label_match("[sol:high] Inspect and report.").group("tier").casefold(), "sol")
        self.assertEqual(plugin._unsupported_label("[sol:bogus] Inspect and report.")[0], "sol")
        self.assertIsNotNone(plugin._label_match("[sol:max] Inspect and report."))
        self.assertEqual(plugin._unsupported_label("[spark:max] Inspect and report.")[0], "spark")

    def test_fallback_literals_are_pinned_to_the_host_model_vocabularies(self):
        self.assertEqual(host_tier_levels(), FALLBACK_TIER_LEVELS)
        self.assertEqual(getattr(router, "_FALLBACK_CODEX_TIER_EFFORTS", None), FALLBACK_TIER_LEVELS)

    def test_models_override_uses_the_overridden_model_vocabulary(self):
        cfg = shipped_config()
        cfg["models"]["sol"] = "gpt-5.3-codex-spark"
        levels = getattr(router, "_codex_tier_efforts", lambda _cfg: {})(cfg)
        self.assertEqual(levels.get("sol"), FALLBACK_TIER_LEVELS["spark"])
        try:
            unsupported = router._unsupported_label("[sol:max] Inspect and report.", cfg)
        except TypeError:
            unsupported = None
        self.assertEqual(unsupported[:2] if unsupported else None, ("sol", "max"))

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

    def test_max_is_valid_for_luna_terra_and_sol_but_not_spark_and_none_is_never_valid(self):
        cfg = shipped_config()
        for tier in ("luna", "terra", "sol"):
            with self.subTest(tier=tier):
                self.assertIsNotNone(router._label_match(f"[{tier}:max] Inspect and report."))
        self.assertEqual(router._unsupported_label("[spark:max] Inspect and report.")[:2], ("spark", "max"))
        for tier in ("luna", "spark", "terra", "sol"):
            with self.subTest(tier=tier):
                self.assertEqual(router._unsupported_label(f"[{tier}:none] Inspect and report.")[:2], (tier, "none"))

    def test_case_insensitive_stated_label_runs_as_the_matching_tier_and_level(self):
        cfg = shipped_config()
        routed = self.route("[SOL:High] Inspect the implementation and report.", cfg)
        self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), ("sol", "high"))

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

    def test_pre_tool_refuses_bad_level_before_triage_charges_the_budget(self):
        triage.reset_state()
        self.addCleanup(triage.reset_state)
        cfg = shipped_config()
        with tempfile.TemporaryDirectory() as directory:
            bounded_path = Path(directory) / "bounded-budget.jsonl"
            cfg["triage"] = {
                "enabled": True, "max_workers_per_turn": 4, "max_children_per_worker": 1,
                "path": str(Path(directory) / "triage.jsonl"),
            }
            cfg["task_budget"] = {"enabled": True, "path": str(bounded_path)}
            with patch.object(router, "_load_config", return_value=cfg), \
                 patch.object(router, "_requested_delegation_targets", return_value=()):
                refused = router.on_pre_tool_call(
                    "delegate_task", {"goal": "[sol:bogus] Inspect and report."}, turn_id="f09-budget",
                )
                self.assertIsNotNone(refused)
                self.assertEqual(refused["action"], "block")
                self.assertIn("Unsupported sol effort 'bogus'", refused["message"])
                self.assertEqual(triage._TURN_WORKERS, {})
                self.assertFalse(bounded_path.exists())
                for _ in range(4):
                    self.assertIsNone(router.on_pre_tool_call(
                        "delegate_task", {"goal": "[sol:high] Inspect and report."}, turn_id="f09-budget",
                    ))
                self.assertEqual(
                    router.on_pre_tool_call(
                        "delegate_task", {"goal": "[sol:high] Inspect and report."}, turn_id="f09-budget",
                    )["action"],
                    "block",
                )

    def test_pre_tool_refuses_a_bad_second_task_without_charging_the_batch(self):
        triage.reset_state()
        self.addCleanup(triage.reset_state)
        cfg = shipped_config()
        cfg["triage"] = {"enabled": True, "max_workers_per_turn": 4, "max_children_per_worker": 1}
        with patch.object(router, "_load_config", return_value=cfg), \
             patch.object(router, "_requested_delegation_targets", return_value=()):
            refused = router.on_pre_tool_call("delegate_task", {
                "tasks": [
                    {"goal": "[sol:high] Inspect and report."},
                    {"goal": "[sol:bogus] Inspect and report."},
                ],
            }, turn_id="f09-second-task")
        self.assertIsNotNone(refused)
        self.assertEqual(refused["action"], "block")
        self.assertEqual(triage._TURN_WORKERS, {})

    def test_malformed_delegation_levels_are_refused_but_root_text_is_not_a_label(self):
        cfg = shipped_config()
        cfg["triage"] = {"enabled": False}
        cfg["task_budget"] = {"enabled": False}
        for level in ("x-high", " high", "high-2"):
            with self.subTest(level=level), patch.object(router, "_load_config", return_value=cfg), \
                 patch.object(router, "_requested_delegation_targets", return_value=()):
                goal = f"[sol:{level}] Inspect and report."
                refused = router.on_pre_tool_call("delegate_task", {"goal": goal}, turn_id=f"f09-{level}")
                self.assertIsNotNone(refused)
                self.assertEqual(refused["action"], "block")
                self.assertIn("Unsupported sol effort", refused["message"])
                self.assertIn("low, medium, high, xhigh, max", refused["message"])
                self.assertIsNone(router._label_match(goal))
                root = router.classify_request(request(goal, cfg), config=cfg)
                unlabelled = router.classify_request(request("Inspect and report.", cfg), config=cfg)
                self.assertEqual((root.tier, root.effort), (unlabelled.tier, unlabelled.effort))

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
