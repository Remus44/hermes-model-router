import contextlib
import copy
import importlib
import importlib.util
import json
import re
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

import model_router as router
from model_router import triage, usage_guard, worker_admission


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


@contextlib.contextmanager
def plugin_without_reasoning_effort():
    """Yield a freshly loaded plugin (or None) while the effort module stays blocked.

    The import blocker is removed only when the ``with`` block ends, so every
    assertion made inside it runs against the missing-host-module state.
    """
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
            plugin = None
            spec = importlib.util.spec_from_file_location(
                module_name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
            )
            if spec is not None and spec.loader is not None:
                candidate = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = candidate
                try:
                    spec.loader.exec_module(candidate)
                    plugin = candidate
                except ModuleNotFoundError:
                    plugin = None
            yield plugin
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
    cfg["preferences"] = {}
    cfg["cooldown"] = {}
    cfg["logging"] = {"enabled": False}
    return cfg


def request(text, cfg):
    return {
        "model": cfg["models"]["terra"],
        "messages": [{"role": "user", "content": text}],
        "temperature": 0.2,
    }


class ExactEffortLabelTests(unittest.TestCase):
    def setUp(self):
        self._reset_effort_label_cache()
        self.addCleanup(self._reset_effort_label_cache)

    @staticmethod
    def _reset_effort_label_cache():
        router._CODEX_TIER_EFFORTS_CACHE.clear()
        router._CODEX_TIER_EFFORTS_WARNING_EMITTED = False

    def route(self, text, cfg):
        with patch.object(router, "_load_config", return_value=cfg), patch.object(router, "_log_decision"):
            return router.route_llm_request(
                request=request(text, cfg),
                provider="openai-codex",
                model=cfg["models"]["terra"],
                api_call_count=1,
                turn_id="f09-label-wire",
            )

    def route_and_record(self, text, cfg, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            route_log = Path(directory) / "router.jsonl"
            cfg["logging"] = {"enabled": True, "path": str(route_log)}
            with patch.object(router, "_load_config", return_value=cfg):
                route_kwargs = {
                    "request": request(text, cfg),
                    "provider": "openai-codex",
                    "model": cfg["models"]["terra"],
                    "api_call_count": 1,
                    "turn_id": "f09-stated-effort",
                }
                route_kwargs.update(kwargs)
                routed = router.route_llm_request(**route_kwargs)
            record = json.loads(route_log.read_text(encoding="utf-8").strip())
        if routed["metadata"]["tier"] != "refused":
            self.assertEqual(record["effort"], routed["metadata"]["effort"])
            self.assertEqual(record["effort"], routed["request"]["reasoning"]["effort"])
        return routed, record

    def test_plugin_import_and_label_validation_fall_back_when_host_effort_module_is_missing(self):
        cfg = shipped_config()
        with plugin_without_reasoning_effort() as plugin:
            self.assertIsNotNone(plugin, "the plugin must load without agent.reasoning_effort")
            assert plugin is not None
            # The blocker is still installed here: the host module really is unreachable.
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module("agent.reasoning_effort")
            plugin._CODEX_TIER_EFFORTS_CACHE.clear()
            plugin._CODEX_TIER_EFFORTS_WARNING_EMITTED = False
            with self.assertLogs("model_router", level="WARNING") as logged:
                self.assertIsNotNone(plugin._label_match("[sol:max] Inspect and report.", cfg))
                self.assertIsNone(plugin._unsupported_label("[sol:max] Inspect and report.", cfg))
                refused = plugin._unsupported_label("[spark:max] Inspect and report.", cfg)
                self.assertIsNotNone(refused)
                self.assertEqual(refused[:2], ("spark", "max"))
                self.assertEqual(refused[2], FALLBACK_TIER_LEVELS["spark"])
                self.assertEqual(plugin._unsupported_label("[sol:bogus] Inspect and report.", cfg)[0], "sol")
                self.assertEqual(plugin._codex_tier_efforts(cfg), FALLBACK_TIER_LEVELS)
            literal_warnings = [r for r in logged.records if "literal Codex effort labels" in r.getMessage()]
            self.assertEqual(len(literal_warnings), 1, [r.getMessage() for r in logged.records])
            self.assertTrue(plugin._CODEX_TIER_EFFORTS_WARNING_EMITTED)

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
                self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), (tier, expected), routed)
                self.assertEqual(routed["request"]["reasoning"]["effort"], expected)

    def test_stated_levels_survive_the_r12_3_route_matrix_and_match_the_route_record(self):
        for level in ("high", "xhigh", "max"):
            with self.subTest(case="same-tier Sol design", level=level):
                cfg = shipped_config()
                routed, _record = self.route_and_record(
                    f"[sol:{level}] Design the responsive dashboard layout.", cfg,
                )
                self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), ("sol", level))

        with self.subTest(case="soft stated level"):
            soft_cfg = shipped_config()
            soft_cfg["effort"]["terra"] = "high"
            reading = usage_guard.Reading(75.0, None, None, None, time.time())
            with patch.object(usage_guard, "peek", return_value=reading):
                routed, _record = self.route_and_record(
                    "[sol:xhigh] Inspect the implementation and report.", soft_cfg, platform="subagent",
                )
            self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), ("sol", "xhigh"))
            self.assertIn("step-down skipped, stated level", routed["reason"])

        with self.subTest(case="disabled Sol fallback"):
            disabled_sol_cfg = shipped_config()
            disabled_sol_cfg["callable"]["sol"] = False
            routed, _record = self.route_and_record(
                "[sol:xhigh] Inspect the implementation and report.", disabled_sol_cfg,
            )
            self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), ("terra", "xhigh"), routed)

        with self.subTest(case="shipped Spark fallback"):
            shipped_spark_cfg = yaml.safe_load((ROOT / "router_config.yaml").read_text(encoding="utf-8"))
            shipped_spark_cfg["triage"] = {"enabled": False}
            shipped_spark_cfg["orchestration"] = {"enabled": False}
            shipped_spark_cfg["session_policy"] = {"pin_root_parent": False}
            shipped_spark_cfg["shadow"] = {"enabled": False}
            routed, _record = self.route_and_record(
                "[spark:xhigh] Inspect the local implementation and report.", shipped_spark_cfg,
            )
            self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), ("luna", "xhigh"))

        with self.subTest(case="read-only substitution"):
            read_only_cfg = shipped_config()
            routed, _record = self.route_and_record("[luna:high] Fix the parser module.", read_only_cfg)
            self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), ("terra", "high"))

    def test_stated_level_dispatch_refuses_an_unavailable_or_non_read_only_label_before_budgeting(self):
        triage.reset_state()
        self.addCleanup(triage.reset_state)
        unavailable_cfg = shipped_config()
        unavailable_cfg["triage"] = {"enabled": True, "max_workers_per_turn": 4, "max_children_per_worker": 1}
        unavailable_cfg["callable"]["sol"] = False
        with patch.object(router, "_load_config", return_value=unavailable_cfg):
            refused = router.on_pre_tool_call(
                "delegate_task", {"goal": "[sol:xhigh] Inspect the implementation and report."}, turn_id="f09-unavailable",
            )
        self.assertIsNotNone(refused)
        self.assertEqual(refused["action"], "block")
        self.assertIn("sol", refused["message"].casefold())
        self.assertEqual(triage._TURN_WORKERS, {})

        read_only_cfg = shipped_config()
        read_only_cfg["triage"] = {"enabled": True, "max_workers_per_turn": 4, "max_children_per_worker": 1}
        with patch.object(router, "_load_config", return_value=read_only_cfg):
            refused = router.on_pre_tool_call(
                "delegate_task", {"goal": "[luna:high] Fix the parser module."}, turn_id="f09-read-only",
            )
        self.assertIsNotNone(refused)
        self.assertEqual(refused["action"], "block")
        self.assertIn("luna", refused["message"].casefold())
        self.assertIn("read-only", refused["message"].casefold())
        self.assertEqual(triage._TURN_WORKERS, {})

    def test_bare_labels_still_use_the_destination_tier_effort_and_match_the_route_record(self):
        cfg = shipped_config()
        cfg["effort"].update({"sol": "low", "terra": "high"})
        for text, tier, effort in (
            ("[sol] Inspect the implementation and report.", "sol", "low"),
            ("[terra] Implement the bounded repair.", "terra", "high"),
        ):
            with self.subTest(text=text):
                routed, _record = self.route_and_record(text, cfg)
                self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), (tier, effort))

    def test_retry_carries_a_stated_level_to_its_substitute_and_logs_only_after_submission(self):
        cfg = shipped_config()
        cfg["callable"]["spark"] = True
        cfg["effort"]["spark"] = "high"
        with tempfile.TemporaryDirectory() as directory:
            route_log = Path(directory) / "router.jsonl"
            cfg["logging"] = {"enabled": True, "path": str(route_log)}
            submitted = []

            def next_call(_request):
                raise RuntimeError("HTTP 503: upstream connect error")

            def retry_call(retry_request):
                self.assertFalse(route_log.exists(), "retry route was logged before submission")
                submitted.append(copy.deepcopy(retry_request))
                return "recovered"

            with patch.object(router, "_load_config", return_value=cfg):
                result = router.run_llm_with_transient_failover(
                    request={**request("[sol:xhigh] Inspect the implementation and report.", cfg), "model": cfg["models"]["sol"]},
                    next_call=next_call,
                    retry_call=retry_call,
                    provider="openai-codex",
                )
            record = json.loads(route_log.read_text(encoding="utf-8").strip())
        self.assertEqual(result, "recovered")
        self.assertEqual(submitted[0]["model"], cfg["models"]["spark"])
        self.assertEqual(submitted[0]["reasoning"]["effort"], "xhigh")
        self.assertEqual(record["effort"], submitted[0]["reasoning"]["effort"])

    def test_retry_logging_failure_does_not_discard_a_successful_retry(self):
        cfg = shipped_config()
        cfg["callable"]["spark"] = True
        try:
            with patch.object(router, "_load_config", return_value=cfg), \
                 patch.object(router, "_log_decision", side_effect=RuntimeError("log preview failed")):
                result = router.run_llm_with_transient_failover(
                    request={**request("[sol:xhigh] Inspect and report.", cfg), "model": cfg["models"]["sol"]},
                    next_call=Mock(side_effect=RuntimeError("HTTP 503: upstream connect error")),
                    retry_call=Mock(return_value="recovered"), provider="openai-codex", platform="subagent",
                )
        except RuntimeError as error:
            self.fail(f"successful retry was discarded by logging: {error}")
        self.assertEqual(result, "recovered")

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

    def test_pre_tool_refuses_bad_level_before_triage_charges_the_worker_budget(self):
        triage.reset_state()
        self.addCleanup(triage.reset_state)
        cfg = shipped_config()
        cfg["triage"] = {
            "enabled": True, "max_workers_per_turn": 4, "max_children_per_worker": 1,
        }
        with patch.object(router, "_load_config", return_value=cfg), \
             patch.object(router, "_requested_delegation_targets", return_value=()):
            before = dict(triage._TURN_WORKERS)
            refused = router.on_pre_tool_call(
                "delegate_task", {"goal": "[sol:bogus] Inspect and report."}, turn_id="f09-budget",
            )
            self.assertIsNotNone(refused)
            self.assertEqual(refused["action"], "block")
            self.assertIn("Unsupported sol effort 'bogus'", refused["message"])
            self.assertEqual(triage._TURN_WORKERS, before)
            self.assertEqual(triage._TURN_WORKERS, {})
            for _ in range(4):
                self.assertIsNone(router.on_pre_tool_call(
                    "delegate_task", {"goal": "[sol:high] Inspect and report."}, turn_id="f09-budget",
                ))
            self.assertEqual(sum(triage._TURN_WORKERS.values()), 4)
            self.assertEqual(
                router.on_pre_tool_call(
                    "delegate_task", {"goal": "[sol:high] Inspect and report."}, turn_id="f09-budget",
                )["action"],
                "block",
            )

    def test_pre_tool_refuses_bad_level_before_the_bounded_task_budget_is_written(self):
        triage.reset_state()
        self.addCleanup(triage.reset_state)
        cfg = shipped_config()
        cfg["triage"] = {"enabled": False}
        with tempfile.TemporaryDirectory() as directory:
            bounded_path = Path(directory) / "bounded-budget.jsonl"
            cfg["task_budget"] = {"enabled": True, "path": str(bounded_path)}
            with patch.object(router, "_load_config", return_value=cfg), \
                 patch.object(router, "_requested_delegation_targets", return_value=()):
                refused = router.on_pre_tool_call(
                    "delegate_task", {"goal": "[sol:bogus] Inspect and report."}, turn_id="f09-bounded",
                )
                self.assertIsNotNone(refused)
                self.assertEqual(refused["action"], "block")
                self.assertIn("Unsupported sol effort 'bogus'", refused["message"])
                self.assertFalse(bounded_path.exists())
                # The refused call used no allowance: the one permitted worker is still free.
                self.assertIsNone(router.on_pre_tool_call(
                    "delegate_task", {"goal": "[sol:high] Inspect and report."}, turn_id="f09-bounded",
                ))
                records = [json.loads(line) for line in bounded_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual([record["event"] for record in records], ["bounded_worker_admitted"])
                second = router.on_pre_tool_call(
                    "delegate_task", {"goal": "[sol:high] Inspect and report."}, turn_id="f09-bounded",
                )
                self.assertIsNotNone(second)
                self.assertEqual(second["action"], "block")
                self.assertIn("Low-risk task routing budget reached", second["message"])
                self.assertEqual(len(bounded_path.read_text(encoding="utf-8").splitlines()), 1)

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

    def test_a_supported_level_followed_by_text_is_not_refused_and_other_malformed_labels_are(self):
        # (goal, expected _unsupported_label (tier, level) or None, is a valid label)
        cases = (
            ("[sol:high]text", None, False),
            ("[sol:high],", None, False),
            ("[sol:HIGH]text", None, False),
            ("[terra:xhigh]: do it", None, False),
            ("[sol:high]. Do x", None, False),
            ("[sol:high] x", None, True),
            ("[sol] x", None, True),
            ("[sol:bogus] x", ("sol", "bogus"), False),
            ("[sol:x-high] x", ("sol", "x-high"), False),
            ("[sol: high] x", ("sol", " high"), False),
            ("[sol:] x", ("sol", ""), False),
            ("[sol:high-2] x", ("sol", "high-2"), False),
            ("[spark:max]text", ("spark", "max"), False),
        )
        cfg = shipped_config()
        for goal, expected, is_label in cases:
            with self.subTest(goal=goal):
                unsupported = router._unsupported_label(goal, cfg)
                self.assertEqual(unsupported[:2] if unsupported else None, expected)
                self.assertEqual(router._label_match(goal, cfg) is not None, is_label)
                if not is_label:
                    stripped = re.sub(r"^\s*\[[^\]]*\]", "", goal)
                    root = router.classify_request(request(goal, cfg), config=cfg)
                    plain = router.classify_request(request(stripped, cfg), config=cfg)
                    self.assertEqual((root.tier, root.effort), (plain.tier, plain.effort))

    def test_guard_dispatches_a_supported_level_followed_by_text_and_blocks_a_malformed_level(self):
        cfg = shipped_config()
        for goal, dispatched in (("[sol:high]text", True), ("[sol:x-high] x", False)):
            with self.subTest(goal=goal):
                call = Mock(return_value="dispatched")
                with patch.object(router, "_load_config", return_value=cfg), \
                     patch.object(worker_admission, "delegate_task_route", return_value=("openai-codex", cfg["models"]["terra"])), \
                     patch.object(router, "_delegation_targets_detail", return_value={}), \
                     patch.object(router, "_delegated_claude_review_status", return_value=(None, "not a review")), \
                     patch.object(worker_admission, "refusal", return_value=""):
                    result = worker_admission.guard_tool_execution(
                        tool_name="delegate_task", args={"tasks": [{"goal": goal}]}, next_call=call,
                    )
                self.assertEqual(call.call_count, 1 if dispatched else 0)
                if dispatched:
                    self.assertEqual(result, "dispatched")
                else:
                    self.assertIn("Unsupported sol effort 'x-high'", result)

    def test_pre_tool_hook_dispatches_a_supported_level_followed_by_text_and_blocks_a_malformed_level(self):
        triage.reset_state()
        self.addCleanup(triage.reset_state)
        cfg = shipped_config()
        cfg["triage"] = {"enabled": False}
        cfg["task_budget"] = {"enabled": False}
        for goal, dispatched in (("[sol:high]text", True), ("[sol:x-high] x", False)):
            with self.subTest(goal=goal), patch.object(router, "_load_config", return_value=cfg), \
                 patch.object(router, "_requested_delegation_targets", return_value=()):
                result = router.on_pre_tool_call("delegate_task", {"goal": goal}, turn_id="f09-boundary")
                if dispatched:
                    self.assertIsNone(result)
                else:
                    self.assertIsNotNone(result)
                    self.assertEqual(result["action"], "block")
                    self.assertIn("Unsupported sol effort 'x-high'", result["message"])

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

    def test_host_skills_delta_note_does_not_change_luna_read_only_classification(self):
        cfg = shipped_config()
        goal = "[luna] Inspect the configuration and report only."
        skills_delta = (
            "[System: The skills available in this conversation changed after its system prompt was built.\n"
            "Now available (load with skill_view(name) when relevant, like the skills listed in the system prompt):\n"
            "    - creative: Create and edit artifacts.]"
        )
        plain = router.classify_request(request(goal, cfg), config=cfg)
        noted = router.classify_request(request(goal + "\n\n" + skills_delta, cfg), config=cfg)
        self.assertEqual((plain.tier, noted.tier), ("luna", "luna"))

    def test_design_gate_carries_any_validated_label_level_to_sol_or_refuses_an_unsupported_sol_level(self):
        cfg = shipped_config()
        routed, _record = self.route_and_record("[terra:xhigh] Design the responsive dashboard layout.", cfg)
        self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), ("sol", "xhigh"))

        unsupported_cfg = shipped_config()
        unsupported_cfg["models"]["sol"] = unsupported_cfg["models"]["spark"]
        with self.assertRaisesRegex(RuntimeError, "cannot run on substitute tier 'sol'"):
            router.classify_request(
                request("[terra:max] Design the responsive dashboard layout.", unsupported_cfg),
                config=unsupported_cfg,
            )

    def test_spark_image_reroute_carries_the_stated_level(self):
        cfg = shipped_config()
        image_request = request("[spark:xhigh] Inspect this image and report.", cfg)
        image_request["messages"] = [{"role": "user", "content": [
            {"type": "input_text", "text": "[spark:xhigh] Inspect this image and report."},
            {"type": "input_image", "image_url": "https://example.invalid/image.png"},
        ]}]
        with patch.object(router, "_load_config", return_value=cfg), patch.object(router, "_log_decision"):
            routed = router.route_llm_request(
                request=image_request, provider="openai-codex", model=cfg["models"]["terra"],
                api_call_count=1, turn_id="f09-image",
            )
        self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), ("terra", "xhigh"), routed)
        self.assertEqual(routed["request"]["reasoning"]["effort"], "xhigh")

    def test_root_spark_label_deferred_to_default_keeps_the_stated_level(self):
        cfg = shipped_config()
        cfg["default_model"] = "terra"
        routed, _record = self.route_and_record("[spark:xhigh] Inspect the local implementation and report.", cfg)
        self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), ("terra", "xhigh"), routed)

    def test_root_spark_label_validates_the_level_on_the_preserved_destination(self):
        """O-1: the shipped cross-provider default is a hop, not the destination."""
        cfg = shipped_config()
        self.assertEqual(cfg["default_model"], "qwen")
        self.assertNotEqual(cfg["tier_providers"]["qwen"], cfg["tier_providers"]["terra"])
        for qwen_callable in (True, False):
            for level in ("low", "high"):
                with self.subTest(qwen_callable=qwen_callable, level=level):
                    cfg["callable"]["qwen"] = qwen_callable
                    routed, _record = self.route_and_record(f"[spark:{level}] Inspect the parser module.", cfg)
                    self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), ("terra", level),
                                     routed)

        levels = {tier: tuple(values) for tier, values in router._codex_tier_efforts(cfg).items()}
        levels["terra"] = tuple(level for level in levels["terra"] if level != "low")
        with patch.object(router, "_codex_tier_efforts", return_value=levels):
            refused, _record = self.route_and_record("[spark:low] Inspect the parser module.", cfg)
        self.assertEqual(refused["metadata"]["tier"], "refused", refused)
        self.assertIn("cannot run on substitute tier 'terra'", refused["metadata"]["refusal"])

    def test_root_spark_label_with_an_unavailable_active_tier_refuses_visibly(self):
        """O-1 review: the deferral must not turn a visible refusal into a fail-open RuntimeError."""
        cfg = shipped_config()
        cfg["callable"].update({"terra": False, "qwen": True})
        try:
            routed, _record = self.route_and_record("[spark:medium] Inspect the parser module.", cfg)
        except RuntimeError as error:
            self.fail(f"fail-open RuntimeError instead of a visible refusal: {error}")
        self.assertEqual(routed["metadata"]["tier"], "refused", routed)

    def test_root_spark_label_on_a_switched_off_default_is_validated_on_its_fallback(self):
        """O-1 review: with Qwen off, a Sol parent's root Spark label lands on the Qwen fallback."""
        cfg = shipped_config()
        cfg["callable"]["qwen"] = False
        routed, _record = self.route_and_record("[spark:high] Inspect the parser module.", cfg,
                                                model=cfg["models"]["sol"])
        self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]),
                         (cfg["fallbacks"]["qwen"], "high"), routed)

    def test_spark_leaf_demotion_uses_the_dispatch_read_only_predicate_and_carries_the_level(self):
        cfg = shipped_config()
        with patch.object(router, "_is_spark_read_only_work", return_value=False), \
             patch.object(router, "_is_spark_read_only_request", return_value=True):
            routed, _record = self.route_and_record(
                "[spark:xhigh] Inspect the module imports.", cfg,
                platform="subagent", turn_id="f09:sa-1",
                model=cfg["models"]["spark"],
            )
        self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), ("terra", "xhigh"), routed)

    def test_spark_quota_fallback_carries_the_stated_level(self):
        cfg = shipped_config()
        with patch.object(router, "_spark_quota_exhausted_for_turn", return_value=True):
            routed, _record = self.route_and_record(
                "[spark:xhigh] Inspect the implementation and report.", cfg,
                platform="subagent", turn_id="f09-quota",
                model=cfg["models"]["spark"],
            )
        self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), ("luna", "xhigh"))

    def test_design_preference_replacement_carries_the_stated_level(self):
        cfg = shipped_config()
        cfg["preferences"] = {"design": ["terra"]}
        routed, _record = self.route_and_record("[sol:xhigh] Design the responsive dashboard layout.", cfg)
        self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), ("terra", "xhigh"), routed)

    def test_cross_provider_preservation_carries_the_stated_level(self):
        cfg = shipped_config()
        source = router.RouteDecision("terra", cfg["models"]["terra"], "explicit", "xhigh", stated_effort="xhigh")
        preserved = router._decision_with_carried_stated_effort(
            "sol", "cross-provider route preserved for the active session", cfg, source,
        )
        self.assertEqual((preserved.tier, preserved.effort), ("sol", "xhigh"))

    def test_request_seam_exposes_an_unsupported_substitute_refusal_before_execution(self):
        cfg = shipped_config()
        cfg["models"]["sol"] = cfg["models"]["spark"]
        with patch.object(router, "_load_config", return_value=cfg), patch.object(router, "_log_decision"):
            routed = router.route_llm_request(
                request=request("[terra:max] Design the responsive dashboard layout.", cfg),
                provider="openai-codex", model=cfg["models"]["terra"], api_call_count=1,
                turn_id="f09-refusal",
            )
        self.assertIn("refusal", routed["metadata"])
        self.assertIn("cannot run on substitute tier 'sol'", routed["metadata"]["refusal"])
        downstream = Mock()
        with patch.object(router, "_load_config", return_value=cfg):
            stopped = router.run_llm_with_transient_failover(
                request=routed["request"], next_call=downstream, provider="openai-codex", platform="subagent",
            )
        downstream.assert_not_called()
        self.assertEqual(stopped.usage.total_tokens, 0)

    def test_retry_unsupported_substitute_is_a_visible_stop_without_a_downstream_call(self):
        cfg = shipped_config()
        cfg["fallbacks"]["sol"] = "spark"
        downstream = Mock(side_effect=RuntimeError("HTTP 503: upstream connect error"))
        retry = Mock()
        with patch.object(router, "_load_config", return_value=cfg):
            stopped = router.run_llm_with_transient_failover(
                request={**request("[sol:max] Inspect and report.", cfg), "model": cfg["models"]["sol"]},
                next_call=downstream, retry_call=retry, provider="openai-codex", platform="subagent",
            )
        downstream.assert_called_once()
        retry.assert_not_called()
        self.assertEqual(stopped.usage.total_tokens, 0)

    def test_stated_soft_step_down_reason_says_it_was_skipped(self):
        cfg = shipped_config()
        cfg["effort"]["terra"] = "high"
        reading = usage_guard.Reading(75.0, None, None, None, time.time())
        with patch.object(usage_guard, "peek", return_value=reading):
            routed, _record = self.route_and_record(
                "[sol:xhigh] Inspect and report.", cfg, platform="subagent",
            )
        self.assertIn("step-down skipped, stated level", routed["reason"])

    def test_stated_label_account_hard_closure_stops_through_execution_middleware_without_provider_call(self):
        from hermes_cli.middleware import run_llm_execution_middleware

        cfg = shipped_config()
        cfg["usage_guard"] = {"accounts": {"openai-codex": {"soft_percent": 70, "hard_percent": 90}}}
        manager = types.SimpleNamespace(_middleware={"llm_execution": [router.run_llm_with_transient_failover]},
                                        _report_hook_failure=Mock())
        downstream = Mock()
        reading = usage_guard.Reading(95.0, 0.0, None, None, time.time())
        worker_request = request("[sol:xhigh] Inspect and report.", cfg)
        with patch.object(router, "_load_config", return_value=cfg), \
             patch.object(usage_guard, "peek", return_value=reading), \
             patch("hermes_cli.plugins._delivery_manager", return_value=manager):
            stopped = run_llm_execution_middleware(
                worker_request, downstream, provider="openai-codex", platform="subagent", turn_id="f09-hard",
            )
        downstream.assert_not_called()
        manager._report_hook_failure.assert_not_called()
        self.assertEqual(stopped.usage.total_tokens, 0)
        self.assertIn("ROUTER WORKER STOPPED", stopped.output_text)

    def test_stated_dispatch_refusal_names_the_exact_tier_for_cooling_and_batches(self):
        cfg = shipped_config()
        cfg["triage"] = {"enabled": True, "max_workers_per_turn": 4, "max_children_per_worker": 1}
        with patch.object(router, "_load_config", return_value=cfg), \
             patch.object(router, "_tier_cooldown_remaining", return_value=30), \
             patch.object(router, "_requested_delegation_targets", return_value=()):
            cooled = router.on_pre_tool_call("delegate_task", {"goal": "[sol:high] Inspect and report."}, turn_id="f09-cool")
            batched = router.on_pre_tool_call("delegate_task", {"tasks": [
                {"goal": "[sol:high] Inspect and report."}, {"goal": "[sol:xhigh] Inspect and report."},
            ]}, turn_id="f09-batch")
        self.assertEqual(cooled["message"], "Stated effort label [sol:high] is refused because sol is cooling down for another 30s.")
        self.assertEqual(batched["action"], "block")

    def test_stated_label_block_applies_to_delegate_task_not_delegate_claude(self):
        cfg = shipped_config()
        cfg["callable"].update({"sol": False, "sonnet5": True})
        cfg["triage"] = {"enabled": False}
        with patch.object(router, "_load_config", return_value=cfg), patch.object(router, "_requested_delegation_targets", return_value=()):
            task = router.on_pre_tool_call("delegate_task", {"goal": "[sol:high] Inspect and report."}, turn_id="f09-task")
            claude = router.on_pre_tool_call("delegate_claude", {"goal": "[sol:high] Inspect and report."}, turn_id="f09-claude")
        self.assertEqual(task["action"], "block")
        self.assertIsNone(claude)

    def test_spark_preservation_after_callable_fallback_keeps_the_stated_level_on_wire_and_route_record(self):
        cfg = shipped_config()
        cfg["callable"]["luna"] = False
        cfg["effort"]["spark"] = "medium"
        routed, record = self.route_and_record(
            "[luna:xhigh] Inspect the parser and report.", cfg,
            platform="subagent", turn_id="f09:sa-spark", model=cfg["models"]["spark"],
        )
        self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), ("spark", "xhigh"), routed)
        self.assertEqual(routed["request"]["reasoning"]["effort"], "xhigh")
        self.assertEqual(record["effort"], "xhigh")

        unsupported_cfg = shipped_config()
        unsupported_cfg["callable"]["luna"] = False
        refused, record = self.route_and_record(
            "[luna:max] Inspect the parser and report.", unsupported_cfg,
            platform="subagent", turn_id="f09:sa-spark-refusal", model=unsupported_cfg["models"]["spark"],
        )
        self.assertEqual(refused["metadata"]["tier"], "refused")
        self.assertIn("cannot run on substitute tier 'spark'", refused["metadata"]["refusal"])
        self.assertEqual(record["tier"], "refused")

    def test_design_destination_validation_applies_only_when_the_design_gate_selects_sol(self):
        cfg = shipped_config()
        cfg["models"]["sol"] = cfg["models"]["spark"]
        unchanged, record = self.route_and_record("[terra:max] Inspect the parser and report.", cfg)
        self.assertEqual((unchanged["metadata"]["tier"], unchanged["metadata"]["effort"]), ("terra", "max"), unchanged)
        self.assertNotIn("refusal", unchanged["metadata"])
        self.assertEqual(record["effort"], "max")

        stopped, record = self.route_and_record("[terra:max] Design the responsive dashboard layout.", cfg)
        self.assertEqual(stopped["metadata"]["tier"], "refused")
        self.assertIn("cannot run on substitute tier 'sol'", stopped["metadata"]["refusal"])
        self.assertEqual(record["tier"], "refused")

    def test_spark_leaf_site_uses_the_plan_read_only_predicate_not_root_request_predicate(self):
        cfg = shipped_config()
        goal = "[spark:xhigh] Assess parser branch risk."
        self.assertTrue(router._is_spark_read_only_work(goal))
        self.assertFalse(router._is_spark_read_only_request(goal))
        routed, _record = self.route_and_record(
            goal, cfg, platform="subagent", turn_id="f09:sa-plan-read-only", model=cfg["models"]["spark"],
        )
        self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), ("spark", "xhigh"), routed)

    def test_cross_provider_preservation_site_keeps_stated_level_and_mutation_loses_it(self):
        cfg = shipped_config()
        cfg["tier_providers"]["sol"] = "anthropic"
        routed, record = self.route_and_record("[sol:xhigh] Inspect the parser and report.", cfg)
        self.assertEqual((routed["metadata"]["tier"], routed["metadata"]["effort"]), ("terra", "xhigh"), routed)
        self.assertEqual((routed["request"]["reasoning"]["effort"], record["effort"]), ("xhigh", "xhigh"))

        def old_plain_decision(target, reason, config, source, *, kind=None):
            return router._decision(target, reason, config, kind=source.kind if kind is None else kind)

        with patch.object(router, "_decision_with_carried_stated_effort", side_effect=old_plain_decision):
            reverted = self.route("[sol:xhigh] Inspect the parser and report.", cfg)
        self.assertEqual((reverted["metadata"]["tier"], reverted["metadata"]["effort"]),
                         ("terra", cfg["effort"]["terra"]))

    def test_preferred_fallback_skips_an_incompatible_effort_candidate(self):
        cfg = shipped_config()
        cfg["callable"]["sol"] = False
        cfg["preferences"] = {"default": ["spark", "terra"]}
        source = router.RouteDecision("sol", cfg["models"]["sol"], "explicit [sol:max] override", "max",
                                      kind="default", stated_effort="max")
        fallback = router._resolve_callable_fallback(source, cfg)
        self.assertEqual((fallback.tier, fallback.effort), ("terra", "max"))

        def old_preference_resolution(target, reason, config, source, *, kind=None):
            return router._decision(target, reason, config, kind=source.kind if kind is None else kind)

        with patch.object(router, "_decision_with_carried_stated_effort", side_effect=old_preference_resolution):
            reverted = router._resolve_callable_fallback(source, cfg)
        self.assertEqual((reverted.tier, reverted.effort), ("spark", cfg["effort"]["spark"]))

    def test_failed_retry_makes_no_route_log_call(self):
        cfg = shipped_config()
        original = Mock(side_effect=RuntimeError("HTTP 503: upstream connect error"))
        retry = Mock(side_effect=RuntimeError("retry unavailable"))
        with patch.object(router, "_load_config", return_value=cfg), \
             patch.object(router, "_log_decision") as logged:
            with self.assertRaisesRegex(RuntimeError, "retry unavailable"):
                router.run_llm_with_transient_failover(
                    request={**request("[sol:xhigh] Inspect and report.", cfg), "model": cfg["models"]["sol"]},
                    next_call=original, retry_call=retry, provider="openai-codex", platform="subagent",
                )
        original.assert_called_once()
        retry.assert_called_once()
        logged.assert_not_called()

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


class NormalizedDelegationPreflightTests(unittest.TestCase):
    """The registered pre-tool hook must parse labels as classification does."""

    def setUp(self):
        from hermes_cli import plugins
        from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

        self.manager = PluginManager()
        self.manager._discovered = True
        context = PluginContext(
            PluginManifest(name="f09-normalized-preflight", source="test", key="f09-normalized-preflight"),
            self.manager,
        )
        self.root = SimpleNamespace(session_id="f09-normalized-root", _delegate_depth=0)
        triage.reset_state()
        self.addCleanup(triage.reset_state)
        manager_patch = patch.object(plugins, "_plugin_manager", self.manager)
        manager_patch.start()
        self.addCleanup(manager_patch.stop)
        delivery_patch = patch("hermes_cli.plugins._delivery_manager", return_value=self.manager)
        delivery_patch.start()
        self.addCleanup(delivery_patch.stop)
        with patch.object(router.claude_delegation, "register", return_value=True):
            router.register(context)
        self.assertIn(router.on_pre_tool_call, self.manager._hooks["pre_tool_call"])
        self.addCleanup(self._dispose_registrations)

    def _dispose_registrations(self):
        for registration in reversed(list(self.manager._registration_order)):
            registration.dispose()

    def _deliver(self, args, cfg, *, turn_id, cooldown=0):
        from hermes_cli import plugins

        raw_dispatch = Mock(return_value="raw dispatch")
        with patch.object(router, "_load_config", return_value=cfg), \
             patch.object(router, "_delegation_targets_detail", return_value={}), \
             patch.object(router, "_tier_cooldown_remaining", return_value=cooldown), \
             patch.object(triage, "_caller", return_value=self.root):
            block, _modified_args = plugins._dispatch_pre_tool_call_hooks(
                "delegate_task", args, session_id="f09-normalized-root", turn_id=turn_id,
            )
        return block, raw_dispatch, ("" if block else raw_dispatch())

    @staticmethod
    def _cfg():
        cfg = shipped_config()
        cfg["triage"] = {"enabled": True, "max_workers_per_turn": 4, "max_children_per_worker": 1}
        return cfg

    def test_registered_hook_refuses_normalized_stated_labels_before_admission_or_raw_dispatch(self):
        cases = (
            ("luna write", "[lúna:high] Fix the parser module.", self._cfg(), 0,
             "restricted to non-design read-only"),
            ("spark write", "[spárk:high] Fix the parser module.", self._cfg(), 0,
             "restricted to non-design read-only"),
            ("unavailable", "[sól:high] Inspect and report.",
             {**self._cfg(), "callable": {**self._cfg()["callable"], "sol": False}}, 0, "sol is switched off"),
            ("cooling", "[térra:high] Inspect and report.", self._cfg(), 30,
             "terra is cooling down for another 30s"),
            ("unsupported", "[sól:bogus] Inspect and report.", self._cfg(), 0,
             "Unsupported sol effort 'bogus'"),
        )
        for name, goal, cfg, cooldown, expected in cases:
            with self.subTest(name=name):
                triage.reset_state()
                block, raw_dispatch, _result = self._deliver({"goal": goal}, cfg, turn_id=f"f09-normalized:{name}",
                                                              cooldown=cooldown)
                self.assertIsNotNone(block, "normalized label escaped registered preflight")
                message = block or ""
                self.assertIn(expected, message)
                raw_dispatch.assert_not_called()
                self.assertEqual(triage._TURN_WORKERS, {}, "preflight refusal consumed worker admission")
                with patch.object(router, "_load_config", return_value=cfg), \
                     patch.object(router, "_tier_cooldown_remaining", return_value=cooldown), \
                     patch.object(triage, "enabled", return_value=False):
                    direct = router.on_pre_tool_call(
                        "delegate_task", {"goal": goal}, turn_id=f"f09-normalized:direct:{name}",
                    )
                self.assertEqual((direct or {}).get("message"), message)

    def test_registered_hook_keeps_normalized_controls_and_boundaries(self):
        cases = (
            ("ascii refusal", {"goal": "[luna:high] Fix the parser module."}, False),
            ("normalized read only", {"goal": "[lúna:high] Inspect the parser and report."}, True),
            ("normalized bare", {"goal": "[sól] Inspect the parser and report."}, True),
            ("non-spawn action", {"action": "stop", "goal": "[sól:bogus] Inspect and report."}, True),
            ("supported no-space boundary", {"goal": "[sól:high]Fix the parser module."}, True),
            ("malformed normalized", {"goal": "[sól:x-high] Inspect and report."}, False),
            ("batched", {"tasks": [
                {"goal": "[sol:high] Inspect and report."},
                {"goal": "[sól:bogus] Inspect and report."},
            ]}, False),
        )
        for name, args, admitted in cases:
            with self.subTest(name=name):
                triage.reset_state()
                block, raw_dispatch, result = self._deliver(args, self._cfg(), turn_id=f"f09-normalized:{name}")
                if admitted:
                    self.assertIsNone(block)
                    self.assertEqual(result, "raw dispatch")
                    raw_dispatch.assert_called_once()
                    expected_admission = {} if name == "non-spawn action" else {f"f09-normalized:{name}": 1}
                    self.assertEqual(triage._TURN_WORKERS, expected_admission)
                else:
                    self.assertIsNotNone(block)
                    raw_dispatch.assert_not_called()
                    self.assertEqual(triage._TURN_WORKERS, {}, "preflight refusal consumed worker admission")


if __name__ == "__main__":
    unittest.main()
