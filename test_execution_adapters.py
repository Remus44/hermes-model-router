"""S05 adapters wrap existing execution seams without shared routing mutation."""
import inspect
import json
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace

from model_router import execution_adapters as adapters
from model_router import execution_contracts as contracts
from model_router import runtime_capabilities as runtime


def identity(transport, *, alias="terra", provider="openai-codex", model="gpt-5.6-terra",
             selection_mode="profile_preferred", effort="unknown"):
    return contracts.TargetIdentity(
        provider=provider, account=provider, transport=transport, alias=alias,
        selection_mode=selection_mode,
        requested=contracts.ModelFact(model, "operator_request"),
        resolved=contracts.ModelFact(model, "configured_target"),
        observed=contracts.ModelFact(),
        effort=contracts.EffortFact(effort, "unknown", "not_observed"),
    )


def request(target, *, attempt="attempt-001"):
    return contracts.ExecutionRequest(
        workflow_id="wf-001", plan_version=1, task_id="task-001", attempt_id=attempt,
        goal="Implement one bounded change.", acceptance_criteria=("tests pass",),
        context_reference="repo:README.md@abc123", target=target,
        repository="repo:/work/router", workspace="workspace:/work/router",
        permissions=("read", "write"), tool_requirements=("terminal",), mutating=True,
        write_scope=("one.py",), timeout_seconds=60, deadline_epoch_ms=999999,
        attempt_budget=1, verification_policy="required", substitution_policy="forbid",
    )


def snapshot(*, codex_model="unsupported", claude_submission="supported", bridge="supported"):
    def cap(name, status, reason="fixture"):
        return runtime.Capability(name, status, reason=reason)
    return runtime.RuntimeSnapshot(
        1, "fixture", runtime.Fact("supported", 2), runtime.Fact("supported", 1),
        runtime.Fact("supported", False), (), (
            runtime.AdapterCapabilities("hermes_codex", (
                cap("submission", "supported"), cap("model_parameter", codex_model),
                cap("effort_application", "unknown"), cap("identity_observability", "unknown"),
                cap("cancellation", "supported"), cap("fallback_ownership", "unknown"),
            )),
            runtime.AdapterCapabilities("hermes_claude", (
                cap("submission", claude_submission), cap("effort_application", bridge),
                cap("identity_observability", "unknown"), cap("cancellation", "unknown"),
                cap("fallback_ownership", "unknown"),
            )),
            runtime.AdapterCapabilities("claude_cli", (
                cap("submission", "supported"), cap("exact_model", "supported"),
                cap("identity_observability", "supported"), cap("effort_application", "unknown"),
                cap("cancellation", "unknown"), cap("fallback_ownership", "unknown"),
            )),
        ), (), True,
    )


class HermesCodexAdapterTests(unittest.TestCase):
    def test_exact_codex_is_unsupported_when_host_cannot_enforce_and_observe_model(self):
        adapter = adapters.HermesCodexAdapter(lambda **_kwargs: "{}", cfg={}, runtime_snapshot=snapshot())
        exact = request(identity("hermes_codex", selection_mode="exact"))
        eligibility = adapter.can_execute(exact, snapshot())
        self.assertEqual(eligibility.status, "unsupported")
        self.assertIn("exact", " ".join(eligibility.reasons))

    def test_profile_codex_uses_only_supported_task_fields_and_keeps_result_unknown(self):
        calls = []
        parent = SimpleNamespace(_delegate_depth=0)
        def delegate_task(**kwargs):
            calls.append(kwargs)
            return json.dumps({"status": "dispatched", "delegation_id": "deleg-1"})
        adapter = adapters.HermesCodexAdapter(delegate_task, cfg={}, runtime_snapshot=snapshot(),
                                              parent_agent=lambda: parent,
                                              admission=lambda *_args, **_kwargs: "")
        submitted = adapter.submit(request(identity("hermes_codex")))
        self.assertTrue(submitted.accepted)
        self.assertEqual(submitted.handle, "deleg-1")
        # The supported middleware selector is the goal label; no model/provider/effort field exists.
        self.assertEqual(calls, [{"tasks": [{"goal": "[terra] Implement one bounded change.",
                                               "context": "repo:README.md@abc123"}],
                                  "parent_agent": parent, "background": True}])
        self.assertEqual(adapter.result("deleg-1").status, "unknown")

    def test_kwargs_bind_to_the_installed_host_delegate_task_signature(self):
        from tools import delegate_tool
        calls = []
        adapter = adapters.HermesCodexAdapter(lambda **kw: calls.append(kw) or json.dumps({"delegation_id": "d"}),
                                              cfg={}, runtime_snapshot=snapshot(),
                                              parent_agent=lambda: SimpleNamespace(_delegate_depth=1),
                                              admission=lambda *_a, **_k: "")
        self.assertTrue(adapter.submit(request(identity("hermes_codex"))).accepted)
        inspect.signature(delegate_tool.delegate_task).bind(**calls[0])
        self.assertIs(calls[0]["background"], False)  # host rule: synchronous below the top level

    def test_no_active_parent_or_unroutable_alias_never_dispatches(self):
        calls = []
        def delegate_task(**kwargs):
            calls.append(kwargs)
            return "{}"
        adapter = adapters.HermesCodexAdapter(delegate_task, cfg={}, runtime_snapshot=snapshot(),
                                              parent_agent=lambda: None, admission=lambda *_a, **_k: "")
        self.assertEqual(adapter.submit(request(identity("hermes_codex"))).rejection.failure_class, "capability")
        with_parent = adapters.HermesCodexAdapter(delegate_task, cfg={}, runtime_snapshot=snapshot(),
                                                  parent_agent=lambda: SimpleNamespace(_delegate_depth=0),
                                                  admission=lambda *_a, **_k: "")
        odd = request(identity("hermes_codex", alias="qwen"))
        self.assertEqual(with_parent.can_execute(odd, snapshot()).status, "unsupported")
        self.assertEqual(calls, [])

    def test_luna_and_spark_do_not_receive_mutating_work_where_router_would_substitute(self):
        adapter = adapters.HermesCodexAdapter(lambda **_kw: "{}", cfg={}, runtime_snapshot=snapshot(),
                                              parent_agent=lambda: SimpleNamespace(_delegate_depth=0),
                                              admission=lambda *_a, **_k: "")
        for alias in ("luna", "spark"):
            eligibility = adapter.can_execute(request(identity("hermes_codex", alias=alias)), snapshot())
            self.assertEqual(eligibility.status, "unsupported", alias)

    def test_host_error_text_is_preserved_in_the_typed_rejection(self):
        adapter = adapters.HermesCodexAdapter(lambda **_kw: json.dumps({"error": "Delegation depth limit reached"}),
                                              cfg={}, runtime_snapshot=snapshot(),
                                              parent_agent=lambda: SimpleNamespace(_delegate_depth=0),
                                              admission=lambda *_a, **_k: "")
        rejected = adapter.submit(request(identity("hermes_codex")))
        self.assertFalse(rejected.accepted)
        self.assertIn("depth limit", rejected.rejection.message.summary)


class HermesClaudeAdapterTests(unittest.TestCase):
    def test_mixed_tiers_use_separate_calls_and_haiku_effort_is_not_applicable(self):
        calls = []
        def delegate_claude(args):
            calls.append(args)
            return json.dumps({"delegation_id": f"deleg-{len(calls)}", "claude_tier": args["tier"]})
        adapter = adapters.HermesClaudeAdapter(delegate_claude, cfg={}, runtime_snapshot=snapshot(), admission=lambda *_args, **_kwargs: "")
        haiku = request(identity("hermes_claude", alias="haiku", provider="anthropic",
                                 model="claude-haiku-4-5", effort="not_applicable"), attempt="attempt-haiku")
        sonnet = request(identity("hermes_claude", alias="sonnet5", provider="anthropic",
                                  model="claude-sonnet-5-5", effort="high"), attempt="attempt-sonnet")
        outcomes = adapter.submit_batch((haiku, sonnet))
        self.assertEqual([outcome.handle for outcome in outcomes], ["deleg-1", "deleg-2"])
        self.assertEqual([call["tier"] for call in calls], ["haiku", "sonnet"])
        self.assertEqual(calls[0]["tasks"][0]["context"], "repo:README.md@abc123")
        self.assertEqual(adapter.applied_effort(haiku).applied, "not_applicable")
        self.assertEqual(adapter.applied_effort(sonnet).applied, "unknown")

    def test_missing_transport_specific_effort_seam_refuses_non_haiku_and_not_haiku(self):
        adapter = adapters.HermesClaudeAdapter(lambda _args: "{}", cfg={}, runtime_snapshot=snapshot(),
                                               admission=lambda *_args, **_kwargs: "")
        sonnet = request(identity("hermes_claude", alias="sonnet5", provider="anthropic",
                                  model="claude-sonnet-5-5", effort="high"))
        haiku = request(identity("hermes_claude", alias="haiku", provider="anthropic",
                                 model="claude-haiku-4-5", effort="not_applicable"))
        self.assertEqual(adapter.can_execute(sonnet, snapshot(bridge="unsupported")).status, "unsupported")
        self.assertEqual(adapter.can_execute(haiku, snapshot(bridge="unsupported")).status, "yes")


class ClaudeCliAdapterTests(unittest.TestCase):
    def test_cli_uses_bridge_result_identity_and_does_not_promote_configured_metadata(self):
        calls = []
        def bridge(**kwargs):
            calls.append(kwargs)
            return {"bridge_run_id": "bridge-1", "result": "reviewed", "effective_model": "claude-other",
                    "identity": None, "num_turns": 2}
        target = identity("claude_cli", alias="sonnet", provider="anthropic", model="claude-sonnet-5-5",
                          selection_mode="exact")
        adapter = adapters.ClaudeCliAdapter(bridge, cfg={}, runtime_snapshot=snapshot(),
                                            admission=lambda *_args, **_kwargs: "")
        submitted = adapter.submit(request(target))
        self.assertTrue(submitted.accepted)
        result = adapter.result(submitted.handle)
        self.assertEqual(calls[0]["model"], "sonnet")
        self.assertEqual(result.observed_target.observed.value, "claude-other")
        self.assertEqual(result.observed_target.observed.source, "claude_cli.result.effective_model")
        self.assertEqual(result.terminal_status, "failed")
        self.assertEqual(result.failure.failure_class, "exact-route-mismatch")


class AdmissionReservationTests(unittest.TestCase):
    def test_shared_reservation_blocks_parallel_last_slot_before_second_dispatch(self):
        calls = []
        calls_lock = threading.Lock()
        def delegate_task(**kwargs):
            with calls_lock:
                calls.append(kwargs)
            return json.dumps({"delegation_id": f"deleg-{len(calls)}"})
        limited = snapshot()
        limited = replace(limited, max_concurrent_children=runtime.Fact("supported", 1))
        reservations = adapters.ReservationBook()
        adapter = adapters.HermesCodexAdapter(delegate_task, cfg={}, runtime_snapshot=limited,
                                              parent_agent=lambda: SimpleNamespace(_delegate_depth=0),
                                              reservations=reservations,
                                              admission=lambda *_args, **_kwargs: "")
        barrier = threading.Barrier(3)
        results = []
        def submit(attempt):
            barrier.wait()
            results.append(adapter.submit(request(identity("hermes_codex"), attempt=attempt)))
        first = threading.Thread(target=submit, args=("attempt-first",))
        second = threading.Thread(target=submit, args=("attempt-second",))
        first.start()
        second.start()
        barrier.wait()
        first.join()
        second.join()
        self.assertEqual(sum(outcome.accepted for outcome in results), 1)
        rejected = next(outcome for outcome in results if not outcome.accepted)
        self.assertEqual(rejected.rejection.failure_class, "concurrency")
        self.assertEqual(len(calls), 1)

    def test_factory_retains_actual_legacy_entrypoints_without_recursive_wrapping(self):
        delegate_task = object()
        delegate_claude = object()
        bridge = object()
        registry = adapters.legacy_adapters(delegate_task=delegate_task, delegate_claude=delegate_claude,
                                            claude_bridge=bridge, cfg={})
        self.assertIs(registry["hermes_codex"]._delegate_task, delegate_task)
        self.assertIs(registry["hermes_claude"]._delegate_claude, delegate_claude)
        self.assertIs(registry["claude_cli"]._bridge, bridge)
        self.assertEqual(set(registry), {"hermes_codex", "hermes_claude", "claude_cli"})

    def test_different_transport_calls_keep_context_and_configuration_isolated(self):
        cfg = {"claude_delegation": {"tiers": {"sonnet": "configured-only"}}, "unchanged": ["value"]}
        codex_calls, claude_calls = [], []
        shared = adapters.ReservationBook()
        roomy = replace(snapshot(), max_concurrent_children=runtime.Fact("supported", 2))
        parent = SimpleNamespace(_delegate_depth=0)
        codex = adapters.HermesCodexAdapter(
            lambda **kwargs: codex_calls.append(kwargs) or json.dumps({"delegation_id": "codex-1"}),
            cfg=cfg, runtime_snapshot=roomy, parent_agent=lambda: parent, reservations=shared,
            admission=lambda *_args, **_kwargs: "",
        )
        claude = adapters.HermesClaudeAdapter(
            lambda args: claude_calls.append(args) or json.dumps({"delegation_id": "claude-1"}),
            cfg=cfg, runtime_snapshot=roomy, reservations=shared,
            admission=lambda *_args, **_kwargs: "",
        )
        codex_request = replace(request(identity("hermes_codex"), attempt="codex-attempt"),
                                context_reference="repo:codex.md@one")
        claude_request = replace(request(identity("hermes_claude", alias="haiku", provider="anthropic",
                                                  model="claude-haiku-4-5", effort="not_applicable"),
                                 attempt="claude-attempt"), context_reference="repo:claude.md@two")
        threads = [threading.Thread(target=adapter.submit, args=(item,))
                   for adapter, item in ((codex, codex_request), (claude, claude_request))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(codex_calls[0]["tasks"][0]["context"], "repo:codex.md@one")
        self.assertEqual(codex_calls[0]["tasks"][0]["goal"], "[terra] Implement one bounded change.")
        self.assertNotIn("credentials_cfg", codex_calls[0])
        self.assertEqual(claude_calls[0], {"tasks": [{"goal": "Implement one bounded change.",
                                                         "context": "repo:claude.md@two"}], "tier": "haiku"})
        self.assertEqual(cfg, {"claude_delegation": {"tiers": {"sonnet": "configured-only"}},
                               "unchanged": ["value"]})


if __name__ == "__main__":
    unittest.main()
