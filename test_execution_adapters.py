"""Offline transport regressions: execute public boundaries, never construct agents."""
import inspect
import json
import threading
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import model_router as router
from model_router import execution_adapters as adapters
from model_router import execution_contracts as contracts
from model_router import runtime_capabilities as runtime
from model_router import claude_delegation as claude
from model_router import claude_opus_bridge as cli
from model_router import worker_admission
from model_router.test_claude_delegation import _cfg, _reading


def identity(transport, *, alias="terra", provider="openai-codex", model="gpt-5.6-terra",
             selection_mode="profile_preferred", effort="unknown"):
    return contracts.TargetIdentity(provider, provider, transport, alias, selection_mode,
        contracts.ModelFact(model, "operator_request"), contracts.ModelFact(model, "configured_target"),
        contracts.ModelFact(), contracts.EffortFact(effort, "unknown", "not_observed"))


@contextmanager
def raw_cli(side_effect):
    """Mock only the CLI's PRIVATE raw subprocess operation (F03): the public
    boundary in ``claude_opus_bridge.dispatch`` stays real. Input validation is
    stubbed so fixture paths/tasks need not exist; yields the raw mock."""
    with patch.object(cli, "_prepare", return_value=SimpleNamespace(alias="opus", adjustment="")), \
            patch.object(cli, "_run_cli", side_effect=side_effect) as raw:
        yield raw


def request(target, *, attempt="attempt-001"):
    return contracts.ExecutionRequest("wf-001", 1, "task-001", attempt,
        "Implement one bounded change.", ("tests pass",), "repo:README.md@abc123", target,
        "repo:/work/router", "workspace:/work/router", ("read", "write"), ("terminal",),
        True, ("one.py",), 60, 999999, 1, "required", "forbid")


def snapshot(*, bridge="supported"):
    def cap(name, status):
        return runtime.Capability(name, status, reason="fixture")
    return runtime.RuntimeSnapshot(1, "fixture", runtime.Fact("supported", 2),
        runtime.Fact("supported", 1), runtime.Fact("supported", False), (), (
        runtime.AdapterCapabilities("hermes_codex", tuple(cap(n, s) for n, s in (
            ("submission", "supported"), ("model_parameter", "unsupported"),
            ("effort_application", "unknown"), ("identity_observability", "unknown"),
            ("cancellation", "supported")))),
        runtime.AdapterCapabilities("hermes_claude", tuple(cap(n, s) for n, s in (
            ("submission", "supported"), ("effort_application", bridge),
            ("identity_observability", "unknown"), ("cancellation", "supported")))),
        runtime.AdapterCapabilities("claude_cli", tuple(cap(n, s) for n, s in (
            ("submission", "supported"), ("exact_model", "supported"),
            ("identity_observability", "supported"), ("effort_application", "unknown"),
            ("cancellation", "supported")))),
        ), (), True)


class StructuredConstraintTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.instances = (
            adapters.HermesCodexAdapter(lambda **kw: self.calls.append(kw), runtime_snapshot=snapshot(),
                parent_agent=lambda: SimpleNamespace(_delegate_depth=0), admission=lambda *a, **k: ""),
            adapters.HermesClaudeAdapter(lambda args: self.calls.append(args), runtime_snapshot=snapshot(),
                admission=lambda *a, **k: ""),
            adapters.ClaudeCliAdapter(lambda **kw: self.calls.append(kw), runtime_snapshot=snapshot(),
                admission=lambda *a, **k: ""))
        self.targets = (identity("hermes_codex"),
            identity("hermes_claude", alias="haiku", provider="anthropic", model="claude-haiku-4-5", effort="not_applicable"),
            identity("claude_cli", alias="sonnet", provider="anthropic", model="claude-sonnet-5-5"))

    def test_restrictive_permissions_do_not_inherit_writes(self):
        for adapter, target in zip(self.instances, self.targets):
            item = replace(request(target), mutating=False, permissions=("read",), write_scope=())
            self.assertEqual(adapter.can_execute(item, snapshot()).status, "unsupported")
            self.assertFalse(adapter.submit(item).accepted)
        self.assertEqual(self.calls, [])

    def test_missing_tools_are_refused_before_dispatch(self):
        for adapter, target in zip(self.instances, self.targets):
            item = replace(request(target), tool_requirements=("missing-tool",))
            decision = adapter.can_execute(item, snapshot())
            self.assertEqual(decision.status, "unsupported")
            self.assertIn("tool", " ".join(decision.reasons))
        self.assertEqual(self.calls, [])

    def test_workspace_and_unresolved_context_are_not_reported_as_used(self):
        for adapter, target in zip(self.instances, self.targets):
            item = replace(request(target), workspace="workspace:/other", context_reference="repo:required.md@revision")
            decision = adapter.can_execute(item, snapshot())
            self.assertEqual(decision.status, "unsupported")
            reasons = " ".join(decision.reasons)
            self.assertIn("workspace", reasons)
            self.assertIn("context", reasons)
            self.assertFalse(adapter.submit(item).accepted)
        self.assertEqual(self.calls, [])

    def test_exact_native_identity_is_not_configured_metadata_proof(self):
        for adapter, target in zip(self.instances[:2], self.targets[:2]):
            decision = adapter.can_execute(request(replace(target, selection_mode="exact")), snapshot())
            self.assertEqual(decision.status, "unsupported")
            self.assertIn("exact", " ".join(decision.reasons))

    def test_cli_exact_effort_and_account_constraints_are_refused(self):
        target = replace(self.targets[2], selection_mode="exact", effort=contracts.EffortFact("high", "unknown", "not_observed"))
        decision = self.instances[2].can_execute(request(target), snapshot())
        self.assertEqual(decision.status, "unsupported")
        reasons = " ".join(decision.reasons)
        self.assertIn("effort", reasons)
        self.assertIn("account", reasons)
        self.assertFalse(self.instances[2].submit(request(target)).accepted)
        self.assertEqual(self.calls, [])

    def test_same_tier_effort_is_refused_not_silently_replaced_with_config(self):
        adapter = self.instances[1]
        for effort in ("low", "high"):
            target = identity("hermes_claude", alias="sonnet5", provider="anthropic", model="claude-sonnet-5-5", effort=effort)
            decision = adapter.can_execute(request(target), snapshot())
            self.assertEqual(decision.status, "unsupported")
            self.assertIn("effort", " ".join(decision.reasons))
        self.assertEqual(self.calls, [])

    def test_missing_effort_seam_is_explicit_and_haiku_has_no_thinking(self):
        adapter = self.instances[1]
        target = identity("hermes_claude", alias="sonnet5", provider="anthropic", model="claude-sonnet-5-5", effort="high")
        decision = adapter.can_execute(request(target), snapshot(bridge="unsupported"))
        self.assertEqual(decision.status, "unsupported")
        self.assertIn("effort", " ".join(decision.reasons))
        self.assertEqual(adapter.applied_effort(request(self.targets[1])).applied, "not_applicable")

    def test_forbidden_native_substitution_is_not_assumed_enforceable(self):
        for adapter, target in zip(self.instances[:2], self.targets[:2]):
            decision = adapter.can_execute(request(target), snapshot())
            self.assertEqual(decision.status, "unsupported")
            self.assertIn("substitution", " ".join(decision.reasons))

    def test_deadline_timeout_and_write_scope_require_real_seams(self):
        for adapter, target in zip(self.instances, self.targets):
            decision = adapter.can_execute(request(target), snapshot())
            reasons = " ".join(decision.reasons)
            for constraint in ("deadline", "timeout", "write_scope"):
                self.assertIn(constraint, reasons)
            self.assertEqual(decision.status, "unsupported")

    def test_advertised_capabilities_match_methods_not_host_cancellation(self):
        for adapter in self.instances:
            self.assertNotIn("cancellation", adapter.capabilities(snapshot()).capabilities)
            self.assertEqual(adapter.cancel("not-running").status, "unsupported")
            self.assertNotIn("submission", adapter.capabilities(snapshot()).capabilities)


class ExistingDefectRegressionTests(unittest.TestCase):
    """These assertion failures reproduce defects using only pre-fix public APIs."""
    def test_reservation_owner_rejects_partial_attempt_keys(self):
        for key in ("attempt", ("workflow", 1, "task"), ("workflow", True, "task", "attempt"),
                    ("", 1, "task", "attempt")):
            with self.subTest(key=key):
                book = adapters.ReservationBook()
                self.assertFalse(book.claim("shared", key, 1))
                self.assertEqual(book.records(), ())

    def test_existing_claim_is_not_a_fresh_dispatch_authorization(self):
        book = adapters.ReservationBook()
        self.assertTrue(book.claim("shared", ("wf", 1, "task", "attempt"), 1))
        self.assertFalse(book.claim("shared", ("wf", 1, "task", "attempt"), 1))

    def test_completed_native_work_is_not_a_safe_submission_rejection(self):
        starts = []
        raw = lambda **kw: starts.append(kw) or json.dumps({"results": [{"status": "completed", "result": "changed"}], "total_duration_seconds": 1})
        adapter = adapters.HermesCodexAdapter(raw, runtime_snapshot=snapshot(),
            parent_agent=lambda: SimpleNamespace(_delegate_depth=1), admission=lambda *a, **k: "")
        submitted = adapter.submit(request(identity("hermes_codex")))
        self.assertFalse(starts and not submitted.accepted, "work already completed but adapter reported safe rejection")

    def test_same_attempt_never_runs_native_transport_twice(self):
        starts = []
        raw = lambda **kw: starts.append(kw) or json.dumps({"delegation_id": "same-child"})
        adapter = adapters.HermesCodexAdapter(raw, runtime_snapshot=snapshot(),
            parent_agent=lambda: SimpleNamespace(_delegate_depth=0), reservations=adapters.ReservationBook(),
            admission=lambda *a, **k: "")
        item = request(identity("hermes_codex"))
        adapter.submit(item)
        adapter.submit(item)
        self.assertLessEqual(len(starts), 1)

    def test_registered_entrypoint_is_the_transport_boundary_not_only_old_guard(self):
        registrations = {}
        ctx = SimpleNamespace(register_hook=lambda *a: None,
            register_middleware=lambda name, fn: registrations.update({name: fn}))
        with patch.object(claude, "register", return_value=True):
            router.register(ctx)
        self.assertIsNot(registrations["tool_execution"], worker_admission.guard_tool_execution)


class LegacyBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.book = adapters.ReservationBook()
        self.parent = SimpleNamespace(_delegate_depth=0, session_id="parent")
        self.shared_patch = patch.object(adapters, "RESERVATIONS", self.book, create=True)
        self.shared_patch.start()
        self.addCleanup(self.shared_patch.stop)

    def dispatch(self, raw, *, key=("workflow", 1, "task", "attempt"), transport="hermes_codex"):
        return adapters.dispatch_legacy(raw, transport=transport, scope="parent:shared", attempt_key=key,
            reservations=self.book)

    def _host_entry(self, result, *, schema=None):
        from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry

        return _build_result_entry(SimpleNamespace(model='gpt-terra'), result, 0, 1.0,
            schema if schema is not None else _SchemaOutcome(None, None, [], 0))

    def _receipt_for(self, results=None, **payload):
        if results is not None:
            payload['results'] = results
        book = adapters.ReservationBook()
        adapters.dispatch_legacy(lambda: json.dumps(payload), transport='hermes_codex',
            scope='parent', reservations=book)
        return book.records()[0]

    def test_a04_native_iteration_exhaustion_is_not_normalized_as_success(self):
        entry = self._host_entry({'final_response': 'partial implementation', 'completed': False, 'api_calls': 3})
        self.assertEqual(entry['exit_reason'], 'max_iterations')
        self.assertTrue(entry['truncated'])

        receipt = self._receipt_for([entry])

        self.assertEqual(receipt.status, 'failed', 'real host partial/budget exhaustion became succeeded')
        self.assertEqual(getattr(receipt, 'exit_reason', None), 'max_iterations')
        self.assertTrue(getattr(receipt, 'truncated', False))
        self.assertEqual(getattr(receipt, 'task_outcomes', ()), ('completed:max_iterations',))
        self.assertEqual(receipt.observed_model, 'unknown')

    def test_native_child_results_keep_terminal_partial_and_verification_evidence(self):
        from tools.delegate_tool_child_run import _SchemaOutcome

        ordinary = self._host_entry({'final_response': 'done', 'completed': True, 'api_calls': 1})
        partial = self._host_entry({'final_response': 'unfinished', 'completed': False, 'api_calls': 3})
        interrupted = self._host_entry({'final_response': 'Operation interrupted.', 'interrupted': True,
            'messages': [{'role': 'assistant', 'content': 'usable partial notes'}]})
        failure = self._host_entry({'final_response': 'provider rejected', 'failed': True,
            'error': 'provider rejected'})
        schema_invalid = self._host_entry({'final_response': '{bad json', 'completed': True},
            schema=_SchemaOutcome({'type': 'object'}, False, ['invalid JSON'], 1))
        timeout = {'status': 'timeout', 'exit_reason': 'timeout', 'truncated': False,
            'error': 'child timed out'}
        cases = (
            ('ordinary', [ordinary], 'succeeded', 'completed', False, 'unknown', ('completed:completed',)),
            ('partial', [partial], 'failed', 'max_iterations', True, 'unknown', ('completed:max_iterations',)),
            ('interrupted', [interrupted], 'cancelled', 'interrupted', False, 'unknown', ('interrupted:interrupted',)),
            ('failure', [failure], 'failed', 'error', False, 'unknown', ('failed:error',)),
            ('timeout', [timeout], 'timed_out', 'timeout', False, 'unknown', ('timeout:timeout',)),
            ('mixed', [ordinary, partial], 'failed', 'mixed', True, 'unknown',
                ('completed:completed', 'completed:max_iterations')),
            ('schema-invalid', [schema_invalid], 'succeeded', 'completed', False, 'failed', ('completed:completed',)),
        )
        for name, entries, status, exit_reason, truncated, verification, outcomes in cases:
            with self.subTest(name=name):
                receipt = self._receipt_for(entries)
                self.assertEqual(receipt.status, status)
                self.assertEqual(getattr(receipt, 'exit_reason', None), exit_reason)
                self.assertEqual(getattr(receipt, 'truncated', None), truncated)
                self.assertEqual(getattr(receipt, 'verification', None), verification)
                self.assertEqual(getattr(receipt, 'task_outcomes', ()), outcomes)
                self.assertEqual(receipt.observed_model, 'unknown')

    def test_background_malformed_and_contradictory_native_results_remain_unknown(self):
        contradictory = self._host_entry({'final_response': 'inconsistent', 'completed': False})
        contradictory['truncated'] = False
        cases = (
            ('background', None, {'status': 'dispatched', 'delegation_id': 'delegation-1'}, 'delegation-1'),
            ('malformed', None, {'results': [{'status': 'completed', 'exit_reason': 'completed',
                'truncated': 'not-a-bool'}]}, None),
            ('contradictory', [contradictory], {}, None),
        )
        for name, results, payload, handle in cases:
            with self.subTest(name=name):
                receipt = self._receipt_for(**payload) if results is None else self._receipt_for(results, **payload)
                self.assertEqual(receipt.status, 'unknown')
                self.assertEqual(getattr(receipt, 'exit_reason', None), 'unknown')
                self.assertEqual(getattr(receipt, 'verification', None), 'unknown')
                self.assertEqual(getattr(receipt, 'task_outcomes', ()), ())
                if handle:
                    self.assertEqual(receipt.handle, handle)

    def test_oversized_handle_is_not_silently_repaired_into_another_identifier(self):
        payload = json.dumps({"delegation_id": "d" * 300})
        self.assertIs(self.dispatch(lambda: payload), payload)
        record = self.book.record(("workflow", 1, "task", "attempt"))
        self.assertTrue(record.handle.startswith("local-"))
        self.assertEqual(record.status, "unknown")

    def test_changed_payload_under_same_public_attempt_is_refused(self):
        calls = []
        with patch.object(router, "_load_config", return_value={"enabled": True}), \
             patch.object(router, "_delegation_targets_detail", return_value={}), \
             patch.object(router, "_delegated_claude_review_status", return_value=(None, "")), \
             patch.object(worker_admission, "delegate_task_route", return_value=("openai-codex", "gpt-5.6-terra")), \
             patch.object(worker_admission, "refusal", return_value=""), \
             patch("agent.subagent_lifecycle.get_active_subagent_parent", return_value=self.parent):
            raw = lambda sent: calls.append(sent) or json.dumps({"results": [{"status": "completed"}], "total_duration_seconds": 1})
            common = {"tool_name": "delegate_task", "next_call": raw, "session_id": "wf", "turn_id": "turn", "tool_call_id": "attempt"}
            adapters.guard_legacy_tool_execution(args={"goal": "read only"}, **common)
            refused = adapters.guard_legacy_tool_execution(args={"goal": "edit different files"}, **common)
        self.assertIn("error", json.loads(refused))
        self.assertEqual(len(calls), 1)

    def test_default_factory_has_one_mandatory_reservation_owner(self):
        registry = adapters.legacy_adapters(delegate_task=lambda **kw: None,
            delegate_claude=lambda args: None, claude_bridge=lambda **kw: None)
        self.assertTrue(all(adapter._reservations is self.book for adapter in registry.values()))

    def test_journal_bound_evicts_lru_finished_records_instead_of_refusing(self):
        # Superseded (fix round 2, N1): a full journal used to refuse new work.
        calls = []
        payload = json.dumps({"results": [{"status": "completed"}], "total_duration_seconds": 1})
        with patch.object(adapters, "_MAX_RECORDS", 2):
            self.dispatch(lambda: calls.append("first") or payload)
            self.dispatch(lambda: calls.append("second") or payload, key=("second", 1, "task", "attempt"))
            self.assertIs(self.dispatch(lambda: calls.append("duplicate") or payload), payload)  # retained: replay
            self.dispatch(lambda: calls.append("third") or payload, key=("third", 1, "task", "attempt"))
            self.assertIsNone(self.book.record(("second", 1, "task", "attempt")))  # least recently used
            self.assertIsNotNone(self.book.record(("workflow", 1, "task", "attempt")))
        self.assertEqual(calls, ["first", "second", "third"])

    def test_journal_bound_never_evicts_in_flight_records(self):
        started, finish = threading.Event(), threading.Event()
        payload = json.dumps({"results": [{"status": "completed"}], "total_duration_seconds": 1})
        def slow():
            started.set()
            self.assertTrue(finish.wait(5))
            return payload
        with patch.object(adapters, "_MAX_RECORDS", 1):
            thread = threading.Thread(target=lambda: self.dispatch(slow))
            thread.start()
            self.assertTrue(started.wait(5))
            try:
                self.assertIs(self.dispatch(lambda: payload, key=("other", 1, "t", "a")), payload)
                self.assertIsNotNone(self.book.record(("workflow", 1, "task", "attempt")))
                duplicate = json.loads(self.dispatch(lambda: "SHOULD NOT START"))
                self.assertIn("error", duplicate)  # in-flight duplicate still blocked
            finally:
                finish.set()
                thread.join(5)

    def test_real_synchronous_shape_has_terminal_evidence_without_top_level_handle(self):
        payload = json.dumps({"results": [{"task_index": 0, "status": "completed", "result": "done",
            "subagent_id": "sa-0-child", "session_id": "child-session"}], "total_duration_seconds": 1.2})
        self.assertIs(self.dispatch(lambda: payload), payload)
        record = self.book.record(("workflow", 1, "task", "attempt"))
        self.assertEqual(record.status, "succeeded")
        self.assertEqual(record.child_ids, ("sa-0-child", "child-session"))
        self.assertTrue(record.handle)
        self.assertEqual(self.dispatch(lambda: "{}", key=("wf2", 1, "t", "a")), "{}")

    def test_background_to_sync_fallback_preserves_host_payload(self):
        payload = json.dumps({"results": [{"task_index": 0, "status": "completed", "result": "done"}],
            "total_duration_seconds": 0.2, "note": "background=true unavailable; ran SYNCHRONOUSLY"})
        self.assertIs(self.dispatch(lambda: payload), payload)
        self.assertEqual(self.book.record(("workflow", 1, "task", "attempt")).status, "succeeded")

    def test_unknown_result_and_start_then_error_release_capacity_after_call(self):
        # Superseded (fix round 2, C1/Minor): ambiguous outcomes stay traceable as
        # unknown but release the slot when the synchronous call returns; a top-level
        # host error payload is a failed outcome.
        for mode, status in (("malformed", "unknown"), ("exception", "unknown"), ("error-after-start", "failed")):
            with self.subTest(mode=mode):
                self.book = adapters.ReservationBook()
                def raw():
                    if mode == "exception":
                        raise RuntimeError("started worker but lost response")
                    return "lost response" if mode == "malformed" else json.dumps({"error": "host failed after building children"})
                if mode == "exception":
                    with self.assertRaises(RuntimeError):
                        self.dispatch(raw)
                else:
                    self.dispatch(raw)
                self.assertEqual(self.book.record(("workflow", 1, "task", "attempt")).status, status)
                self.assertEqual(self.dispatch(lambda: "later", key=("wf2", 1, "t", "a")), "later")
                restarted = []
                self.dispatch(lambda: restarted.append(1) or "SHOULD NOT START")  # same attempt: replay/refuse
                self.assertEqual(restarted, [])

    def test_same_full_attempt_is_dispatched_only_once_even_after_terminal(self):
        calls = []
        payload = json.dumps({"results": [{"status": "completed", "result": "done"}], "total_duration_seconds": 1})
        raw = lambda: calls.append("start") or payload
        self.assertIs(self.dispatch(raw), payload)
        self.assertIs(self.dispatch(raw), payload)
        self.assertEqual(calls, ["start"])

    def test_cross_workflow_same_local_attempt_does_not_share_claim(self):
        calls = []
        raw = lambda: calls.append("start") or json.dumps({"status": "dispatched", "delegation_id": "d"})
        self.dispatch(raw)
        self.dispatch(raw, key=("other-workflow", 1, "task", "attempt"))
        self.dispatch(raw)  # same full key: replayed, not re-dispatched
        self.assertEqual(len(calls), 2)
        self.assertIsNot(self.book.record(("workflow", 1, "task", "attempt")),
                         self.book.record(("other-workflow", 1, "task", "attempt")))
        self.assertEqual(len(self.book.records()), 2)

    def test_pending_same_attempt_is_blocked_during_dispatch(self):
        started, finish = threading.Event(), threading.Event()
        calls = []
        def raw():
            calls.append("start")
            started.set()
            self.assertTrue(finish.wait(5))
            return json.dumps({"delegation_id": "d"})
        thread = threading.Thread(target=lambda: self.dispatch(raw))
        thread.start()
        self.assertTrue(started.wait(5))
        try:
            duplicate = json.loads(self.dispatch(raw))
            self.assertIn("error", duplicate)
        finally:
            finish.set()
            thread.join(5)
        self.assertEqual(len(calls), 1)

    def test_shared_owner_admits_distinct_concurrent_calls_and_dedups_same_attempt(self):
        # Superseded (fix round 3, R1): legacy dispatches get no plugin capacity
        # check; the host is the sole admission authority. Only the same attempt
        # is deduplicated while it is in flight.
        started, release = threading.Event(), threading.Event()
        calls, responses = [], []
        def slow():
            calls.append("hermes_codex")
            started.set()
            self.assertTrue(release.wait(5))
            return json.dumps({"delegation_id": "hermes_codex"})
        codex_key = ("hermes_codex", 1, "task", "attempt")
        thread = threading.Thread(target=lambda: responses.append(self.dispatch(slow, key=codex_key)))
        thread.start()
        self.assertTrue(started.wait(5))
        try:
            claude_raw = json.dumps({"delegation_id": "hermes_claude"})
            self.assertIs(self.dispatch(lambda: calls.append("hermes_claude") or claude_raw,
                transport="hermes_claude", key=("hermes_claude", 1, "task", "attempt")), claude_raw)
            self.assertEqual(self.dispatch(lambda: calls.append("wide") or "{}",
                key=("wide", 1, "t", "a")), "{}")
            duplicate = json.loads(self.dispatch(lambda: calls.append("duplicate") or "{}", key=codex_key))
            self.assertIn("error", duplicate)  # same in-flight attempt: never a second dispatch
        finally:
            release.set()
            thread.join(5)
        self.assertEqual(responses, [json.dumps({"delegation_id": "hermes_codex"})])
        self.assertEqual(calls, ["hermes_codex", "hermes_claude", "wide"])

    def test_empty_and_ambiguous_partial_sync_results_stay_unknown_without_holding_capacity(self):
        for results in ([], [{"status": "unknown", "result": "lost"}], [{"status": "completed"}, {"status": "running"}]):
            self.book = adapters.ReservationBook()
            self.dispatch(lambda: json.dumps({"results": results, "total_duration_seconds": 1}))
            self.assertEqual(self.book.record(("workflow", 1, "task", "attempt")).status, "unknown")
            self.assertEqual(self.dispatch(lambda: "next", key=("w2", 1, "t", "a")), "next")

    def test_claude_adjustment_evidence_is_retained_not_served_identity(self):
        payload = json.dumps({"delegation_id": "d", "claude_tier": "sonnet", "tier_adjusted": "opus→sonnet (weekly usage 75.0)",
            "reasoning_effort": "not applied"})
        self.dispatch(lambda: payload, transport="hermes_claude")
        record = self.book.record(("workflow", 1, "task", "attempt"))
        self.assertEqual(record.adjustment, "opus→sonnet (weekly usage 75.0)")
        self.assertEqual(record.resolved_tier, "sonnet")
        self.assertEqual(record.observed_model, "unknown")
        self.assertEqual(record.effort, "not applied")

    def test_registered_codex_boundary_preserves_authority_and_single_admission(self):
        registrations = {}
        ctx = SimpleNamespace(register_hook=lambda *a: None,
            register_middleware=lambda name, fn: registrations.update({name: fn}))
        with patch.object(claude, "register", return_value=True):
            router.register(ctx)
        entrypoint = registrations["tool_execution"]
        admissions, dispatches = [], []
        args = {"tasks": [{"goal": "original bounded goal", "context": "original context", "output_schema": {"type": "object"}}]}
        result = json.dumps({"results": [{"status": "completed", "result": "done"}], "total_duration_seconds": 1})
        def raw(sent):
            dispatches.append(sent)
            return result
        with patch.object(router, "_load_config", return_value={"enabled": True}), \
             patch.object(router, "_delegation_targets_detail", return_value={}), \
             patch.object(router, "_delegated_claude_review_status", return_value=(None, "")), \
             patch.object(worker_admission, "delegate_task_route", return_value=("openai-codex", "gpt-5.6-terra")), \
             patch.object(worker_admission, "refusal", side_effect=lambda *a, **k: admissions.append(a) or ""), \
             patch("agent.subagent_lifecycle.get_active_subagent_parent", return_value=self.parent):
            self.assertIs(entrypoint(tool_name="delegate_task", args=args, next_call=raw), result)
        self.assertEqual(len(admissions), 1)
        self.assertEqual(dispatches, [args])
        self.assertIs(dispatches[0], args)
        self.assertTrue(self.book.records())

    def test_public_codex_attempt_metadata_deduplicates_dispatch(self):
        cfg = {"enabled": True}
        calls = []
        args = {"goal": "original goal", "context": "original context"}
        raw = lambda sent: calls.append(sent) or json.dumps({"results": [{"status": "completed"}], "total_duration_seconds": 1})
        with patch.object(router, "_load_config", return_value=cfg), \
             patch.object(router, "_delegation_targets_detail", return_value={}), \
             patch.object(router, "_delegated_claude_review_status", return_value=(None, "")), \
             patch.object(worker_admission, "delegate_task_route", return_value=("openai-codex", "gpt-5.6-terra")), \
             patch.object(worker_admission, "refusal", return_value=""), \
             patch("agent.subagent_lifecycle.get_active_subagent_parent", return_value=self.parent):
            entry = getattr(adapters, "guard_legacy_tool_execution", worker_admission.guard_tool_execution)
            first = entry(tool_name="delegate_task", args=args, next_call=raw,
                session_id="workflow", turn_id="turn", tool_call_id="attempt")
            second = entry(tool_name="delegate_task", args=args, next_call=raw,
                session_id="workflow", turn_id="turn", tool_call_id="attempt")
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)

    def test_public_claude_attempt_metadata_deduplicates_without_second_raw_launch(self):
        calls = []
        raw = lambda **kw: calls.append(kw) or json.dumps({"results": [{"status": "completed"}], "total_duration_seconds": 1})
        with patch.object(router, "_load_config", return_value=_cfg()), \
             patch.object(claude, "_host", return_value=(raw, lambda: self.parent)), \
             patch.object(claude.usage_guard, "read", return_value=_reading(10.0)) as read:
            entry = getattr(adapters, "guard_legacy_tool_execution", worker_admission.guard_tool_execution)
            results = []
            for _ in range(2):
                payload = entry(tool_name="delegate_claude", args={"goal": "g", "tier": "haiku"},
                    next_call=claude.handle_delegate_claude, session_id="workflow", turn_id="turn", tool_call_id="attempt")
                results.append(payload)
                self.assertNotIn("error", json.loads(payload))
        self.assertEqual(len(calls), 1)
        self.assertEqual(read.call_count, 1)
        self.assertEqual(results[0], results[1])

    def test_non_spawn_tools_and_control_actions_bypass_boundary(self):
        for name, args in (("terminal", {"command": "unchanged"}), ("delegate_task", {"action": "list"})):
            sentinel = object()
            entry = getattr(adapters, "guard_legacy_tool_execution", worker_admission.guard_tool_execution)
            self.assertIs(entry(tool_name=name, args=args, next_call=lambda a: sentinel), sentinel)
            self.assertEqual(self.book.records(), ())

    def test_actual_claude_entrypoint_single_guard_and_raw_dispatch_with_step_down(self):
        cfg = _cfg()
        raw_calls = []
        parent = self.parent
        raw_result = json.dumps({"results": [{"status": "completed", "result": "done"}], "total_duration_seconds": 1})
        def raw(**kw):
            raw_calls.append(kw)
            inspect.signature(__import__("tools.delegate_tool", fromlist=["delegate_task"]).delegate_task).bind(**kw)
            return raw_result
        with patch.object(router, "_load_config", return_value=cfg), \
             patch.object(claude, "_host", return_value=(raw, lambda: parent)), \
             patch.object(claude, "install_reasoning_bridge", return_value=(True, "")), \
             patch.object(claude.usage_guard, "read", return_value=_reading(75.0)) as read, \
             patch.object(claude.usage_guard, "apply", wraps=claude.usage_guard.apply) as apply:
            args = {"tasks": [{"goal": "g", "context": "original context"}], "tier": "opus"}
            payload = json.loads(claude.handle_delegate_claude(args))
        self.assertEqual(read.call_count, 1)
        self.assertEqual(apply.call_count, 1)
        self.assertEqual(len(raw_calls), 1)
        self.assertEqual(raw_calls[0]["credentials_cfg"], {"provider": "anthropic", "model": "claude-sonnet-5-5", "fallback_providers": []})
        self.assertEqual(raw_calls[0]["tasks"], args["tasks"])
        self.assertEqual(payload["claude_tier"], "sonnet")
        self.assertIn("opus→sonnet", payload["tier_adjusted"])
        records = self.book.records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].resolved_tier, "sonnet")
        self.assertIn("opus→sonnet", records[0].adjustment)
        self.assertEqual(records[0].status, "succeeded")

    def test_real_claude_handler_preserves_scoped_config_effort_and_credentials(self):
        from tools import delegate_tool, delegate_tool_config
        from hermes_constants import parse_reasoning_effort
        cfg = _cfg()
        cfg["claude_delegation"]["reasoning_effort"] = {"sonnet": "high", "opus": "low"}
        self.addCleanup(claude._reset_reasoning_bridge_for_tests)
        parent = SimpleNamespace(provider="openai-codex", reasoning_config={"effort": "medium"}, _delegate_depth=0)
        seen = []
        def resolver(parent_agent, delegation_cfg, parent_api_key, *, model=None, override_provider=None,
                     override_base_url=None, override_api_key=None, override_api_mode=None,
                     override_acp_command=None, override_acp_args=None, routing_cfg=None):
            return {"provider": override_provider, "model": model, "reasoning_config": parent.reasoning_config}
        def raw(**kw):
            creds = kw["credentials_cfg"]
            seen.append(delegate_tool._resolve_child_runtime(parent, {}, None, model=creds["model"], override_provider=creds["provider"]))
            return json.dumps({"results": [{"status": "completed", "result": "done"}], "total_duration_seconds": 1})
        with patch.object(delegate_tool, "_resolve_child_runtime", resolver), \
             patch.object(delegate_tool_config, "_resolve_child_runtime", resolver), \
             patch.object(router, "_load_config", return_value=cfg), \
             patch.object(claude, "_host", return_value=(raw, lambda: parent)), \
             patch.object(claude.usage_guard, "read", return_value=_reading(10.0)):
            for tier in ("sonnet", "opus"):
                metadata = {"tool_name": "delegate_claude", "args": {"goal": "g", "tier": tier},
                    "next_call": claude.handle_delegate_claude, "session_id": "workflow", "turn_id": "turn", "tool_call_id": tier}
                first = adapters.guard_legacy_tool_execution(**metadata)
                duplicate = adapters.guard_legacy_tool_execution(**metadata)
                self.assertNotIn("error", json.loads(first))
                self.assertNotIn("reasoning_effort", json.loads(first))  # real wrapped resolver saw application
                self.assertEqual(first, duplicate)  # replay must not falsely annotate effort 'not applied'
        self.assertEqual([r["reasoning_config"] for r in seen], [parse_reasoning_effort("high"), parse_reasoning_effort("low")])
        self.assertEqual(parent.reasoning_config, {"effort": "medium"})

    def test_requested_same_tier_efforts_never_reach_real_configured_medium_handler(self):
        cfg = _cfg()
        cfg["claude_delegation"]["reasoning_effort"] = {"sonnet": "medium"}
        starts = []
        adapter = adapters.HermesClaudeAdapter(claude.handle_delegate_claude,
            runtime_snapshot=snapshot(), admission=lambda *a, **k: "")
        with patch.object(router, "_load_config", return_value=cfg), \
             patch.object(claude, "_host", return_value=(lambda **kw: starts.append(kw) or json.dumps({"delegation_id": "d"}), lambda: self.parent)), \
             patch.object(claude, "install_reasoning_bridge", return_value=(True, "")), \
             patch.object(claude.usage_guard, "read", return_value=_reading(10.0)):
            for effort in ("low", "high"):
                target = identity("hermes_claude", alias="sonnet5", provider="anthropic", model="claude-sonnet-5-5", effort=effort)
                self.assertFalse(adapter.submit(request(target, attempt=effort)).accepted)
        self.assertEqual(starts, [])
        self.assertEqual(cfg["claude_delegation"]["reasoning_effort"], {"sonnet": "medium"})

    def test_forbidden_step_down_never_reaches_real_handler_but_legacy_records_adjustment(self):
        cfg = _cfg()
        starts = []
        adapter = adapters.HermesClaudeAdapter(claude.handle_delegate_claude,
            runtime_snapshot=snapshot(), admission=lambda *a, **k: "")
        with patch.object(router, "_load_config", return_value=cfg), \
             patch.object(claude, "_host", return_value=(lambda **kw: starts.append(kw) or json.dumps({"results": [{"status": "completed"}], "total_duration_seconds": 1}), lambda: self.parent)), \
             patch.object(claude, "install_reasoning_bridge", return_value=(True, "")), \
             patch.object(claude.usage_guard, "read", return_value=_reading(75.0)):
            target = identity("hermes_claude", alias="opus5", provider="anthropic", model="claude-opus-5-5")
            self.assertFalse(adapter.submit(request(target)).accepted)
            self.assertEqual(starts, [])
            allowed = json.loads(claude.handle_delegate_claude({"goal": "g", "tier": "opus"}))
        self.assertEqual(len(starts), 1)
        self.assertEqual(allowed["claude_tier"], "sonnet")
        self.assertIn("opus→sonnet", allowed["tier_adjusted"])
        self.assertEqual(self.book.records()[0].resolved_tier, "sonnet")

    def test_missing_capacity_seam_refuses_before_native_dispatch(self):
        import sys
        import types
        starts = []
        with patch.dict(sys.modules, {"tools.delegate_tool": types.ModuleType("tools.delegate_tool")}):
            raw = adapters.native_legacy_dispatch(lambda: starts.append("start"), parent=self.parent,
                transport="hermes_codex")
        self.assertIn("error", json.loads(raw))
        self.assertEqual(starts, [])
        self.assertEqual(self.book.records(), ())

    def test_real_cli_wrapper_passes_exact_canonical_identity_and_read_only_tools(self):
        target = identity("claude_cli", alias="sonnet", provider="anthropic", model="claude-sonnet-5-5", selection_mode="exact")
        completed = SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({"type": "result", "subtype": "success", "result": "reviewed", "modelUsage": {"claude-sonnet-5-5": {}}, "num_turns": 1}))
        with patch.object(cli.subprocess, "run", return_value=completed) as run, \
             patch.object(cli, "_load_config", return_value={}), \
             patch.object(cli, "classify_review_dispatch", return_value=(True, "")), \
             patch("model_router.target_identity._cli_exact_capability",
                   return_value=("supported", "fixture capability evidence")):
            result = router._run_opus5_bridge(repo=str(Path(__file__).parent), task="[sonnet-review] inspect file", write=False,
                review=True, model="sonnet", requested_alias="sonnet", identity=target, cfg={})
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--model") + 1], "claude-sonnet-5-5")
        self.assertEqual(command[command.index("--tools") + 1], "Read")
        self.assertIn("Bash", command[command.index("--disallowedTools") + 1])
        self.assertEqual(run.call_args.kwargs["cwd"], str(Path(__file__).parent))
        self.assertEqual(result["effective_model"], "claude-sonnet-5-5")
        self.assertEqual(len(self.book.records()), 1)
        self.assertEqual(self.book.records()[0].observed_model, "claude-sonnet-5-5")

    def test_cli_failure_records_terminal_receipt_and_propagates_typed_exception(self):
        with raw_cli(cli.ClaudeBridgeFailure("model mismatch", "model-mismatch")):
            with self.assertRaises(cli.ClaudeBridgeFailure):
                router._run_opus5_bridge(repo="/work", task="review", write=False, cfg={})
        self.assertEqual(len(self.book.records()), 1)
        self.assertEqual(self.book.records()[0].status, "failed")

    def test_untyped_cli_exception_stays_unknown(self):
        with raw_cli(OSError("CLI missing")):
            with self.assertRaises(OSError):
                router._run_opus5_bridge(repo="/work", task="review", write=False, cfg={})
        self.assertEqual(self.book.records()[0].status, "unknown")


class UnchangedHostAdmissionTests(unittest.TestCase):
    """Fix round 2: the legacy boundary never refuses what the unchanged host accepts."""
    def setUp(self):
        self.book = adapters.ReservationBook()
        self.parent = SimpleNamespace(_delegate_depth=0, session_id="parent")
        owner = patch.object(adapters, "RESERVATIONS", self.book)
        owner.start()
        self.addCleanup(owner.stop)

    def background(self, index):
        return json.dumps({"status": "dispatched", "delegation_id": f"deleg_{index}", "goals": ["g"]})

    def test_background_dispatches_beyond_limit_across_time_are_admitted(self):
        # C1: accepted background work has no usable completion correlation, so it must
        # not keep the parent's slot after the synchronous dispatch window closes.
        starts = []
        with patch("tools.delegate_tool._get_max_concurrent_children", return_value=2):
            for index in range(7):
                transport = "hermes_claude" if index % 2 else "hermes_codex"
                raw = adapters.native_legacy_dispatch(lambda i=index: starts.append(i) or self.background(i),
                    parent=self.parent, transport=transport)
                self.assertNotIn("error", json.loads(raw))
        self.assertEqual(starts, list(range(7)))
        self.assertTrue(all(r.status == "unknown" for r in self.book.records()))  # accepted, not completed

    def test_registered_codex_boundary_admits_background_calls_past_limit(self):
        calls = []
        with patch.object(router, "_load_config", return_value={"enabled": True}), \
             patch.object(router, "_delegation_targets_detail", return_value={}), \
             patch.object(router, "_delegated_claude_review_status", return_value=(None, "")), \
             patch.object(worker_admission, "delegate_task_route", return_value=("openai-codex", "gpt-5.6-terra")), \
             patch.object(worker_admission, "refusal", return_value=""), \
             patch("tools.delegate_tool._get_max_concurrent_children", return_value=1), \
             patch("agent.subagent_lifecycle.get_active_subagent_parent", return_value=self.parent):
            for index in range(4):
                raw = lambda sent, i=index: calls.append(i) or self.background(i)
                response = adapters.guard_legacy_tool_execution(tool_name="delegate_task",
                    args={"goal": f"goal {index}", "background": True}, next_call=raw,
                    session_id="s", turn_id=f"turn-{index}", tool_call_id=f"call-{index}")
                self.assertNotIn("error", json.loads(response))
        self.assertEqual(calls, [0, 1, 2, 3])

    def test_journal_is_bounded_and_never_refuses_legacy_calls(self):
        # N1: more calls than the journal bound, mixing terminal, background and error payloads.
        payloads = (json.dumps({"results": [{"status": "completed"}], "total_duration_seconds": 1}),
            self.background(0), json.dumps({"error": "validation"}), "lost response")
        starts = 0
        with patch.object(adapters, "_MAX_RECORDS", 8):
            for index in range(40):
                def raw(i=index):
                    nonlocal starts
                    starts += 1
                    return payloads[i % len(payloads)]
                response = adapters.dispatch_legacy(raw, transport="hermes_codex", scope="parent:x")
                self.assertIs(response, payloads[index % len(payloads)])
                self.assertLessEqual(len(self.book.records()), 8)
        self.assertEqual(starts, 40)

    def test_default_journal_bound_admits_4097_legacy_calls(self):
        payload = json.dumps({"results": [{"status": "completed"}], "total_duration_seconds": 1})
        starts = []
        for index in range(adapters._MAX_RECORDS + 1):
            self.assertIs(adapters.dispatch_legacy(lambda: starts.append(1) or payload, transport="hermes_codex",
                scope="parent:x"), payload)
        self.assertEqual(len(starts), adapters._MAX_RECORDS + 1)
        self.assertLessEqual(len(self.book.records()), adapters._MAX_RECORDS)

    def test_sealed_public_record_replays_while_retained_and_expires_after_ttl(self):
        clock = [1000.0]
        calls = []
        payload = json.dumps({"results": [{"status": "completed"}], "total_duration_seconds": 1})
        key = ("session", 0, "turn", "call")
        with patch.object(adapters, "_now", lambda: clock[0], create=True):
            first = adapters.dispatch_legacy(lambda: calls.append(1) or payload, transport="hermes_codex",
                scope="p", attempt_key=key, fingerprint="f")
            clock[0] += 60
            self.assertIs(adapters.dispatch_legacy(lambda: calls.append(2) or payload, transport="hermes_codex",
                scope="p", attempt_key=key, fingerprint="f"), first)
            self.assertEqual(calls, [1])
            clock[0] += getattr(adapters, "_RECORD_TTL_SECONDS", 3600) + 1
            adapters.dispatch_legacy(lambda: calls.append(3) or payload, transport="hermes_codex",
                scope="p", attempt_key=key, fingerprint="f")
        self.assertEqual(calls, [1, 3])

    def test_cli_call_after_typed_bridge_failure_is_not_refused(self):
        # N2: ClaudeBridgeFailure is a typed terminal outcome and releases capacity.
        ok = {"bridge_run_id": "run-2", "effective_model": "claude-opus-5-5", "result": "ok"}
        failures = (cli.ClaudeBridgeFailure("model mismatch", "model-mismatch"),
                    cli.ClaudeBridgeFailure("timed out", "timeout"))
        for failure in failures:
            with self.subTest(kind=failure.failure_kind):
                self.book = adapters.ReservationBook()
                with patch.object(adapters, "RESERVATIONS", self.book), \
                     raw_cli([failure, ok]) as dispatch:
                    with self.assertRaises(cli.ClaudeBridgeFailure) as raised:
                        router._run_opus5_bridge(repo="/work", task="review", write=False, cfg={})
                    self.assertIs(raised.exception, failure)
                    self.assertEqual(router._run_opus5_bridge(repo="/work", task="review", write=False, cfg={}), ok)
                self.assertEqual(dispatch.call_count, 2)
                statuses = [r.status for r in self.book.records()]
                self.assertEqual(statuses, ["timed_out" if failure.failure_kind == "timeout" else "failed", "succeeded"])

    def test_two_concurrent_cli_calls_are_not_refused(self):
        entered, release = threading.Event(), threading.Event()
        results, errors = [], []
        def dispatch(*args, **kwargs):
            if not entered.is_set():
                entered.set()
                self.assertTrue(release.wait(5))
            return {"bridge_run_id": f"run-{len(results)}", "effective_model": "claude-opus-5-5"}
        def call():
            try:
                results.append(router._run_opus5_bridge(repo="/work", task="review", write=False, cfg={}))
            except Exception as exc:  # pragma: no cover - failure path of the regression
                errors.append(exc)
        with raw_cli(dispatch):
            first = threading.Thread(target=call)
            first.start()
            self.assertTrue(entered.wait(5))
            try:
                call()  # while the first CLI call is still in flight
            finally:
                release.set()
                first.join(5)
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)

    def test_error_payload_releases_and_records_failed_receipt(self):
        calls = []
        error = json.dumps({"error": "Too many tasks"})
        self.assertIs(adapters.dispatch_legacy(lambda: calls.append(1) or error, transport="hermes_codex",
            scope="parent:y", attempt_key=("w", 1, "t", "a")), error)
        self.assertEqual(self.book.record(("w", 1, "t", "a")).status, "failed")
        ok = json.dumps({"results": [{"status": "completed"}], "total_duration_seconds": 1})
        self.assertIs(adapters.dispatch_legacy(lambda: calls.append(2) or ok, transport="hermes_codex",
            scope="parent:y", attempt_key=("w2", 1, "t", "a")), ok)
        self.assertEqual(calls, [1, 2])

    def test_batch_wider_than_limit_reaches_host_for_its_own_refusal(self):
        calls = []
        host_error = json.dumps({"error": "Too many tasks: 3 provided, but max_concurrent_children is 2."})
        with patch("tools.delegate_tool._get_max_concurrent_children", return_value=2):
            raw = adapters.native_legacy_dispatch(lambda: calls.append(1) or host_error, parent=self.parent,
                transport="hermes_codex")
        self.assertIs(raw, host_error)
        self.assertEqual(calls, [1])


class PublicClaimIdentityTests(unittest.TestCase):
    """Fix round 3: claim identity (N3) and host-only legacy admission (R1).

    Every interleaving goes through the registered public middleware
    ``guard_legacy_tool_execution``. Ordering uses events only, never sleeps.
    """
    def setUp(self):
        self.book = adapters.ReservationBook()
        self.parent = SimpleNamespace(_delegate_depth=0, session_id="parent")
        self.clock = [1000.0]
        self.gates = {}  # goal -> (reached, resume): pause after raw return, before public sealing
        self.events = []
        real_guard = worker_admission.guard_tool_execution
        def pausing_guard(**kw):
            result = real_guard(**kw)
            gate = self.gates.get((kw.get("args") or {}).get("goal"))
            if gate is not None:
                gate[0].set()
                if not gate[1].wait(5):
                    raise AssertionError("paused invocation was never resumed")
                if len(gate) > 2:
                    raise gate[2]
            return result
        for patcher in (patch.object(adapters, "RESERVATIONS", self.book),
                        patch.object(adapters, "_now", lambda: self.clock[0]),
                        patch.object(router, "_load_config", return_value={"enabled": True}),
                        patch.object(router, "_delegation_targets_detail", return_value={}),
                        patch.object(router, "_delegated_claude_review_status", return_value=(None, "")),
                        patch.object(worker_admission, "delegate_task_route", return_value=("openai-codex", "gpt-5.6-terra")),
                        patch.object(worker_admission, "refusal", return_value=""),
                        patch.object(worker_admission, "guard_tool_execution", pausing_guard),
                        patch("tools.delegate_tool._get_max_concurrent_children", return_value=1),
                        patch("agent.subagent_lifecycle.get_active_subagent_parent", return_value=self.parent)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def event(self):
        event = threading.Event()
        self.events.append(event)
        return event

    def spawn(self, fn):
        box = {}
        def run():
            try:
                box["result"] = fn()
            except BaseException as exc:  # surfaced by the test's own assertions
                box["error"] = exc
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(lambda: [event.set() for event in self.events])  # runs before the join
        return thread, box

    def call(self, call_id, goal, raw):
        return adapters.guard_legacy_tool_execution(tool_name="delegate_task", args={"goal": goal},
            next_call=raw, session_id="s", turn_id="t", tool_call_id=call_id)

    @staticmethod
    def result(name):
        return json.dumps({"results": [{"status": "completed", "result": name}], "total_duration_seconds": 1})

    def pause(self, goal):
        reached, resume = self.event(), self.event()
        self.gates[goal] = (reached, resume)
        return reached, resume

    def test_a03_concurrent_claude_lookup_miss_replays_final_public_response(self):
        book = adapters.ReservationBook()
        parent = SimpleNamespace(_delegate_depth=0, session_id='parent')
        begin = book._begin
        both_missed, release_claim = threading.Event(), threading.Event()
        arrivals, arrival_lock = [0], threading.Lock()
        results, errors, calls, audits = {}, [], [], []

        def gated_begin(*args, **kwargs):
            with arrival_lock:
                arrivals[0] += 1
                if arrivals[0] == 2:
                    both_missed.set()
            if not release_claim.wait(5):
                raise AssertionError('both public callers did not reach atomic claim')
            return begin(*args, **kwargs)

        def raw(**kwargs):
            calls.append(kwargs)
            claude._REASONING_SCOPE.get().applied[0] += 1
            return json.dumps({'results': [{'status': 'completed', 'summary': 'done'}]})

        def invoke(label):
            try:
                results[label] = adapters.guard_legacy_tool_execution(tool_name='delegate_claude',
                    args={'tier': 'sonnet', 'goal': 'review'}, next_call=claude.handle_delegate_claude,
                    session_id='wf', turn_id='turn', tool_call_id='same-call')
            except BaseException as exc:
                errors.append(exc)

        with patch.object(adapters, 'RESERVATIONS', book), \
             patch.object(book, '_begin', side_effect=gated_begin), \
             patch.object(router, '_load_config', return_value=_cfg()), \
             patch.object(claude, '_host', return_value=(raw, lambda: parent)), \
             patch.object(claude, 'install_reasoning_bridge', return_value=(True, '')), \
             patch.object(claude.usage_guard, 'read', return_value=_reading(10)), \
             patch.object(claude.usage_guard, 'apply', wraps=claude.usage_guard.apply) as apply, \
             patch.object(claude, '_log', side_effect=lambda _cfg, entry: audits.append(entry)):
            first = threading.Thread(target=invoke, args=('first',), name='first')
            second = threading.Thread(target=invoke, args=('second',), name='second')
            first.start()
            second.start()
            self.assertTrue(both_missed.wait(5))
            release_claim.set()
            first.join(6)
            second.join(6)
            replay = adapters.guard_legacy_tool_execution(tool_name='delegate_claude',
                args={'tier': 'sonnet', 'goal': 'review'}, next_call=claude.handle_delegate_claude,
                session_id='wf', turn_id='turn', tool_call_id='same-call')
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertFalse(errors, repr(errors))
        self.assertEqual(len(calls), 1)
        self.assertEqual(apply.call_count, 1)
        self.assertEqual(len(audits), 1)
        pending = json.dumps({'error': 'Invocation already started/unknown, response unavailable, or payload changed. Nothing was spawned by this call.'})
        self.assertTrue(all(result in (replay, pending) for result in results.values()))
        self.assertIn(replay, results.values())

    def test_public_haiku_receipt_keeps_not_applicable_effort(self):
        raw_calls = []
        def raw(**kwargs):
            raw_calls.append(kwargs)
            return self.result("haiku")
        with patch.object(router, "_load_config", return_value=_cfg()), \
             patch.object(claude, "_host", return_value=(raw, lambda: self.parent)), \
             patch.object(claude.usage_guard, "read", return_value=_reading(10.0)):
            response = adapters.guard_legacy_tool_execution(tool_name="delegate_claude",
                args={"goal": "receipt", "tier": "haiku"}, next_call=claude.handle_delegate_claude,
                session_id="receipt", turn_id="turn", tool_call_id="haiku")
        self.assertNotIn("error", json.loads(response))
        receipt = self.book.record(("receipt", 0, "turn", "haiku"))
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.effort, "not_applicable")
        self.assertEqual(receipt.resolved_tier, "haiku")
        self.assertEqual(len(raw_calls), 1)

    def test_public_claude_non_json_result_keeps_resolved_tier_and_adjustment(self):
        raw_calls = []
        def raw(**kwargs):
            raw_calls.append(kwargs)
            claude._REASONING_SCOPE.get().applied[0] += 1
            return "not json"
        with patch.object(router, "_load_config", return_value=_cfg()), \
             patch.object(claude, "_host", return_value=(raw, lambda: self.parent)), \
             patch.object(claude, "install_reasoning_bridge", return_value=(True, "")), \
             patch.object(claude.usage_guard, "read", return_value=_reading(75.0)):
            response = adapters.guard_legacy_tool_execution(tool_name="delegate_claude",
                args={"goal": "receipt", "tier": "opus"}, next_call=claude.handle_delegate_claude,
                session_id="receipt", turn_id="turn", tool_call_id="non-json")
        self.assertEqual(response, "not json")
        receipt = self.book.record(("receipt", 0, "turn", "non-json"))
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.resolved_tier, "sonnet")
        self.assertIn("opus", receipt.adjustment)
        self.assertEqual(len(raw_calls), 1)

    def test_identical_claude_duplicate_before_seal_is_pending_then_replays_once(self):
        raw_calls, scopes, audits = [], [], []
        paused, release = self.event(), self.event()
        real_annotate, real_scope = claude._annotate, claude.reasoning_scope
        @contextmanager
        def counted_scope(*args, **kwargs):
            scopes.append(1)
            with real_scope(*args, **kwargs) as scope:
                yield scope
        def raw(**kwargs):
            raw_calls.append(kwargs)
            claude._REASONING_SCOPE.get().applied[0] += 1
            return self.result("one")
        def paused_annotate(*args, **kwargs):
            paused.set()
            self.assertTrue(release.wait(5))
            return real_annotate(*args, **kwargs)
        def call():
            return adapters.guard_legacy_tool_execution(tool_name="delegate_claude",
                args={"goal": "one", "tier": "sonnet"}, next_call=claude.handle_delegate_claude,
                session_id="duplicate", turn_id="turn", tool_call_id="before-seal")
        with patch.object(router, "_load_config", return_value=_cfg()), \
             patch.object(claude, "_host", return_value=(raw, lambda: self.parent)), \
             patch.object(claude, "install_reasoning_bridge", return_value=(True, "")), \
             patch.object(claude, "reasoning_scope", counted_scope), \
             patch.object(claude, "_annotate", paused_annotate), \
             patch.object(claude.usage_guard, "read", return_value=_reading(10.0)) as read, \
             patch.object(claude.usage_guard, "apply", wraps=claude.usage_guard.apply) as apply, \
             patch.object(claude, "_log", side_effect=lambda _cfg, entry: audits.append(entry)):
            thread, box = self.spawn(call)
            self.assertTrue(paused.wait(5))
            pending = call()
            self.assertEqual(pending, json.dumps({"error":
                "Invocation already started/unknown, response unavailable, or payload changed. Nothing was spawned by this call."}))
            release.set()
            thread.join(5)
            first = box.get("result")
            self.assertEqual(call(), first)
        self.assertEqual((len(raw_calls), read.call_count, apply.call_count, len(scopes), len(audits)), (1, 1, 1, 1, 1))

    def test_loser_gated_in_begin_after_seal_replays_without_reannotation(self):
        raw_calls, scopes, audits = [], [], []
        annotation_paused, release_annotation = self.event(), self.event()
        loser_entered, release_loser = self.event(), self.event()
        real_begin, real_annotate, real_scope = self.book._begin, claude._annotate, claude.reasoning_scope
        public_calls, begin_lock = [0], threading.Lock()
        @contextmanager
        def counted_scope(*args, **kwargs):
            scopes.append(1)
            with real_scope(*args, **kwargs) as scope:
                yield scope
        def gated_begin(*args, **kwargs):
            if args[0].startswith("public:"):
                with begin_lock:
                    public_calls[0] += 1
                    loser = public_calls[0] == 2
                if loser:
                    loser_entered.set()
                    self.assertTrue(release_loser.wait(5))
            return real_begin(*args, **kwargs)
        def raw(**kwargs):
            raw_calls.append(kwargs)
            claude._REASONING_SCOPE.get().applied[0] += 1
            return self.result("sealed")
        def paused_annotate(*args, **kwargs):
            annotation_paused.set()
            self.assertTrue(release_annotation.wait(5))
            return real_annotate(*args, **kwargs)
        def call():
            return adapters.guard_legacy_tool_execution(tool_name="delegate_claude",
                args={"goal": "sealed", "tier": "sonnet"}, next_call=claude.handle_delegate_claude,
                session_id="duplicate", turn_id="turn", tool_call_id="after-seal")
        with patch.object(self.book, "_begin", side_effect=gated_begin), \
             patch.object(router, "_load_config", return_value=_cfg()), \
             patch.object(claude, "_host", return_value=(raw, lambda: self.parent)), \
             patch.object(claude, "install_reasoning_bridge", return_value=(True, "")), \
             patch.object(claude, "reasoning_scope", counted_scope), \
             patch.object(claude, "_annotate", paused_annotate), \
             patch.object(claude.usage_guard, "read", return_value=_reading(10.0)) as read, \
             patch.object(claude.usage_guard, "apply", wraps=claude.usage_guard.apply) as apply, \
             patch.object(claude, "_log", side_effect=lambda _cfg, entry: audits.append(entry)):
            owner, owner_box = self.spawn(call)
            self.assertTrue(annotation_paused.wait(5))
            loser, loser_box = self.spawn(call)
            self.assertTrue(loser_entered.wait(5))
            release_annotation.set()
            owner.join(5)
            release_loser.set()
            loser.join(5)
        self.assertEqual(loser_box.get("result"), owner_box.get("result"))
        self.assertEqual((len(raw_calls), read.call_count, apply.call_count, len(scopes), len(audits)), (1, 1, 1, 1, 1))

    def test_sealed_claude_duplicate_ignores_later_usage_and_settings_changes(self):
        cfg, raw_calls, audits = _cfg(), [], []
        def raw(**kwargs):
            raw_calls.append(kwargs)
            claude._REASONING_SCOPE.get().applied[0] += 1
            return self.result("original")
        def call():
            return adapters.guard_legacy_tool_execution(tool_name="delegate_claude",
                args={"goal": "original", "tier": "opus"}, next_call=claude.handle_delegate_claude,
                session_id="duplicate", turn_id="turn", tool_call_id="settings")
        with patch.object(router, "_load_config", return_value=cfg), \
             patch.object(claude, "_host", return_value=(raw, lambda: self.parent)), \
             patch.object(claude, "install_reasoning_bridge", return_value=(True, "")), \
             patch.object(claude.usage_guard, "read", return_value=_reading(10.0)) as read, \
             patch.object(claude.usage_guard, "apply", wraps=claude.usage_guard.apply) as apply, \
             patch.object(claude, "_log", side_effect=lambda _cfg, entry: audits.append(entry)):
            first = call()
            cfg["callable"]["opus5"] = cfg["callable"]["sonnet5"] = False
            read.return_value = _reading(95.0)
            duplicate = call()
        self.assertEqual(duplicate, first)
        self.assertEqual((len(raw_calls), read.call_count, apply.call_count, len(audits)), (1, 1, 1, 1))

    def test_public_claude_base_exception_unwinds_to_unknown_and_blocks_duplicate(self):
        raw_calls = []
        class Fatal(BaseException):
            pass
        def raw(**kwargs):
            raw_calls.append(kwargs)
            raise Fatal("raw lost")
        def call():
            return adapters.guard_legacy_tool_execution(tool_name="delegate_claude",
                args={"goal": "fatal", "tier": "haiku"}, next_call=claude.handle_delegate_claude,
                session_id="duplicate", turn_id="turn", tool_call_id="fatal")
        with patch.object(router, "_load_config", return_value=_cfg()), \
             patch.object(claude, "_host", return_value=(raw, lambda: self.parent)), \
             patch.object(claude.usage_guard, "read", return_value=_reading(10.0)):
            with self.assertRaises(Fatal):
                call()
            duplicate = call()
        receipt = self.book.record(("duplicate", 0, "turn", "fatal"))
        self.assertEqual(receipt.status, "unknown")
        with self.book._lock:
            claim = self.book._claims[("duplicate", 0, "turn", "fatal")]
            self.assertFalse(claim.protected)
            self.assertFalse(claim.in_flight)
        self.assertIn("error", json.loads(duplicate))
        self.assertEqual(len(raw_calls), 1)

    def test_oversized_public_responses_return_to_owner_and_block_duplicates(self):
        oversized = "x" * (adapters._MAX_CACHED_RESPONSE + 1)
        codex_calls, claude_calls = [], []
        codex = adapters.guard_legacy_tool_execution(tool_name="delegate_task", args={"goal": "large"},
            next_call=lambda sent: codex_calls.append(sent) or oversized,
            session_id="oversized", turn_id="turn", tool_call_id="codex")
        codex_duplicate = adapters.guard_legacy_tool_execution(tool_name="delegate_task", args={"goal": "large"},
            next_call=lambda sent: codex_calls.append(sent) or "SHOULD NOT START",
            session_id="oversized", turn_id="turn", tool_call_id="codex")
        with patch.object(router, "_load_config", return_value=_cfg()), \
             patch.object(claude, "_host", return_value=(lambda **kwargs: claude_calls.append(kwargs) or oversized, lambda: self.parent)), \
             patch.object(claude.usage_guard, "read", return_value=_reading(10.0)):
            claude_response = adapters.guard_legacy_tool_execution(tool_name="delegate_claude",
                args={"goal": "large", "tier": "haiku"}, next_call=claude.handle_delegate_claude,
                session_id="oversized", turn_id="turn", tool_call_id="claude")
            claude_duplicate = adapters.guard_legacy_tool_execution(tool_name="delegate_claude",
                args={"goal": "large", "tier": "haiku"}, next_call=claude.handle_delegate_claude,
                session_id="oversized", turn_id="turn", tool_call_id="claude")
        self.assertEqual((codex, claude_response), (oversized, oversized))
        pending = json.dumps({"error":
            "Invocation already started/unknown, response unavailable, or payload changed. Nothing was spawned by this call."})
        self.assertEqual(codex_duplicate, pending)
        self.assertEqual(claude_duplicate, pending)
        self.assertEqual((len(codex_calls), len(claude_calls)), (1, 1))

    def test_changed_transport_under_public_ids_is_rejected_before_claude_usage_read(self):
        args, codex_calls = {"goal": "same", "tier": "haiku"}, []
        first = adapters.guard_legacy_tool_execution(tool_name="delegate_task", args=args,
            next_call=lambda sent: codex_calls.append(sent) or self.result("codex"),
            session_id="transport", turn_id="turn", tool_call_id="same")
        with patch.object(claude.usage_guard, "read", side_effect=AssertionError("duplicate read usage")) as read:
            changed = adapters.guard_legacy_tool_execution(tool_name="delegate_claude", args=args,
                next_call=claude.handle_delegate_claude,
                session_id="transport", turn_id="turn", tool_call_id="same")
        self.assertEqual(first, self.result("codex"))
        self.assertEqual(changed, json.dumps({"error":
            "Invocation already started/unknown, response unavailable, or payload changed. Nothing was spawned by this call."}))
        self.assertEqual((len(codex_calls), read.call_count), (1, 0))

    def test_old_public_completion_never_overwrites_or_unprotects_replacement_claim(self):
        # N3 interleaving from re-review 2, driven through the public middleware.
        key, calls = ("s", 0, "t", "K"), []
        a_reached, a_resume = self.pause("goal A")
        a_thread, a_box = self.spawn(lambda: self.call("K", "goal A", lambda sent: calls.append("A") or self.result("A")))
        self.assertTrue(a_reached.wait(5))  # A: raw returned, public seal not yet written
        # Force real TTL expiry plus LRU pressure (this evicts A on 1aad967) ...
        self.clock[0] += adapters._RECORD_TTL_SECONDS + 1
        with patch.object(adapters, "_MAX_RECORDS", 1):
            self.assertEqual(self.call("F0", "filler 0", lambda sent: calls.append("F0") or self.result("F0")), self.result("F0"))
        # ... then remove A's record directly, so the claim-identity check (not just
        # A's protection) is what keeps A's late seal away from the replacement claim.
        with self.book._lock:
            self.book._claims.pop(key, None)
        b_started, b_release = self.event(), self.event()
        def raw_b(sent):
            calls.append("B")
            b_started.set()
            if not b_release.wait(5):
                raise AssertionError("B never released")
            return self.result("B")
        b_thread, b_box = self.spawn(lambda: self.call("K", "goal B", raw_b))
        self.assertTrue(b_started.wait(5))
        b_record = self.book.record(key)
        self.assertIsNotNone(b_record)
        a_resume.set()
        a_thread.join(5)
        self.assertEqual(a_box.get("result"), self.result("A"))
        # B's record is untouched by A's completion and seal.
        self.assertEqual(self.book.record(key), b_record)
        # An identical repeat of B never receives A's response and never re-dispatches B.
        repeat = self.call("K", "goal B", lambda sent: calls.append("B again") or "SHOULD NOT START")
        self.assertNotEqual(repeat, self.result("A"))
        self.assertIn("error", json.loads(repeat))
        # B stays protected from LRU and TTL eviction while it runs.
        self.clock[0] += adapters._RECORD_TTL_SECONDS + 1
        with patch.object(adapters, "_MAX_RECORDS", 1):
            self.assertEqual(self.call("F", "filler", lambda sent: calls.append("F") or self.result("F")), self.result("F"))
        self.assertEqual(self.book.record(key), b_record)
        self.assertIn("error", json.loads(self.call("K", "goal B", lambda sent: calls.append("B again") or "SHOULD NOT START")))
        b_release.set()
        b_thread.join(5)
        self.assertEqual(b_box.get("result"), self.result("B"))
        self.assertEqual(self.call("K", "goal B", lambda sent: calls.append("B again") or "SHOULD NOT START"), self.result("B"))
        self.assertEqual(calls.count("B"), 1)
        self.assertNotIn("B again", calls)

    def test_own_record_is_not_lru_evicted_between_raw_return_and_seal(self):
        key, calls = ("s", 0, "t", "K"), []
        a_reached, a_resume = self.pause("goal A")
        a_thread, a_box = self.spawn(lambda: self.call("K", "goal A", lambda sent: calls.append("A") or self.result("A")))
        self.assertTrue(a_reached.wait(5))
        with patch.object(adapters, "_MAX_RECORDS", 1):
            self.assertEqual(self.call("F", "filler", lambda sent: calls.append("F") or self.result("F")), self.result("F"))
            self.assertIsNotNone(self.book.record(key))
            changed = self.call("K", "goal B", lambda sent: calls.append("B") or self.result("B"))
            self.assertIn("error", json.loads(changed))
            a_resume.set()
            a_thread.join(5)
        self.assertEqual(a_box.get("result"), self.result("A"))
        self.assertEqual(self.call("K", "goal A", lambda sent: calls.append("A again") or "NO"), self.result("A"))
        self.assertEqual(calls, ["A", "F"])

    def test_own_record_does_not_expire_between_raw_return_and_seal(self):
        key, calls = ("s", 0, "t", "K"), []
        a_reached, a_resume = self.pause("goal A")
        a_thread, a_box = self.spawn(lambda: self.call("K", "goal A", lambda sent: calls.append("A") or self.result("A")))
        self.assertTrue(a_reached.wait(5))
        self.clock[0] += adapters._RECORD_TTL_SECONDS + 1
        changed = self.call("K", "goal B", lambda sent: calls.append("B") or self.result("B"))
        self.assertIn("error", json.loads(changed))
        self.assertIsNotNone(self.book.record(key))
        a_resume.set()
        a_thread.join(5)
        self.assertEqual(a_box.get("result"), self.result("A"))
        self.assertEqual(self.call("K", "goal A", lambda sent: calls.append("A again") or "NO"), self.result("A"))
        self.assertEqual(calls, ["A"])

    def test_claude_handler_post_processing_keeps_record_protected(self):
        # The public delegate_claude handler annotates after raw return (claude._annotate).
        calls, key = [], ("s", 0, "t", "C")
        reached, resume = self.event(), self.event()
        real_annotate = claude._annotate
        def paused_annotate(*args, **kwargs):
            if not reached.is_set():  # pause only invocation "a"
                reached.set()
                if not resume.wait(5):
                    raise AssertionError("annotation never resumed")
            return real_annotate(*args, **kwargs)
        raw = lambda **kw: calls.append(kw["goal"]) or self.result(kw["goal"])
        def call(goal):
            return adapters.guard_legacy_tool_execution(tool_name="delegate_claude", args={"goal": goal, "tier": "haiku"},
                next_call=claude.handle_delegate_claude, session_id="s", turn_id="t", tool_call_id="C")
        with patch.object(router, "_load_config", return_value=_cfg()), \
             patch.object(claude, "_host", return_value=(raw, lambda: self.parent)), \
             patch.object(claude.usage_guard, "read", return_value=_reading(10.0)), \
             patch.object(claude, "_annotate", paused_annotate):
            thread, box = self.spawn(lambda: call("a"))
            self.assertTrue(reached.wait(5))
            self.clock[0] += adapters._RECORD_TTL_SECONDS + 1
            self.assertIn("error", json.loads(call("b")))
            self.assertIsNotNone(self.book.record(key))
            resume.set()
            thread.join(5)
            first = box.get("result")
            self.assertEqual(json.loads(first)["claude_tier"], "haiku")
            self.assertEqual(call("a"), first)
        self.assertEqual(calls, ["a"])

    def test_unwound_public_invocation_releases_its_protection(self):
        # Protection lasts until the seal or the invocation unwinds (exception path).
        calls = []
        reached, resume = self.event(), self.event()
        self.gates["boom"] = (reached, resume, RuntimeError("post-processing failed"))
        resume.set()
        with self.assertRaises(RuntimeError):
            self.call("E1", "boom", lambda sent: calls.append("E1") or self.result("E1"))
        def raising(sent):
            calls.append("E2")
            raise RuntimeError("raw dispatch lost its response")
        with self.assertRaises(RuntimeError):
            self.call("E2", "raw raises", raising)
        self.assertEqual(self.book.record(("s", 0, "t", "E2")).status, "unknown")
        with patch.object(adapters, "_MAX_RECORDS", 1):
            self.assertEqual(self.call("F", "filler", lambda sent: calls.append("F") or self.result("F")), self.result("F"))
        self.assertIsNone(self.book.record(("s", 0, "t", "E1")))
        self.assertIsNone(self.book.record(("s", 0, "t", "E2")))
        self.assertEqual(calls, ["E1", "E2", "F"])

    def test_public_changed_payload_after_ttl_expiry_is_a_new_call(self):
        calls = []
        first = self.call("K", "P1", lambda sent: calls.append("P1") or self.result("P1"))
        self.assertIn("error", json.loads(self.call("K", "P2", lambda sent: calls.append("early") or "NO")))
        self.assertEqual(self.call("K", "P1", lambda sent: calls.append("again") or "NO"), first)
        self.clock[0] += adapters._RECORD_TTL_SECONDS + 1
        second = self.call("K", "P2", lambda sent: calls.append("P2") or self.result("P2"))
        self.assertEqual(second, self.result("P2"))
        self.assertEqual(self.call("K", "P2", lambda sent: calls.append("again") or "NO"), second)
        self.assertIn("error", json.loads(self.call("K", "P1", lambda sent: calls.append("again") or "NO")))
        self.assertEqual(calls, ["P1", "P2"])

    def test_distinct_concurrent_public_calls_on_one_parent_are_admitted_at_limit_one(self):
        # R1: the host is the sole admission authority for legacy dispatches.
        calls = []
        started, release = self.event(), self.event()
        def slow(sent):
            calls.append("first")
            started.set()
            if not release.wait(5):
                raise AssertionError("first call never released")
            return self.result("first")
        thread, box = self.spawn(lambda: self.call("c1", "first", slow))
        self.assertTrue(started.wait(5))
        second = self.call("c2", "second", lambda sent: calls.append("second") or self.result("second"))
        self.assertEqual(second, self.result("second"))
        release.set()
        thread.join(5)
        self.assertEqual(box.get("result"), self.result("first"))
        self.assertEqual(calls, ["first", "second"])

    def test_distinct_concurrent_native_calls_mixed_transports_are_admitted_at_limit_one(self):
        calls = []
        started, release = self.event(), self.event()
        def slow():
            calls.append("codex")
            started.set()
            if not release.wait(5):
                raise AssertionError("codex call never released")
            return self.result("codex")
        thread, box = self.spawn(lambda: adapters.native_legacy_dispatch(slow, parent=self.parent,
            transport="hermes_codex"))
        self.assertTrue(started.wait(5))
        claude_raw = adapters.native_legacy_dispatch(lambda: calls.append("claude") or self.result("claude"),
            parent=self.parent, transport="hermes_claude")
        self.assertEqual(claude_raw, self.result("claude"))
        release.set()
        thread.join(5)
        self.assertEqual(box.get("result"), self.result("codex"))
        self.assertEqual(calls, ["codex", "claude"])

    def test_wide_batch_through_public_middleware_reaches_host_refusal(self):
        calls = []
        host_error = json.dumps({"error": "Too many tasks: 3 provided, but max_concurrent_children is 1."})
        response = adapters.guard_legacy_tool_execution(tool_name="delegate_task",
            args={"tasks": [{"goal": "a"}, {"goal": "b"}, {"goal": "c"}]},
            next_call=lambda sent: calls.append(1) or host_error, session_id="s", turn_id="t", tool_call_id="wide")
        self.assertIs(response, host_error)
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()
