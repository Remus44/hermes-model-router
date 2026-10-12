"""S04 execution-record and adapter-double contract tests (I03-I10)."""
import json
import unittest
from dataclasses import replace

from model_router import execution_contracts as contracts
from model_router import execution_test_doubles as doubles


IDENTITY = contracts.TargetIdentity(
    provider="openai-codex", account="codex-primary", transport="hermes_codex",
    alias="terra", selection_mode="exact",
    requested=contracts.ModelFact("gpt-5.6-terra", "operator_request"),
    resolved=contracts.ModelFact("gpt-5.6-terra", "host_target"),
    observed=contracts.ModelFact(), effort=contracts.EffortFact("high", "unknown", "not_observed"),
)


def request(**changes):
    fields = {
        "workflow_id": "wf-001", "plan_version": 1, "task_id": "task-001", "attempt_id": "attempt-001",
        "goal": "Implement one bounded change.", "acceptance_criteria": ("tests pass",),
        "context_reference": "repo:README.md@abc123", "target": IDENTITY,
        "repository": "repo:/work/router", "workspace": "workspace:/work/router",
        "permissions": ("read", "write"), "tool_requirements": ("terminal",), "mutating": True,
        "write_scope": ("execution_contracts.py",), "timeout_seconds": 60, "deadline_epoch_ms": 999999,
        "attempt_budget": 2, "verification_policy": "required", "substitution_policy": "forbid",
    }
    fields.update(changes)
    return contracts.ExecutionRequest(**fields)


class RecordValidationTests(unittest.TestCase):
    def test_request_round_trips_stably_with_unknown_effort_and_usage(self):
        original = request()
        data = original.as_dict()
        self.assertEqual(contracts.ExecutionRequest.from_dict(json.loads(json.dumps(data))).as_dict(), data)
        self.assertEqual(original.target.effort.applied, "unknown")
        result = contracts.WorkerResult(
            workflow_id=original.workflow_id, task_id=original.task_id, attempt_id=original.attempt_id,
            handle="handle-001", terminal_status="succeeded", summary="completed",
            output=contracts.OutputReference("short output", "artifact:output-001", True),
            requested_target=IDENTITY, resolved_target=IDENTITY, observed_target=IDENTITY,
            usage={"input_tokens": "unknown", "future_provider_meter": "unknown"},
            validation_evidence=("test: OK",), artifacts=("artifact:output-001",),
        )
        data = result.as_dict()
        self.assertEqual(contracts.WorkerResult.from_dict(json.loads(json.dumps(data))).as_dict(), data)
        self.assertEqual(data["usage"]["future_provider_meter"], "unknown")

    def test_invalid_types_identifiers_and_unknown_routing_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            request(workflow_id="bad id")
        with self.assertRaises(TypeError):
            request(timeout_seconds="60")
        data = request().as_dict()
        data["permissions"] = "write"
        with self.assertRaises(TypeError):
            contracts.ExecutionRequest.from_dict(data)
        data = request().as_dict()
        data["invented_route"] = "terra"
        with self.assertRaises(ValueError):
            contracts.ExecutionRequest.from_dict(data)
        data = request().as_dict()
        data["target"] = {"alias": "terra"}
        with self.assertRaises(ValueError):
            contracts.ExecutionRequest.from_dict(data)
        result = contracts.WorkerResult(
            workflow_id="wf-001", task_id="task-001", attempt_id="attempt-001", handle="handle-001",
            terminal_status="succeeded", summary="ok", output=contracts.OutputReference("ok"),
            requested_target=IDENTITY, resolved_target=IDENTITY, observed_target=IDENTITY,
        ).as_dict()
        result["artifacts"] = "artifact:wrong-shape"
        with self.assertRaises(TypeError):
            contracts.WorkerResult.from_dict(result)

    def test_legacy_direct_tool_identity_needs_no_task_graph(self):
        legacy = contracts.legacy_direct_tool_request(
            tool_invocation_id="tool-abc", goal="Read one file.", target=IDENTITY,
            repository="repo:/work/router", workspace="workspace:/work/router",
            acceptance_criteria=("file read",), permissions=("read",), tool_requirements=("read_file",),
            mutating=False, write_scope=(), timeout_seconds=60, deadline_epoch_ms=2000000000000,
            attempt_budget=1, verification_policy="required", substitution_policy="forbid",
        )
        self.assertEqual(legacy.workflow_id, "legacy:tool-abc")
        self.assertEqual(legacy.attempt_id, "legacy:tool-abc:attempt")
        self.assertEqual(legacy.plan_version, 0)
        self.assertEqual(legacy.task_id, "legacy:tool-abc:task")

    def test_output_and_artifact_bounds_do_not_discard_failure_or_verification_evidence(self):
        with self.assertRaises(ValueError):
            contracts.OutputReference("x" * (contracts.MAX_OUTPUT_SUMMARY_CHARS + 1))
        with self.assertRaises(ValueError):
            contracts.WorkerResult(
                workflow_id="wf-001", task_id="task-001", attempt_id="attempt-001", handle="handle-001",
                terminal_status="succeeded", summary="ok", output=contracts.OutputReference("x"),
                requested_target=IDENTITY, resolved_target=IDENTITY, observed_target=IDENTITY,
                artifacts=("artifact:" + "x" * contracts.MAX_ARTIFACT_REFERENCE_CHARS,),
            )
        result = contracts.WorkerResult(
            workflow_id="wf-001", task_id="task-001", attempt_id="attempt-001", handle="handle-001",
            terminal_status="failed", summary="failed", output=contracts.OutputReference("x", "artifact:full-log", truncated=True),
            requested_target=IDENTITY, resolved_target=IDENTITY, observed_target=IDENTITY,
            failure=contracts.FailureDetail("execution-error", False, "inspect artifact"),
            validation_evidence=("verification failed",), artifacts=("artifact:full-log",),
        )
        self.assertEqual(result.failure.failure_class, "execution-error")
        self.assertEqual(result.validation_evidence, ("verification failed",))


class LifecycleTests(unittest.TestCase):
    def test_only_declared_lifecycle_transitions_are_allowed(self):
        state = contracts.AttemptLifecycle.created("wf-001", "task-001", "attempt-001")
        state = state.transition("submitted", handle="handle-001")
        state = state.transition("running")
        self.assertEqual(state.transition("succeeded").status, "succeeded")
        with self.assertRaises(ValueError):
            state.transition("created")
        with self.assertRaises(ValueError):
            contracts.AttemptLifecycle.created("wf-001", "task-001", "attempt-001").transition("succeeded")
        self.assertEqual(state.transition("unknown").status, "unknown")
        with self.assertRaises(ValueError):
            contracts.AttemptLifecycle("wf-001", "task-001", "attempt-001", "submitted")

    def test_lifecycle_and_submission_round_trip_without_cross_consumption(self):
        lifecycle = contracts.AttemptLifecycle.created("wf-001", "task-001", "attempt-001").transition(
            "submitted", handle="handle-001")
        lifecycle_data = json.loads(json.dumps(lifecycle.as_dict()))
        self.assertEqual(contracts.AttemptLifecycle.from_dict(lifecycle_data), lifecycle)
        submission = contracts.Submission.for_acceptance("wf-001", "task-001", "attempt-001", "handle-001")
        submission_data = json.loads(json.dumps(submission.as_dict()))
        self.assertEqual(contracts.Submission.from_dict(submission_data), submission)
        submission_data["unrecognized_route"] = "x"
        with self.assertRaises(ValueError):
            contracts.Submission.from_dict(submission_data)

    def test_accepted_submission_is_not_a_terminal_worker_result(self):
        accepted = contracts.Submission.for_acceptance("wf-001", "task-001", "attempt-001", "handle-001")
        self.assertTrue(accepted.accepted)
        self.assertFalse(accepted.terminal)
        self.assertNotIsInstance(accepted, contracts.WorkerResult)
        with self.assertRaises(ValueError):
            contracts.WorkerResult.from_dict(accepted.as_dict())


class AdapterDoubleTests(unittest.TestCase):
    def test_delayed_completion_is_pending_then_terminal_without_calls(self):
        result = doubles.succeeded_result(request(), "handle-delay")
        adapter = doubles.FakeExecutionAdapter.delayed(result, pending_polls=2)
        submission = adapter.submit(request())
        self.assertTrue(submission.accepted)
        self.assertEqual(adapter.result(submission.handle).status, "pending")
        self.assertEqual(adapter.result(submission.handle).status, "pending")
        self.assertEqual(adapter.result(submission.handle).terminal_status, "succeeded")
        self.assertEqual(adapter.calls, {"capabilities": 0, "can_execute": 0, "submit": 1, "result": 3, "cancel": 0})

    def test_partial_batch_capability_refusal_identity_mismatch_unknown_and_cancel(self):
        first, second = request(attempt_id="attempt-002"), request(attempt_id="attempt-003")
        adapter = doubles.FakeExecutionAdapter.partial_batch(
            succeeded=doubles.succeeded_result(first, "handle-ok"),
            failed=doubles.failed_result(second, "handle-fail", "provider-transient"),
        )
        outcomes = adapter.submit_batch((first, second))
        self.assertEqual([outcome.accepted for outcome in outcomes], [True, True])
        self.assertEqual(adapter.result("handle-ok").terminal_status, "succeeded")
        self.assertEqual(adapter.result("handle-fail").terminal_status, "failed")

        refused = doubles.FakeExecutionAdapter.capability_refusal("required workspace unavailable")
        self.assertEqual(refused.can_execute(request(), {}).status, "unsupported")
        self.assertFalse(refused.submit(request()).accepted)

        mismatch = doubles.FakeExecutionAdapter.identity_mismatch(request(), "gpt-other")
        observed = mismatch.result(mismatch.submit(request()).handle).observed_target
        self.assertEqual(observed.observed.value, "gpt-other")

        unknown = doubles.FakeExecutionAdapter.unknown_liveness(request())
        self.assertEqual(unknown.result(unknown.submit(request()).handle).status, "unknown")

        cancellable = doubles.FakeExecutionAdapter.cancellable(request())
        handle = cancellable.submit(request()).handle
        self.assertEqual(cancellable.cancel(handle).status, "acknowledged")
        self.assertEqual(cancellable.result(handle).terminal_status, "cancelled")


class ReviewRegressionTests(unittest.TestCase):
    def result(self, **changes):
        return replace(doubles.succeeded_result(request(), "handle-review"), **changes)

    def test_request_sequences_are_owned_and_round_trip_equal(self):
        criteria = ["tests pass"]
        original = request(acceptance_criteria=criteria, permissions=["read"],
                           tool_requirements=["terminal"], write_scope=["file"])
        criteria.append("unauthorized")
        self.assertEqual(original.acceptance_criteria, ("tests pass",))
        self.assertEqual(contracts.ExecutionRequest.from_dict(original.as_dict()), original)
        self.assertEqual(hash(contracts.ExecutionRequest.from_dict(original.as_dict())), hash(original))

    def test_result_nested_data_and_sequences_are_defensively_owned(self):
        usage = {"meter": {"samples": [1, None]}}
        evidence, artifacts = ["check"], ["artifact:log"]
        result = self.result(usage=usage, validation_evidence=evidence, artifacts=artifacts)
        usage["meter"]["samples"].append("x" * 10000)
        evidence.append("forged")
        artifacts.append("forged")
        self.assertEqual(result.as_dict()["usage"], {"meter": {"samples": [1, None]}})
        self.assertEqual(result.validation_evidence, ("check",))
        self.assertEqual(result.artifacts, ("artifact:log",))
        with self.assertRaises(TypeError):
            result.usage["meter"]["samples"][0] = 2
        wire = result.as_dict()
        wire["usage"]["meter"]["samples"].append(2)
        self.assertEqual(contracts.WorkerResult.from_dict(result.as_dict()), result)

    def test_truncation_requires_reference(self):
        with self.assertRaisesRegex(ValueError, "artifact"):
            contracts.OutputReference("preview", truncated=True)

    def test_oversized_details_are_explicit_references_not_dropped(self):
        long = "x" * (contracts.MAX_EVIDENCE_CHARS + 1)
        output = contracts.OutputReference.bounded("x" * 5000, artifact_reference="artifact:output")
        self.assertTrue(output.truncated)
        self.assertEqual(output.artifact_reference, "artifact:output")
        evidence = contracts.bounded_evidence(long, artifact_reference="artifact:evidence")
        self.assertIn("artifact:evidence", evidence)
        failure = contracts.FailureDetail("execution-error", False,
            message=contracts.OutputReference.bounded(long, artifact_reference="artifact:message", limit=2048),
            details=contracts.OutputReference.bounded(long, artifact_reference="artifact:details", limit=2048),
            reset_hint=contracts.bounded_evidence(long, artifact_reference="artifact:reset"))
        result = self.result(terminal_status="failed", failure=failure, output=output,
                             validation_evidence=[evidence])
        self.assertEqual(contracts.WorkerResult.from_dict(result.as_dict()), result)
        with self.assertRaisesRegex(ValueError, "artifact"):
            contracts.bounded_evidence(long)

    def test_artifact_limit_has_valid_control(self):
        self.result(artifacts=("x" * contracts.MAX_ARTIFACT_REFERENCE_CHARS,))
        with self.assertRaisesRegex(ValueError, "artifacts"):
            self.result(artifacts=("x" * (contracts.MAX_ARTIFACT_REFERENCE_CHARS + 1),))

    def test_cancel_terminal_and_unknown_preserves_outcome(self):
        result = self.result()
        adapter = doubles.FakeExecutionAdapter.delayed(result, pending_polls=0)
        handle = adapter.submit(request()).handle
        self.assertEqual(adapter.cancel(handle).status, "unsupported")
        self.assertEqual(adapter.result(handle), result)
        self.assertEqual(adapter.cancel(handle).status, "unsupported")
        unknown = doubles.FakeExecutionAdapter.unknown_liveness(request())
        handle = unknown.submit(request()).handle
        self.assertEqual(unknown.cancel(handle).status, "unknown")
        self.assertEqual(unknown.result(handle).status, "unknown")

    def test_explicit_reconciliation_preserves_handle_and_version(self):
        running = contracts.AttemptLifecycle("wf", "task", "attempt", "running", "handle", plan_version=7)
        unknown = running.transition("unknown")
        with self.assertRaises(ValueError):
            unknown.transition("succeeded")
        for status in ("running", "succeeded", "failed", "cancelled", "timed_out"):
            reconciled = unknown.reconcile(status)
            self.assertEqual((reconciled.handle, reconciled.plan_version), ("handle", 7))
            self.assertEqual(contracts.AttemptLifecycle.from_dict(reconciled.as_dict()), reconciled)
        with self.assertRaises(ValueError):
            running.transition("unknown", handle="other")

    def test_lifecycle_state_handle_consistency(self):
        for status in ("submitted", "running", "unknown", "succeeded", "failed", "cancelled", "timed_out"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                contracts.AttemptLifecycle("wf", "task", "attempt", status)
        for status in ("created", "unavailable", "unsupported"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                contracts.AttemptLifecycle("wf", "task", "attempt", status, "handle")

    def test_direct_identity_and_fact_validation(self):
        for changes in ({"provider": 4}, {"account": "x" * 129}, {"alias": ""},
                        {"requested": "model"}, {"effort": None}, {"schema_version": True}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                replace(IDENTITY, **changes)
        for constructor, kwargs in ((contracts.ModelFact, {"value": []}),
                                    (contracts.ModelFact, {"canonical": 1}),
                                    (contracts.EffortFact, {"applied": "x" * 129})):
            with self.assertRaises((ValueError, TypeError)):
                constructor(**kwargs)

    def test_exact_mismatch_is_not_success_but_verification_is_separate(self):
        observed = replace(IDENTITY, observed=contracts.ModelFact("other", "result"))
        with self.assertRaisesRegex(ValueError, "exact"):
            self.result(observed_target=observed)
        effort = replace(IDENTITY, effort=contracts.EffortFact("high", "low", "wire"))
        with self.assertRaisesRegex(ValueError, "exact"):
            self.result(observed_target=effort)
        adapter = doubles.FakeExecutionAdapter.identity_mismatch(request(), "other")
        outcome = adapter.result(adapter.submit(request()).handle)
        self.assertEqual(outcome.terminal_status, "failed")
        self.assertEqual(outcome.failure.failure_class, "exact-route-mismatch")
        self.assertEqual(self.result(validation_evidence=("verification failed",)).terminal_status, "succeeded")

    def test_a07_exact_worker_result_rejects_known_account_mismatch(self):
        wanted = contracts.TargetIdentity(
            'anthropic', 'account-A', 'hermes_claude', 'sonnet', 'exact',
            requested=contracts.ModelFact('claude-sonnet-5-5', 'request'),
            resolved=contracts.ModelFact('claude-sonnet-5-5', 'config'),
            observed=contracts.ModelFact('claude-sonnet-5-5', 'response'),
        )
        other = replace(wanted, account='account-B')
        with self.assertRaises(ValueError):
            contracts.WorkerResult('wf', 'task', 'attempt', 'handle', 'succeeded', 'done',
                contracts.OutputReference('done'), wanted, other, other)

    def test_a07_exact_worker_result_rejects_known_transport_mismatch(self):
        wanted = self.result().requested_target
        other = replace(wanted, transport='claude_cli')
        with self.assertRaisesRegex(ValueError, 'exact'):
            self.result(resolved_target=other, observed_target=other)

    def test_a07_exact_worker_result_accepts_unknown_account_observation(self):
        wanted = self.result().requested_target
        unknown = replace(wanted, account=contracts.UNKNOWN)
        self.assertEqual(self.result(resolved_target=unknown, observed_target=unknown).terminal_status, 'succeeded')

    def test_a07_unconstrained_request_accepts_known_observed_account(self):
        wanted = self.result().requested_target
        unconstrained = replace(wanted, account=contracts.UNKNOWN)
        observed = replace(wanted, observed=contracts.ModelFact())
        self.assertEqual(self.result(requested_target=unconstrained, resolved_target=observed,
                                     observed_target=observed).terminal_status, 'succeeded')

    def test_a07_unconstrained_request_accepts_known_resolved_account(self):
        wanted = self.result().requested_target
        unconstrained = replace(wanted, account=contracts.UNKNOWN)
        resolved = replace(wanted, observed=contracts.ModelFact())
        observed = replace(unconstrained, observed=contracts.ModelFact())
        self.assertEqual(self.result(requested_target=unconstrained, resolved_target=resolved,
                                     observed_target=observed).terminal_status, 'succeeded')

    def test_a07_exact_worker_result_accepts_matching_account_and_transport(self):
        wanted = self.result().requested_target
        matching = replace(wanted, observed=contracts.ModelFact())
        self.assertEqual(self.result(resolved_target=matching, observed_target=matching).terminal_status, 'succeeded')

    def test_exact_authoritative_request_checks_resolved_model_and_provider(self):
        for changes in ({"resolved": contracts.ModelFact("other", "host")},
                        {"provider": "anthropic"}):
            target = replace(IDENTITY, selection_mode="profile_preferred", **changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "exact"):
                self.result(resolved_target=target)
        preferred = replace(IDENTITY, selection_mode="profile_preferred")
        self.result(requested_target=preferred,
                    observed_target=replace(preferred, observed=contracts.ModelFact("other", "result")))

    def test_boundary_unknown_fields_and_boolean_schema_are_rejected(self):
        records = [contracts.PendingResult("pending"), contracts.Eligibility("yes"),
                   contracts.CancelOutcome("unknown"), contracts.AdapterCapabilities("fake", ("result",))]
        for record in records:
            wire = record.as_dict()
            wire["invented_route"] = "other"
            with self.assertRaises(ValueError):
                type(record).from_dict(wire)
        for record in (request(), self.result(), contracts.Submission.for_acceptance("wf", "task", "attempt", "handle")):
            wire = record.as_dict()
            wire["schema_version"] = True
            with self.assertRaises((ValueError, TypeError)):
                type(record).from_dict(wire)

    def test_failure_status_consistency(self):
        with self.assertRaises(ValueError):
            self.result(failure=contracts.FailureDetail("execution-error", False))
        for status, failure_class in (("failed", "execution-error"), ("cancelled", "cancelled"), ("timed_out", "timeout")):
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.result(terminal_status=status)
            self.result(terminal_status=status, failure=contracts.FailureDetail(failure_class, False))
        with self.assertRaises(ValueError):
            self.result(terminal_status="cancelled", failure=contracts.FailureDetail("timeout", False))

    def test_legacy_requires_authority_and_supports_long_ids(self):
        base = dict(tool_invocation_id="t" * 128, goal="Existing tool", target=IDENTITY,
                    repository="repo", workspace="workspace")
        with self.assertRaises(TypeError):
            contracts.legacy_direct_tool_request(**base)
        explicit = dict(acceptance_criteria=("explicit check",), permissions=("write",),
                        tool_requirements=("terminal",), mutating=True, write_scope=("file",),
                        timeout_seconds=60, deadline_epoch_ms=2000000000000, attempt_budget=1,
                        verification_policy="required", substitution_policy="forbid")
        legacy = contracts.legacy_direct_tool_request(**base, **explicit)
        self.assertTrue(legacy.mutating)
        self.assertEqual(legacy.permissions, ("write",))
        self.assertEqual(legacy.deadline_epoch_ms, explicit["deadline_epoch_ms"])
        self.assertEqual(contracts.legacy_direct_tool_request(**base, **explicit), legacy)
        for value in (legacy.workflow_id, legacy.task_id, legacy.attempt_id):
            self.assertLessEqual(len(value), 128)

    def test_recursive_redaction_is_stored_and_immutable(self):
        metadata = {"nested": [{"api_key": "synthetic-key", "Cookie": "synthetic-cookie",
                                "bearer": "synthetic-bearer", "safe": "kept"}],
                    "authorization": "synthetic-auth", "access_token": "synthetic-token"}
        result = self.result(provider_metadata=metadata)
        metadata["nested"][0]["safe"] = "changed"
        self.assertEqual(result.provider_metadata["nested"][0]["api_key"], "[redacted]")
        wire = result.as_dict()["provider_metadata"]
        self.assertEqual(wire["nested"][0]["safe"], "kept")
        self.assertNotIn("synthetic", json.dumps(wire))
        with self.assertRaises(TypeError):
            result.provider_metadata["nested"][0]["safe"] = "forged"

    def test_json_finite_string_and_aggregate_bounds(self):
        for usage in ({"bad": float("nan")}, {"bad": float("inf")}, {"bad": "x" * 10000},
                      {str(i): list(range(64)) for i in range(64)}):
            with self.subTest(usage_type=type(usage)), self.assertRaises(ValueError):
                self.result(usage=usage)
        with self.assertRaises(TypeError):
            self.result(usage=[])

    def test_versions_changed_files_and_boundary_round_trips(self):
        result = self.result(plan_version=7, changed_files=["file.py"])
        self.assertEqual(result.changed_files, ("file.py",))
        self.assertEqual(contracts.WorkerResult.from_dict(result.as_dict()), result)
        records = [contracts.PendingResult("unknown", "lost"), contracts.Eligibility("unavailable", ["quota"]),
                   contracts.CancelOutcome("unsupported", "no seam"),
                   contracts.AdapterCapabilities("fake", ["result"])]
        for record in records:
            self.assertEqual(type(record).from_dict(record.as_dict()), record)
        for record in (request(), result, IDENTITY, records[-1]):
            with self.assertRaises((ValueError, TypeError)):
                replace(record, schema_version=True)
        lifecycle = contracts.AttemptLifecycle.created("wf", "task", "attempt").as_dict()
        lifecycle["schema_version"] = True
        with self.assertRaises((ValueError, TypeError)):
            contracts.AttemptLifecycle.from_dict(lifecycle)

    def test_fake_unavailable_and_retryability_not_authorized(self):
        adapter = doubles.FakeExecutionAdapter.unavailable("capacity unavailable")
        self.assertEqual(adapter.can_execute(request(), {}).status, "unavailable")
        self.assertFalse(adapter.submit(request()).accepted)
        self.assertFalse(doubles.failed_result(request(), "handle", "provider-transient").failure.retryable)


if __name__ == "__main__":
    unittest.main()
