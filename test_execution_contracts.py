"""S04 execution-record and adapter-double contract tests (I03-I10)."""
import json
import unittest

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
                terminal_status="failed", summary="failed", output=contracts.OutputReference("x"),
                requested_target=IDENTITY, resolved_target=IDENTITY, observed_target=IDENTITY,
                artifacts=("artifact:" + "x" * contracts.MAX_ARTIFACT_REFERENCE_CHARS,),
            )
        result = contracts.WorkerResult(
            workflow_id="wf-001", task_id="task-001", attempt_id="attempt-001", handle="handle-001",
            terminal_status="failed", summary="failed", output=contracts.OutputReference("x", truncated=True),
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


if __name__ == "__main__":
    unittest.main()
