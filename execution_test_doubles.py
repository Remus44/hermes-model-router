"""Deterministic S04 adapter test doubles; they never invoke providers or tools."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, Tuple

from . import execution_contracts as contracts


class FakeExecutionAdapter:
    """Scripted implementation of the frozen adapter boundary for offline tests."""

    def __init__(self, *, eligibility=None, script=None):
        self._eligibility = eligibility or contracts.Eligibility("yes")
        self._script = dict(script or {})
        self._handles: Dict[str, list] = {}
        self._requests: Dict[str, contracts.ExecutionRequest] = {}
        self.calls = {"capabilities": 0, "can_execute": 0, "submit": 0, "result": 0, "cancel": 0}

    @classmethod
    def delayed(cls, result: contracts.WorkerResult, *, pending_polls: int) -> "FakeExecutionAdapter":
        return cls(script={result.attempt_id: (result.handle, [contracts.PendingResult("pending")] * pending_polls + [result])})

    @classmethod
    def partial_batch(cls, *, succeeded: contracts.WorkerResult, failed: contracts.WorkerResult) -> "FakeExecutionAdapter":
        return cls(script={succeeded.attempt_id: (succeeded.handle, [succeeded]),
                           failed.attempt_id: (failed.handle, [failed])})

    @classmethod
    def capability_refusal(cls, reason: str) -> "FakeExecutionAdapter":
        return cls(eligibility=contracts.Eligibility("unsupported", (reason,)))

    @classmethod
    def identity_mismatch(cls, request: contracts.ExecutionRequest, observed_model: str) -> "FakeExecutionAdapter":
        result = succeeded_result(request, "handle-mismatch")
        observed = replace(result.observed_target,
                           observed=contracts.ModelFact(observed_model, "fake:result", canonical=True))
        return cls(script={request.attempt_id: (result.handle, [replace(result, observed_target=observed)])})

    @classmethod
    def unknown_liveness(cls, request: contracts.ExecutionRequest) -> "FakeExecutionAdapter":
        return cls(script={request.attempt_id: ("handle-unknown", [contracts.PendingResult("unknown", "fake lost observation")])})

    @classmethod
    def cancellable(cls, request: contracts.ExecutionRequest) -> "FakeExecutionAdapter":
        pending = contracts.PendingResult("pending")
        return cls(script={request.attempt_id: ("handle-cancel", [pending])})

    def capabilities(self, runtime_snapshot):
        self.calls["capabilities"] += 1
        return contracts.AdapterCapabilities("fake", ("submission", "result", "cancellation"))

    def can_execute(self, request: contracts.ExecutionRequest, runtime_snapshot) -> contracts.Eligibility:
        self.calls["can_execute"] += 1
        return self._eligibility

    def submit(self, request: contracts.ExecutionRequest) -> contracts.Submission:
        self.calls["submit"] += 1
        if self._eligibility.status != "yes":
            return contracts.Submission(request.workflow_id, request.task_id, request.attempt_id, False,
                                        rejection=contracts.FailureDetail("capability", False,
                                                                          self._eligibility.reasons[0]))
        handle, results = self._script.get(request.attempt_id, (f"handle-{request.attempt_id}",
                                                                  [succeeded_result(request, f"handle-{request.attempt_id}")]))
        self._handles[handle] = list(results)
        self._requests[handle] = request
        return contracts.Submission.for_acceptance(request.workflow_id, request.task_id, request.attempt_id, handle)

    def submit_batch(self, requests: Iterable[contracts.ExecutionRequest]) -> Tuple[contracts.Submission, ...]:
        return tuple(self.submit(request) for request in requests)

    def result(self, handle: str) -> contracts.AdapterResult:
        self.calls["result"] += 1
        entries = self._handles.get(handle)
        if not entries:
            return contracts.PendingResult("unknown", "fake handle not observed")
        outcome = entries.pop(0)
        if not entries:
            entries.append(outcome)
        return outcome

    def cancel(self, handle: str) -> contracts.CancelOutcome:
        self.calls["cancel"] += 1
        if handle not in self._handles:
            return contracts.CancelOutcome("unknown", "fake handle not observed")
        queued = self._handles[handle]
        result = next((item for item in queued if isinstance(item, contracts.WorkerResult)), None)
        if result is None:
            self._handles[handle] = [cancelled_result(self._requests[handle], handle)]
        else:
            self._handles[handle] = [replace(result, terminal_status="cancelled", summary="cancelled", failure=None)]
        return contracts.CancelOutcome("acknowledged")


def succeeded_result(request: contracts.ExecutionRequest, handle: str) -> contracts.WorkerResult:
    return contracts.WorkerResult(request.workflow_id, request.task_id, request.attempt_id, handle, "succeeded", "completed",
                                  contracts.OutputReference("completed"), request.target, request.target, request.target)


def failed_result(request: contracts.ExecutionRequest, handle: str, failure_class: str) -> contracts.WorkerResult:
    return contracts.WorkerResult(request.workflow_id, request.task_id, request.attempt_id, handle, "failed", "failed",
                                  contracts.OutputReference("failed"), request.target, request.target, request.target,
                                  failure=contracts.FailureDetail(failure_class, True, "retry fixture"))


def cancelled_result(request: contracts.ExecutionRequest, handle: str) -> contracts.WorkerResult:
    return contracts.WorkerResult(request.workflow_id, request.task_id, request.attempt_id, handle, "cancelled", "cancelled",
                                  contracts.OutputReference("cancelled"), request.target, request.target, request.target)
