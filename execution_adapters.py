"""S05 wrappers for the existing Hermes/Codex, Hermes/Claude and Claude CLI seams.

These adapters translate the frozen S04 records into the narrow, already-supported
transport calls.  They deliberately do not alter global delegation configuration,
construct agents, or infer an observed identity from a configured target.
"""
from __future__ import annotations

import json
import threading
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from . import execution_contracts as contracts
from . import worker_admission


def _cap(snapshot: Any, transport: str, name: str) -> Any:
    return snapshot.adapter(transport).capability(name)


def _known_handle(raw: Any, *keys: str) -> str:
    payload: Any = raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            payload = None
    if not isinstance(payload, Mapping):
        return ""
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _transport_error(raw: Any) -> str:
    """Return an existing transport refusal verbatim without treating it as a handle."""
    payload: Any = raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return ""
    if not isinstance(payload, Mapping):
        return ""
    error = payload.get("error")
    return str(error) if isinstance(error, str) and error else ""


def _rejection(request: contracts.ExecutionRequest, failure_class: str, reason: str) -> contracts.Submission:
    return contracts.Submission(request.workflow_id, request.task_id, request.attempt_id, False,
                                rejection=contracts.FailureDetail(failure_class, False,
                                                                  message=contracts.OutputReference(reason)))


def _unknown_target(target: contracts.TargetIdentity) -> contracts.TargetIdentity:
    return replace(target, observed=contracts.ModelFact(),
                   effort=contracts.EffortFact(target.effort.requested, "unknown", "not_observed"))


def _failed_result(request: contracts.ExecutionRequest, handle: str, failure_class: str, reason: str,
                   observed: Optional[contracts.TargetIdentity] = None) -> contracts.WorkerResult:
    target = observed or _unknown_target(request.target)
    return contracts.WorkerResult(
        request.workflow_id, request.task_id, request.attempt_id, handle, "failed", reason,
        contracts.OutputReference(reason), request.target, request.target, target,
        workspace=request.workspace, plan_version=request.plan_version,
        failure=contracts.FailureDetail(failure_class, False, message=contracts.OutputReference(reason)),
    )


class ReservationBook:
    """Atomic local claims for a host concurrency slot until S06 observes completion."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._claims: Dict[str, set[str]] = {}

    def claim(self, scope: str, attempt_id: str, limit: int) -> bool:
        with self._lock:
            attempts = self._claims.setdefault(scope, set())
            if attempt_id in attempts:
                return True
            if len(attempts) >= limit:
                return False
            attempts.add(attempt_id)
            return True

    def release(self, scope: str, attempt_id: str) -> None:
        with self._lock:
            attempts = self._claims.get(scope)
            if attempts is None:
                return
            attempts.discard(attempt_id)
            if not attempts:
                self._claims.pop(scope, None)


class _Adapter:
    transport = ""

    def __init__(self, *, cfg: Optional[Dict[str, Any]] = None, runtime_snapshot: Any = None,
                 admission: Callable[..., str] = worker_admission.refusal,
                 reservations: Optional[ReservationBook] = None) -> None:
        self._cfg = cfg if isinstance(cfg, dict) else {}
        self._runtime_snapshot = runtime_snapshot
        self._admission = admission
        self._reservations = reservations
        self._pending: Dict[str, contracts.AdapterResult] = {}

    def capabilities(self, runtime_snapshot: Any) -> contracts.AdapterCapabilities:
        names = tuple(cap.name for cap in runtime_snapshot.adapter(self.transport).capabilities
                      if cap.status == "supported")
        return contracts.AdapterCapabilities(self.transport, names)

    def _snapshot(self) -> Any:
        if self._runtime_snapshot is None:
            raise RuntimeError("adapter requires a runtime capability snapshot")
        return self._runtime_snapshot

    def _base_eligibility(self, request: contracts.ExecutionRequest, snapshot: Any) -> Optional[contracts.Eligibility]:
        if request.target.transport != self.transport:
            return contracts.Eligibility("unsupported", (f"request transport is {request.target.transport}, not {self.transport}",))
        submission = _cap(snapshot, self.transport, "submission")
        if submission.status != "supported":
            return contracts.Eligibility("unsupported", (f"submission is {submission.status}: {submission.reason}",))
        try:
            refusal = self._admission(request.target.provider, request.target.resolved.value, self._cfg, blocking=True)
        except TypeError:
            refusal = self._admission(request.target.provider, request.target.resolved.value, self._cfg)
        if refusal:
            return contracts.Eligibility("unavailable", (str(refusal),))
        return None

    def result(self, handle: str) -> contracts.AdapterResult:
        return self._pending.get(handle, contracts.PendingResult("unknown", "adapter has no observed handle"))

    def _reservation_scope(self, request: contracts.ExecutionRequest) -> str:
        if self.transport in ("hermes_codex", "hermes_claude"):
            return "hermes_children"
        return f"{self.transport}:{request.target.provider}:{request.target.account}"

    def _reserve(self, request: contracts.ExecutionRequest, snapshot: Any) -> bool:
        """Claim the host slot before dispatch; S06 owns eventual completion release."""
        if self._reservations is None:
            return True
        limit = snapshot.max_concurrent_children
        if limit.status != "supported" or not isinstance(limit.value, int) or limit.value < 1:
            return False
        return self._reservations.claim(self._reservation_scope(request), request.attempt_id, limit.value)

    def _release(self, request: contracts.ExecutionRequest) -> None:
        if self._reservations is not None:
            self._reservations.release(self._reservation_scope(request), request.attempt_id)


class HermesCodexAdapter(_Adapter):
    """Wrap the host's supported ``delegate_task(tasks=...)`` route.

    The installed host's batch schema does not establish per-call model, provider,
    effort or completion identity.  Exact requests are therefore refused rather
    than simulated through host/global configuration mutation.
    """
    transport = "hermes_codex"

    _GOAL_PREFIX_TARGETS = ("luna", "spark", "terra", "sol")

    def __init__(self, delegate_task: Callable[..., Any], *, cfg: Optional[Dict[str, Any]] = None,
                 runtime_snapshot: Any = None, admission: Callable[..., str] = worker_admission.refusal,
                 parent_agent: Callable[[], Any] = lambda: None,
                 reservations: Optional[ReservationBook] = None) -> None:
        super().__init__(cfg=cfg, runtime_snapshot=runtime_snapshot, admission=admission,
                         reservations=reservations)
        self._delegate_task = delegate_task
        self._parent_agent = parent_agent

    def can_execute(self, request: contracts.ExecutionRequest, runtime_snapshot: Any) -> contracts.Eligibility:
        base = self._base_eligibility(request, runtime_snapshot)
        if base is not None:
            return base
        if request.target.alias not in self._GOAL_PREFIX_TARGETS:
            return contracts.Eligibility("unsupported", (
                f"Hermes/Codex host selection supports only goal-prefix targets: {', '.join(self._GOAL_PREFIX_TARGETS)}",))
        if request.mutating and request.target.alias in ("luna", "spark"):
            return contracts.Eligibility("unsupported", (
                f"{request.target.alias} is not an eligible mutating Hermes/Codex worker target",))
        if request.target.selection_mode == "exact":
            model = _cap(runtime_snapshot, self.transport, "model_parameter")
            observed = _cap(runtime_snapshot, self.transport, "identity_observability")
            return contracts.Eligibility("unsupported", (
                "exact Hermes/Codex selection requires supported per-call model and served-identity seams; "
                f"model_parameter={model.status}, identity_observability={observed.status}",
            ))
        if request.target.effort.requested not in ("unknown", "not_applicable"):
            return contracts.Eligibility("unsupported", (
                "Hermes/Codex effort is not a supported per-call delegate_task field on this host",))
        return contracts.Eligibility("yes")

    def submit(self, request: contracts.ExecutionRequest) -> contracts.Submission:
        snapshot = self._snapshot()
        eligibility = self.can_execute(request, snapshot)
        if eligibility.status != "yes":
            return _rejection(request, "capability" if eligibility.status == "unsupported" else "concurrency",
                              "; ".join(eligibility.reasons))
        parent = self._parent_agent()
        if parent is None:
            return _rejection(request, "capability", "delegate_task requires an active Hermes parent")
        if not self._reserve(request, snapshot):
            return _rejection(request, "concurrency", "no atomic Hermes child slot is available")
        try:
            raw = self._delegate_task(
                tasks=[{"goal": f"[{request.target.alias}] {request.goal}", "context": request.context_reference}],
                parent_agent=parent,
                background=not getattr(parent, "_delegate_depth", 0) > 0,
            )
        except Exception as exc:
            self._release(request)
            return _rejection(request, "execution-error", f"delegate_task failed: {type(exc).__name__}: {exc}")
        error = _transport_error(raw)
        if error:
            self._release(request)
            return _rejection(request, "execution-error", error)
        handle = _known_handle(raw, "delegation_id", "subagent_id", "id")
        if not handle:
            self._release(request)
            return _rejection(request, "execution-error", "delegate_task accepted no observable delegation handle")
        self._pending[handle] = contracts.PendingResult("unknown", "host completion correlation is owned by S06")
        return contracts.Submission.for_acceptance(request.workflow_id, request.task_id, request.attempt_id, handle)

    def cancel(self, handle: str) -> contracts.CancelOutcome:
        return contracts.CancelOutcome("unsupported", "legacy delegate_task stop ownership is not wrapped by S05")


class HermesClaudeAdapter(_Adapter):
    """Wrap ``delegate_claude`` while keeping its per-call credentials/effort scope."""
    transport = "hermes_claude"
    _TIER_FOR_ALIAS = {"haiku": "haiku", "sonnet5": "sonnet", "opus5": "opus", "sonnet": "sonnet", "opus": "opus"}

    def __init__(self, delegate_claude: Callable[[Dict[str, Any]], Any], *, cfg: Optional[Dict[str, Any]] = None,
                 runtime_snapshot: Any = None, admission: Callable[..., str] = worker_admission.refusal,
                 reservations: Optional[ReservationBook] = None) -> None:
        super().__init__(cfg=cfg, runtime_snapshot=runtime_snapshot, admission=admission,
                         reservations=reservations)
        self._delegate_claude = delegate_claude

    def _tier(self, request: contracts.ExecutionRequest) -> str:
        return self._TIER_FOR_ALIAS.get(request.target.alias, "")

    def applied_effort(self, request: contracts.ExecutionRequest) -> contracts.EffortFact:
        return contracts.EffortFact(request.target.effort.requested,
                                    "not_applicable" if self._tier(request) == "haiku" else "unknown",
                                    "haiku_no_thinking" if self._tier(request) == "haiku" else "not_observed")

    def can_execute(self, request: contracts.ExecutionRequest, runtime_snapshot: Any) -> contracts.Eligibility:
        base = self._base_eligibility(request, runtime_snapshot)
        if base is not None:
            return base
        tier = self._tier(request)
        if not tier:
            return contracts.Eligibility("unsupported", (f"no delegate_claude tier for {request.target.alias!r}",))
        if tier == "haiku":
            if request.target.effort.requested not in ("unknown", "not_applicable"):
                return contracts.Eligibility("unsupported", ("Haiku has no reasoning-effort transport seam",))
        else:
            effort = _cap(runtime_snapshot, self.transport, "effort_application")
            if effort.status != "supported":
                return contracts.Eligibility("unsupported", (
                    f"Claude reasoning-effort seam is {effort.status}: {effort.reason}",))
            if request.target.selection_mode == "exact" and request.target.effort.requested not in ("unknown", "not_applicable"):
                return contracts.Eligibility("unsupported", (
                    "exact Claude effort is not observable from the legacy delegate_claude response",))
        if request.target.selection_mode == "exact":
            observed = _cap(runtime_snapshot, self.transport, "identity_observability")
            return contracts.Eligibility("unsupported", (
                f"exact Hermes/Claude selection needs served identity evidence; identity_observability={observed.status}",))
        return contracts.Eligibility("yes")

    def submit(self, request: contracts.ExecutionRequest) -> contracts.Submission:
        snapshot = self._snapshot()
        eligibility = self.can_execute(request, snapshot)
        if eligibility.status != "yes":
            return _rejection(request, "capability" if eligibility.status == "unsupported" else "concurrency",
                              "; ".join(eligibility.reasons))
        if not self._reserve(request, snapshot):
            return _rejection(request, "concurrency", "no atomic Hermes child slot is available")
        try:
            raw = self._delegate_claude({"tasks": [{"goal": request.goal, "context": request.context_reference}],
                                         "tier": self._tier(request)})
        except Exception as exc:
            self._release(request)
            return _rejection(request, "execution-error", f"delegate_claude failed: {type(exc).__name__}: {exc}")
        error = _transport_error(raw)
        if error:
            self._release(request)
            return _rejection(request, "execution-error", error)
        handle = _known_handle(raw, "delegation_id", "subagent_id", "id")
        if not handle:
            self._release(request)
            return _rejection(request, "execution-error", "delegate_claude accepted no observable delegation handle")
        self._pending[handle] = contracts.PendingResult("unknown", "host completion correlation is owned by S06")
        return contracts.Submission.for_acceptance(request.workflow_id, request.task_id, request.attempt_id, handle)

    def submit_batch(self, requests: Iterable[contracts.ExecutionRequest]) -> Tuple[contracts.Submission, ...]:
        # Mixed tiers/efforts become individual calls; delegate_claude accepts one tier per call.
        return tuple(self.submit(request) for request in requests)

    def cancel(self, handle: str) -> contracts.CancelOutcome:
        return contracts.CancelOutcome("unsupported", "delegate_claude cancellation seam is not observed by S05")


class ClaudeCliAdapter(_Adapter):
    """Wrap the existing synchronous Claude CLI bridge with result-model evidence."""
    transport = "claude_cli"
    _TIER_FOR_ALIAS = {"opus": "opus", "opus5": "opus", "sonnet": "sonnet", "sonnet5": "sonnet"}

    def __init__(self, bridge: Callable[..., Mapping[str, Any]], *, cfg: Optional[Dict[str, Any]] = None,
                 runtime_snapshot: Any = None, admission: Callable[..., str] = worker_admission.refusal,
                 reservations: Optional[ReservationBook] = None) -> None:
        super().__init__(cfg=cfg, runtime_snapshot=runtime_snapshot, admission=admission,
                         reservations=reservations)
        self._bridge = bridge

    def can_execute(self, request: contracts.ExecutionRequest, runtime_snapshot: Any) -> contracts.Eligibility:
        base = self._base_eligibility(request, runtime_snapshot)
        if base is not None:
            return base
        tier = self._TIER_FOR_ALIAS.get(request.target.alias)
        if not tier:
            return contracts.Eligibility("unsupported", (f"no Claude CLI tier for {request.target.alias!r}",))
        if request.target.selection_mode == "exact":
            exact = _cap(runtime_snapshot, self.transport, "exact_model")
            observed = _cap(runtime_snapshot, self.transport, "identity_observability")
            if exact.status != "supported" or observed.status != "supported":
                return contracts.Eligibility("unsupported", (
                    f"exact CLI route needs exact_model and identity_observability evidence; "
                    f"exact_model={exact.status}, identity_observability={observed.status}",))
        return contracts.Eligibility("yes")

    def submit(self, request: contracts.ExecutionRequest) -> contracts.Submission:
        eligibility = self.can_execute(request, self._snapshot())
        if eligibility.status != "yes":
            return _rejection(request, "capability" if eligibility.status == "unsupported" else "concurrency",
                              "; ".join(eligibility.reasons))
        tier = self._TIER_FOR_ALIAS[request.target.alias]
        try:
            payload = self._bridge(repo=request.repository.removeprefix("repo:"), task=request.goal,
                                   write=request.mutating, review=not request.mutating, model=tier,
                                   requested_alias=tier, cfg=self._cfg)
        except Exception as exc:
            return _rejection(request, getattr(exc, "failure_class", "execution-error"), str(exc))
        handle = _known_handle(payload, "bridge_run_id")
        if not handle:
            return _rejection(request, "execution-error", "Claude CLI bridge returned no observable bridge_run_id")
        model = str(payload.get("effective_model") or "") if isinstance(payload, Mapping) else ""
        observed = replace(request.target, observed=contracts.ModelFact(
            model or "unknown", "claude_cli.result.effective_model" if model else "not_observed", canonical=bool(model)),
            effort=contracts.EffortFact(request.target.effort.requested, "unknown", "not_observed"))
        if request.target.selection_mode == "exact" and (not model or model != request.target.requested.value):
            result = _failed_result(request, handle, "exact-route-mismatch",
                                    "Claude CLI served-model evidence did not match the exact request", observed)
        else:
            result = contracts.WorkerResult(
                request.workflow_id, request.task_id, request.attempt_id, handle, "succeeded", "completed",
                contracts.OutputReference(str(payload.get("result") or "")), request.target, request.target, observed,
                workspace=request.workspace, plan_version=request.plan_version,
            )
        self._pending[handle] = result
        return contracts.Submission.for_acceptance(request.workflow_id, request.task_id, request.attempt_id, handle)

    def cancel(self, handle: str) -> contracts.CancelOutcome:
        return contracts.CancelOutcome("unsupported", "synchronous Claude CLI process cancellation is not wrapped by S05")


def legacy_adapters(*, delegate_task: Callable[..., Any], delegate_claude: Callable[[Dict[str, Any]], Any],
                    claude_bridge: Callable[..., Mapping[str, Any]], cfg: Optional[Dict[str, Any]] = None,
                    runtime_snapshot: Any = None,
                    admission: Callable[..., str] = worker_admission.refusal,
                    parent_agent: Callable[[], Any] = lambda: None,
                    reservations: Optional[ReservationBook] = None) -> Dict[str, _Adapter]:
    """Expose existing legacy transports through the adapter boundary without interception.

    Callers keep ownership of their current entrypoints; this factory only passes the
    original callable through, so it cannot recurse through a tool wrapper or perform
    a second admission/dispatch.
    """
    kwargs = {"cfg": cfg, "runtime_snapshot": runtime_snapshot, "admission": admission,
              "reservations": reservations}
    return {"hermes_codex": HermesCodexAdapter(delegate_task, parent_agent=parent_agent, **kwargs),
            "hermes_claude": HermesClaudeAdapter(delegate_claude, **kwargs),
            "claude_cli": ClaudeCliAdapter(claude_bridge, **kwargs)}
