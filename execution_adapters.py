"""Transport boundaries, not a second scheduler or a source of tool authority.

Legacy calls keep their host-owned payload/selection/budgets. Their receipts are
separate from ExecutionRequest: missing authority is never filled with defaults.
Structured requests fail closed where the installed transports cannot enforce the
whole request. S06 owns durable receipts and reconciliation of unknown workers.
"""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from . import execution_contracts as contracts
from . import worker_admission

AttemptKey = Tuple[str, int, str, str]
_MAX_RECORDS = 4096
_MAX_CACHED_RESPONSE = 8192
_NATIVE_INVOCATION: ContextVar[Any] = ContextVar("model_router_legacy_invocation", default=None)


@dataclass(frozen=True)
class LegacyReceipt:
    """Bounded execution evidence only; no invented permissions or observed identity."""
    attempt_key: AttemptKey
    transport: str
    scope: str
    handle: str
    status: str = "unknown"
    child_ids: Tuple[str, ...] = ()
    resolved_tier: str = "unknown"
    adjustment: str = ""
    effort: str = "unknown"
    observed_model: str = "unknown"


@dataclass
class _Claim:
    scope: str
    weight: int
    receipt: LegacyReceipt
    response: Any = None
    cached: bool = False
    fingerprint: str = ""
    sealed: bool = False


class ReservationBook:
    """One locked process-local owner for slot claims and attempt deduplication.

    Unknown/in-flight claims never expire. Terminal receipts also retain their
    attempt keys: reaching the bounded journal limit refuses new work rather than
    silently forgetting a key and permitting duplicate mutation. S06 supplies
    persistence/reconciliation; this class never automatically replays an attempt.
    """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._claims: Dict[Any, _Claim] = {}

    def _begin(self, scope: str, key: AttemptKey, limit: int, weight: int,
               transport: str, fingerprint: str = "") -> Tuple[bool, Optional[_Claim]]:
        with self._lock:
            if not isinstance(key, tuple) or len(key) != 4:
                return False, None  # an attempt ID alone is not a full ownership key
            workflow, plan_version, task, attempt = key
            if (isinstance(plan_version, bool) or not isinstance(plan_version, int) or plan_version < 0
                    or not all(isinstance(value, str) and 0 < len(value) <= 256
                               for value in (workflow, task, attempt))):
                return False, None
            existing = self._claims.get(key)
            if existing is not None:
                if (existing.scope != scope or existing.receipt.transport != transport
                        or existing.fingerprint != fingerprint):
                    return False, None  # changed authority under an already used attempt ID
                return False, existing
            if (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
                    or isinstance(weight, bool) or not isinstance(weight, int) or weight < 1):
                return False, None
            used = sum(c.weight for c in self._claims.values() if c.scope == scope)
            if used + weight > limit or len(self._claims) >= _MAX_RECORDS:
                return False, None
            entry = _Claim(scope, weight, LegacyReceipt(key, transport, scope, "local-" + uuid.uuid4().hex), fingerprint=fingerprint)
            self._claims[key] = entry
            return True, entry

    def claim(self, scope: str, attempt_key: AttemptKey, limit: int) -> bool:
        """True means a *fresh* authorization, never a previously claimed attempt."""
        return self._begin(scope, attempt_key, limit, 1, "hermes_codex")[0]

    def record(self, key: AttemptKey) -> Optional[LegacyReceipt]:
        with self._lock:
            entry = self._claims.get(key)
            return entry.receipt if entry else None

    def records(self) -> Tuple[LegacyReceipt, ...]:
        with self._lock:
            return tuple(entry.receipt for entry in self._claims.values())

    def _tool_response(self, key: AttemptKey, fingerprint: str, transport: str) -> Tuple[bool, Any]:
        with self._lock:
            entry = self._claims.get(key)
            if entry is None:
                return False, None
            if (entry.fingerprint == fingerprint and entry.receipt.transport == transport
                    and entry.sealed and entry.cached):
                return True, entry.response
            return True, json.dumps({"error": "Invocation already started/unknown, response unavailable, or payload changed. Nothing was spawned by this call."})

    def _seal_tool_response(self, key: AttemptKey, raw: Any) -> None:
        # Native raw completion precedes Claude's effort/step-down annotation.
        # Seal only the complete public response so replay neither re-admits nor
        # enters a fresh reasoning scope that could falsely report 'not applied'.
        with self._lock:
            if key not in self._claims:
                return  # established refusal before raw dispatch
        self._finish(key, raw)
        with self._lock:
            self._claims[key].sealed = True

    def _finish(self, key: AttemptKey, raw: Any, *, exceptional: bool = False,
                evidence: Optional[Mapping[str, Any]] = None) -> None:
        with self._lock:
            entry = self._claims[key]
            entry.response, entry.cached = None, False
            entry.receipt = _normalize(entry.receipt, raw, exceptional=exceptional, evidence=evidence)
            if entry.receipt.status in ("succeeded", "failed", "cancelled", "timed_out"):
                entry.weight = 0
            # Cache only bounded legacy responses. Oversized duplicates refuse,
            # never re-dispatch; the original caller still receives the full raw result.
            try:
                size = len(raw) if isinstance(raw, str) else len(json.dumps(raw))
            except (TypeError, ValueError):
                size = _MAX_CACHED_RESPONSE + 1
            if not exceptional and size <= _MAX_CACHED_RESPONSE:
                entry.response, entry.cached = raw, True


RESERVATIONS = ReservationBook()


def _payload(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return {}
    return raw if isinstance(raw, Mapping) else {}


def _bounded(value: Any, limit: int = 256) -> str:
    return value[:limit] if isinstance(value, str) else ""


def _identifier(value: Any) -> str:
    # Reject unretainable identifiers instead of truncating them into a different
    # handle/model. The original legacy response is still returned verbatim.
    return value if isinstance(value, str) and 0 < len(value) <= 256 else ""


def _normalize(receipt: LegacyReceipt, raw: Any, *, exceptional: bool = False,
               evidence: Optional[Mapping[str, Any]] = None) -> LegacyReceipt:
    payload = _payload(raw)
    facts = evidence or payload
    receipt = replace(receipt, resolved_tier=_bounded(facts.get("claude_tier")) or receipt.resolved_tier,
        adjustment=_bounded(facts.get("tier_adjusted"), 512),
        effort=_bounded(facts.get("reasoning_effort")) or receipt.effort)
    if exceptional:
        return receipt
    # Only the validated CLI bridge result supplies served-model evidence.
    if receipt.transport == "claude_cli" and _identifier(payload.get("bridge_run_id")):
        return replace(receipt, handle=_identifier(payload["bridge_run_id"]), status="succeeded",
            observed_model=_identifier(payload.get("effective_model")) or "unknown")
    results = payload.get("results")
    if isinstance(results, list) and results:
        statuses = [r.get("status") if isinstance(r, Mapping) else None for r in results]
        terminal = {"completed", "failed", "error", "cancelled", "timeout"}
        ids = tuple(dict.fromkeys(_identifier(r.get(k)) for r in results if isinstance(r, Mapping)
            for k in ("subagent_id", "session_id") if _identifier(r.get(k))))[:128]
        receipt = replace(receipt, child_ids=ids)
        if all(isinstance(s, str) and s in terminal for s in statuses) and not payload.get("delegation_id"):
            state = "succeeded" if all(s == "completed" for s in statuses) else "failed"
            return replace(receipt, status=state)
    # A handle proves dispatch acceptance, not completion. Malformed responses,
    # top-level errors and partial results may follow a start; keep them reserved.
    handle = next((_identifier(payload.get(k)) for k in ("delegation_id", "subagent_id", "id")
                   if _identifier(payload.get(k))), receipt.handle)
    return replace(receipt, handle=handle)


def dispatch_legacy(raw_dispatch: Callable[[], Any], *, transport: str, scope: str,
                    limit: int, weight: int = 1, attempt_key: Optional[AttemptKey] = None,
                    reservations: Optional[ReservationBook] = None,
                    evidence: Optional[Mapping[str, Any]] = None, fingerprint: str = "") -> Any:
    """Call an explicitly raw seam exactly once and return its public payload unchanged.

    Admission remains with the named legacy owner upstream of this call. An
    observation-only invocation ID is created when the host has no attempt IDs;
    repeated *new* legacy tool calls are not invented to be the same attempt.
    """
    owner = reservations if reservations is not None else RESERVATIONS
    key = attempt_key if attempt_key is not None else ("legacy", 0, transport, uuid.uuid4().hex)
    fresh, entry = owner._begin(scope, key, limit, weight, transport, fingerprint)
    if not fresh:
        if entry is not None and entry.cached:
            return entry.response
        message = "No fresh execution reservation: attempt already started/unknown, capacity unavailable, or receipt journal full. Nothing was spawned by this call."
        if transport == "claude_cli":
            from .claude_opus_bridge import ClaudeBridgeFailure
            raise ClaudeBridgeFailure(message, "concurrency")
        return json.dumps({"error": message})
    try:
        raw = raw_dispatch()
    except BaseException:
        owner._finish(key, None, exceptional=True, evidence=evidence)
        raise
    owner._finish(key, raw, evidence=evidence)
    return raw


def native_legacy_dispatch(raw_dispatch: Callable[[], Any], *, parent: Any, tasks: Any = None,
                           transport: str, evidence: Optional[Mapping[str, Any]] = None) -> Any:
    """No tools/workspaces/routes are invented: the host remains their authority."""
    try:
        from tools.delegate_tool import _get_max_concurrent_children
        limit = _get_max_concurrent_children()
    except Exception:
        limit = 0  # missing host capacity seam cannot authorize a launch
    weight = len(tasks) if isinstance(tasks, list) and tasks else 1
    # Both native transports share this parent's child capacity. No parent field
    # or config mutation is used to simulate a route, a workspace, or isolation.
    scope = f"hermes_children:{getattr(parent, 'session_id', '')}:{id(parent)}"
    invocation = _NATIVE_INVOCATION.get()
    key, fingerprint, started = invocation if invocation is not None else (None, "", None)
    def launch() -> Any:
        if started is not None:
            started[0] = True
        return raw_dispatch()
    return dispatch_legacy(launch, transport=transport, scope=scope, limit=limit,
                           weight=weight, evidence=evidence, attempt_key=key, fingerprint=fingerprint)


def guard_legacy_tool_execution(**kwargs: Any) -> Any:
    """Registered native boundaries; each legacy transport retains one admission owner."""
    args = kwargs.get("args") or {}
    name = str(kwargs.get("tool_name") or "").removeprefix("mcp__")
    if name not in ("delegate_task", "delegate_claude") or args.get("action", "spawn") in ("list", "steer", "stop"):
        return kwargs["next_call"](args)
    raw_next = kwargs["next_call"]
    identifiers = tuple(kwargs.get(k) for k in ("session_id", "turn_id", "tool_call_id"))
    invocation = None
    if all(isinstance(value, str) and value for value in identifiers):
        session, turn, tool_call = identifiers
        key = (session, 0, turn, tool_call)
        try:
            fingerprint = hashlib.sha256(json.dumps(args, sort_keys=True, allow_nan=False).encode()).hexdigest()
        except (TypeError, ValueError):
            return json.dumps({"error": "Legacy invocation is not a JSON-compatible tool payload. Nothing was spawned."})
        invocation = (key, fingerprint, [False])
        transport = "hermes_claude" if name == "delegate_claude" else "hermes_codex"
        exists, response = RESERVATIONS._tool_response(key, fingerprint, transport)
        if exists:
            return response
    token = _NATIVE_INVOCATION.set(invocation)
    try:
        if name == "delegate_claude":
            # Its real handler performs admission, effort scoping and raw native
            # dispatch. Only correlation metadata crosses this outer boundary.
            response = raw_next(args)
        else:
            def admitted(sent: Dict[str, Any]) -> Any:
                from agent.subagent_lifecycle import get_active_subagent_parent
                return native_legacy_dispatch(lambda: raw_next(sent), parent=get_active_subagent_parent(),
                    tasks=sent.get("tasks"), transport="hermes_codex")
            response = worker_admission.guard_tool_execution(**{**kwargs, "next_call": admitted})
        if invocation is not None and invocation[2][0]:
            RESERVATIONS._seal_tool_response(invocation[0], response)
        return response
    finally:
        _NATIVE_INVOCATION.reset(token)


def _cap(snapshot: Any, transport: str, name: str) -> Any:
    return snapshot.adapter(transport).capability(name)


class _Adapter:
    transport = ""
    def __init__(self, *, cfg: Optional[Dict[str, Any]] = None, runtime_snapshot: Any = None,
                 admission: Callable[..., str] = worker_admission.refusal,
                 reservations: Optional[ReservationBook] = None) -> None:
        self._cfg = cfg if isinstance(cfg, dict) else {}
        self._runtime_snapshot = runtime_snapshot
        self._admission = admission
        self._reservations = reservations if reservations is not None else RESERVATIONS

    def capabilities(self, runtime_snapshot: Any) -> contracts.AdapterCapabilities:
        # Host seams are not adapter methods. No structured submit cell is fully
        # enforceable yet; result lookup and cancellation are likewise not implemented.
        return contracts.AdapterCapabilities(self.transport, ())

    def _constraint_reasons(self, request: contracts.ExecutionRequest, snapshot: Any) -> Tuple[str, ...]:
        reasons = []
        if request.target.transport != self.transport:
            reasons.append("request transport does not match adapter")
        submission = _cap(snapshot, self.transport, "submission")
        if submission.status != "supported":
            reasons.append(f"submission seam is {submission.status}: {submission.reason}")
        # These mandatory S04 fields have no proven per-call enforcement seams.
        # Prompting a worker to comply is not permission, workspace or budget enforcement.
        reasons.extend(("permissions/tool requirements cannot be enforced by this legacy transport",
            "repository/workspace selection or confinement is not verified for this request",
            "context reference has no trusted content/revision resolver; acceptance criteria cannot replace it",
            "per-attempt timeout/deadline and attempt budget lack an enforcing seam"))
        if request.write_scope:
            reasons.append("write_scope cannot be enforced by the legacy tool permission contract")
        if request.target.account:
            reasons.append("account credential binding is not verified for the authoritative request")
        if request.target.effort.requested not in ("unknown", "not_applicable"):
            effort = _cap(snapshot, self.transport, "effort_application")
            reasons.append(f"requested effort has no proven per-request application/observation seam (host={effort.status})")
        if request.substitution_policy == "forbid":
            reasons.append("forbidden substitution cannot be guaranteed by legacy routing/admission")
        if request.target.selection_mode == "exact":
            reasons.append("exact provider/account/model/effort request is not fully enforceable and observable")
        return tuple(reasons)

    def can_execute(self, request: contracts.ExecutionRequest, runtime_snapshot: Any) -> contracts.Eligibility:
        return contracts.Eligibility("unsupported", self._constraint_reasons(request, runtime_snapshot))

    def submit(self, request: contracts.ExecutionRequest) -> contracts.Submission:
        if self._runtime_snapshot is None:
            reasons = ("adapter requires a runtime capability snapshot",)
        else:
            reasons = self.can_execute(request, self._runtime_snapshot).reasons
        return contracts.Submission(request.workflow_id, request.task_id, request.attempt_id, False,
            rejection=contracts.FailureDetail("capability", False,
                message=contracts.OutputReference("; ".join(reasons))))

    def result(self, handle: str) -> contracts.AdapterResult:
        return contracts.PendingResult("unknown", "no structured execution was dispatched; legacy receipts are separate")

    def cancel(self, handle: str) -> contracts.CancelOutcome:
        return contracts.CancelOutcome("unsupported", "S05 does not implement cancellation/reconciliation")


class HermesCodexAdapter(_Adapter):
    transport = "hermes_codex"
    def __init__(self, delegate_task: Callable[..., Any], *, parent_agent: Callable[[], Any] = lambda: None,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._delegate_task, self._parent_agent = delegate_task, parent_agent


class HermesClaudeAdapter(_Adapter):
    transport = "hermes_claude"
    _TIER_FOR_ALIAS = {"haiku": "haiku", "sonnet5": "sonnet", "opus5": "opus", "sonnet": "sonnet", "opus": "opus"}
    def __init__(self, delegate_claude: Callable[..., Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._delegate_claude = delegate_claude

    def applied_effort(self, request: contracts.ExecutionRequest) -> contracts.EffortFact:
        haiku = self._TIER_FOR_ALIAS.get(request.target.alias) == "haiku"
        return contracts.EffortFact(request.target.effort.requested, "not_applicable" if haiku else "unknown",
            "haiku_no_thinking" if haiku else "not_observed")

    def submit_batch(self, requests: Iterable[contracts.ExecutionRequest]) -> Tuple[contracts.Submission, ...]:
        return tuple(self.submit(request) for request in requests)


class ClaudeCliAdapter(_Adapter):
    transport = "claude_cli"
    def __init__(self, bridge: Callable[..., Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._bridge = bridge


def legacy_adapters(*, delegate_task: Callable[..., Any], delegate_claude: Callable[..., Any],
                    claude_bridge: Callable[..., Any], cfg: Optional[Dict[str, Any]] = None,
                    runtime_snapshot: Any = None, admission: Callable[..., str] = worker_admission.refusal,
                    parent_agent: Callable[[], Any] = lambda: None,
                    reservations: Optional[ReservationBook] = None) -> Dict[str, _Adapter]:
    """Compatibility factory; actual legacy entrypoints use dispatch_legacy directly."""
    kwargs = {"cfg": cfg, "runtime_snapshot": runtime_snapshot, "admission": admission,
              "reservations": reservations if reservations is not None else RESERVATIONS}
    return {"hermes_codex": HermesCodexAdapter(delegate_task, parent_agent=parent_agent, **kwargs),
            "hermes_claude": HermesClaudeAdapter(delegate_claude, **kwargs),
            "claude_cli": ClaudeCliAdapter(claude_bridge, **kwargs)}
