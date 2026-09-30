"""Versioned execution contracts.

Pure, JSON-compatible records only: no config reads, model calls, subprocesses or
scheduler behavior.  S02 target-identity shapes remain compatible; S04 adds the
request, lifecycle, submission, result, and adapter-boundary contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

SCHEMA_VERSION = 1
TRANSPORTS = ("hermes_codex", "hermes_claude", "claude_cli")
SELECTION_EXACT, SELECTION_PREFERRED = "exact", "profile_preferred"
SELECTION_MODES = (SELECTION_EXACT, SELECTION_PREFERRED)
UNKNOWN = "unknown"
NOT_APPLICABLE = "not_applicable"
SOURCE_NOT_OBSERVED = "not_observed"
FAILURE_EXACT_ROUTE_MISMATCH = "exact-route-mismatch"
FAILURE_CAPABILITY = "capability"

MAX_IDENTIFIER_CHARS = 128
MAX_GOAL_CHARS = 4096
MAX_ACCEPTANCE_CRITERIA = 32
MAX_CRITERION_CHARS = 1024
MAX_OUTPUT_SUMMARY_CHARS = 4096
MAX_ARTIFACT_REFERENCE_CHARS = 1024
MAX_EVIDENCE_ITEMS = 64
MAX_EVIDENCE_CHARS = 2048
MAX_USAGE_FIELDS = 64
MAX_METADATA_FIELDS = 32
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_TERMINAL = ("succeeded", "failed", "cancelled", "timed_out")
_LIFECYCLE = ("created", "submitted", "running", "succeeded", "failed", "cancelled", "timed_out", "unavailable", "unsupported", "unknown")
_ALLOWED_TRANSITIONS = {
    "created": ("submitted", "unavailable", "unsupported"),
    "submitted": ("running", "unknown"),
    "running": ("succeeded", "failed", "cancelled", "timed_out", "unknown"),
    "unknown": (), "succeeded": (), "failed": (), "cancelled": (), "timed_out": (),
    "unavailable": (), "unsupported": (),
}
_FAILURE_CLASSES = (
    "capability", "authentication", "exact-route-mismatch", "rate-limit-transient",
    "quota-exhausted", "concurrency", "provider-transient", "invalid-request", "timeout",
    "cancelled", "execution-error", "verification-failed", "partial-mutation", "unknown",
)


def _string(name: str, value: Any, limit: int, *, identifier: bool = False, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if (not value and not allow_empty) or len(value) > limit:
        raise ValueError(f"{name} must be non-empty and at most {limit} characters")
    if identifier and not _ID.fullmatch(value):
        raise ValueError(f"{name} must be a bounded identifier")
    return value


def _integer(name: str, value: Any, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _string_tuple(name: str, value: Any, limit: int, item_limit: int) -> Tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a sequence")
    if len(value) > limit:
        raise ValueError(f"{name} has too many entries")
    return tuple(_string(name, item, item_limit) for item in value)


def _exact_keys(data: Mapping[str, Any], allowed: Sequence[str], record: str) -> None:
    if not isinstance(data, Mapping):
        raise TypeError(f"{record} must be an object")
    unknown = set(data) - set(allowed)
    missing = set(allowed) - set(data)
    if unknown:
        raise ValueError(f"{record} has unknown fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"{record} is missing fields: {', '.join(sorted(missing))}")


def _json_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        raise ValueError("JSON metadata is too deeply nested")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_USAGE_FIELDS:
            raise ValueError("JSON object has too many fields")
        return { _string("JSON key", key, MAX_IDENTIFIER_CHARS): _json_value(item, depth=depth + 1)
                 for key, item in value.items() }
    if isinstance(value, (tuple, list)):
        if len(value) > MAX_USAGE_FIELDS:
            raise ValueError("JSON sequence has too many items")
        return [_json_value(item, depth=depth + 1) for item in value]
    raise TypeError("value must be JSON-compatible")


def _redact(value: Mapping[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, item in value.items():
        key = _string("metadata key", key, MAX_IDENTIFIER_CHARS)
        if any(word in key.lower() for word in ("token", "secret", "password", "authorization", "credential")):
            result[key] = "[redacted]"
        else:
            result[key] = _json_value(item)
    return result


@dataclass(frozen=True)
class ModelFact:
    value: str = UNKNOWN
    source: str = SOURCE_NOT_OBSERVED
    canonical: bool = True

    @property
    def known(self) -> bool:
        return bool(self.value) and self.value != UNKNOWN

    def as_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "source": self.source, "canonical": self.canonical}


@dataclass(frozen=True)
class EffortFact:
    requested: str = NOT_APPLICABLE
    applied: str = NOT_APPLICABLE
    source: str = SOURCE_NOT_OBSERVED

    def as_dict(self) -> Dict[str, Any]:
        return {"requested": self.requested, "applied": self.applied, "source": self.source}


@dataclass(frozen=True)
class TargetIdentity:
    provider: str
    account: str
    transport: str
    alias: str
    selection_mode: str
    requested: ModelFact = field(default_factory=ModelFact)
    resolved: ModelFact = field(default_factory=ModelFact)
    observed: ModelFact = field(default_factory=ModelFact)
    effort: EffortFact = field(default_factory=EffortFact)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.transport not in TRANSPORTS:
            raise ValueError(f"unknown transport {self.transport!r}")
        if self.selection_mode not in SELECTION_MODES:
            raise ValueError(f"unknown selection mode {self.selection_mode!r}")

    def as_dict(self) -> Dict[str, Any]:
        return {"schema_version": self.schema_version, "provider": self.provider, "account": self.account,
                "transport": self.transport, "alias": self.alias, "selection_mode": self.selection_mode,
                "requested": self.requested.as_dict(), "resolved": self.resolved.as_dict(),
                "observed": self.observed.as_dict(), "effort": self.effort.as_dict()}


@dataclass(frozen=True)
class ExactRouteMismatch:
    stage: str
    alias: str
    transport: str
    requested: str
    requested_source: str
    actual: str
    actual_source: str

    @property
    def failure_class(self) -> str:
        return FAILURE_EXACT_ROUTE_MISMATCH

    def as_dict(self) -> Dict[str, Any]:
        return {"failure_class": self.failure_class, "stage": self.stage, "alias": self.alias,
                "transport": self.transport, "requested": self.requested,
                "requested_source": self.requested_source, "actual": self.actual,
                "actual_source": self.actual_source}


def check_exact(identity: TargetIdentity) -> Optional[ExactRouteMismatch]:
    if identity.selection_mode != SELECTION_EXACT or not identity.requested.known:
        return None
    wanted = identity.requested
    if identity.observed.known and identity.observed.canonical and identity.observed.value != wanted.value:
        actual, stage = identity.observed, "observed"
    elif identity.resolved.known and identity.resolved.canonical and identity.resolved.value != wanted.value:
        actual, stage = identity.resolved, "resolved"
    else:
        return None
    return ExactRouteMismatch(stage, identity.alias, identity.transport, wanted.value, wanted.source,
                              actual.value, actual.source)


def _fact(data: Any) -> ModelFact:
    _exact_keys(data, ("value", "source", "canonical"), "model fact")
    if not isinstance(data["canonical"], bool):
        raise TypeError("model fact canonical must be a boolean")
    return ModelFact(_string("model fact value", data["value"], MAX_IDENTIFIER_CHARS),
                     _string("model fact source", data["source"], MAX_IDENTIFIER_CHARS), data["canonical"])


def _effort(data: Any) -> EffortFact:
    _exact_keys(data, ("requested", "applied", "source"), "effort fact")
    return EffortFact(_string("effort requested", data["requested"], MAX_IDENTIFIER_CHARS),
                      _string("effort applied", data["applied"], MAX_IDENTIFIER_CHARS),
                      _string("effort source", data["source"], MAX_IDENTIFIER_CHARS))


def _identity(data: Any) -> TargetIdentity:
    keys = ("schema_version", "provider", "account", "transport", "alias", "selection_mode",
            "requested", "resolved", "observed", "effort")
    _exact_keys(data, keys, "target identity")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported target identity schema version")
    return TargetIdentity(_string("provider", data["provider"], MAX_IDENTIFIER_CHARS),
                          _string("account", data["account"], MAX_IDENTIFIER_CHARS), data["transport"],
                          _string("alias", data["alias"], MAX_IDENTIFIER_CHARS), data["selection_mode"],
                          _fact(data["requested"]), _fact(data["resolved"]), _fact(data["observed"]),
                          _effort(data["effort"]), data["schema_version"])


@dataclass(frozen=True)
class ExecutionRequest:
    workflow_id: str
    plan_version: int
    task_id: str
    attempt_id: str
    goal: str
    acceptance_criteria: Tuple[str, ...]
    context_reference: str
    target: TargetIdentity
    repository: str
    workspace: str
    permissions: Tuple[str, ...]
    tool_requirements: Tuple[str, ...]
    mutating: bool
    write_scope: Tuple[str, ...]
    timeout_seconds: int
    deadline_epoch_ms: int
    attempt_budget: int
    verification_policy: str
    substitution_policy: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("workflow_id", "task_id", "attempt_id"):
            _string(name, getattr(self, name), MAX_IDENTIFIER_CHARS, identifier=True)
        _integer("plan_version", self.plan_version, 0)
        _string("goal", self.goal, MAX_GOAL_CHARS)
        criteria = _string_tuple("acceptance_criteria", self.acceptance_criteria,
                                 MAX_ACCEPTANCE_CRITERIA, MAX_CRITERION_CHARS)
        if not criteria:
            raise ValueError("acceptance_criteria must not be empty")
        for name in ("context_reference", "repository", "workspace", "verification_policy", "substitution_policy"):
            _string(name, getattr(self, name), MAX_ARTIFACT_REFERENCE_CHARS)
        if not isinstance(self.target, TargetIdentity):
            raise TypeError("target must be a TargetIdentity")
        for name in ("permissions", "tool_requirements", "write_scope"):
            _string_tuple(name, getattr(self, name), MAX_EVIDENCE_ITEMS, MAX_ARTIFACT_REFERENCE_CHARS)
        if not isinstance(self.mutating, bool):
            raise TypeError("mutating must be a boolean")
        _integer("timeout_seconds", self.timeout_seconds, 1)
        _integer("deadline_epoch_ms", self.deadline_epoch_ms, 1)
        _integer("attempt_budget", self.attempt_budget, 1)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported execution request schema version")

    def as_dict(self) -> Dict[str, Any]:
        return {"schema_version": self.schema_version, "workflow_id": self.workflow_id, "plan_version": self.plan_version,
                "task_id": self.task_id, "attempt_id": self.attempt_id, "goal": self.goal,
                "acceptance_criteria": list(self.acceptance_criteria), "context_reference": self.context_reference,
                "target": self.target.as_dict(), "repository": self.repository, "workspace": self.workspace,
                "permissions": list(self.permissions), "tool_requirements": list(self.tool_requirements),
                "mutating": self.mutating, "write_scope": list(self.write_scope), "timeout_seconds": self.timeout_seconds,
                "deadline_epoch_ms": self.deadline_epoch_ms, "attempt_budget": self.attempt_budget,
                "verification_policy": self.verification_policy, "substitution_policy": self.substitution_policy}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionRequest":
        keys = tuple(cls.__dataclass_fields__)
        _exact_keys(data, keys, "execution request")
        return cls(data["workflow_id"], data["plan_version"], data["task_id"], data["attempt_id"], data["goal"],
                   data["acceptance_criteria"], data["context_reference"], _identity(data["target"]),
                   data["repository"], data["workspace"], data["permissions"], data["tool_requirements"],
                   data["mutating"], data["write_scope"], data["timeout_seconds"], data["deadline_epoch_ms"],
                   data["attempt_budget"], data["verification_policy"], data["substitution_policy"], data["schema_version"])


def legacy_direct_tool_request(*, tool_invocation_id: str, goal: str, target: TargetIdentity,
                               repository: str, workspace: str) -> ExecutionRequest:
    tool = _string("tool_invocation_id", tool_invocation_id, MAX_IDENTIFIER_CHARS, identifier=True)
    workflow = f"legacy:{tool}"
    return ExecutionRequest(workflow, 0, f"{workflow}:task", f"{workflow}:attempt", goal, ("legacy tool completed",),
                            f"legacy-tool:{tool}", target, repository, workspace, ("read",), (), False, (),
                            300, 1, 1, "pending", "forbid")


@dataclass(frozen=True)
class OutputReference:
    summary: str
    artifact_reference: Optional[str] = None
    truncated: bool = False

    def __post_init__(self) -> None:
        _string("output summary", self.summary, MAX_OUTPUT_SUMMARY_CHARS, allow_empty=True)
        if self.artifact_reference is not None:
            _string("output artifact reference", self.artifact_reference, MAX_ARTIFACT_REFERENCE_CHARS)
        if not isinstance(self.truncated, bool):
            raise TypeError("output truncated must be a boolean")

    def as_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary, "artifact_reference": self.artifact_reference, "truncated": self.truncated}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OutputReference":
        _exact_keys(data, ("summary", "artifact_reference", "truncated"), "output reference")
        return cls(data["summary"], data["artifact_reference"], data["truncated"])


@dataclass(frozen=True)
class FailureDetail:
    failure_class: str
    retryable: bool
    reset_hint: Optional[str] = None

    def __post_init__(self) -> None:
        if self.failure_class not in _FAILURE_CLASSES:
            raise ValueError("unknown failure class")
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a boolean")
        if self.reset_hint is not None:
            _string("reset hint", self.reset_hint, MAX_EVIDENCE_CHARS)

    def as_dict(self) -> Dict[str, Any]:
        return {"failure_class": self.failure_class, "retryable": self.retryable, "reset_hint": self.reset_hint}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FailureDetail":
        _exact_keys(data, ("failure_class", "retryable", "reset_hint"), "failure detail")
        return cls(data["failure_class"], data["retryable"], data["reset_hint"])


@dataclass(frozen=True)
class WorkerResult:
    workflow_id: str
    task_id: str
    attempt_id: str
    handle: str
    terminal_status: str
    summary: str
    output: OutputReference
    requested_target: TargetIdentity
    resolved_target: TargetIdentity
    observed_target: TargetIdentity
    usage: Mapping[str, Any] = field(default_factory=dict)
    validation_evidence: Tuple[str, ...] = ()
    artifacts: Tuple[str, ...] = ()
    workspace: str = UNKNOWN
    base_revision: str = UNKNOWN
    failure: Optional[FailureDetail] = None
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("workflow_id", "task_id", "attempt_id", "handle"):
            _string(name, getattr(self, name), MAX_IDENTIFIER_CHARS, identifier=True)
        if self.terminal_status not in _TERMINAL:
            raise ValueError("terminal_status must be terminal")
        _string("summary", self.summary, MAX_OUTPUT_SUMMARY_CHARS, allow_empty=True)
        if not isinstance(self.output, OutputReference):
            raise TypeError("output must be an OutputReference")
        for name in ("requested_target", "resolved_target", "observed_target"):
            if not isinstance(getattr(self, name), TargetIdentity):
                raise TypeError(f"{name} must be a TargetIdentity")
        _json_value(self.usage)
        _string_tuple("validation_evidence", self.validation_evidence, MAX_EVIDENCE_ITEMS, MAX_EVIDENCE_CHARS)
        _string_tuple("artifacts", self.artifacts, MAX_EVIDENCE_ITEMS, MAX_ARTIFACT_REFERENCE_CHARS)
        _string("workspace", self.workspace, MAX_ARTIFACT_REFERENCE_CHARS)
        _string("base_revision", self.base_revision, MAX_IDENTIFIER_CHARS)
        if self.failure is not None and not isinstance(self.failure, FailureDetail):
            raise TypeError("failure must be a FailureDetail")
        if self.terminal_status == "failed" and self.failure is None:
            raise ValueError("failed result requires failure detail")
        if len(self.provider_metadata) > MAX_METADATA_FIELDS:
            raise ValueError("provider metadata has too many fields")
        _redact(self.provider_metadata)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported worker result schema version")

    def as_dict(self) -> Dict[str, Any]:
        return {"schema_version": self.schema_version, "workflow_id": self.workflow_id, "task_id": self.task_id,
                "attempt_id": self.attempt_id, "handle": self.handle, "terminal_status": self.terminal_status,
                "summary": self.summary, "output": self.output.as_dict(),
                "requested_target": self.requested_target.as_dict(), "resolved_target": self.resolved_target.as_dict(),
                "observed_target": self.observed_target.as_dict(), "usage": _json_value(self.usage),
                "validation_evidence": list(self.validation_evidence), "artifacts": list(self.artifacts),
                "workspace": self.workspace, "base_revision": self.base_revision,
                "failure": None if self.failure is None else self.failure.as_dict(),
                "provider_metadata": _redact(self.provider_metadata)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkerResult":
        keys = tuple(cls.__dataclass_fields__)
        _exact_keys(data, keys, "worker result")
        failure = None if data["failure"] is None else FailureDetail.from_dict(data["failure"])
        return cls(data["workflow_id"], data["task_id"], data["attempt_id"], data["handle"], data["terminal_status"],
                   data["summary"], OutputReference.from_dict(data["output"]), _identity(data["requested_target"]),
                   _identity(data["resolved_target"]), _identity(data["observed_target"]), data["usage"],
                   data["validation_evidence"], data["artifacts"], data["workspace"], data["base_revision"],
                   failure, data["provider_metadata"], data["schema_version"])


@dataclass(frozen=True)
class AttemptLifecycle:
    workflow_id: str
    task_id: str
    attempt_id: str
    status: str = "created"
    handle: Optional[str] = None

    @classmethod
    def created(cls, workflow_id: str, task_id: str, attempt_id: str) -> "AttemptLifecycle":
        return cls(workflow_id, task_id, attempt_id)

    def __post_init__(self) -> None:
        for name in ("workflow_id", "task_id", "attempt_id"):
            _string(name, getattr(self, name), MAX_IDENTIFIER_CHARS, identifier=True)
        if self.status not in _LIFECYCLE:
            raise ValueError("unknown lifecycle status")
        if self.handle is not None:
            _string("handle", self.handle, MAX_IDENTIFIER_CHARS, identifier=True)
        if self.status in ("submitted", "running") and self.handle is None:
            raise ValueError(f"{self.status} lifecycle requires a handle")
        if self.status == "created" and self.handle is not None:
            raise ValueError("created lifecycle cannot have a handle")

    def as_dict(self) -> Dict[str, Any]:
        return {"record_type": "attempt_lifecycle", "schema_version": SCHEMA_VERSION,
                "workflow_id": self.workflow_id, "task_id": self.task_id, "attempt_id": self.attempt_id,
                "status": self.status, "handle": self.handle}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AttemptLifecycle":
        keys = ("record_type", "schema_version", "workflow_id", "task_id", "attempt_id", "status", "handle")
        _exact_keys(data, keys, "attempt lifecycle")
        if data["record_type"] != "attempt_lifecycle" or data["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported attempt lifecycle record")
        return cls(data["workflow_id"], data["task_id"], data["attempt_id"], data["status"], data["handle"])

    def transition(self, status: str, *, handle: Optional[str] = None) -> "AttemptLifecycle":
        if status not in _ALLOWED_TRANSITIONS[self.status]:
            raise ValueError(f"invalid lifecycle transition {self.status} -> {status}")
        if self.status == "created" and status == "submitted" and handle is None:
            raise ValueError("submitted lifecycle requires a handle")
        return AttemptLifecycle(self.workflow_id, self.task_id, self.attempt_id, status,
                                handle if handle is not None else self.handle)


@dataclass(frozen=True)
class Submission:
    workflow_id: str
    task_id: str
    attempt_id: str
    accepted: bool
    handle: Optional[str] = None
    rejection: Optional[FailureDetail] = None

    @classmethod
    def for_acceptance(cls, workflow_id: str, task_id: str, attempt_id: str, handle: str) -> "Submission":
        return cls(workflow_id, task_id, attempt_id, True, handle)

    @property
    def terminal(self) -> bool:
        return False

    def __post_init__(self) -> None:
        for name in ("workflow_id", "task_id", "attempt_id"):
            _string(name, getattr(self, name), MAX_IDENTIFIER_CHARS, identifier=True)
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a boolean")
        if self.accepted:
            if self.handle is None or self.rejection is not None:
                raise ValueError("accepted submission requires only a handle")
            _string("handle", self.handle, MAX_IDENTIFIER_CHARS, identifier=True)
        elif self.handle is not None or self.rejection is None:
            raise ValueError("rejected submission requires only a typed rejection")

    def as_dict(self) -> Dict[str, Any]:
        return {"record_type": "submission", "schema_version": SCHEMA_VERSION,
                "workflow_id": self.workflow_id, "task_id": self.task_id,
                "attempt_id": self.attempt_id, "accepted": self.accepted, "handle": self.handle,
                "rejection": None if self.rejection is None else self.rejection.as_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Submission":
        keys = ("record_type", "schema_version", "workflow_id", "task_id", "attempt_id", "accepted", "handle", "rejection")
        _exact_keys(data, keys, "submission")
        if data["record_type"] != "submission" or data["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported submission record")
        rejection = None if data["rejection"] is None else FailureDetail.from_dict(data["rejection"])
        return cls(data["workflow_id"], data["task_id"], data["attempt_id"], data["accepted"], data["handle"], rejection)


@dataclass(frozen=True)
class PendingResult:
    status: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in ("pending", "unknown"):
            raise ValueError("pending result status must be pending or unknown")
        _string("pending reason", self.reason, MAX_EVIDENCE_CHARS, allow_empty=True)


@dataclass(frozen=True)
class Eligibility:
    status: str
    reasons: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("yes", "unavailable", "unsupported"):
            raise ValueError("unknown eligibility status")
        _string_tuple("eligibility reasons", self.reasons, MAX_EVIDENCE_ITEMS, MAX_EVIDENCE_CHARS)


@dataclass(frozen=True)
class CancelOutcome:
    status: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in ("acknowledged", "unsupported", "unknown"):
            raise ValueError("unknown cancel outcome")
        _string("cancel reason", self.reason, MAX_EVIDENCE_CHARS, allow_empty=True)


@dataclass(frozen=True)
class AdapterCapabilities:
    """Versioned, offline description of an adapter's supported contract actions."""
    transport: str
    capabilities: Tuple[str, ...]

    def __post_init__(self) -> None:
        _string("adapter transport", self.transport, MAX_IDENTIFIER_CHARS)
        _string_tuple("adapter capabilities", self.capabilities, MAX_EVIDENCE_ITEMS, MAX_IDENTIFIER_CHARS)

    def as_dict(self) -> Dict[str, Any]:
        return {"transport": self.transport, "capabilities": list(self.capabilities)}


AdapterResult = Union[PendingResult, WorkerResult]
