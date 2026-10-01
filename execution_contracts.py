"""Versioned execution contracts.

Pure, JSON-compatible records only: no config reads, model calls, subprocesses or
scheduler behavior.  S02 target-identity shapes remain compatible; S04 adds the
request, lifecycle, submission, result, and adapter-boundary contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import hashlib
import math
from types import MappingProxyType
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


MAX_JSON_NODES = 1024
MAX_JSON_CHARS = 32768


def _version(value: Any) -> None:
    _integer("schema_version", value)
    if value != SCHEMA_VERSION:
        raise ValueError("unsupported schema version")


def _json_value(value: Any, *, depth: int = 0, budget=None, redact=False) -> Any:
    # Shared traversal bounds the whole payload, not just each container.
    if budget is None:
        budget = [0, 0]
    budget[0] += 1
    if depth > 8 or budget[0] > MAX_JSON_NODES:
        raise ValueError("JSON metadata exceeds depth or aggregate node bounds")
    if isinstance(value, str):
        _string("JSON string", value, MAX_EVIDENCE_CHARS, allow_empty=True)
        budget[1] += len(value)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    elif isinstance(value, int) and not isinstance(value, bool) and value.bit_length() > 64:
        raise ValueError("JSON integer exceeds 64-bit bounds")
    if budget[1] > MAX_JSON_CHARS:
        raise ValueError("JSON metadata exceeds aggregate character bounds")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_USAGE_FIELDS:
            raise ValueError("JSON object has too many fields")
        result = {}
        for key, item in value.items():
            key = _string("JSON key", key, MAX_IDENTIFIER_CHARS)
            budget[1] += len(key)
            if redact and any(word in key.lower().replace("-", "_") for word in
                    ("token", "secret", "password", "authorization", "credential", "api_key", "apikey", "cookie", "bearer")):
                item = "[redacted]"
            result[key] = _json_value(item, depth=depth + 1, budget=budget, redact=redact)
        return result
    if isinstance(value, (tuple, list)):
        if len(value) > MAX_USAGE_FIELDS:
            raise ValueError("JSON sequence has too many items")
        return [_json_value(item, depth=depth + 1, budget=budget, redact=redact) for item in value]
    raise TypeError("value must be JSON-compatible")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _owned_mapping(value: Any, *, redact=False) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("usage/provider metadata must be an object")
    return _freeze_json(_json_value(value, redact=redact))


def _redact(value: Mapping[str, Any]) -> Dict[str, Any]:
    return _json_value(value, redact=True)


@dataclass(frozen=True)
class ModelFact:
    """One model identity with its evidence source.

    ``canonical`` is False for a value that is an alias handed to a tool (the
    Claude CLI ``--model sonnet``): it names a request, not a served model.
    """
    value: str = UNKNOWN
    source: str = SOURCE_NOT_OBSERVED
    canonical: bool = True

    def __post_init__(self) -> None:
        _string("model fact value", self.value, MAX_IDENTIFIER_CHARS)
        _string("model fact source", self.source, MAX_IDENTIFIER_CHARS)
        if not isinstance(self.canonical, bool):
            raise TypeError("model fact canonical must be a boolean")

    @property
    def known(self) -> bool:
        return bool(self.value) and self.value != UNKNOWN

    def as_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "source": self.source, "canonical": self.canonical}


@dataclass(frozen=True)
class EffortFact:
    """Requested versus applied effort; ``unknown``/``not_applicable`` are explicit."""
    requested: str = NOT_APPLICABLE
    applied: str = NOT_APPLICABLE
    source: str = SOURCE_NOT_OBSERVED

    def __post_init__(self) -> None:
        for name in ("requested", "applied", "source"):
            _string("effort " + name, getattr(self, name), MAX_IDENTIFIER_CHARS)

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
        _version(self.schema_version)
        for name in ("provider", "account", "alias"):
            _string(name, getattr(self, name), MAX_IDENTIFIER_CHARS)
        for name in ("requested", "resolved", "observed"):
            if not isinstance(getattr(self, name), ModelFact):
                raise TypeError(name + " must be a ModelFact")
        if not isinstance(self.effort, EffortFact):
            raise TypeError("effort must be an EffortFact")
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
    """Typed refusal of an exact request whose identity disagrees (I04)."""
    stage: str  # "resolved" | "observed"
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
    """The mismatch an exact identity carries, if any; always None for preferred.

    The observed model is the strongest evidence, so it is checked first. A
    resolved or observed value counts only when it is canonical: a CLI alias
    argument is not evidence of the served model.
    """
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
    _version(data["schema_version"])
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
        object.__setattr__(self, "acceptance_criteria", criteria)
        if not criteria:
            raise ValueError("acceptance_criteria must not be empty")
        for name in ("context_reference", "repository", "workspace", "verification_policy", "substitution_policy"):
            _string(name, getattr(self, name), MAX_ARTIFACT_REFERENCE_CHARS)
        if not isinstance(self.target, TargetIdentity):
            raise TypeError("target must be a TargetIdentity")
        for name in ("permissions", "tool_requirements", "write_scope"):
            object.__setattr__(self, name, _string_tuple(name, getattr(self, name), MAX_EVIDENCE_ITEMS, MAX_ARTIFACT_REFERENCE_CHARS))
        if not isinstance(self.mutating, bool):
            raise TypeError("mutating must be a boolean")
        _integer("timeout_seconds", self.timeout_seconds, 1)
        _integer("deadline_epoch_ms", self.deadline_epoch_ms, 1)
        _integer("attempt_budget", self.attempt_budget, 1)
        _version(self.schema_version)

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
                               repository: str, workspace: str, acceptance_criteria: Tuple[str, ...],
                               permissions: Tuple[str, ...], tool_requirements: Tuple[str, ...],
                               mutating: bool, write_scope: Tuple[str, ...], timeout_seconds: int,
                               deadline_epoch_ms: int, attempt_budget: int, verification_policy: str,
                               substitution_policy: str) -> ExecutionRequest:
    """Assign observation IDs, never invent tool authority or execution budgets."""
    tool = _string("tool_invocation_id", tool_invocation_id, MAX_IDENTIFIER_CHARS, identifier=True)
    suffix = tool if len(tool) <= MAX_IDENTIFIER_CHARS - len("legacy::attempt") else hashlib.sha256(tool.encode()).hexdigest()
    workflow = f"legacy:{suffix}"
    return ExecutionRequest(workflow, 0, f"{workflow}:task", f"{workflow}:attempt", goal, acceptance_criteria,
                            f"legacy-tool:{tool}", target, repository, workspace, permissions,
                            tool_requirements, mutating, write_scope, timeout_seconds, deadline_epoch_ms,
                            attempt_budget, verification_policy, substitution_policy)


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
        if self.truncated and self.artifact_reference is None:
            raise ValueError("truncated output requires an artifact reference")

    @classmethod
    def bounded(cls, text: str, *, artifact_reference: Optional[str] = None,
                limit: int = MAX_OUTPUT_SUMMARY_CHARS) -> "OutputReference":
        """Caller owns persistence of the full text; this helper does no I/O."""
        _integer("summary limit", limit, 1)
        if limit > MAX_OUTPUT_SUMMARY_CHARS:
            raise ValueError("summary limit exceeds contract bound")
        if not isinstance(text, str):
            raise TypeError("output text must be a string")
        return cls(text[:limit], artifact_reference, len(text) > limit)

    def as_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary, "artifact_reference": self.artifact_reference, "truncated": self.truncated}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OutputReference":
        _exact_keys(data, ("summary", "artifact_reference", "truncated"), "output reference")
        return cls(data["summary"], data["artifact_reference"], data["truncated"])


def bounded_evidence(text: str, *, artifact_reference: Optional[str] = None) -> str:
    """Retain short evidence or explicitly reference caller-persisted full detail."""
    if not isinstance(text, str):
        raise TypeError("evidence must be a string")
    if len(text) <= MAX_EVIDENCE_CHARS:
        return _string("evidence", text, MAX_EVIDENCE_CHARS)
    if artifact_reference is None:
        raise ValueError("oversized evidence requires an artifact reference")
    _string("evidence artifact reference", artifact_reference, MAX_ARTIFACT_REFERENCE_CHARS)
    return "artifact-reference:" + artifact_reference


@dataclass(frozen=True)
class FailureDetail:
    failure_class: str
    retryable: bool
    reset_hint: Optional[str] = None
    message: Optional[OutputReference] = None
    details: Optional[OutputReference] = None

    def __post_init__(self) -> None:
        if self.failure_class not in _FAILURE_CLASSES:
            raise ValueError("unknown failure class")
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a boolean")
        if self.reset_hint is not None:
            _string("reset hint", self.reset_hint, MAX_EVIDENCE_CHARS)
        for name in ("message", "details"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), OutputReference):
                raise TypeError(name + " must be an OutputReference")

    def as_dict(self) -> Dict[str, Any]:
        return {"failure_class": self.failure_class, "retryable": self.retryable, "reset_hint": self.reset_hint,
                "message": None if self.message is None else self.message.as_dict(),
                "details": None if self.details is None else self.details.as_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FailureDetail":
        _exact_keys(data, tuple(cls.__dataclass_fields__), "failure detail")
        return cls(data["failure_class"], data["retryable"], data["reset_hint"],
                   None if data["message"] is None else OutputReference.from_dict(data["message"]),
                   None if data["details"] is None else OutputReference.from_dict(data["details"]))


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
    plan_version: int = 0
    changed_files: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _integer("plan_version", self.plan_version)
        object.__setattr__(self, "changed_files", _string_tuple("changed_files", self.changed_files, MAX_EVIDENCE_ITEMS, MAX_ARTIFACT_REFERENCE_CHARS))
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
        object.__setattr__(self, "usage", _owned_mapping(self.usage))
        object.__setattr__(self, "validation_evidence", _string_tuple("validation_evidence", self.validation_evidence, MAX_EVIDENCE_ITEMS, MAX_EVIDENCE_CHARS))
        object.__setattr__(self, "artifacts", _string_tuple("artifacts", self.artifacts, MAX_EVIDENCE_ITEMS, MAX_ARTIFACT_REFERENCE_CHARS))
        _string("workspace", self.workspace, MAX_ARTIFACT_REFERENCE_CHARS)
        _string("base_revision", self.base_revision, MAX_IDENTIFIER_CHARS)
        if self.failure is not None and not isinstance(self.failure, FailureDetail):
            raise TypeError("failure must be a FailureDetail")
        if self.terminal_status != "succeeded" and self.failure is None:
            raise ValueError("non-success result requires failure detail")
        if self.terminal_status == "succeeded" and self.failure is not None:
            raise ValueError("succeeded result cannot carry failure")
        expected_class = {"cancelled": "cancelled", "timed_out": "timeout"}.get(self.terminal_status)
        if expected_class is not None and self.failure.failure_class != expected_class:
            raise ValueError("terminal status contradicts failure class")
        if self.terminal_status == "succeeded" and self.requested_target.selection_mode == SELECTION_EXACT:
            wanted = self.requested_target
            # Evaluate the authoritative request against retained resolved/observed facts.
            combined = TargetIdentity(wanted.provider, wanted.account, wanted.transport, wanted.alias,
                                      wanted.selection_mode, wanted.requested, self.resolved_target.resolved,
                                      self.observed_target.observed, self.observed_target.effort)
            effort = self.observed_target.effort.applied
            if (check_exact(combined) is not None
                    or any(target.provider != wanted.provider for target in (self.resolved_target, self.observed_target))
                    or any(target.account not in (UNKNOWN, wanted.account)
                           for target in (self.resolved_target, self.observed_target))
                    or any(target.transport != wanted.transport
                           for target in (self.resolved_target, self.observed_target))
                    or (wanted.effort.requested not in (UNKNOWN, NOT_APPLICABLE)
                        and effort not in (UNKNOWN, NOT_APPLICABLE) and effort != wanted.effort.requested)):
                raise ValueError("exact route mismatch cannot be reported as success")
        if len(self.provider_metadata) > MAX_METADATA_FIELDS:
            raise ValueError("provider metadata has too many fields")
        object.__setattr__(self, "provider_metadata", _owned_mapping(self.provider_metadata, redact=True))
        _version(self.schema_version)

    def as_dict(self) -> Dict[str, Any]:
        return {"schema_version": self.schema_version, "workflow_id": self.workflow_id, "task_id": self.task_id,
                "attempt_id": self.attempt_id, "handle": self.handle, "terminal_status": self.terminal_status,
                "summary": self.summary, "output": self.output.as_dict(),
                "requested_target": self.requested_target.as_dict(), "resolved_target": self.resolved_target.as_dict(),
                "observed_target": self.observed_target.as_dict(), "usage": _json_value(self.usage),
                "validation_evidence": list(self.validation_evidence), "artifacts": list(self.artifacts),
                "workspace": self.workspace, "base_revision": self.base_revision,
                "failure": None if self.failure is None else self.failure.as_dict(),
                "provider_metadata": _redact(self.provider_metadata),
                "plan_version": self.plan_version, "changed_files": list(self.changed_files)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkerResult":
        keys = tuple(cls.__dataclass_fields__)
        _exact_keys(data, keys, "worker result")
        failure = None if data["failure"] is None else FailureDetail.from_dict(data["failure"])
        return cls(data["workflow_id"], data["task_id"], data["attempt_id"], data["handle"], data["terminal_status"],
                   data["summary"], OutputReference.from_dict(data["output"]), _identity(data["requested_target"]),
                   _identity(data["resolved_target"]), _identity(data["observed_target"]), data["usage"],
                   data["validation_evidence"], data["artifacts"], data["workspace"], data["base_revision"],
                   failure, data["provider_metadata"], data["schema_version"], data["plan_version"], data["changed_files"])


@dataclass(frozen=True)
class AttemptLifecycle:
    workflow_id: str
    task_id: str
    attempt_id: str
    status: str = "created"
    handle: Optional[str] = None
    plan_version: int = 0

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
        _integer("plan_version", self.plan_version)
        pre_submission = self.status in ("created", "unavailable", "unsupported")
        if not pre_submission and self.handle is None:
            raise ValueError(f"{self.status} lifecycle requires a handle")
        if pre_submission and self.handle is not None:
            raise ValueError(f"{self.status} lifecycle cannot have a handle")

    def as_dict(self) -> Dict[str, Any]:
        return {"record_type": "attempt_lifecycle", "schema_version": SCHEMA_VERSION,
                "workflow_id": self.workflow_id, "task_id": self.task_id, "attempt_id": self.attempt_id,
                "status": self.status, "handle": self.handle, "plan_version": self.plan_version}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AttemptLifecycle":
        keys = ("record_type", "schema_version", "workflow_id", "task_id", "attempt_id", "status", "handle", "plan_version")
        _exact_keys(data, keys, "attempt lifecycle")
        _version(data["schema_version"])
        if data["record_type"] != "attempt_lifecycle":
            raise ValueError("unsupported attempt lifecycle record")
        return cls(data["workflow_id"], data["task_id"], data["attempt_id"], data["status"], data["handle"], data["plan_version"])

    def transition(self, status: str, *, handle: Optional[str] = None) -> "AttemptLifecycle":
        if status not in _ALLOWED_TRANSITIONS[self.status]:
            raise ValueError(f"invalid lifecycle transition {self.status} -> {status}")
        if self.status == "created" and status == "submitted" and handle is None:
            raise ValueError("submitted lifecycle requires a handle")
        if self.handle is not None and handle is not None and handle != self.handle:
            raise ValueError("lifecycle handle cannot change")
        return AttemptLifecycle(self.workflow_id, self.task_id, self.attempt_id, status,
                                handle if handle is not None else self.handle, self.plan_version)

    def reconcile(self, status: str) -> "AttemptLifecycle":
        """Explicit caller-attested reconciliation; no ownership release or I/O."""
        if self.status != "unknown" or status not in ("running",) + _TERMINAL:
            raise ValueError("reconciliation requires unknown -> running/terminal")
        return AttemptLifecycle(self.workflow_id, self.task_id, self.attempt_id, status,
                                self.handle, self.plan_version)


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
        elif self.handle is not None or not isinstance(self.rejection, FailureDetail):
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
        _version(data["schema_version"])
        if data["record_type"] != "submission":
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


    def as_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PendingResult":
        _exact_keys(data, ("status", "reason"), "PendingResult")
        return cls(data["status"], data["reason"])


@dataclass(frozen=True)
class Eligibility:
    status: str
    reasons: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("yes", "unavailable", "unsupported"):
            raise ValueError("unknown eligibility status")
        object.__setattr__(self, "reasons", _string_tuple("eligibility reasons", self.reasons, MAX_EVIDENCE_ITEMS, MAX_EVIDENCE_CHARS))


    def as_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "reasons": list(self.reasons)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Eligibility":
        _exact_keys(data, ("status", "reasons"), "Eligibility")
        return cls(data["status"], data["reasons"])


@dataclass(frozen=True)
class CancelOutcome:
    status: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in ("acknowledged", "unsupported", "unknown"):
            raise ValueError("unknown cancel outcome")
        _string("cancel reason", self.reason, MAX_EVIDENCE_CHARS, allow_empty=True)


    def as_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CancelOutcome":
        _exact_keys(data, ("status", "reason"), "CancelOutcome")
        return cls(data["status"], data["reason"])


@dataclass(frozen=True)
class AdapterCapabilities:
    """Versioned, offline description of an adapter's supported contract actions."""
    transport: str
    capabilities: Tuple[str, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version)
        _string("adapter transport", self.transport, MAX_IDENTIFIER_CHARS)
        object.__setattr__(self, "capabilities", _string_tuple("adapter capabilities", self.capabilities, MAX_EVIDENCE_ITEMS, MAX_IDENTIFIER_CHARS))

    def as_dict(self) -> Dict[str, Any]:
        return {"schema_version": self.schema_version, "transport": self.transport, "capabilities": list(self.capabilities)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdapterCapabilities":
        _exact_keys(data, tuple(cls.__dataclass_fields__), "adapter capabilities")
        return cls(data["transport"], data["capabilities"], data["schema_version"])


AdapterResult = Union[PendingResult, WorkerResult]
