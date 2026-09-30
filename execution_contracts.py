"""Versioned execution contracts (S02: target identity; S04 extends this module).

Pure, JSON-compatible, frozen records. Nothing here reads config, touches the
network or dispatches anything.

Identity concepts are kept apart on purpose (plan 4.1): provider/account,
transport, operator alias, requested/resolved/observed model (each with the
evidence source that produced it), selection mode and effort. ``exact`` never
substitutes silently: a disagreement becomes a typed ``ExactRouteMismatch``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

SCHEMA_VERSION = 1

TRANSPORTS = ("hermes_codex", "hermes_claude", "claude_cli")
SELECTION_EXACT, SELECTION_PREFERRED = "exact", "profile_preferred"
SELECTION_MODES = (SELECTION_EXACT, SELECTION_PREFERRED)

UNKNOWN = "unknown"
NOT_APPLICABLE = "not_applicable"
SOURCE_NOT_OBSERVED = "not_observed"

FAILURE_EXACT_ROUTE_MISMATCH = "exact-route-mismatch"
FAILURE_CAPABILITY = "capability"


@dataclass(frozen=True)
class ModelFact:
    """One model identity with its evidence source.

    ``canonical`` is False for a value that is an alias handed to a tool (the
    Claude CLI ``--model sonnet``): it names a request, not a served model.
    """
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
    """Requested versus applied effort; ``unknown``/``not_applicable`` are explicit."""
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
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "account": self.account,
            "transport": self.transport,
            "alias": self.alias,
            "selection_mode": self.selection_mode,
            "requested": self.requested.as_dict(),
            "resolved": self.resolved.as_dict(),
            "observed": self.observed.as_dict(),
            "effort": self.effort.as_dict(),
        }


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
    resolved value counts only when it is canonical: a CLI alias argument is not
    evidence of the served model.
    """
    if identity.selection_mode != SELECTION_EXACT or not identity.requested.known:
        return None
    wanted = identity.requested
    if identity.observed.known and identity.observed.value != wanted.value:
        actual, stage = identity.observed, "observed"
    elif identity.resolved.known and identity.resolved.canonical and identity.resolved.value != wanted.value:
        actual, stage = identity.resolved, "resolved"
    else:
        return None
    return ExactRouteMismatch(stage, identity.alias, identity.transport, wanted.value, wanted.source,
                              actual.value, actual.source)
