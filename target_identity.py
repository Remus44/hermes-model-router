"""Resolved target identity and configuration drift diagnostic (S02; I03, I04).

Four config owners name Claude models and can disagree: the router's
``claude_delegation.tiers``, the CLI alias map (``claude_opus_bridge``), the
host's named targets/fallbacks (``~/.hermes/config.yaml``) and what the CLI
actually served. This module compares them without any network, subprocess or
model call, names the owner of every value, and never writes config.

Rules kept here:
* ``exact`` is honored or refused with a typed ``ExactRouteMismatch``; it never
  substitutes silently. ``profile_preferred`` records the substitution.
* The Claude CLI is invoked with an alias, so its resolved identity is the alias
  (non-canonical). Exact canonical CLI selection needs ``exact_model`` capability
  evidence of ``supported``; ``unknown``/``unsupported`` refuse it (plan 9).
* Resolutions are cached by a fingerprint of every input (TTL, size bound, lock),
  so a changed alias, tier or capability never reuses a stale answer. Not on the
  routing hot path.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import claude_delegation, runtime_capabilities
from .execution_contracts import (
    FAILURE_CAPABILITY, FAILURE_EXACT_ROUTE_MISMATCH, NOT_APPLICABLE, SELECTION_EXACT,
    SELECTION_MODES, SELECTION_PREFERRED, TRANSPORTS, UNKNOWN, EffortFact, ExactRouteMismatch,
    ModelFact, TargetIdentity, check_exact,
)

CACHE_TTL_SECONDS = 30.0
CACHE_MAX_ENTRIES = 16
SCHEMA_VERSION = 1

RESOLVED, EXACT_MISMATCH, UNSUPPORTED = "resolved", "exact_mismatch", "unsupported"

_LOCK = threading.Lock()
_CACHE: "OrderedDict[str, Tuple[float, ResolutionResult]]" = OrderedDict()


@dataclass(frozen=True)
class ResolutionResult:
    identity: TargetIdentity
    status: str  # resolved | exact_mismatch | unsupported
    failure_class: str = ""
    reasons: Tuple[str, ...] = ()
    mismatch: Optional[ExactRouteMismatch] = None
    substitution: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "failure_class": self.failure_class,
                "reasons": list(self.reasons), "identity": self.identity.as_dict(),
                "mismatch": self.mismatch.as_dict() if self.mismatch else None,
                "substitution": dict(self.substitution) if self.substitution else None}


# ------------------------------------------------------------------- helpers
def _cli_alias_map() -> Dict[str, str]:
    # Lazy: claude_opus_bridge imports the package root, which imports this module.
    from . import claude_opus_bridge
    return dict(claude_opus_bridge.CLAUDE_REVIEW_MODELS)


def _tier_of(alias: str) -> Optional[str]:
    if alias in claude_delegation.TIERS:
        return alias
    return claude_delegation.TIER_FOR_TARGET.get(alias)


def _router_target(alias: str) -> str:
    tier = _tier_of(alias)
    return claude_delegation.TARGET_FOR_TIER[tier] if tier else alias


def _tier_model(tier: str, cfg: Mapping[str, Any]) -> Tuple[str, str]:
    """(model, source) for a Claude tier, naming router config or the shipped default."""
    raw = (cfg or {}).get("claude_delegation")
    explicit = ((raw or {}).get("tiers") or {}) if isinstance(raw, dict) else {}
    if isinstance(explicit, dict) and str(explicit.get(tier) or "").strip():
        return str(explicit[tier]).strip(), f"router_config:claude_delegation.tiers.{tier}"
    return (str(claude_delegation.DEFAULTS["tiers"].get(tier) or "").strip(),
            f"claude_delegation.DEFAULTS:tiers.{tier}")


def _host_target(host_cfg: Mapping[str, Any], name: str) -> Optional[Tuple[str, str, str]]:
    targets = ((host_cfg or {}).get("delegation") or {}).get("targets") or {}
    for key, spec in targets.items() if isinstance(targets, dict) else ():
        if str(key).strip().casefold() == name and isinstance(spec, dict):
            model = str(spec.get("model") or "").strip()
            if model:
                return (model, str(spec.get("provider") or "").strip().casefold(),
                        f"host_config:delegation.targets.{key}.model")
    return None


def _provider(alias: str, cfg: Mapping[str, Any], host_provider: str, transport: str) -> str:
    mapped = str(((cfg or {}).get("tier_providers") or {}).get(_router_target(alias)) or "").strip()
    if mapped:
        return mapped
    if host_provider:
        return host_provider
    if transport in ("hermes_claude", "claude_cli"):
        return "anthropic"
    return str((cfg or {}).get("provider") or UNKNOWN)


def _cli_exact_capability(snapshot: Optional[runtime_capabilities.RuntimeSnapshot],
                          cfg: Mapping[str, Any]) -> Tuple[str, str]:
    snap = snapshot if snapshot is not None else runtime_capabilities.snapshot(None, dict(cfg or {}))
    cap = snap.adapter("claude_cli").capability("exact_model")
    return cap.status, cap.reason


def _fingerprint(parts: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=repr).encode("utf-8")).hexdigest()


def _substitution(requested: ModelFact, resolved: ModelFact, policy: str, reason: str) -> Dict[str, Any]:
    return {"policy": policy, "requested": requested.value, "requested_source": requested.source,
            "resolved": resolved.value, "resolved_source": resolved.source, "reason": reason}


# ----------------------------------------------------------------- resolution
def _compute(alias: str, transport: str, mode: str, requested_model: str, effort: str,
             account: str, cfg: Mapping[str, Any], host_cfg: Mapping[str, Any],
             cli_exact: Tuple[str, str]) -> ResolutionResult:
    tier = _tier_of(alias)
    target = _router_target(alias)
    reasons: List[str] = []
    host_hit = _host_target(host_cfg, target)
    effort_fact = (EffortFact(effort, UNKNOWN, "not_observed") if effort
                   else EffortFact(NOT_APPLICABLE, NOT_APPLICABLE))
    cli_map = _cli_alias_map()

    # Requested: the operator's ask, else the model the router's own config names.
    if requested_model:
        requested = ModelFact(requested_model, "operator_request")
    elif transport == "claude_cli":
        value = cli_map.get(tier or alias)
        requested = (ModelFact(value, f"cli_alias_map:claude_opus_bridge.CLAUDE_REVIEW_MODELS.{tier or alias}")
                     if value else ModelFact())
    elif tier:
        model, source = _tier_model(tier, cfg)
        requested = ModelFact(model, source) if model else ModelFact()
    else:
        model = str(((cfg or {}).get("models") or {}).get(alias) or "").strip()
        requested = ModelFact(model, f"router_config:models.{alias}") if model else ModelFact()

    def build(resolved: ModelFact, provider: str) -> TargetIdentity:
        return TargetIdentity(provider=provider, account=account, transport=transport, alias=alias,
                              selection_mode=mode, requested=requested, resolved=resolved,
                              effort=effort_fact)

    def refuse(resolved: ModelFact, provider: str, why: str) -> ResolutionResult:
        return ResolutionResult(build(resolved, provider), UNSUPPORTED, FAILURE_CAPABILITY, tuple(reasons + [why]))

    if transport == "claude_cli":
        provider = _provider(alias, cfg, "anthropic", transport)
        if not tier or tier not in cli_map:
            return refuse(ModelFact(), provider, f"claude_cli has no alias for {alias!r}")
        if mode == SELECTION_EXACT:
            status, why = cli_exact
            if status != "supported":
                return refuse(ModelFact(tier, "cli_alias_argument", canonical=False), provider,
                              f"exact canonical CLI selection needs exact_model evidence; status {status}"
                              + (f" ({why})" if why else ""))
            resolved = ModelFact(requested.value, "cli_canonical_model_argument")
        else:
            resolved = ModelFact(tier, "cli_alias_argument", canonical=False)
            reasons.append(f"CLI alias {tier!r} to canonical model is not verified until observed")
    elif transport == "hermes_claude":
        provider = _provider(alias, cfg, "anthropic", transport)
        if not tier:
            return refuse(ModelFact(), provider, f"hermes_claude has no Claude tier for {alias!r}")
        model, source = _tier_model(tier, cfg)
        resolved = ModelFact(model, source) if model else ModelFact()
        if not resolved.known:
            return refuse(resolved, provider, f"no model configured for Claude tier {tier!r}")
    else:  # hermes_codex: delegate_task(model=alias) resolves through the host's named target
        if not requested.known and not host_hit:
            return refuse(ModelFact(), _provider(alias, cfg, "", transport), f"unknown alias {alias!r}")
        provider = _provider(alias, cfg, host_hit[1] if host_hit else "", transport)
        if not host_hit:
            return refuse(ModelFact(), provider, f"host has no delegation target {target!r}")
        resolved = ModelFact(host_hit[0], host_hit[2])

    identity = build(resolved, provider)
    if not requested.known:
        requested = ModelFact(resolved.value, resolved.source) if resolved.canonical else requested
        identity = build(resolved, provider)
    mismatch = check_exact(identity)
    if mismatch:
        return ResolutionResult(identity, EXACT_MISMATCH, FAILURE_EXACT_ROUTE_MISMATCH,
                                tuple(reasons + [f"exact request {mismatch.requested!r} resolves to "
                                                 f"{mismatch.actual!r} via {mismatch.actual_source}"]),
                                mismatch=mismatch)
    substitution = None
    if (mode == SELECTION_PREFERRED and requested.known and resolved.canonical
            and resolved.value != requested.value):
        substitution = _substitution(requested, resolved, SELECTION_PREFERRED,
                                     "resolved model differs from the requested model")
    return ResolutionResult(identity, RESOLVED, "", tuple(reasons), substitution=substitution)


def resolve_target(alias: str, *, transport: str, selection_mode: str = SELECTION_PREFERRED,
                   requested_model: Optional[str] = None, effort: Optional[str] = None,
                   account: str = UNKNOWN, cfg: Optional[Mapping[str, Any]] = None,
                   host_cfg: Optional[Mapping[str, Any]] = None,
                   snapshot: Optional[runtime_capabilities.RuntimeSnapshot] = None) -> ResolutionResult:
    """Resolve an operator alias on one transport into a typed identity record.

    Read-only and cached by a fingerprint of every input, including the CLI alias
    map and the CLI ``exact_model`` capability evidence.
    """
    if transport not in TRANSPORTS:
        raise ValueError(f"unknown transport {transport!r}")
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"unknown selection mode {selection_mode!r}")
    if cfg is None:
        from . import _load_config
        cfg = _load_config()
    if host_cfg is None:
        from . import _read_host_config
        host_cfg = _read_host_config()
    alias = str(alias or "").strip().casefold()
    cli_exact = (_cli_exact_capability(snapshot, cfg)
                 if transport == "claude_cli" and selection_mode == SELECTION_EXACT else ("n/a", ""))
    args = (alias, transport, selection_mode, str(requested_model or "").strip(),
            str(effort or "").strip(), str(account or UNKNOWN))
    key = _fingerprint({
        "args": args, "cli_exact": cli_exact, "cli_map": _cli_alias_map(),
        "router": {k: (cfg or {}).get(k) for k in ("provider", "models", "tier_providers")},
        "tiers": claude_delegation.delegation_config(dict(cfg or {}))["tiers"],
        "claude_delegation": (cfg or {}).get("claude_delegation"),
        "host_targets": ((host_cfg or {}).get("delegation") or {}).get("targets"),
    })
    now = time.monotonic()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < CACHE_TTL_SECONDS:
            _CACHE.move_to_end(key)
            return hit[1]
    result = _compute(*args, dict(cfg or {}), dict(host_cfg or {}), cli_exact)
    with _LOCK:
        _CACHE[key] = (now, result)
        _CACHE.move_to_end(key)
        while len(_CACHE) > CACHE_MAX_ENTRIES:
            _CACHE.popitem(last=False)
    return result


def verify_observed(identity: TargetIdentity, observed_model: str, source: str
                    ) -> Tuple[TargetIdentity, Optional[ExactRouteMismatch], Optional[Dict[str, Any]]]:
    """Attach per-call observed evidence: (identity, exact mismatch, recorded substitution)."""
    from dataclasses import replace
    seen = replace(identity, observed=ModelFact(str(observed_model or UNKNOWN) or UNKNOWN, source))
    mismatch = check_exact(seen)
    substitution = None
    if (identity.selection_mode == SELECTION_PREFERRED and seen.observed.known
            and identity.requested.known and seen.observed.value != identity.requested.value):
        substitution = {"policy": SELECTION_PREFERRED, "requested": identity.requested.value,
                        "requested_source": identity.requested.source,
                        "observed": seen.observed.value, "observed_source": source,
                        "reason": "observed model differs from the requested model"}
    return seen, mismatch, substitution


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def cache_info() -> Dict[str, Any]:
    with _LOCK:
        return {"entries": len(_CACHE), "max_entries": CACHE_MAX_ENTRIES, "ttl_seconds": CACHE_TTL_SECONDS}


# ------------------------------------------------------------- drift diagnostic
def _family(model: Any) -> str:
    """The Claude tier a model id belongs to, or ``""`` for anything else."""
    text = str(model or "").casefold()
    if not text.startswith("claude"):
        return ""
    for tier in claude_delegation.TIERS:
        if tier in text:
            return tier
    return ""


def _host_claude_values(host_cfg: Mapping[str, Any]) -> List[Tuple[str, str, str]]:
    """(tier, key, model) for every Claude model the host config names outside targets."""
    found: List[Tuple[str, str, str]] = []
    model = (host_cfg or {}).get("model")
    if isinstance(model, dict) and _family(model.get("default")):
        found.append((_family(model["default"]), "model.default", str(model["default"]).strip()))
    lists = (("fallback_providers", (host_cfg or {}).get("fallback_providers")),
             ("delegation.fallback_providers", ((host_cfg or {}).get("delegation") or {}).get("fallback_providers")))
    for prefix, entries in lists:
        for index, entry in enumerate(entries if isinstance(entries, list) else []):
            if isinstance(entry, dict) and _family(entry.get("model")):
                found.append((_family(entry["model"]), f"{prefix}[{index}].model", str(entry["model"]).strip()))
    return found


def drift_diagnostic(cfg: Mapping[str, Any], host_cfg: Optional[Mapping[str, Any]] = None,
                     observed: Optional[Mapping[str, Sequence[Tuple[str, str]]]] = None,
                     snapshot: Optional[runtime_capabilities.RuntimeSnapshot] = None) -> Dict[str, Any]:
    """Compare every config owner's Claude model per tier; needs no network request.

    ``observed`` maps tier -> [(served_model, evidence_source)] from recorded
    reviews. Returns plain JSON-compatible data; inputs are not mutated.
    """
    host_cfg = host_cfg if isinstance(host_cfg, dict) else {}
    cli_map = _cli_alias_map()
    host_extra = _host_claude_values(host_cfg)
    tiers: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []
    for tier in claude_delegation.TIERS:
        model, source = _tier_model(tier, cfg)
        owner, _, key = source.partition(":")
        values: List[Dict[str, str]] = []
        if model:
            values.append({"owner": owner, "key": key, "value": model})
        if cli_map.get(tier):
            values.append({"owner": "cli_alias_map", "key": f"claude_opus_bridge.CLAUDE_REVIEW_MODELS.{tier}",
                           "value": cli_map[tier]})
        hit = _host_target(host_cfg, claude_delegation.TARGET_FOR_TIER[tier])
        if hit:
            values.append({"owner": "host_config", "key": hit[2].split(":", 1)[1], "value": hit[0]})
        values.extend({"owner": "host_config", "key": k, "value": v} for t, k, v in host_extra if t == tier)
        expected = values[0] if model else None
        disagreeing = [v for v in values[1:] if expected and v["value"] != expected["value"]]
        tiers.append({"tier": tier, "agree": not disagreeing, "values": values})
        if disagreeing:
            findings.append({"kind": "config_drift", "tier": tier, "expected": expected,
                             "disagreeing": disagreeing})
        seen = [{"value": m, "source": s} for m, s in (observed or {}).get(tier, ())
                if expected and m != expected["value"]]
        if seen:
            findings.append({"kind": "observed_mismatch", "tier": tier, "expected": expected, "observed": seen})
    status, reason = _cli_exact_capability(snapshot, cfg)
    return {"schema_version": SCHEMA_VERSION, "network_used": False,
            "host_config": {"available": bool(host_cfg)},
            "cli_exact_model": {"status": status, "reason": reason},
            "tiers": tiers, "findings": findings}


_MODEL_LINE = re.compile(r"^(\s*(?:model|default):\s*)([^\s#]+)(.*)$")


def propose_host_migration(host_text: str, cfg: Mapping[str, Any]) -> str:
    """Unified diff aligning drifting host Claude models with the router's tier models.

    Text only: nothing is applied or written. Empty string when nothing drifts.
    Only ``model:``/``default:`` lines holding a Claude model of a tier family
    that differs from that tier's router-owned model are proposed.
    """
    expected = {tier: _tier_model(tier, cfg)[0] for tier in claude_delegation.TIERS}
    old = host_text.splitlines(keepends=True)
    new: List[str] = []
    for line in old:
        match = _MODEL_LINE.match(line.rstrip("\n"))
        tier = _family(match.group(2)) if match else None
        if match and tier and expected.get(tier) and match.group(2) != expected[tier]:
            line = f"{match.group(1)}{expected[tier]}{match.group(3)}" + ("\n" if line.endswith("\n") else "")
        new.append(line)
    return "".join(difflib.unified_diff(old, new, "config.yaml", "config.yaml (proposed)"))
