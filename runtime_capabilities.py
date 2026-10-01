"""Read-only runtime capability snapshot and worker-topology resolution (S01).

Capability *evidence* is not dispatch and not admission: a snapshot says what the
host schema and checked seams could support. Fresh admission (worker_admission,
usage_guard) is still required immediately before any execution.

Rules kept here:
* No model probe, subprocess, network call or write. Only in-process reads of the
  request's tool schema, the host's delegation settings and callable seams (I14).
* A missing, false or unverifiable seam is ``unsupported`` or ``unknown`` with a
  reason -- never ``supported``.
* The default topology is ``parent_direct``; ``nested_conductor`` is granted only
  when depth, the orchestrator switch, the seams and the transport all check out.
  Nothing here ever raises or writes the host's spawn depth.
* Snapshots are frozen and cached by a fingerprint of host limits, schema, seams and
  routing config, with a TTL, a size bound and a lock.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from . import claude_delegation

SCHEMA_VERSION = 1
CACHE_TTL_SECONDS = 30.0
CACHE_MAX_ENTRIES = 16

SUPPORTED, UNSUPPORTED, UNKNOWN = "supported", "unsupported", "unknown"
TOPOLOGIES = ("parent_direct", "nested_conductor")
TRANSPORTS = ("hermes_codex", "hermes_claude", "claude_cli")
DEFAULT_TOPOLOGY = "parent_direct"

_DELEGATE_NAMES = ("delegate_task", "mcp__delegate_task")
_CLAUDE_NAMES = ("delegate_claude", "mcp__delegate_claude")
_HOST_SEAMS = ("delegate_task", "_resolve_child_toolsets", "_build_child_agent")
_CONFIG_KEYS = ("orchestration", "callable", "claude_delegation")


@dataclass(frozen=True)
class Fact:
    """A host-derived number with explicit provenance."""
    status: str
    value: Any = None
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "value": self.value, "reason": self.reason}


@dataclass(frozen=True)
class Capability:
    name: str
    status: str
    value: Any = None
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "value": self.value, "reason": self.reason}


@dataclass(frozen=True)
class AdapterCapabilities:
    transport: str
    capabilities: Tuple[Capability, ...]

    def capability(self, name: str) -> Capability:
        for item in self.capabilities:
            if item.name == name:
                return item
        return Capability(name, UNKNOWN, reason="capability not modelled")


@dataclass(frozen=True)
class RuntimeSnapshot:
    schema_version: int
    fingerprint: str
    max_spawn_depth: Fact
    max_concurrent_children: Fact
    orchestrator_enabled: Fact
    seams: Tuple[Tuple[str, bool], ...]
    adapters: Tuple[AdapterCapabilities, ...]
    configured: Tuple[Tuple[str, Any], ...]
    admission_required: bool = True

    def adapter(self, transport: str) -> AdapterCapabilities:
        for item in self.adapters:
            if item.transport == transport:
                return item
        return AdapterCapabilities(transport, ())

    def seam_ok(self, name: str) -> bool:
        return dict(self.seams).get(name, False)


@dataclass(frozen=True)
class TopologyChoice:
    requested: str
    selected: str
    status: str  # supported | unsupported
    reason: str = ""
    transport: str = "hermes_codex"

    def as_dict(self) -> Dict[str, Any]:
        return {"requested": self.requested, "selected": self.selected, "status": self.status,
                "reason": self.reason, "transport": self.transport}


# --------------------------------------------------------------- host reading
def _module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception:  # ImportError, or a module patched to None
        return None


def _fact(getter_module: Any, getter: str, coerce) -> Fact:
    if getter_module is None:
        return Fact(UNKNOWN, None, "host delegation settings are not importable")
    fn = getattr(getter_module, getter, None)
    if not callable(fn):
        return Fact(UNKNOWN, None, f"host has no callable {getter}")
    try:
        return Fact(SUPPORTED, coerce(fn()), "")
    except Exception as exc:
        return Fact(UNKNOWN, None, f"{getter} failed: {type(exc).__name__}")


def _delegate_tool(request: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(request, dict):
        return None
    for tool in request.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        name = tool.get("name") or (fn.get("name") if isinstance(fn, dict) else None)
        if isinstance(name, str) and name in _DELEGATE_NAMES:
            return tool
    return None


def _tool_names(request: Any) -> Tuple[str, ...]:
    names = []
    for tool in (request.get("tools") if isinstance(request, dict) else None) or []:
        if isinstance(tool, dict):
            fn = tool.get("function")
            name = tool.get("name") or (fn.get("name") if isinstance(fn, dict) else None)
            if isinstance(name, str):
                names.append(name)
    return tuple(sorted(set(names)))


def _schema_properties(tool: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The schema's top-level properties, or None when absent/malformed."""
    if tool is None:
        return None
    owner = tool["function"] if isinstance(tool.get("function"), dict) else tool
    for key in ("parameters", "input_schema"):
        schema = owner.get(key)
        if isinstance(schema, dict):
            props = schema.get("properties")
            return props if isinstance(props, dict) else None
    return None


def _config_view(cfg: Any) -> Dict[str, Any]:
    cfg = cfg if isinstance(cfg, dict) else {}
    return {key: cfg.get(key) for key in _CONFIG_KEYS if key in cfg}


def _digest(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, default=repr, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------- snapshot build
def _gather(request: Any, cfg: Any) -> Dict[str, Any]:
    """Everything a snapshot depends on; cheap, in-process, no side effects."""
    host = _module("tools.delegate_tool")
    host_cfg = _module("tools.delegate_tool_config")
    seams = tuple((name, callable(getattr(host, name, None)) if host is not None else False)
                  for name in _HOST_SEAMS)
    tool = _delegate_tool(request)
    props = _schema_properties(tool)
    try:
        bridge_ok, bridge_reason = claude_delegation.reasoning_bridge_status()
    except Exception as exc:
        bridge_ok, bridge_reason = False, f"effort seam check failed: {type(exc).__name__}"
    try:
        claude_active = bool(claude_delegation.is_active())
    except Exception:
        claude_active = False
    names = _tool_names(request)
    return {
        "host": host, "host_cfg": host_cfg, "seams": seams,
        "tool_present": tool is not None,
        "properties": None if props is None else tuple(sorted(props)),
        "tool_names": names,
        "bridge": (bool(bridge_ok), str(bridge_reason)),
        "claude_active": claude_active,
        "interrupt": callable(getattr(host, "interrupt_subagent", None)),
        "async_getter": callable(getattr(host_cfg, "_get_max_async_children", None)),
        "stop_hook": callable(getattr(sys.modules.get(__package__ or ""), "on_subagent_stop", None)),
        "config": _config_view(cfg),
    }


def _cap(name: str, status: str, value: Any = None, reason: str = "") -> Capability:
    return Capability(name, status, value, reason)


def _pair(name: str, outcome: Tuple[str, str]) -> Capability:
    """Capability from a (status, reason) pair; the reason must not land in ``value``."""
    return Capability(name, outcome[0], None, outcome[1])


def _build_adapters(g: Dict[str, Any]) -> Tuple[AdapterCapabilities, ...]:
    seams_ok = all(ok for _n, ok in g["seams"])
    host_missing = g["host"] is None
    props = g["properties"]

    if host_missing:
        submission = (UNSUPPORTED, "Hermes delegation host is not importable")
    elif not seams_ok:
        missing = ", ".join(n for n, ok in g["seams"] if not ok)
        submission = (UNSUPPORTED, f"host seam missing or not callable: {missing}")
    elif not g["tool_present"]:
        submission = (UNKNOWN, "request does not carry delegate_task (absent or deferred)")
    elif props is None:
        submission = (UNKNOWN, "delegate_task schema is missing or malformed")
    else:
        submission = (SUPPORTED, "delegate_task schema present and host seams callable")

    if props is None:
        model_param = (UNKNOWN, "delegate_task schema unavailable")
    elif "model" in props:
        model_param = (SUPPORTED, "schema declares a model parameter")
    else:
        model_param = (UNSUPPORTED, "schema has no model parameter")

    bridge_ok, bridge_reason = g["bridge"]
    effort_host = (UNKNOWN, "effort seam compatible; applied effort is observed per call only") \
        if bridge_ok else (UNSUPPORTED, bridge_reason or "reasoning-effort seam unavailable")
    cancellation = (SUPPORTED, "cooperative interrupt via host interrupt_subagent; no forced kill") \
        if g["interrupt"] else (UNKNOWN, "host exposes no interrupt_subagent seam")
    if g["async_getter"] and g["stop_hook"]:
        async_delivery = (UNKNOWN, "async child limit seam and router on_subagent_stop hook exist, but they "
                                   "do not prove completion delivery; observation is owned by S06")
    elif not g["stop_hook"]:
        async_delivery = (UNSUPPORTED, "router on_subagent_stop lifecycle hook is not callable")
    else:
        async_delivery = (UNSUPPORTED, "host exposes no async child limit seam")
    claude_seam_reason = "delegate_claude is not shown to use the host interrupt/async seams (S05)"
    common_unknown = {
        "exact_model": "exact wire identity needs per-call observation, not a schema",
        "identity_observability": "served model identity is not derivable from a schema",
        "permissions": "child tool/permission contract is not verified by discovery",
        "workspace_isolation": "host isolation is not proven available by discovery",
    }

    def unknowns():
        return [_cap(n, UNKNOWN, reason=r) for n, r in common_unknown.items()]

    if model_param[0] == SUPPORTED:
        codex_batch = (UNKNOWN, "schema declares a model parameter; per-task split is not verified (S04)")
    elif model_param[0] == UNSUPPORTED:
        codex_batch = (UNSUPPORTED, "needs a model parameter per task")
    else:
        codex_batch = (UNKNOWN, model_param[1])
    if host_missing or not seams_ok:
        codex_fallback = (UNSUPPORTED, submission[1])
    else:
        codex_fallback = (UNKNOWN, "no host seam reports fallback ownership; determined per call (S05)")

    codex = AdapterCapabilities("hermes_codex", tuple([
        _pair("submission", submission),
        _pair("model_parameter", model_param),
        _pair("mixed_target_batch", codex_batch),
        _pair("effort_application", effort_host),
        _pair("cancellation", cancellation),
        _pair("async_delivery", async_delivery),
        _pair("fallback_ownership", codex_fallback),
    ] + unknowns()))

    claude_name = claude_delegation.TOOL_NAME
    if host_missing or not seams_ok:
        claude_submission = (UNSUPPORTED, submission[1])
    elif not g["claude_active"]:
        claude_submission = (UNSUPPORTED, "Claude delegation is not active")
    elif claude_name in g["tool_names"] or any(n in g["tool_names"] for n in _CLAUDE_NAMES):
        claude_submission = (SUPPORTED, f"{claude_name} is visible in the request")
    else:
        claude_submission = (UNKNOWN, f"{claude_name} not visible in this request (deferred or absent)")

    if claude_submission[0] == UNSUPPORTED:
        claude_fallback = (UNSUPPORTED, claude_submission[1])
    else:
        claude_fallback = (UNKNOWN, "no seam reports Claude fallback ownership; determined per call (S05)")

    claude = AdapterCapabilities("hermes_claude", tuple([
        _pair("submission", claude_submission),
        _cap("model_parameter", UNSUPPORTED, reason="route is chosen per call by tier, not a model parameter"),
        _cap("mixed_target_batch", UNSUPPORTED, reason="one tier and effort per delegate_claude call"),
        _pair("effort_application", effort_host),
        _cap("cancellation", UNKNOWN, reason=claude_seam_reason),
        _cap("async_delivery", UNKNOWN, reason=claude_seam_reason),
        _pair("fallback_ownership", claude_fallback),
    ] + unknowns()))

    cli_reason = "CLI availability is not probed on the request path"
    cli = AdapterCapabilities("claude_cli", tuple([
        _cap("submission", UNKNOWN, reason=cli_reason),
        _cap("model_parameter", UNKNOWN, reason=cli_reason),
        _cap("mixed_target_batch", UNSUPPORTED, reason="independent runs, no shared batch"),
        _cap("effort_application", UNKNOWN, reason="capability-dependent on the installed CLI"),
        _cap("cancellation", UNKNOWN, reason="process lifecycle not verified"),
        _cap("async_delivery", UNKNOWN, reason="process lifecycle not verified"),
        _cap("fallback_ownership", UNKNOWN, "cli_native", "CLI-native fallback not verified"),
    ] + unknowns()))
    return (codex, claude, cli)


def _fingerprint(g: Dict[str, Any], depth: Fact, conc: Fact, orch: Fact) -> str:
    return _digest({
        "v": SCHEMA_VERSION, "depth": [depth.status, depth.value], "conc": [conc.status, conc.value],
        "orch": [orch.status, orch.value], "seams": g["seams"], "tool": g["tool_present"],
        "props": g["properties"], "names": g["tool_names"], "bridge": g["bridge"],
        "claude": g["claude_active"], "interrupt": g["interrupt"], "async": g["async_getter"], "stop_hook": g["stop_hook"],
        "config": g["config"], "host": g["host"] is not None,
    })


_CACHE: "OrderedDict[str, Tuple[float, RuntimeSnapshot]]" = OrderedDict()
_LOCK = threading.Lock()


def snapshot(request: Any = None, cfg: Any = None) -> RuntimeSnapshot:
    """Return the (cached) capability snapshot for this request/config/host."""
    g = _gather(request, cfg)
    depth = _fact(g["host_cfg"], "_get_max_spawn_depth", int)
    conc = _fact(g["host_cfg"], "_get_max_concurrent_children", int)
    orch = _fact(g["host_cfg"], "_get_orchestrator_enabled", bool)
    key = _fingerprint(g, depth, conc, orch)
    now = time.monotonic()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and now - hit[0] <= CACHE_TTL_SECONDS:
            _CACHE.move_to_end(key)
            return hit[1]
    cfg_view = g["config"]
    orchestration = cfg_view.get("orchestration") if isinstance(cfg_view.get("orchestration"), dict) else {}
    claude_cfg = cfg_view.get("claude_delegation") if isinstance(cfg_view.get("claude_delegation"), dict) else {}
    configured = (
        ("orchestration_enabled", bool(orchestration.get("enabled"))),
        ("conductor", orchestration.get("conductor")),
        ("claude_delegation_enabled", bool(claude_cfg.get("enabled"))),
    )
    snap = RuntimeSnapshot(
        schema_version=SCHEMA_VERSION, fingerprint=key, max_spawn_depth=depth,
        max_concurrent_children=conc, orchestrator_enabled=orch, seams=g["seams"],
        adapters=_build_adapters(g), configured=configured,
    )
    with _LOCK:
        _CACHE[key] = (now, snap)
        _CACHE.move_to_end(key)
        while len(_CACHE) > CACHE_MAX_ENTRIES:
            _CACHE.popitem(last=False)
    return snap


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def cache_info() -> Dict[str, Any]:
    with _LOCK:
        return {"size": len(_CACHE), "max_entries": CACHE_MAX_ENTRIES, "ttl_seconds": CACHE_TTL_SECONDS}


# ------------------------------------------------------------------- topology
def resolve_topology(snap: RuntimeSnapshot, requested: str = DEFAULT_TOPOLOGY,
                     transport: str = "hermes_codex") -> TopologyChoice:
    """Pick a topology. An unsupported request falls back to ``parent_direct`` with a reason."""
    def refused(reason: str) -> TopologyChoice:
        return TopologyChoice(str(requested), DEFAULT_TOPOLOGY, UNSUPPORTED, reason, str(transport))

    if requested not in TOPOLOGIES:
        return refused(f"unknown topology {requested!r}")
    if transport not in TRANSPORTS:
        return refused(f"unknown transport {transport!r}")
    if requested == DEFAULT_TOPOLOGY:
        return TopologyChoice(requested, DEFAULT_TOPOLOGY, SUPPORTED, "", transport)
    depth = snap.max_spawn_depth
    if depth.status != SUPPORTED:
        return refused(f"host depth unknown: {depth.reason}")
    if depth.value < 2:
        return refused(f"host max spawn depth is {depth.value}; a nested conductor needs depth >= 2")
    if snap.orchestrator_enabled.status != SUPPORTED or not snap.orchestrator_enabled.value:
        return refused("host orchestrator role is disabled or unknown")
    missing = [n for n, ok in snap.seams if not ok]
    if missing:
        return refused("host seam missing: " + ", ".join(missing))
    submission = snap.adapter(transport).capability("submission")
    if submission.status != SUPPORTED:
        return refused(f"{transport} submission is {submission.status}: {submission.reason}")
    return TopologyChoice(requested, requested, SUPPORTED, "", transport)


# ----------------------------------------------------------------- diagnostic
def diagnostic(snap: RuntimeSnapshot, requested: str = DEFAULT_TOPOLOGY,
               transport: str = "hermes_codex") -> Dict[str, Any]:
    """A detached JSON-able view separating configured / runtime / topology."""
    choice = resolve_topology(snap, requested, transport)
    return {
        "schema_version": snap.schema_version,
        "fingerprint": snap.fingerprint,
        "configured": dict(snap.configured),
        "runtime": {
            "max_spawn_depth": snap.max_spawn_depth.as_dict(),
            "max_concurrent_children": snap.max_concurrent_children.as_dict(),
            "orchestrator_enabled": snap.orchestrator_enabled.as_dict(),
            "seams": dict(snap.seams),
            "admission_required": snap.admission_required,
            "adapters": {a.transport: {c.name: c.as_dict() for c in a.capabilities} for a in snap.adapters},
        },
        "topology": {**choice.as_dict(), "active": "unknown",
                     "active_reason": "a snapshot cannot observe the running topology"},
    }


# ------------------------------------------------------------- section-7 matrix
# (case, capability read by S01 or None, owning slice, reason when S01 cannot evaluate)
MATRIX_CASES = (
    ("submission", "submission", "S01", ""),
    ("exact_model", "exact_model", "S02", ""),
    ("exact_effort", "effort_application", "S05", ""),
    ("mixed_target_batch", "mixed_target_batch", "S04", ""),
    ("tool_context_permissions", "permissions", "S04", ""),
    ("usage_admission", None, "S03", "shared admission is evaluated by worker_admission at dispatch, not discovery"),
    ("quota_vs_pacing", None, "S03", "failure classification needs a runtime failure, not discovery"),
    ("native_fallback", "fallback_ownership", "S05", ""),
    ("background_completion", "async_delivery", "S06", ""),
    ("failure_cancel_timeout", "cancellation", "S06", ""),
    ("concurrent_calls", None, "S05", ""),
    ("restart_duplicate_hook", None, "S06", "needs lifecycle ingestion, not discovery"),
    ("parent_stop_amend", None, "S09", "needs plan amendment and launch suppression, not discovery"),
    ("verification_substitution", None, "S14", "needs reviewer identity records, not discovery"),
)


def compatibility_matrix(snap: RuntimeSnapshot, *, parallel: bool = False) -> Tuple[Dict[str, Any], ...]:
    """Every plan-section-7 case x transport with an explicit status, reason and owning slice.

    Parent provider is deliberately not an input: the host seams do not depend on it, and
    what a Codex or Claude parent can reach is already visible as the request's tool list
    (``delegate_claude`` present or deferred) that the snapshot fingerprints.
    ``parallel`` is accepted for compatibility and no longer changes any status. The host
    ``max_concurrent_children`` is a per-call batch width: not a limit on distinct legacy
    invocations and not a CLI concurrency limit (the host is the sole admission authority
    for legacy dispatches). ``concurrent_calls`` therefore stays ``unknown`` and reports
    the observed width as its own ``observed`` fact.
    """
    rows = []
    for case, cap_name, owner, why in MATRIX_CASES:
        for adapter in snap.adapters:
            extra: Dict[str, Any] = {}
            if cap_name is not None:
                cap = adapter.capability(cap_name)
                status, reason = cap.status, cap.reason
            elif case == "concurrent_calls":
                conc = snap.max_concurrent_children
                width = conc.value if conc.status == SUPPORTED else UNKNOWN
                extra["observed"] = {"host_batch_width": width}
                status = UNKNOWN
                reason = (f"host per-call batch width is {width} ({conc.status}); it is not a limit on "
                          "distinct legacy invocations or on the CLI, and the host alone admits legacy "
                          "dispatches, so distinct-call capacity is not reported")
            else:
                status, reason = UNKNOWN, why
            rows.append({"case": case, "transport": adapter.transport, "status": status,
                         "reason": reason, "owner": owner, **extra})
    return tuple(rows)
