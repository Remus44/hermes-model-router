"""Claude delegation: Claude workers as real Hermes children, next to the Codex workforce.

Hermes's ``delegate_task`` has one delegation route per process, and on this host
it is pinned to Codex. ``delegate_claude`` reaches Claude by calling the same
``delegate_task`` with a per-call route (``credentials_cfg``) pinned to the
``anthropic`` provider -- the mechanism Hermes's own /review uses. Pinning the
provider, rather than inheriting the parent's, is what keeps Claude delegation working
while the parent itself is on a Codex fallback.

Nothing here is imported from the router at module level: the router imports
this module, so router helpers are pulled in function-locally.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
import threading
import types
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple

from . import usage_guard
from .hermes_paths import hermes_path

_logger = logging.getLogger("model_router.claude_delegation")

TIERS: Tuple[str, ...] = ("haiku", "sonnet", "opus")
# Router target names stay what the config, cooldowns and dashboard already use;
# the tool speaks in short tier names. This table is the only place they meet.
TARGET_FOR_TIER: Dict[str, str] = {"haiku": "haiku", "sonnet": "sonnet5", "opus": "opus5"}
TIER_FOR_TARGET: Dict[str, str] = {target: tier for tier, target in TARGET_FOR_TIER.items()}

EDITABLE_REASONING_TIERS: Tuple[str, ...] = ("sonnet", "opus")
REASONING_LEVELS: Tuple[str, ...] = ("low", "medium", "high", "xhigh")
DEFAULT_REASONING_EFFORT: Dict[str, str] = {"sonnet": "medium", "opus": "medium"}

DEFAULTS: Dict[str, Any] = {
    "tiers": {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"},
    "default_tier": "sonnet",
    "log_path": "",
    "reasoning_effort": dict(DEFAULT_REASONING_EFFORT),
}

_ACTIVE = False
# Set by the router for the request it is routing: whether *this* request can use
# delegate_claude. A session's tool list is fixed when its agent is built, so the
# live Claude switches and the tools the request actually carries can disagree; the
# request is what the conductor sees, so it decides.
_REQUEST_ACTIVE: ContextVar[Optional[bool]] = ContextVar("claude_delegation_request_active", default=None)
# The last availability the router saw, so a flip can drop Hermes's tool-list memo.
_LAST_AVAILABLE: Optional[bool] = None


def is_active() -> bool:
    """Whether Claude delegation may be offered right now.

    Inside a routed request: the request's own answer (a Claude model is switched
    on AND the request offers the tool). Outside one: whether the tool is registered.
    """
    scoped = _REQUEST_ACTIVE.get()
    return _ACTIVE if scoped is None else scoped


@contextmanager
def request_scope(active: bool) -> Iterator[None]:
    token = _REQUEST_ACTIVE.set(bool(active))
    try:
        yield
    finally:
        _REQUEST_ACTIVE.reset(token)


# ---------------------------------------------------------------------------
# Reasoning-effort compatibility bridge
#
# Hermes copies a delegating parent's ``reasoning_config`` verbatim into every
# child it spawns (see ``tools.delegate_tool_config._resolve_child_runtime``).
# That leaves no way for this plugin to give a Claude Sonnet/Opus child its own
# per-tier reasoning effort without either forking Hermes's private resolver or
# mutating shared parent/config state (which would leak across concurrent
# delegations). Instead, install_reasoning_bridge() wraps the *module global*
# Hermes's own call site (tools.delegate_tool._resolve_child_runtime) actually
# calls, and the wrapper only ever substitutes ``reasoning_config`` while a
# ContextVar scope set by the active delegate_claude call is live, and only for
# the exact child it was set for. Outside that narrow window -- including any
# other concurrent delegation, any non-Anthropic child, or any host that lacks
# this private seam -- the original Hermes behavior is untouched.


@dataclass(frozen=True)
class _ReasoningScope:
    """One delegate_claude call's claim on the next matching child's reasoning.

    ``parent`` is compared by identity, so unrelated concurrent delegations from
    other parents (or other tasks under the same parent) never match.
    """

    parent: Any
    tier: str
    model: str
    reasoning_config: Dict[str, Any]
    # The resolver runs synchronously while the scope is live.  A list keeps the
    # counter mutable despite this scope being frozen and prevents a host seam
    # that never reaches the wrapper from being logged as a successful effort run.
    applied: list[int] = field(default_factory=lambda: [0])


_REASONING_BRIDGE_STATE_KEY = "_hermes_model_router_claude_reasoning_state"


def _new_reasoning_bridge_state() -> Any:
    state = types.ModuleType(_REASONING_BRIDGE_STATE_KEY)
    state.scope = ContextVar("claude_delegation_reasoning_scope", default=None)
    state.lock = threading.Lock()
    state.installed = False
    state.reason = "not yet installed"
    state.original = None
    state.wrapper = None
    return state


# ``sys.modules`` is a per-process identity registry: this holder is a real
# ``types.ModuleType`` (not a plain namespace object) so code that walks
# ``sys.modules`` expecting module objects -- ``importlib.reload``-style
# tooling, some warning/pickle helpers, ``inspect.getmodule`` -- does not trip
# over it. A newer copy backfills fields it knows about on an older holder; a
# non-holder or incompatible field type is still unsupported rather than
# silently replaced.
_REASONING_BRIDGE_STATE: Any = sys.modules.setdefault(
    _REASONING_BRIDGE_STATE_KEY,
    _new_reasoning_bridge_state(),
)
for _state_field, _state_default in (
    ("scope", lambda: ContextVar("claude_delegation_reasoning_scope", default=None)),
    ("lock", threading.Lock),
    ("installed", lambda: False),
    ("reason", lambda: "not yet installed"),
    ("original", lambda: None),
    ("wrapper", lambda: None),
):
    _REASONING_BRIDGE_STATE.__dict__.setdefault(_state_field, _state_default())

# These aliases retain the test-visible names. They are safe to alias directly
# because ContextVar and Lock are already shared objects; plain state stays on
# the per-process holder above for reloads and alternate imports.
_REASONING_SCOPE: ContextVar[Optional[_ReasoningScope]] = _REASONING_BRIDGE_STATE.scope
_REASONING_BRIDGE_LOCK = _REASONING_BRIDGE_STATE.lock


def __getattr__(name: str) -> Any:
    """Keep private bridge-state reads compatible while the holder owns mutation."""
    state_fields = {
        "_REASONING_BRIDGE_INSTALLED": "installed",
        "_REASONING_BRIDGE_REASON": "reason",
        "_REASONING_BRIDGE_ORIGINAL": "original",
        "_REASONING_BRIDGE_WRAPPER": "wrapper",
    }
    try:
        return getattr(_REASONING_BRIDGE_STATE, state_fields[name])
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


@contextmanager
def reasoning_scope(parent: Any, tier: str, model: str, reasoning_config: Dict[str, Any]) -> Iterator[_ReasoningScope]:
    """Claim the next matching Anthropic child's reasoning_config for this call.

    Held for the duration of the ``delegate_task`` call inside ``_dispatch``:
    the wrapped ``_resolve_child_runtime`` only applies ``reasoning_config``
    when it sees a call whose ``parent_agent``/``model``/provider match this
    scope, so the claim cannot leak onto some other concurrent delegation.
    """
    scope = _ReasoningScope(parent, tier, model, dict(reasoning_config))
    token = _REASONING_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _REASONING_SCOPE.reset(token)


def _wrap_resolve_child_runtime(original: Callable[..., Any]) -> Callable[..., Any]:
    signature = inspect.signature(original)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        scope = _REASONING_SCOPE.get()
        if scope is None:
            return result
        try:
            bound = signature.bind_partial(*args, **kwargs)
        except TypeError:
            return result
        arguments = bound.arguments
        parent_agent = arguments.get("parent_agent")
        override_provider = arguments.get("override_provider")
        if parent_agent is not scope.parent or override_provider != "anthropic":
            return result
        model = None
        if isinstance(result, dict):
            model = result.get("model")
        if model is None:
            model = arguments.get("model")
        if model != scope.model:
            return result
        if isinstance(result, dict):
            scope.applied[0] += 1
            return dict(result, reasoning_config=dict(scope.reasoning_config))
        return result

    wrapper.__model_router_original__ = original
    return wrapper


def _validate_reasoning_bridge_seam() -> Tuple[bool, str, Optional[Any], Optional[Callable[..., Any]]]:
    """Check whether the host seam this bridge wraps still looks the way it must.

    Side-effect-free: only imports and inspects ``tools.delegate_tool`` /
    ``tools.delegate_tool_config`` (both cheap and side-effect-free imports on
    this host), never assigns anything. Returns
    ``(ok, reason, delegate_tool_module_or_None, current_resolver_or_None)``;
    the last two let a caller that wants to actually install reuse the same
    lookup instead of re-importing.

    This is the single source of truth for "is the seam compatible" -- both
    ``install_reasoning_bridge()`` (which then also wraps it) and
    ``reasoning_bridge_compatibility()`` (which never does) call this, so the
    two validations cannot drift apart.
    """
    try:
        import sys as _sys

        # Read back through sys.modules rather than `import ... as name`: once a
        # submodule has been imported, a plain dotted import can bind through the
        # parent package's cached attribute instead of a swapped-in sys.modules
        # entry (as tests do via unittest.mock.patch.dict(sys.modules, ...)).
        # _import_hermes also finds them when the venv's install map is stale
        # (the standalone dashboard has no PYTHONPATH to fall back on).
        usage_guard._import_hermes("tools.delegate_tool")
        usage_guard._import_hermes("tools.delegate_tool_config")
        delegate_tool = _sys.modules["tools.delegate_tool"]
        delegate_tool_config = _sys.modules["tools.delegate_tool_config"]
    except Exception as exc:
        return False, f"Hermes delegation API not importable ({type(exc).__name__}: {exc})", None, None

    current = getattr(delegate_tool, "_resolve_child_runtime", None)
    if current is None:
        return False, "tools.delegate_tool has no _resolve_child_runtime to wrap", delegate_tool, None
    if not callable(current):
        return False, "tools.delegate_tool._resolve_child_runtime is not callable", delegate_tool, None

    already_wrapped = (
        _REASONING_BRIDGE_STATE.original is not None
        and current is _REASONING_BRIDGE_STATE.wrapper
    )
    if already_wrapped:
        return True, "", delegate_tool, current

    # A foreign wrapper -- e.g. installed by an earlier copy of this same module
    # after a reload/re-import -- carries its own claim on the seam via
    # ``__model_router_original__``. Unwrap it before comparing: the underlying
    # original, not the foreign wrapper, is the real host function, so install
    # can re-wrap that original with THIS module's wrapper.
    current_original = getattr(current, "__model_router_original__", current)

    original_from_config = getattr(delegate_tool_config, "_resolve_child_runtime", None)
    if current_original is not original_from_config:
        return False, (
            "tools.delegate_tool._resolve_child_runtime is not the same callable as "
            "tools.delegate_tool_config._resolve_child_runtime; the host seam has moved"
        ), delegate_tool, current

    try:
        signature = inspect.signature(current_original)
    except (TypeError, ValueError) as exc:
        return False, f"could not inspect _resolve_child_runtime's signature: {exc}", delegate_tool, current

    required = ("parent_agent", "model", "override_provider")
    missing = [name for name in required if name not in signature.parameters]
    if missing:
        return False, "_resolve_child_runtime lacks " + ", ".join(missing), delegate_tool, current

    return True, "", delegate_tool, current


def reasoning_bridge_compatibility() -> Tuple[bool, str]:
    """Whether THIS host's seam is compatible with the reasoning-effort bridge.

    Side-effect-free: never installs, wraps or mutates anything, including the
    bridge's own module globals -- safe to call from a process (e.g. the
    standalone dashboard) that must never construct agents or otherwise touch
    Hermes's delegation machinery. Runs exactly the same checks
    ``install_reasoning_bridge()`` does, via ``_validate_reasoning_bridge_seam()``,
    so "the seam is compatible" and "the wrapper installed cleanly" cannot
    silently diverge.
    """
    ok, reason, _delegate_tool, _current = _validate_reasoning_bridge_seam()
    return ok, reason


def install_reasoning_bridge() -> Tuple[bool, str]:
    """Idempotently install the reasoning-effort bridge onto the real host seam.

    Safe to call repeatedly (e.g. once per registration): a prior successful
    install is a no-op, and a prior failure is retried since the host may have
    changed (mainly relevant to tests that swap ``sys.modules`` entries).

    The bridge state is process-wide so a plugin reload observes and uses the
    same wrapper, lock, and active scope rather than creating a disconnected
    bridge that can silently lose a child's configured reasoning effort.
    """
    state = _REASONING_BRIDGE_STATE
    with _REASONING_BRIDGE_LOCK:
        delegate_tool = sys.modules.get("tools.delegate_tool")
        if (state.installed and state.wrapper is not None
                and delegate_tool is not None
                and getattr(delegate_tool, "_resolve_child_runtime", None) is state.wrapper):
            return True, ""
        ok, reason, delegate_tool, current = _validate_reasoning_bridge_seam()
        if not ok:
            state.installed = False
            state.reason = reason
            return False, state.reason

        already_wrapped = state.original is not None and current is state.wrapper
        if already_wrapped:
            state.installed = True
            state.reason = ""
            return True, ""

        original = getattr(current, "__model_router_original__", current)
        try:
            wrapper = _wrap_resolve_child_runtime(original)
            setattr(delegate_tool, "_resolve_child_runtime", wrapper)
        except Exception as exc:
            state.installed = False
            state.reason = f"installing the reasoning bridge failed: {type(exc).__name__}: {exc}"
            return False, state.reason

        state.original = original
        state.wrapper = wrapper
        state.installed = True
        state.reason = ""
        return True, ""


def reasoning_bridge_status() -> Tuple[bool, str]:
    """Whether the reasoning-effort bridge is usable right now, and why not when it isn't.

    ``(True, "")`` when THIS process already installed it *and* the host's
    ``tools.delegate_tool._resolve_child_runtime`` is still the installed
    wrapper -- a later replacement (a host refactor, a reload, another copy's
    reinstall) means the bridge is no longer the seam actually in effect, even
    though ``state.installed`` was never reset. Otherwise falls back to
    ``reasoning_bridge_compatibility()``'s side-effect-free probe, so a process
    that queries status before ever installing (the standalone dashboard)
    still reports "available" whenever the host seam this bridge needs is
    actually compatible -- "available" means "the host seam is compatible",
    not "this process installed the wrapper".
    """
    state = _REASONING_BRIDGE_STATE
    if state.installed:
        delegate_tool = sys.modules.get("tools.delegate_tool")
        current = getattr(delegate_tool, "_resolve_child_runtime", None) if delegate_tool is not None else None
        if current is state.wrapper:
            return True, ""
        return False, (
            "tools.delegate_tool._resolve_child_runtime is no longer the installed "
            "reasoning-effort wrapper; the host seam moved after install"
        )
    return reasoning_bridge_compatibility()


def _reset_reasoning_bridge_for_tests() -> None:
    """Test-only: restore the real host's ``_resolve_child_runtime`` and bridge state.

    Never leaves a wrapped host function installed for later, unrelated test
    modules or a live process.
    """
    state = _REASONING_BRIDGE_STATE
    with _REASONING_BRIDGE_LOCK:
        original = state.original
        if original is not None:
            try:
                import tools.delegate_tool as delegate_tool
                if getattr(delegate_tool, "_resolve_child_runtime", None) is state.wrapper:
                    delegate_tool._resolve_child_runtime = original
            except Exception:
                pass
        state.original = None
        state.wrapper = None
        state.installed = False
        state.reason = "not yet installed"


# Hermes's Tool Search defers plugin tools: a live parent request carries only the
# tool_search/tool_describe/tool_call bridge, and delegate_claude is reached through
# tool_call. Whether it is in that session's deferred scope follows tool_available().
_BRIDGE_CALL_NAMES = frozenset({"tool_call", "mcp__tool_call"})


def offered(tool_names: Iterable[str]) -> bool:
    """Whether a request can reach delegate_claude: listed itself, or through tool_call."""
    names = set(tool_names)
    return bool(names & ({TOOL_NAME, f"mcp__{TOOL_NAME}"} | _BRIDGE_CALL_NAMES))


def delegation_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The ``claude_delegation`` block with defaults filled in.

    The retired ``enabled`` flag is dropped: availability comes from ``callable``.
    """
    raw = (cfg or {}).get("claude_delegation")
    raw = raw if isinstance(raw, dict) else {}
    merged = deepcopy(DEFAULTS)
    for key, value in raw.items():
        if key == "enabled":
            continue
        if key in ("tiers",):
            if isinstance(value, dict):
                merged[key] = {**DEFAULTS[key], **value}
        else:
            merged[key] = value
    return merged


def tier_model(tier: str, cfg: Dict[str, Any]) -> str:
    return str(delegation_config(cfg)["tiers"].get(tier) or "").strip()


def reasoning_effort_config(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Normalized editable per-tier reasoning levels: only ``sonnet`` and ``opus``.

    Unset or invalid values (including ``haiku``, which is not editable here)
    fall back to the safe default rather than raising or propagating garbage
    into a child's ``reasoning_config``.
    """
    raw = delegation_config(cfg).get("reasoning_effort")
    raw = raw if isinstance(raw, dict) else {}
    values = dict(DEFAULT_REASONING_EFFORT)
    for tier in EDITABLE_REASONING_TIERS:
        value = raw.get(tier)
        if isinstance(value, str) and value.strip().casefold() in REASONING_LEVELS:
            values[tier] = value.strip().casefold()
    return values


def target_names(cfg: Dict[str, Any]) -> Tuple[str, ...]:
    """Router target names for every tier that has a model configured."""
    return tuple(sorted(TARGET_FOR_TIER[tier] for tier in TIERS if tier_model(tier, cfg)))


def target_for_model(model: str, cfg: Dict[str, Any]) -> Optional[str]:
    if not model:
        return None
    for tier in TIERS:
        if tier_model(tier, cfg) == model:
            return TARGET_FOR_TIER[tier]
    return None


def host_check() -> Tuple[bool, str]:
    """Whether this Hermes still has the two internals Claude delegation stands on.

    ``credentials_cfg`` is commented "internal callers only" upstream, and the
    active-parent lookup is not on the plugin context. If either moves,
    delegate_claude must not register rather than fail at call time.
    """
    try:
        usage_guard._import_hermes("tools.delegate_tool")
        usage_guard._import_hermes("agent.subagent_lifecycle")
        from tools.delegate_tool import delegate_task
        from agent.subagent_lifecycle import get_active_subagent_parent  # noqa: F401
    except Exception as exc:
        return False, f"Hermes delegation API not importable ({type(exc).__name__}: {exc})"
    parameters = inspect.signature(delegate_task).parameters
    missing = [name for name in ("tasks", "parent_agent", "credentials_cfg") if name not in parameters]
    if missing:
        return False, "delegate_task lacks " + ", ".join(missing)
    return True, ""


def availability_block(cfg: Dict[str, Any]) -> str:
    """Why ``delegate_claude`` must not be offered now, or "" when it may be.

    Config only, so it is cheap enough to run on every tool-list build. Claude is
    available exactly while one of its models is switched on in ``callable``; the
    router's ``_load_config`` has already turned a legacy ``workflow`` into those
    switches.
    """
    switches = cfg.get("callable") or {}
    if not any(switches.get(target) is True for target in TARGET_FOR_TIER.values()):
        return "every Claude target is switched off in `callable`"
    return ""


def tool_available() -> bool:
    """delegate_claude's check_fn: Hermes offers the tool only while this is True."""
    from . import _load_config

    return availability_block(_load_config()) == ""


def note_availability(available: bool) -> None:
    """Drop Hermes's memoized tool list when availability flips.

    ``model_tools`` memoizes whole tool lists without re-running check_fns, so
    flipping the Claude switches would otherwise reach new sessions only after a
    restart.
    ``_clear_tool_defs_cache`` is private upstream; without it, new sessions
    still follow the switch once the memo is rebuilt for another reason.
    """
    global _LAST_AVAILABLE
    previous, _LAST_AVAILABLE = _LAST_AVAILABLE, bool(available)
    if previous is None or previous == _LAST_AVAILABLE:
        return
    try:
        import model_tools

        clear = getattr(model_tools, "_clear_tool_defs_cache", None)
        if callable(clear):
            clear()
    except Exception as exc:
        _logger.debug("claude_delegation: could not clear the host tool-list memo: %s", exc)


def _uncached(fn: Callable[[], bool]) -> Callable[[], bool]:
    """Exempt a check_fn from Hermes's 30-second TTL cache, where the host supports it."""
    try:
        from tools.registry import no_cache_check_fn
    except Exception:
        return fn
    return no_cache_check_fn(fn)


# ---------------------------------------------------------------------------
# The delegate_claude tool

TOOL_NAME = "delegate_claude"
_DESCRIPTION = (
    "Spawn subagents on the Claude subscription -- a separate quota from delegate_task, which runs its "
    "workers on Codex. Same tasks shape as delegate_task, plus one tier for the whole call: \"haiku\" for "
    "quick lookups and exploration, \"sonnet\" (the default) as the everyday worker, \"opus\" for hard or "
    "consequential work. Use it when the router's note recommends a Claude target or the work needs Claude. "
    "Top-level calls run in the background and report back like delegate_task; list, steer or stop Claude "
    "children with delegate_task(action=...)."
)
_FALLBACK_TASKS: Dict[str, Any] = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "What this subagent should accomplish. Be specific "
                                                      "and self-contained -- it knows nothing of your conversation."},
            "context": {"type": "string", "description": "Background this child needs: file paths, error "
                                                         "messages, constraints."},
        },
        "required": ["goal"],
    },
}
_MODEL_HIDDEN_TASK_FIELDS = ("acp_command", "acp_args")
_AUDIT_LOCK = threading.Lock()


def _host_delegate_task() -> Callable[..., str]:
    from tools.delegate_tool import delegate_task
    return delegate_task


def _host() -> Tuple[Callable[..., str], Callable[[], Any]]:
    from agent.subagent_lifecycle import get_active_subagent_parent
    return _host_delegate_task(), get_active_subagent_parent


def _independent_completions() -> bool:
    try:
        from tools.delegate_tool_config import _get_independent_completions
        return bool(_get_independent_completions())
    except Exception:
        return False


def _tasks_schema() -> Dict[str, Any]:
    """delegate_task's own ``tasks`` item shape, so the two tools cannot drift apart."""
    try:
        from tools.delegate_tool import DELEGATE_TASK_SCHEMA
        tasks = deepcopy(DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"])
    except Exception:
        tasks = deepcopy(_FALLBACK_TASKS)
    if not _independent_completions():
        ((tasks.get("items") or {}).get("properties") or {}).pop("group", None)
    tasks["description"] = (
        "One entry per Claude worker. Entries run in parallel, all on the tier chosen for this call."
    )
    return tasks


def build_schema(cfg: Dict[str, Any]) -> Dict[str, Any]:
    default_tier = delegation_config(cfg).get("default_tier") or "sonnet"
    return {
        "name": TOOL_NAME,
        "description": _DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": _tasks_schema(),
                "tier": {
                    "type": "string",
                    "enum": list(TIERS),
                    "description": f"Claude tier for every task in this call. Default \"{default_tier}\".",
                },
            },
            "required": ["tasks"],
        },
    }


def _error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


def next_codex_route(target: str, cfg: Dict[str, Any]) -> str:
    """The Codex tier to name when Claude delegation cannot take this target's work."""
    from . import _is_callable_tier, _is_routable_tier, _peers_for, _account_of, worker_admission

    def usable(name: str) -> bool:
        return (bool(name) and _is_routable_tier(name, cfg) and _is_callable_tier(name, cfg)
                and not worker_admission.refusal(_account_of(name, cfg), name, cfg))

    chain = cfg.get("fallbacks") or {}
    seen, current = {target}, target
    for _ in range(4):
        following = str(chain.get(current) or "")
        if not following or following in seen:
            break
        if usable(following):
            return following
        seen.add(following)
        current = following
    for peer in _peers_for(target, cfg):
        if usable(peer):
            return peer
    default = str(cfg.get("default_model") or "")
    return default if usable(default) else ""


def _pointer(target: str, cfg: Dict[str, Any], *, claude_ok: bool) -> str:
    from . import _is_callable_tier, _peers_for

    options = []
    if claude_ok:
        for peer in _peers_for(target, cfg):
            if peer in TIER_FOR_TARGET and _is_callable_tier(peer, cfg):
                options.append(f'delegate_claude with tier "{TIER_FOR_TARGET[peer]}"')
                break
    codex = next_codex_route(target, cfg)
    options.append(f"delegate_task with a goal prefixed [{codex}]" if codex else "delegate_task")
    return "Use " + " or ".join(options) + " instead."


def _unavailable(target: str, cfg: Dict[str, Any]) -> str:
    from . import _tier_cooldown_remaining

    if (cfg.get("callable") or {}).get(target) is not True:
        return "switched off in the dashboard"
    remaining = _tier_cooldown_remaining(target, cfg)
    return f"cooling down for another {int(remaining // 60) + 1} min" if remaining > 0 else ""


def _strip_hidden(tasks: Any) -> Any:
    if not isinstance(tasks, list):
        return tasks
    return [
        {k: v for k, v in task.items() if k not in _MODEL_HIDDEN_TASK_FIELDS} if isinstance(task, dict) else task
        for task in tasks
    ]


ACCOUNT = "anthropic"


def _log(cfg: Dict[str, Any], entry: Dict[str, Any]) -> None:
    configured = str(delegation_config(cfg).get("log_path") or "").strip()
    if not configured:
        return
    line = {"timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), **entry}
    try:
        path = hermes_path(configured)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOCK, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _audit(cfg: Dict[str, Any], parent: Any, requested: str, used: str,
           outcome: "usage_guard.GuardOutcome", result: str, message: str = "") -> None:
    # No "tier" key on purpose: the router's per-account load counts lines by tier,
    # and the children's own calls are already counted through the middleware.
    entry = {
        "event": TOOL_NAME,
        "tier_requested": requested,
        "tier_used": used,
        "target": TARGET_FOR_TIER.get(used, ""),
        "model": tier_model(used, cfg),
        "usage": outcome.usage,
        "outcome": result,
    }
    session_id = str(getattr(parent, "session_id", "") or "")
    turn_id = str(getattr(parent, "_current_turn_id", "") or "")
    if session_id:
        entry["session_id"] = session_id
    if turn_id:
        entry["turn_id"] = turn_id
    if outcome.adjusted:
        entry["adjusted"] = outcome.adjusted
    if message:
        entry["message"] = message
    _log(cfg, entry)


def _raw_error(raw: Any) -> Optional[str]:
    """The "error" text when ``raw`` parses as a JSON object carrying one, else None."""
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    return str(error) if error else None


def _annotate(raw: Any, tier: str, outcome: "usage_guard.GuardOutcome",
              reasoning_effort: Optional[str] = None) -> Any:
    try:
        payload = json.loads(raw)
    except Exception:
        return raw
    if not isinstance(payload, dict):
        return raw
    payload["claude_tier"] = tier
    if outcome.adjusted:
        payload["tier_adjusted"] = outcome.adjusted
    if outcome.usage == "unknown":
        payload["usage"] = "unknown"
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    return json.dumps(payload, ensure_ascii=False)


def _active_parent() -> Any:
    """The active Hermes parent, without going through ``_host()``.

    Used only to attribute the availability-block refusal's audit line to a
    session/turn before the tier is even known to be a legitimate request --
    unlike the rest of ``_dispatch``, which reuses the single ``_host()`` call.
    """
    try:
        from agent.subagent_lifecycle import get_active_subagent_parent
        return get_active_subagent_parent()
    except Exception:
        return None


def _dispatch(args: Dict[str, Any]) -> str:
    from . import _load_config

    cfg = _load_config()
    settings = delegation_config(cfg)
    requested = str(args.get("tier") or settings.get("default_tier") or "sonnet").strip().casefold()
    if availability_block(cfg):
        message = ("Claude delegation is off: every Claude model is switched off in Settings. "
                   "Use delegate_task, which runs on the Codex route.")
        _audit(cfg, _active_parent(), requested, requested,
              usage_guard.GuardOutcome(TARGET_FOR_TIER.get(requested, "")), "refused", message)
        return _error(message)
    if requested not in TIERS:
        return _error(f"Unknown tier {requested!r}; use one of: {', '.join(TIERS)}.")
    delegate_task, active_parent = _host()
    parent = active_parent()
    if parent is None:
        return _error("delegate_claude must be called from an agent turn; no active Hermes parent was found.")

    guarded = usage_guard.apply(ACCOUNT, TARGET_FOR_TIER[requested], cfg, usage_guard.read(ACCOUNT, cfg))
    tier = TIER_FOR_TARGET.get(guarded.tier, requested)
    outcome = replace(guarded, tier=tier,
                      adjusted=f"{requested}→{tier} (weekly usage {guarded.usage})" if guarded.adjusted else "")
    if outcome.refused:
        message = f"{outcome.refused} {_pointer(TARGET_FOR_TIER[requested], cfg, claude_ok=False)}"
        _audit(cfg, parent, requested, requested, outcome, "refused", message)
        return _error(message)
    target = TARGET_FOR_TIER[tier]
    unavailable = _unavailable(target, cfg)
    if unavailable:
        message = f"Claude tier \"{tier}\" ({target}) is {unavailable}. {_pointer(target, cfg, claude_ok=True)}"
        _audit(cfg, parent, requested, tier, outcome, "refused", message)
        return _error(message)
    model = tier_model(tier, cfg)
    if not model:
        message = f"Claude tier \"{tier}\" has no model under claude_delegation.tiers."
        _audit(cfg, parent, requested, tier, outcome, "refused", message)
        return _error(message)

    # Admission/step-down above is the single owner for this legacy entrypoint.
    # Keep an explicit raw host callable: the adapter must not call this handler
    # again or route through the delegate_task tool-execution middleware.
    raw_delegate_task = delegate_task
    def delegate_task(**kwargs: Any) -> Any:
        from .execution_adapters import native_legacy_dispatch
        evidence = {"claude_tier": tier, "tier_adjusted": outcome.adjusted}
        if tier == "haiku":
            evidence["reasoning_effort"] = "not_applicable"
        return native_legacy_dispatch(lambda: raw_delegate_task(**kwargs), parent=parent,
            tasks=kwargs.get("tasks"), transport="hermes_claude", evidence=evidence)

    # Haiku has no extended-thinking support (see agent.anthropic_adapter.build_anthropic_kwargs),
    # so the reasoning-effort bridge is simply irrelevant to it: a Haiku call never sets a scope
    # and must delegate normally even when the bridge is unavailable on this host.
    reasoning_applied: Optional[bool] = None
    if tier == "haiku":
        raw = delegate_task(
            goal=args.get("goal"),
            context=args.get("context"),
            tasks=_strip_hidden(args.get("tasks")),
            parent_agent=parent,
            # Hermes's own rule (run_agent._dispatch_delegate_task): background at the
            # top level, synchronous for an orchestrator child that needs its results.
            background=not getattr(parent, "_delegate_depth", 0) > 0,
            credentials_cfg={"provider": "anthropic", "model": model, "fallback_providers": []},
        )
    else:
        bridge_ok, bridge_reason = install_reasoning_bridge()
        if not bridge_ok:
            message = f"Claude reasoning effort is unavailable: {bridge_reason}"
            _audit(cfg, parent, requested, tier, outcome, "refused", message)
            return _error(message)
        from hermes_constants import parse_reasoning_effort

        level = reasoning_effort_config(cfg).get(tier)
        if level is None:
            message = f"Claude reasoning effort for {tier} is not configured"
            _audit(cfg, parent, requested, tier, outcome, "refused", message)
            return _error(message)
        reasoning_config = parse_reasoning_effort(level)
        if reasoning_config is None:
            message = f"Claude reasoning effort for {tier} is invalid"
            _audit(cfg, parent, requested, tier, outcome, "refused", message)
            return _error(message)
        with reasoning_scope(parent, tier, model, reasoning_config) as scope:
            raw = delegate_task(
                goal=args.get("goal"),
                context=args.get("context"),
                tasks=_strip_hidden(args.get("tasks")),
                parent_agent=parent,
                # Hermes's own rule (run_agent._dispatch_delegate_task): background at the
                # top level, synchronous for an orchestrator child that needs its results.
                background=not getattr(parent, "_delegate_depth", 0) > 0,
                credentials_cfg={"provider": "anthropic", "model": model, "fallback_providers": []},
            )
        reasoning_applied = scope.applied[0] > 0
    error_message = _raw_error(raw)
    if error_message is not None:
        _audit(cfg, parent, requested, tier, outcome, "error", error_message[:300])
    elif reasoning_applied is False:
        message = "reasoning effort not applied: host seam did not see the child"
        _audit(cfg, parent, requested, tier, outcome, "error", message)
        return _annotate(raw, tier, outcome, reasoning_effort="not applied")
    else:
        _audit(cfg, parent, requested, tier, outcome, "lowered" if outcome.adjusted else "ran")
    return _annotate(raw, tier, outcome)


def handle_delegate_claude(args: Dict[str, Any], **_kwargs: Any) -> str:
    """Tool handler. Never raises into the turn: every failure is a tool error."""
    try:
        return _dispatch(args if isinstance(args, dict) else {})
    except Exception as exc:
        return _error(f"delegate_claude failed: {type(exc).__name__}: {exc}")


def _exempt_from_sequential_deadline() -> bool:
    """Exempt delegate_claude from Hermes's 420s sequential tool deadline.

    A background delegate_task batch runs synchronously when the async pool is
    full or the session can't take async completions, and such a batch can run
    long. delegate_task and manage_connections are already exempt
    (``agent.tool_executor._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS``); without the same
    exemption a real delegate_claude batch times out and orphans its workers.
    Never raises: a host that no longer exposes this set must not block
    registration.
    """
    try:
        import importlib
        tool_executor = importlib.import_module("agent.tool_executor")
    except Exception as exc:
        _logger.warning(
            "claude_delegation: could not exempt delegate_claude from the sequential tool deadline: %s", exc
        )
        return False
    existing = getattr(tool_executor, "_SEQUENTIAL_DEADLINE_EXEMPT_TOOLS", None)
    if not isinstance(existing, frozenset):
        _logger.warning(
            "claude_delegation: could not exempt delegate_claude from the sequential tool deadline: "
            "_SEQUENTIAL_DEADLINE_EXEMPT_TOOLS is missing or not a frozenset"
        )
        return False
    tool_executor._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS = frozenset(existing | {TOOL_NAME})
    return True


def register(ctx: Any, cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Register delegate_claude whenever the host can carry it.

    Registered even while every Claude model is switched off: its check_fn decides,
    per tool-list build, whether Hermes offers it, so the switches need no restart.
    """
    global _ACTIVE
    if cfg is None:
        from . import _load_config
        cfg = _load_config()
    ok, reason = host_check()
    if not ok:
        _ACTIVE = False
        _logger.info("claude_delegation: delegate_claude not registered: %s", reason)
        _log(cfg, {"event": "registration", "registered": False, "reason": reason})
        return False
    try:
        handle = ctx.register_tool(name=TOOL_NAME, toolset="delegation", schema=build_schema(cfg),
                                   handler=handle_delegate_claude, check_fn=_uncached(tool_available),
                                   description=_DESCRIPTION, emoji="🪶")
    except Exception as exc:
        _ACTIVE = False
        reason = f"registering delegate_claude failed: {exc}"
        _logger.warning("claude_delegation: %s", reason)
        _log(cfg, {"event": "registration", "registered": False, "reason": reason})
        return False
    if handle is None:
        _ACTIVE = False
        reason = "ctx.register_tool returned None"
        _logger.info("claude_delegation: delegate_claude not registered: %s", reason)
        _log(cfg, {"event": "registration", "registered": False, "reason": reason})
        return False
    _ACTIVE = True
    _exempt_from_sequential_deadline()
    _log(cfg, {"event": "registration", "registered": True, "reason": "",
               "available": availability_block(cfg) == ""})
    return True
