"""Conservative GPT-6 Luna/GPT-5.6 Terra/GPT-6 Sol + Codex-Spark request router."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import unicodedata
from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - Hermes includes PyYAML
    yaml = None

from . import claude_delegation
from . import runtime_capabilities
from . import target_identity
from . import usage_guard
from . import worker_admission
from .hermes_paths import hermes_path

_logger = logging.getLogger("model_router")

_PLUGIN_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _PLUGIN_DIR / "router_config.yaml"
_LOG_LOCK = threading.Lock()
_SHADOW_LOCK = threading.RLock()
_QUOTA_LOCK = threading.Lock()
_SPARK_QUOTA_EXHAUSTED_TURNS: set[str] = set()
_MAX_REMEMBERED_QUOTA_TURNS = 2048

_DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "provider": "openai-codex",
    "models": {
        "luna": "gpt-6-luna",
        "spark": "gpt-5.3-codex-spark",
        "terra": "gpt-5.6-terra",
        "sol": "gpt-6.1-sol",
        "qwen": "qwen3.7-plus",
        "grok": "grok-4.7",
    },
    # Which tiers are callable — togglable from the web dashboard.
    # When a tier is disabled, any route that selected it falls back
    # according to the `fallbacks` map below.
    "callable": {
        "luna": True,
        "spark": True,
        "terra": True,
        "sol": True,
        # Claude needs a Claude subscription and login, so its models ship off;
        # switch them on from the dashboard once logged in. Claude delegation is
        # available exactly while at least one of them is on.
        "opus5": False,
        "sonnet5": False,
        "haiku": False,
        "qwen": True,
        # Needs a SuperGrok subscription (`hermes auth add xai-oauth`), so it
        # ships off; switch it on from the dashboard once logged in.
        "grok": False,
    },
    # When a callable tier is disabled, routes that selected it fall back here.
    "fallbacks": {
        "spark": "luna",
        "luna": "terra",
        "sol": "terra",
        "opus5": "sol",
        "qwen": "terra",
        "grok": "terra",
    },
    # The default model is both the general-purpose route destination and the
    # orchestration owner. Changing it rewrites this file and the Hermes config.
    "default_model": "terra",
    # spark_max_chars / spark_dev_max_chars removed: no automatic Spark route
    # exists for them to bound, so they only advertised a control that was never
    # consulted. Spark is reached via an explicit [spark] label or delegation.
    "thresholds": {
        "luna_max_chars": 700,
        "sol_min_chars": 3500,
    },
    # Keyed by tier, by ``explicit_<tier>`` for a labelled route, and by
    # ``explicit_<tier>_xhigh`` for one that asked to be escalated. Unset keys
    # degrade to the plain tier, so adding a tier here is optional.
    "effort": {
        "luna": "low", "spark": "low", "terra": "medium", "sol": "high", "qwen": "medium",
        "grok": "medium",
        "explicit_sol": "xhigh",
        "explicit_luna_xhigh": "high", "explicit_spark_xhigh": "high",
        "explicit_terra_xhigh": "high", "explicit_sol_xhigh": "xhigh",
    },
    # Provider mapping per tier: which Hermes provider handles each tier.
    "tier_providers": {
        "luna": "openai-codex",
        "spark": "openai-codex",
        "terra": "openai-codex",
        "sol": "openai-codex",
        "opus5": "openai-codex",
        "qwen": "qwen-token",
        "grok": "xai-oauth",
    },
    "quota_fallbacks": {"spark": {"model": "luna", "effort": "medium"}},
    "logging": {
        "enabled": True,
        "path": "~/.hermes/logs/model-router.jsonl",
    },
    "delegation": {"preserve_spark_subagents": True},
    "coding_agent": {
        "enabled": False,
        "tier": "opus5",
        "model": "claude-opus-5-5",
        "default_repo": "",
        "max_turns": 8,
        "max_budget_usd": 5.0,
        "timeout_seconds": 300,
        "lifecycle_path": "~/.hermes/logs/claude-code-bridge.jsonl",
        "reviewer": {"enabled": False, "max_chars": 8000},
        # Delegated read-only Claude review, off by default and independent of
        # ``enabled`` above, which also arms the label-free coding classifier.
        "delegated_review": {"enabled": False, "max_chars": 8000, "models": ["opus", "sonnet"]},
    },
    # A user-facing session has one durable parent.  The router may still
    # classify specialist *workers*, but it must not turn each user message into
    # a cold planner/model handoff.
    "session_policy": {
        "pin_root_parent": True,
    },
    "orchestration": {
        # Automatic planner fan-out is opt-in.  Parent agents delegate only
        # when they identify a genuinely independent bounded worker task.
        "enabled": False,
        "min_chars": 180,
        "max_tasks": 1,
        # Where the forced conductor runs. Unset means default_model: coordination
        # follows the account the session is already on rather than borrowing the
        # answer from a question about leaf routing.
        "conductor": None,
        # Recovery gate for an active Terra tool loop whose initial preflight
        # was missed (for example, a process that loaded an older plugin).
        "rescue_min_calls": 6,
        "path": "~/.hermes/logs/terra-spark-orchestration.jsonl",
    },
    # A short, clear request must not pay for a planner plus a recursive tree of
    # workers.  This guard applies only to bounded low-risk dispatches; complex
    # and consequential work keeps the normal host limits and explicit workflow.
    "task_budget": {
        "enabled": True,
        "low_risk_max_chars": 1200,
        "max_routing_decisions": 1,
        "max_depth": 1,
        "require_evidence_for_second_worker": True,
        "path": "~/.hermes/logs/model-router-task-budget.jsonl",
    },
    # A tier that just rejected a call for quota is not a candidate for the next
    # one. Held on disk because the interactive TUI and the gateway are separate
    # processes: an in-memory note would not be seen by the other one.
    "cooldown": {
        "enabled": True,
        "path": "~/.hermes/state/model-router-cooldowns.json",
        "quota_seconds": 900,
        "allowed_fails": 3,
        "failure_window_seconds": 60,
        "failure_seconds": 60,
    },
    "usage_report": {"enabled": True, "window_seconds": 3600},
    # Targets of comparable strength, on deliberately different accounts. Used
    # to move work off a loaded or cooling target rather than queueing on it.
    # "Comparable in strength" is stated to the conductor verbatim, so a wrong
    # grouping is an instruction to misroute: sonnet5 sat in the light group and
    # the conductor duly substituted Luna for it whenever the Codex account
    # looked loaded -- implementation leaves on the 700-char, low-effort tier.
    # Luna's real peer is Spark; sonnet5's are the heavy implementation targets.
    "peer_groups": {
        "heavy": ["terra", "opus5", "grok", "qwen", "sonnet5"],
        "light": ["luna", "spark"],
    },
    "shadow": {
        "enabled": False,
        "limit": 10,
        "path": "~/.hermes/logs/spark-shadow-benchmark.jsonl",
    },
}


@dataclass(frozen=True)
class RouteDecision:
    tier: str
    model: str
    reason: str
    effort: str = "medium"
    # Tiers this request was independently eligible for but did not get. The
    # route log records only the winning reason, which hides why an alternative
    # never fires: an eligible-but-never-chosen tier looks identical to one whose
    # preconditions are never met. These signals separate the two.
    vetoed_by: Tuple[str, ...] = ()
    # True when the tier was chosen by policy or a hard capability limit rather
    # than preference. Such a route must not be satisfied by the fallback chain:
    # falling back from it grants exactly the access the decision denied.
    mandatory: bool = False
    # The work kind the gate recognised ("design", "code", "explore", ...). It is
    # what a user preference list is keyed on, and it is recorded in the route log
    # so a surprising route can be traced back to the category that produced it.
    kind: str = ""
    # The external delegation target preferred for this kind, when the preference
    # list names one. The router cannot route across providers, so this travels
    # as advice to the conductor rather than as the route itself.
    prefer_target: str = ""


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# Work kinds a preference list may be keyed on. Kept as an explicit tuple so the
# dashboard, the config validator and the router cannot drift apart on the names.
WORK_KINDS: Tuple[str, ...] = (
    "design", "code", "explore", "review", "sensitive", "critical", "long", "chat", "default",
)


def _preference_list(kind: str, cfg: Dict[str, Any]) -> Tuple[str, ...]:
    """The configured order of preferred tiers for one work kind, or () when unset.

    Unset is meaningful: it means "keep the built-in route", which is why a missing
    or malformed entry never silently becomes an empty preference.
    """
    if not kind:
        return ()
    raw = (cfg.get("preferences") or {}).get(kind)
    if not isinstance(raw, list):
        return ()
    seen: list[str] = []
    for item in raw:
        name = str(item or "").strip().casefold()
        if name and name not in seen:
            seen.append(name)
    return tuple(seen)


def _is_routable_tier(tier: str, cfg: Dict[str, Any]) -> bool:
    """Whether the router itself can serve this tier by rewriting the model name.

    ``route_llm_request`` runs after the provider is chosen, so it can only swap
    models inside its own provider. Anything else (Claude, Qwen) reaches work
    through delegation, never through a route.
    """
    return tier in (cfg.get("models") or {})


def _preferred_route(kind: str, cfg: Dict[str, Any]) -> Optional[str]:
    """First routable+callable tier of the kind's preference list, else None."""
    for tier in _preference_list(kind, cfg):
        if _is_routable_tier(tier, cfg) and _is_callable_tier(tier, cfg):
            return tier
    return None


def _preferred_target(kind: str, cfg: Dict[str, Any]) -> str:
    """First callable EXTERNAL entry of the kind's preference list, else "".

    Ranked above the routable tiers on purpose: if the user put ``opus5`` first for
    design work, the router cannot honour that as a route, but it can tell the
    conductor that design leaves belong on Opus.
    """
    for tier in _preference_list(kind, cfg):
        if _is_routable_tier(tier, cfg):
            continue
        if _is_callable_tier(tier, cfg):
            return tier
    return ""


def _apply_preferences(decision: RouteDecision, cfg: Dict[str, Any]) -> RouteDecision:
    """Overlay the user's per-kind preference onto a built-in decision.

    A configured list is authoritative: it replaces the tier AND the fallback
    chain, and it clears ``mandatory`` because the built-in policy it would have
    enforced is exactly what the user chose to override. With no list configured
    the decision is returned untouched, so the shipped defaults still apply.
    """
    prefs = _preference_list(decision.kind, cfg)
    if not prefs:
        return decision
    target = _preferred_target(decision.kind, cfg)
    tier = _preferred_route(decision.kind, cfg)
    if tier is None or tier == decision.tier:
        # Nothing routable in the list (or it already agrees): keep the route and
        # carry only the delegation advice.
        return replace(decision, prefer_target=target) if target else decision
    try:
        preferred = _decision(tier, f"preferred {decision.kind} route", cfg, kind=decision.kind)
    except (KeyError, ValueError):
        return decision
    return replace(preferred, vetoed_by=decision.vetoed_by, prefer_target=target)


def _resolve_callable_fallback(
    decision: RouteDecision, cfg: Dict[str, Any]
) -> RouteDecision:
    """If the chosen tier is not callable, follow the fallback chain once."""
    # Fail closed: a tier is routable only when the live config explicitly says
    # callable: true and it is not cooling down. Going through
    # ``_is_callable_tier`` rather than reading the flag directly is what makes
    # a cooling tier follow the same path as a disabled one.
    chosen_tier = decision.tier
    if _is_callable_tier(chosen_tier, cfg):
        return decision

    # A configured preference list IS the fallback chain for its kind: the user
    # wrote the order, so walk it before anything built-in. A list with no
    # routable+callable entry (say, only Claude targets while Claude is switched
    # off) has nothing to offer, so the built-in chain below takes over.
    prefs = _preference_list(decision.kind, cfg)
    for tier in prefs:
        if tier != chosen_tier and _is_routable_tier(tier, cfg) and _is_callable_tier(tier, cfg):
            try:
                return _decision(
                    tier, f"preferred {decision.kind} fallback from {chosen_tier}",
                    cfg, kind=decision.kind,
                )
            except (KeyError, ValueError):
                continue

    # A policy route is not a preference. Design work reaches Sol because only
    # Sol may do it, so answering "Sol is unavailable" with Terra performs the
    # work on the tier the rule exists to keep it away from -- and it does so
    # exactly when Sol has run out of quota, which is when the rule matters
    # most. Decline the chain and let the caller fail loudly instead.
    if decision.mandatory:
        return decision

    # Tier is disabled — follow fallback chain (max 3 hops to prevent cycles)
    fallbacks = cfg.get("fallbacks") or {}
    visited = {chosen_tier}
    current = chosen_tier
    for _ in range(3):
        next_tier = fallbacks.get(current)
        if next_tier and next_tier not in visited:
            if _is_callable_tier(next_tier, cfg):
                try:
                    return _decision(next_tier, f"fallback from disabled {chosen_tier}", cfg,
                                     kind=decision.kind)
                except (KeyError, ValueError):
                    break
            visited.add(next_tier)
            current = next_tier

    # No valid fallback found — return original decision unchanged
    return decision


# The route log reaches tens of megabytes; the recent window lives in its tail.
_USAGE_TAIL_BYTES = 1_000_000

_COOLDOWN_LOCK = threading.Lock()
_COOLDOWN_CACHE: Dict[str, Any] = {"key": None, "state": {}}


def _cooldown_path(cfg: Dict[str, Any]) -> Optional[Path]:
    """The shared state file, or None when this config did not name one.

    Deliberately not defaulted to the production path. A component that writes
    to a shared location must take that location from the config it was handed;
    inventing one means any caller with a partial config -- a test, a probe --
    silently writes to the real file and its state leaks into unrelated runs.
    """
    configured = str((cfg.get("cooldown") or {}).get("path") or "").strip()
    return hermes_path(configured) if configured else None


def _read_cooldown_state(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load the shared cooldown file, cached on its own mtime and size.

    Read on every routing decision, so it must not cost a parse per call; it
    must also not go stale, because the process that recorded the cooldown is
    usually not the process that needs to honour it.
    """
    path = _cooldown_path(cfg)
    if path is None:
        return {}
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return {}
    if _COOLDOWN_CACHE.get("key") == key:
        return _COOLDOWN_CACHE["state"]
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        state = state if isinstance(state, dict) else {}
    except Exception:
        state = {}
    _COOLDOWN_CACHE["key"] = key
    _COOLDOWN_CACHE["state"] = state
    return state


def _write_cooldown_state(cfg: Dict[str, Any], state: Dict[str, Any]) -> None:
    path = _cooldown_path(cfg)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        # A cooldown that cannot be persisted must never break routing.
        pass


def _tier_cooldown_remaining(tier: str, cfg: Dict[str, Any]) -> float:
    """Seconds left on this tier's cooldown, or 0.0 when it is available."""
    if not (cfg.get("cooldown") or {}).get("enabled", True):
        return 0.0
    entry = (_read_cooldown_state(cfg).get("tiers") or {}).get(tier)
    if not isinstance(entry, dict):
        return 0.0
    remaining = float(entry.get("until", 0) or 0) - datetime.now(timezone.utc).timestamp()
    return remaining if remaining > 0 else 0.0


def _enter_cooldown(tier: str, cfg: Dict[str, Any], *, seconds: float, reason: str) -> None:
    if not tier or not (cfg.get("cooldown") or {}).get("enabled", True):
        return
    now = datetime.now(timezone.utc).timestamp()
    with _COOLDOWN_LOCK:
        state = dict(_read_cooldown_state(cfg))
        tiers = dict(state.get("tiers") or {})
        current = tiers.get(tier) or {}
        # Never shorten a cooldown already in force: a transient blip arriving
        # during a quota cooldown must not release the tier early.
        until = max(float(current.get("until", 0) or 0), now + float(seconds))
        tiers[tier] = {"until": until, "reason": reason, "recorded_at": now}
        state["tiers"] = tiers
        _write_cooldown_state(cfg, state)


def _reset_hint_seconds(error: BaseException) -> Optional[float]:
    """Seconds until the provider says the quota returns, from the error body.

    Codex answers a usage-limit 429 with ``resets_in_seconds`` and ``resets_at``.
    Benching for a fixed 15 minutes against a three-hour reset is what turns one
    refusal into a loop: the cooldown lapses, the tier is offered again, and the
    next leaf spends its retries rediscovering the same wall.
    """
    text = str(error)
    match = re.search(r"'?\"?resets_in_seconds\"?'?\s*:\s*([0-9]+)", text)
    if match:
        return float(match.group(1))
    match = re.search(r"'?\"?resets_at\"?'?\s*:\s*([0-9]{9,13})", text)
    if match:
        value = float(match.group(1))
        if value > 1e11:  # milliseconds
            value /= 1000.0
        remaining = value - datetime.now(timezone.utc).timestamp()
        return remaining if remaining > 0 else None
    return None


def _account_siblings(tier: str, cfg: Dict[str, Any]) -> Tuple[str, ...]:
    """Other tiers billed to the same account as ``tier``.

    A session/usage quota belongs to the account, not the model, so benching only
    the tier that happened to ask leaves its siblings looking available — and the
    next leaf burns another call learning what this one already established.
    """
    providers = cfg.get("tier_providers") or {}
    account = providers.get(tier)
    if not account:
        return ()
    # Not restricted to routable models: a delegation-only target such as sonnet5
    # shares Opus's account, and benching it is what stops the conductor being
    # advised to send the next leaf into the same exhausted subscription.
    known = set(cfg.get("models") or {}) | set(cfg.get("callable") or {})
    return tuple(
        name for name, owner in providers.items()
        if owner == account and name != tier and name in known
    )


def _record_tier_failure(
    tier: str, cfg: Dict[str, Any], *, quota: bool, error: Optional[BaseException] = None
) -> None:
    """Cool a tier down: at once for quota, or after repeated recent failures."""
    policy = cfg.get("cooldown") or {}
    if not tier or not policy.get("enabled", True):
        return
    if quota:
        # Prefer what the provider actually said over the configured guess, capped so
        # a malformed or absurd hint cannot bench a tier for a day.
        hint = _reset_hint_seconds(error) if error is not None else None
        configured = float(policy.get("quota_seconds", 900) or 900)
        cap = float(policy.get("quota_max_seconds", 21600) or 21600)
        seconds = min(hint, cap) if hint else configured
        reason = "quota exhausted (provider reset)" if hint else "quota exhausted"
        _enter_cooldown(tier, cfg, seconds=seconds, reason=reason)
        # A usage quota is the account's, not the model's.
        for sibling in _account_siblings(tier, cfg):
            if _tier_cooldown_remaining(sibling, cfg) < seconds:
                _enter_cooldown(
                    sibling, cfg, seconds=seconds,
                    reason=f"{reason}; shares an account with {tier}",
                )
        return
    now = datetime.now(timezone.utc).timestamp()
    window = float(policy.get("failure_window_seconds", 60) or 60)
    allowed = max(1, int(policy.get("allowed_fails", 3) or 3))
    with _COOLDOWN_LOCK:
        state = dict(_read_cooldown_state(cfg))
        failures = dict(state.get("failures") or {})
        recent = [float(ts) for ts in (failures.get(tier) or []) if now - float(ts) < window]
        recent.append(now)
        failures[tier] = recent[-allowed:]
        state["failures"] = failures
        _write_cooldown_state(cfg, state)
    if len(recent) >= allowed:
        _enter_cooldown(
            tier, cfg,
            seconds=float(policy.get("failure_seconds", 60) or 60),
            reason=f"{len(recent)} failures within {int(window)}s",
        )


def _is_callable_tier(tier: str, cfg: Dict[str, Any]) -> bool:
    # Live router policy is explicit: missing or malformed entries are disabled.
    if (cfg.get("callable") or {}).get(tier) is not True:
        return False
    # A tier serving 429s is not available, whatever the dashboard says. Routing
    # this through callability means the existing fallback chain and the
    # mandatory-route rule both apply with no further wiring.
    return _tier_cooldown_remaining(tier, cfg) <= 0


def _require_callable(decision: RouteDecision, cfg: Dict[str, Any]) -> RouteDecision:
    """Never emit a route for a tier disabled in the dashboard."""
    resolved = _resolve_callable_fallback(decision, cfg)
    if not _is_callable_tier(resolved.tier, cfg):
        if decision.mandatory:
            cooling = _tier_cooldown_remaining(decision.tier, cfg)
            unavailable = (
                f"cooling down for another {int(cooling)}s" if cooling else "disabled"
            )
            raise RuntimeError(
                f"'{decision.tier}' is required for this request ({decision.reason}) but is "
                f"{unavailable}; no fallback may take its place."
            )
        raise RuntimeError(
            f"No enabled ModelRouter tier is available for requested '{decision.tier}'"
        )
    return resolved


def _usage_step_down(decision: RouteDecision, cfg: Dict[str, Any]) -> RouteDecision:
    """At an account's soft or hard limit, step its heaviest routed tier down.

    Codex is the only account this router routes itself, so this is the Codex
    counterpart of delegate_claude's opus→sonnet. Never touches a mandatory
    decision, never moves work onto an unavailable tier, and never lets a
    malformed usage_guard block disable routing -- any failure here fails open
    and leaves the decision exactly as it arrived.
    """
    if decision.mandatory:
        return decision
    try:
        account = _account_of(decision.tier, cfg)
        if not account or not usage_guard.guarded(account, cfg):
            return decision
        reading = usage_guard.peek(account, cfg)
        outcome = usage_guard.apply(account, decision.tier, cfg, reading)
        if outcome.adjusted:
            label, target = "soft", outcome.tier
            window_reason = f"weekly {outcome.usage}"
        elif outcome.refused:
            # Account closure is enforced before spawning and at worker execution.
            # Moving to another model on the same account cannot rescue it.
            return decision
        else:
            return decision
        if not _is_routable_tier(target, cfg) or not _is_callable_tier(target, cfg):
            return replace(decision, reason=f"{decision.reason}; usage {label} limit: "
                                            f"{decision.tier}→{target} skipped ({target} unavailable)")
        step_reason = f"usage {label} limit: {decision.tier}→{target} ({window_reason})"
        stepped = _decision(target, f"{decision.reason}; {step_reason}", cfg)
        return replace(stepped, kind=decision.kind)
    except Exception:
        _logger.warning("_usage_step_down failed; routing continues without a step-down", exc_info=True)
        return decision


# The Claude models' callable switches. Claude is available exactly while one is on.
_CLAUDE_SWITCHES: Tuple[str, ...] = ("opus5", "sonnet5", "haiku")


def _legacy_claude_verdict(local: Dict[str, Any]) -> Optional[bool]:
    """What a pre-1.21 local file said about Claude, or None when it said nothing.

    ``workflow: codex`` means off and ``workflow: claude_delegation`` means on.
    Any other value, null or blank included, counts as absent, so
    ``claude_delegation.enabled`` then decides if it is a bool (as in 1.19, where
    an unknown workflow ran Claude only through that flag). Read from
    router_config.local.yaml only: the shipped file no longer has either key.
    """
    name = str(local.get("workflow") or "").strip().casefold()
    if name in ("codex", "claude_delegation"):
        return name == "claude_delegation"
    block = local.get("claude_delegation")
    flag = block.get("enabled") if isinstance(block, dict) else None
    return flag if isinstance(flag, bool) else None


def _apply_legacy_claude_switches(cfg: Dict[str, Any], local: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the retired ``workflow`` / ``claude_delegation.enabled`` keys, in memory.

    Off turns every Claude model off whatever the local file says for them. On
    turns on each Claude model the local file does not set itself, so an operator
    who ran Claude delegation keeps it after the shipped default became off. The
    legacy keys are then dropped so no reader acts on them; the file is never
    written here (the dashboard's next save materialises the result).
    """
    verdict = _legacy_claude_verdict(local)
    if verdict is not None:
        switches = dict(cfg.get("callable") or {})
        local_switches = local.get("callable") if isinstance(local.get("callable"), dict) else {}
        for name in _CLAUDE_SWITCHES:
            if verdict is False:
                switches[name] = False
            elif name not in local_switches:
                switches[name] = True
        cfg["callable"] = switches
    cfg.pop("workflow", None)
    block = cfg.get("claude_delegation")
    if isinstance(block, dict) and "enabled" in block:
        cfg["claude_delegation"] = {k: v for k, v in block.items() if k != "enabled"}
    return cfg


def _local_config_path() -> Path:
    """The operator's own settings: git-ignored, beside the shipped router_config.yaml."""
    return _CONFIG_PATH.with_name("router_config.local.yaml")


def _local_overrides() -> Dict[str, Any]:
    """router_config.local.yaml as a mapping; {} when absent or unreadable.

    A broken local file must not take the shipped settings down with it, so it is
    skipped with a warning rather than failing the whole load.
    """
    path = _local_config_path()
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        _logger.warning("router_config.local.yaml ignored: %s", exc)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_config() -> Dict[str, Any]:
    """Built-in defaults, then the shipped router_config.yaml, then router_config.local.yaml."""
    if not _CONFIG_PATH.exists() or yaml is None:
        return _deep_merge({}, _DEFAULT_CONFIG)
    try:
        loaded = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            return _deep_merge({}, _DEFAULT_CONFIG)
        local = _local_overrides()
        return _apply_legacy_claude_switches(_deep_merge(_deep_merge(_DEFAULT_CONFIG, loaded), local), local)
    except Exception:
        return _deep_merge({}, _DEFAULT_CONFIG)


def _normalise(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", without_accents.casefold()).strip()


_DESIGN_WORK = re.compile(
    r"\b(visual\s+design|product\s+design|ui\s*(?:/|and)?\s*ux|ux\s*(?:/|and)?\s*ui|"
    r"ui|ux|css|stylesheet|styling|layout|elrendezes|tipograf|typography|"
    r"wireframe|mockup|figma|design\s+system|brand(?:ing)?|responsive\s+(?:ui|layout|card|component)|"
    r"frontend\s+design|visualis\s+terv(?:ezes)?|felulet(?:et|i)?\s+terv(?:ezes)?|"
    r"look\s+and\s+feel|visual\s+appearance|vizualis\s+megjelenes|arculat|"
    r"(?:color|colour)\s+palette|szinpaletta|font|betutipus)\b"
)
_ACKNOWLEDGEMENT_ONLY = re.compile(
    r"^(?:(?:a|az|this|that)\s+)?(?:(?:ui|ux|design|layout|css|frontend)\s+){0,3}"
    r"(?:(?:nagyon\s+)?(?:jo|szuper|remek|kivalo|nagyszeru|tokeletes)\s+lett|"
    r"(?:koszonom|koszi|thanks|thank\s+you|rendben\s+van|oke|ok|"
    r"jovahagyom|elfogadom|approved|accepted))"
    r"(?:\s*[,!.]\s*(?:(?:nagyon\s+)?(?:jo|szuper|remek|kivalo|nagyszeru|tokeletes)\s+lett|"
    r"koszonom|koszi|thanks|thank\s+you|rendben\s+van|oke|ok|"
    r"jovahagyom|elfogadom|approved|accepted))*[!. ]*$"
)
# The write side of the read-only test. Gaps here are silent: a leaf whose only
# write verb is missing reads as read-only, which is how "[luna] Stabilize,
# correct, test, and commit the dirty foundation" passed a guard designed to
# stop exactly that. "commit" in particular cannot be anything but a write, and
# the Hungarian imperatives were absent altogether even though the goals that
# reach this router are routinely written in Hungarian.
_SPARK_MUTATING_VERBS = (
    r"add|create|implement|modify|change|edit|write|patch|delete|remove|"
    r"deploy|publish|send|restart|configure|install|fix|refactor|javitsd|"
    r"modositsd|hozd\s+letre|torold|telepitsd|allitsd\s+be|"
    r"commit|stabili[sz]e|rewrite|rename|update|upgrade|merge|push|revert|"
    r"apply|eliminate|replace|scaffold|migrate|"
    r"implementald|valositsd\s+meg|keszitsd\s+el|epitsd\s+meg|frissitsd|"
    r"commitold|stabilizald|tavolitsd\s+el|nevezd\s+at|alakitsd\s+at|"
    r"refaktorald|csereld|irasd\s+at"
)
_SPARK_MUTATING_WORK = re.compile(rf"\b({_SPARK_MUTATING_VERBS})\b")
_SPARK_READ_ONLY_WORK = re.compile(
    r"\b(inspect|read|review|audit|report|analy[sz]e|compare|search|find|"
    r"identify|list|check|investigate|research|explore|trace|map|survey|"
    r"enumerate|discover|determine|locate|gather|document|test[- ]case\s+design|"
    r"nezd\s+meg|nezd\s+at|olvasd|ellenorizd|elemezd|jelentsd|keresd|azonositsd|"
    r"hasonlitsd\s+ossze|kutass|tard\s+fel|deritsd\s+ki|vizsgald|tekintsd\s+at|"
    r"gyujtsd\s+ossze|terkepezd\s+fel|merd\s+fel|listazd|allapitsd\s+meg)\b"
)
_SPARK_CONSEQUENTIAL_WORK = re.compile(
    r"\b(production|prod|security|biztonsag|auth(?:entication|orization)?|"
    r"credential|jelszo|password|payment|fizetes|migration|migrate|deploy|"
    r"szerver|server|database|adatbazis)\b"
)


_PLAN_LABEL = re.compile(r"^\s*\[(luna|spark|terra|sol)(?::xhigh)?\](?:\s|$)")
_CLAUDE_REVIEW_LABEL = re.compile(r"^\s*\[(opus|sonnet)5?-review\](?:\s|$)")
# A goal that names an external target the way a Sol or Spark goal names its
# tier. It is not a route and never has been -- the override regex knows only
# the four tiers of the default provider -- so the leaf runs on whatever the
# classifier makes of the rest of the text. The bracket closes on the name, so
# the legitimate [opus5-review] / [sonnet-review] labels do not match.
_EXTERNAL_TARGET_LABEL = re.compile(r"^\s*\[(opus5?|sonnet5?|qwen|grok)\](?:\s|$)", re.I)


def _misdispatched_external_label(text: str, active_model: str, cfg: Dict[str, Any]) -> str:
    """The external target a goal names in its text while running somewhere else.

    Two facts together are proof of a wrong dispatch, and neither is enough
    alone: the goal opens with an external target's name, and the leaf carrying
    it is on one of this provider's models. The dispatcher meant Claude and used
    the prefix mechanism, which only renames a model inside one provider.

    Returns "" when the leaf is already on the named account -- there the prefix
    is redundant, not wrong.
    """
    match = _EXTERNAL_TARGET_LABEL.match(text or "")
    if not match:
        return ""
    name = match.group(1).casefold()
    if name.startswith(("opus", "sonnet")) and not name.endswith("5"):
        name = f"{name}5"
    models = (cfg.get("models") or {})
    # Qwen (and Grok) are the case that made "not one of ours" the wrong test: a tier
    # in ``models`` on a separate provider, unlike the Claude targets, so a
    # correctly dispatched [qwen] leaf would otherwise read as misdispatched.
    if active_model == str(models.get(name, "")):
        return ""
    if active_model not in set(models.values()):
        return ""
    return name


def _is_plan_labelled_worker(text: str) -> bool:
    """True when this text is a delegated worker goal labelled by this router.

    The keyword test predates the planner. It exists because the router once had
    to guess a tier from raw prompt text; a labelled worker goal is instead the
    output of a planner that saw the screenshot, the objective and the repo, and
    that routes design to a [sol] leaf under a schema-enforced contract. Re-deciding
    that with forty keywords overrides a better-informed decision with a worse one
    -- and it cannot even tell the cases apart: `_is_design_request` is true both
    for "identify the layout branches" and for "implement a responsive CSS card".
    """
    return bool(_PLAN_LABEL.match(text))


def _is_design_request(text: str) -> bool:
    return bool(_DESIGN_WORK.search(_normalise(text)))


_EXPLICIT_OPUS_REQUEST = re.compile(
    r"(?:\[(?:opus|opus5)\]|\b(?:let|have)\s+opus\s+(?:work|handle|do)|"
    r"\bopus\s+(?:work|handle|do)|\bopus\s+dolgozzon|\bopussal\b)"
)
_DIRECT_OPUS_UI_RISK = re.compile(
    r"\b(production|prod(?:ra|on|ban|ba|ot)?|deploy|security|biztonsag|auth|oauth|"
    r"credential|jelszo|password|payment|fizetes|billing|database|adatbazis|migration|migracio)\b"
)


def _is_explicit_bounded_opus_ui_request(text: str, cfg: Dict[str, Any]) -> bool:
    """Recognise only a user's explicit, small, low-risk Opus UI request.

    Classification ownership remains Sol; this predicate merely authorises one
    external Opus execution bridge instead of a planner/preflight fan-out.
    """
    policy = ((cfg.get("coding_agent") or {}).get("explicit_ui") or {})
    normalised = _normalise(text)
    return bool(
        policy.get("enabled")
        and _is_design_request(text)
        and _EXPLICIT_OPUS_REQUEST.search(normalised)
        and len(text or "") <= int(policy.get("max_chars", 1200) or 1200)
        and not _DIRECT_OPUS_UI_RISK.search(_normalise(_without_negated_safety_constraints(text)))
    )


def _is_acknowledgement_only(text: str) -> bool:
    """Recognise a closed praise/approval follow-up with no requested action."""
    return bool(_ACKNOWLEDGEMENT_ONLY.fullmatch(_normalise(text)))


# "at commit 7abc123", "the commit it builds on", "base commit": a reference to a
# commit, not an act of committing. Stripped before the write-verb test because
# the goal contract *requires* a base commit, so every well-formed read-only goal
# now names one -- and `commit` was added to the write verbs in the same series
# of changes. The better the goal, the more certainly it read as mutating.
#
# The leading-preposition arm only fires when the determiner sits directly on the
# word, so "the *relevant* commit is 54e23a4" fell through both arms and read as
# an instruction. A copula before the hash carries the reference just as plainly
# as juxtaposition does, and the hash is what makes either one a reference -- so
# the second arm accepts the linking verb rather than the first accepting
# arbitrary filler, which would swallow "fix the parser and commit the change".
_COMMIT_REFERENCE = re.compile(
    r"\b(?:at|base|the|from|on|since|after|before|parent|head|onto|against)\s+commit\b"
    r"|\bcommit\b(?:\s+is|\s+was|\s*[:=])?\s+(?:hash|sha|id|[0-9a-f]{6,40})\b",
    re.I,
)


def _without_commit_references(text: str) -> str:
    """Remove commit *references* so only an instruction to commit is a write."""
    return _COMMIT_REFERENCE.sub(" ", text or "")


# "admin save/update API", "the create/delete endpoints": a write verb used to
# *name* the thing to inspect, not to ask for it. The same shape as the commit
# reference above -- a well-formed read-only goal has to say what it looks at,
# and the things worth looking at are called things like "save/update API", so
# the better the goal, the more certainly it read as mutating.
#
# The artifact noun is what makes it a name, so it is required. The phrase has
# to end there too: "update API and the public DTO" names an endpoint, while
# "update API to v2" is still an instruction and stays a write.
_ARTIFACT_NOUN = (
    r"api|apis|endpoint|endpoints|route|routes|handler|handlers|"
    r"resolver|resolvers|controller|controllers|mutation|mutations|dto|dtos"
)
_VERB_AS_ARTIFACT_NAME = re.compile(
    rf"\b(?:\w+\s*/\s*)*(?:{_SPARK_MUTATING_VERBS})(?:\s*/\s*\w+)*"
    rf"\s+(?:{_ARTIFACT_NOUN})\b"
    rf"(?=\s*(?:[,.;:)\]]|and\b|or\b|es\b|vagy\b|$))",
    re.I,
)


def _without_artifact_names(text: str) -> str:
    """Remove write verbs that *name* an artifact instead of asking for one."""
    return _VERB_AS_ARTIFACT_NAME.sub(" ", text or "")


# "write [REDACTED]", "replace it with [MASKED]": the write verb governs the
# *report*, and specifically the part of the report that refuses to carry a
# secret. It is the strictest sentence in a careful evidence goal, and it read as
# mutation -- the safety half ("Never print a secret value;") is dropped as a
# prohibition, which leaves the redaction half standing alone as an instruction.
#
# The bracketed token is what makes it a placeholder, so it is required, exactly
# as the artifact noun is required above. "Write the masked config to disk" has
# no brackets and stays a write.
_REDACTION_PLACEHOLDER = r"redacted|masked|elided|omitted|secret|placeholder"
_VERB_AS_REDACTION = re.compile(
    rf"\b(?:{_SPARK_MUTATING_VERBS})\s+(?:it|them|that|those)?\s*(?:as|with|to)?\s*"
    rf"[`'\"]*\[\s*(?:{_REDACTION_PLACEHOLDER})\s*\][`'\"]*",
    re.I,
)


def _without_redaction_placeholders(text: str) -> str:
    """Remove write verbs whose object is a redaction placeholder."""
    return _VERB_AS_REDACTION.sub(" ", text or "")


# "What does szamlazz-agent.ts actually implement today?": the write verb belongs
# to the subject under inspection, not to the leaf. A question about what code
# already does is the purest form of read-only work, and naming the behaviour
# accurately requires the same verbs that describe doing it -- so, once more, the
# more precise the question, the more certainly it read as mutating.
#
# The interrogative opener and the auxiliary together are what make it a
# description; either alone is not enough ("Update the DTO, which does matter"
# has both words and no question). The spans between them are bounded so the
# rule cannot reach across a sentence into an unrelated instruction.
_DESCRIPTIVE_QUESTION = re.compile(
    rf"\b(?:what|how|which|whether|where)\b(?:\s+\w+){{0,2}}\s+"
    rf"\b(?:does|do|did)\b(?:\s+\S+){{0,4}}\s+"
    rf"(?:{_SPARK_MUTATING_VERBS})\b",
    re.I,
)


def _without_descriptive_questions(text: str) -> str:
    """Remove write verbs that describe the subject's behaviour in a question."""
    return _DESCRIPTIVE_QUESTION.sub(" ", text or "")


# What may stand between a determiner and the verb-as-noun. A closed list, not
# free words: an open filler walks over the noun in front of a real instruction
# and swallows it, so "before the audit rewrite the config" read as read-only,
# and "no edits but rewrite the config" hid its rewrite behind the refusal.
# These adjectives are the ones that actually qualify a change, and none of them
# can be the object of the verb that follows.
_CHANGE_ADJECTIVE = (
    r"requested|proposed|planned|intended|suggested|upcoming|pending|eventual|"
    r"further|additional|subsequent|later|final|actual|next|same|initial|other|new"
)


# "a test case that would fail before the requested edit": the write verb is a
# *noun* -- the thing the report is measured against, not work to do. A read-only
# goal asking for a currently-failing test has to name the change that will
# eventually make it pass, and naming it read as an instruction to make it.
#
# Observed 2026-09-21: "[luna] Map the exact existing issuer-mode behavior ... a
# concrete test case that would fail before the requested edit; make no edits."
# was escalated with "consequential Luna task requires Sol" -- a goal that
# forbids editing in its last clause, sent to Sol for the word "edit" in its
# second, because "authorization" supplied the consequential half.
#
# Two arms, each with its own corroborating token, because a bare determiner is
# not enough: "Make the change" is an instruction and must stay one. A temporal
# preposition puts the verb outside the work being asked for; failing that, an
# adjective ("the requested edit") marks it as something decided elsewhere --
# which is why the second arm requires one and the first does not.
#
# Two more arms cover the verb as the *subject or modifier* of a noun phrase.
# Observed 2026-09-30: "[luna] Inspect the service color create/edit data flow
# ... identify the probable root cause for a new service ... failing while an
# existing service color update works." ran on Terra, reason "Luna is restricted
# to non-design read-only subtasks" -- a bug report cannot describe the working
# path without naming it. Both arms need a determiner in front, so an imperative
# ("update the handler") never matches, and each has its own corroboration: a
# behavioural predicate right after the verb ("color update works"), or a noun it
# qualifies ("create/edit data flow", "the edit form"). The predicate arm also
# needs a word between determiner and verb: "the fix works" is too often a
# request to make it so.
_NOUN_PHRASE_FILLER = (
    rf"(?!(?:{_SPARK_MUTATING_VERBS}|to|should|must|will|would|can|could|may|"
    rf"might|shall|please|then|and|or|but|so|also|not|do|does|did)\b)[\w-]+"
)
_NOUN_PHRASE_LEAD_WORDS = r"\b(?:the|a|an|this|that|its|their|any|our|your|each)\s+"
_FINITE_PREDICATE = (
    r"works|worked|succeeds|succeeded|fails|failed|breaks|broke|passes|passed|"
    r"errors|errored|crashes|crashed|behaves|behaved|persists|persisted|"
    r"returns|returned|throws|threw|rejects|rejected"
)
_COMPOUND_HEAD = (
    r"flow|path|handler|endpoint|form|route|request|payload|modal|dialog|logic|"
    r"mutation|action|button|screen|page|api|call|operation|scenario|case|branch"
)
_VERB_AS_NOUN_SUBJECT_LEAD = rf"{_NOUN_PHRASE_LEAD_WORDS}(?:{_NOUN_PHRASE_FILLER}\s+){{1,3}}"
_VERB_AS_NOUN_MODIFIER_LEAD = rf"{_NOUN_PHRASE_LEAD_WORDS}(?:{_NOUN_PHRASE_FILLER}\s+){{0,3}}"
_VERB_AS_NOUN = re.compile(
    rf"\b(?:before|after|prior\s+to|following|since|once|until)\s+"
    rf"(?:the|a|an|this|that|its|their|any)\s+(?:(?:{_CHANGE_ADJECTIVE})\s+){{0,2}}"
    rf"(?:{_SPARK_MUTATING_VERBS})s?\b"
    rf"|\b(?:the|a|an|this|that|its|their|any)\s+"
    rf"(?:(?:{_CHANGE_ADJECTIVE})\s+){{1,2}}"
    rf"(?:{_SPARK_MUTATING_VERBS})s?\b"
    rf"|{_VERB_AS_NOUN_SUBJECT_LEAD}(?:{_SPARK_MUTATING_VERBS})s?"
    rf"(?=\s+(?:{_FINITE_PREDICATE})\b)"
    rf"|{_VERB_AS_NOUN_MODIFIER_LEAD}(?:{_SPARK_MUTATING_VERBS})(?:/(?:{_SPARK_MUTATING_VERBS}))*"
    rf"(?=\s+(?:data\s+)?(?:{_COMPOUND_HEAD})s?\b)",
    re.I,
)


def _without_verbs_as_nouns(text: str) -> str:
    """Remove write verbs naming what a read-only report is measured against."""
    return _VERB_AS_NOUN.sub(" ", text or "")


# "make no edits", "no changes to the schema": an explicit refusal to write, in a
# shape the prohibition-clause filter cannot see. That filter knows "do not",
# "never" and "without", so a goal that says "no edits" instead kept the verb and
# read as mutating. The same goal above escaped only by accident -- the plural
# "edits" missed a pattern written in the singular.
#
# Stripped here as a phrase rather than added to the prohibition clauses on
# purpose: dropping the whole clause would hide a real instruction standing next
# to it, so "make no edits but rewrite the config" must still read as a write.
#
# One shared exception alternation, reused by all three regexes, so the set of
# words that authorise a change after a refusal cannot drift between them.
# "but" alone is deliberately included (the plan's ruling, Terra-safe
# direction): "change nothing, but report ..." over-matches to a write rather
# than under-matching a real exception clause to read-only.
_REFUSAL_EXCEPTION_WORDS = (
    r"except|but|save|other\s+th(?:a|e)n|apart\s+from|aside\s+from|besides"
)
_NEGATED_VERB = re.compile(
    rf"\bno\s+(?:(?:{_CHANGE_ADJECTIVE})\s+){{0,2}}(?:{_SPARK_MUTATING_VERBS})s?\b"
    rf"(?!\s*,?\s*(?:else\s+)?(?:{_REFUSAL_EXCEPTION_WORDS})\b)", re.I)
# "change nothing", "modify absolutely nothing": the same refusal with the verb
# first. Only a verb whose direct object is "nothing" is stripped, so "edit the
# schema so that nothing breaks" and "fix the parser; nothing else" still write.
_NOTHING_OBJECT_VERB = re.compile(
    rf"\b(?:{_SPARK_MUTATING_VERBS})\s+(?:absolutely\s+)?nothing\b"
    rf"(?!\s*,?\s*(?:else\s+)?(?:{_REFUSAL_EXCEPTION_WORDS})\b)", re.I)
# A refusal with an immediate exception still authorises a change. This must be
# checked before refusal phrases are stripped, while the strippers retain their
# narrow lookaheads so only the refusal itself is removed in ordinary prose.
_REFUSAL_WITH_EXCEPTION = re.compile(
    rf"\b(?:no\s+(?:(?:{_CHANGE_ADJECTIVE})\s+){{0,2}}(?:{_SPARK_MUTATING_VERBS})s?"
    rf"|(?:{_SPARK_MUTATING_VERBS})\s+(?:absolutely\s+)?nothing)"
    rf"\s*,?\s*(?:else\s+)?(?:{_REFUSAL_EXCEPTION_WORDS})\b", re.I)


def _without_negated_verbs(text: str) -> str:
    """Remove write verbs a goal explicitly refuses ("make no edits", "change nothing")."""
    return _NOTHING_OBJECT_VERB.sub(" ", _NEGATED_VERB.sub(" ", text or ""))


def _without_non_instructing_verbs(text: str) -> str:
    """Strip write verbs that name, quote, describe or refuse instead of instructing.

    Six shapes, one failure: a read-only goal cannot say what it looks at
    without using the vocabulary of changing it. Each stripper requires its own
    corroborating token -- a hash, an artifact noun, a bracketed placeholder, an
    interrogative, a temporal preposition or participle, an explicit "no" -- so
    an unadorned instruction still reads as a write.
    """
    return _without_negated_verbs(
        _without_verbs_as_nouns(
            _without_descriptive_questions(
                _without_redaction_placeholders(
                    _without_artifact_names(_without_commit_references(text))
                )
            )
        )
    )


def _is_spark_read_only_work(text: str) -> bool:
    """A plan-labelled leaf is read-only unless it says otherwise.

    Separated from the design test because mixing them made the question
    unanswerable: "identify the layout branches" and "implement a CSS card"
    both mention design, so a combined predicate rejected both. The verbs
    separate them cleanly -- one reads, the other writes.

    This side looks for *contradiction*, not corroboration. The conductor has
    already declared the leaf read-only by labelling it, so demanding a second
    positive signal means the router overrules that claim whenever the phrasing
    falls outside a hand-written verb list -- which a Hungarian goal did on its
    first outing ("Tárd fel..." reads nothing but says so with a verb the list
    never had). Write verbs are the small, stable set worth enumerating;
    read-only phrasings are open-ended.
    """
    affirmative = _normalise(_without_negated_safety_constraints(text))
    return bool(
        affirmative
        and not _REFUSAL_WITH_EXCEPTION.search(affirmative)
        and not _SPARK_MUTATING_WORK.search(_without_non_instructing_verbs(affirmative))
    )


def _is_spark_read_only_request(text: str) -> bool:
    """Spark may receive only affirmative, bounded non-design evidence work.

    The stricter form, for a claim no conductor vouched for: a bare ``[spark]``
    on a root turn is a label someone typed, so here a positive read-only signal
    is still required.
    """
    affirmative = _normalise(_without_negated_safety_constraints(text))
    return bool(
        _is_spark_read_only_work(text)
        and _SPARK_READ_ONLY_WORK.search(affirmative)
        and not _is_design_request(affirmative)
    )


def _is_consequential_spark_request(text: str) -> bool:
    return bool(_SPARK_CONSEQUENTIAL_WORK.search(_normalise(_without_negated_safety_constraints(text))))


def _without_negated_safety_constraints(text: str) -> str:
    """Remove standalone prohibition clauses before judging a child as risky.

    Delegation goals routinely say things like ``Do not restart services``.
    Those exclusions must not combine with an earlier harmless ``config`` noun
    and become a false consequential-system action. Affirmative clauses are
    retained, so a real production/deploy request still escalates to Sol.
    """
    clauses = re.split(r"(?<=[.!?;])\s+|[\r\n]+", text or "")
    prohibition = re.compile(
        r"\b(do not|don't|must not|never|without|prohibit(?:ed)?|"
        r"ne\s+|tilos|nem szabad)\b",
        re.IGNORECASE,
    )
    return " ".join(clause for clause in clauses if clause.strip() and not prohibition.search(clause)).strip()


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                for key in ("text", "input_text", "content"):
                    value = item.get(key)
                    if isinstance(value, str):
                        chunks.append(value)
                        break
        return "\n".join(chunks)
    if isinstance(content, dict):
        for key in ("text", "input_text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value
    return ""


def _request_items(request: Dict[str, Any]) -> list:
    for key in ("messages", "input"):
        value = request.get(key)
        if isinstance(value, list):
            return value
    return []


_IMAGE_TYPES = {"image", "image_url", "input_image", "image_file", "input_image_file"}
_IMAGE_TEXT_MARKERS = ("[image attached", "[screenshot]", "data:image/")


def _contains_image_attachment(value: Any) -> bool:
    """Recognise textual and structured image parts in one prompt item."""
    if isinstance(value, str):
        return any(marker in value.casefold() for marker in _IMAGE_TEXT_MARKERS)
    if isinstance(value, list):
        return any(_contains_image_attachment(item) for item in value)
    if not isinstance(value, dict):
        return False
    if str(value.get("type") or "").casefold() in _IMAGE_TYPES:
        return True
    if any(value.get(key) for key in ("image_url", "input_image", "image_file")):
        return True
    return any(_contains_image_attachment(item) for item in value.values())


def _request_has_image_attachment(request: Dict[str, Any]) -> bool:
    """Return true only when the latest user turn itself contains an image.

    A historical image is context, not a permanent vision requirement. The
    router must classify the current objective, otherwise one screenshot would
    pin every later text-only prompt and delegated child to Terra forever.
    Before Luna/Spark dispatch, historical image parts are removed from the
    outgoing request by ``_strip_historical_image_attachments`` so text-only
    models never receive unsupported media.
    """
    items = _request_items(request)
    _, user_index = _last_user_text_and_index(items)
    if user_index < 0:
        return False
    item = items[user_index]
    return _contains_image_attachment(item.get("content", "")) if isinstance(item, dict) else False


def _strip_historical_image_attachments(request: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a request without visual media from turns preceding the current user.

    This preserves the current turn exactly. It only runs for Luna/Spark after
    routing has established that the current objective is text-only, preventing
    stale session screenshots from leaking to text-only providers.
    """
    cleaned = deepcopy(request)
    items = _request_items(cleaned)
    _, user_index = _last_user_text_and_index(items)
    if user_index <= 0:
        return cleaned

    def clean(value: Any) -> Any:
        if isinstance(value, str):
            result = value
            for marker in _IMAGE_TEXT_MARKERS:
                result = re.sub(re.escape(marker) + r"[^\]\n]*\]", "[Earlier image omitted]", result, flags=re.IGNORECASE)
            return result
        if isinstance(value, list):
            return [next_value for item in value if (next_value := clean(item)) is not None]
        if not isinstance(value, dict):
            return value
        if str(value.get("type") or "").casefold() in _IMAGE_TYPES:
            return None
        if any(value.get(key) for key in ("image_url", "input_image", "image_file")):
            return None
        return {key: next_value for key, item in value.items() if (next_value := clean(item)) is not None}

    for item in items[:user_index]:
        if isinstance(item, dict) and "content" in item:
            item["content"] = clean(item["content"])
    return cleaned


def _last_user_text_and_index(items: Iterable[Any]) -> tuple[str, int]:
    sequence = list(items)
    for index in range(len(sequence) - 1, -1, -1):
        item = sequence[index]
        if isinstance(item, dict) and item.get("role") == "user":
            return _text_from_content(item.get("content", "")), index
    return "", -1


def _has_prior_exchange(items: Iterable[Any], user_index: int) -> bool:
    """Whether the latest user turn follows earlier assistant work in this request."""
    for item in list(items)[:max(0, user_index)]:
        if isinstance(item, dict) and (
            item.get("role") == "assistant"
            or item.get("type") in {"function_call", "function_call_output"}
        ):
            return True
    return False


def _prompt_preview(request: Any) -> str:
    """Return the complete latest user prompt on a single line."""
    if not isinstance(request, dict):
        return ""
    text, _ = _last_user_text_and_index(_request_items(request))
    return re.sub(r"\s+", " ", text).strip()


def _redacted_preview(value: Any, limit: int = 280) -> str:
    """Bound a local observability description without retaining secrets."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    text = re.sub(
        r"(?i)\b(password|passwd|api[ _-]?key|secret|token)\s*([=:])\s*([^\s,;]+)",
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)\bauthorization\s*:\s*bearer\s+[^\s,;]+", "Authorization: Bearer ***", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "sk-[REDACTED]", text)
    return text if len(text) <= limit else f"{text[:limit - 1]}…"


def _logged_prompt_preview(request: Any, log_cfg: Dict[str, Any]) -> str:
    """Return the local audit preview without retaining full user content."""
    preview = _prompt_preview(request)
    if bool(log_cfg.get("redact_prompt_preview", True)):
        preview = _redacted_preview(preview, limit=max(1, int(log_cfg.get("prompt_preview_chars", 240) or 240)))
    limit = max(1, int(log_cfg.get("prompt_preview_chars", 240) or 240))
    return preview[:limit]


def _lifecycle_event_kind(request: Any) -> Optional[str]:
    """Classify synthetic internal turns without exposing their raw envelope."""
    text = _prompt_preview(request).casefold()
    if text.startswith("[async delegation batch complete") or text.startswith("[async delegation complete"):
        return "async_delegation_completion"
    if text.startswith("[context compaction"):
        return "context_compaction"
    return None


def _completion_delegation_id(request: Any) -> str:
    match = re.match(
        r"^\[ASYNC DELEGATION(?: BATCH)? COMPLETE\s+[—-]\s*([^\]\s]+)",
        _prompt_preview(request),
        re.I,
    )
    return match.group(1) if match else ""


# One task's header line in a delegation completion envelope. The icon carries
# the outcome (✓ done, ✗ failed, ⚠ truncated) and the goal is repeated in full,
# which is what makes a re-dispatch expressible without any state of our own.
_TASK_HEADER = re.compile(
    r"^---\s*(?P<icon>\S)\s*TASK\s+\d+/\d+(?::\s*(?P<goal>.*?))?\s*\(status=(?P<status>[^,)]*)",
    re.M,
)
# What an account limit looks like in the error the envelope quotes.
_QUOTA_STOP_WORDING = re.compile(
    r"(429|quota|usage limit|limit has been reached|rate[ _-]?limit|throttl)", re.I
)
_QUOTA_STOP_HEAD_CHARS = 400


def _delegation_failure_reason(block: str) -> str:
    """The generated status/error lines of a failed task, without its summary.

    Scoped this tightly because the obvious version -- search the block -- reads
    the worker's own partial output too, and a leaf whose subject *is* quota
    handling then reports itself as quota-stopped. The envelope emits the reason
    as parenthesised or ``Error:``-prefixed lines ahead of "Partial output:", so
    that prefix is the whole boundary.
    """
    head = block.split("Partial output:")[0]
    return "\n".join(
        line for line in head.splitlines()
        if line.startswith("(") or line.startswith("Error:")
    )[:_QUOTA_STOP_HEAD_CHARS]


# The immediate notice for one child of a still-running fan-out, which arrives
# while the siblings are still working and says in as many words that it exists
# so the conductor can re-dispatch now rather than at batch end. That is the
# moment this whole notice is for, so it is matched alongside the batch envelope.
_SINGLE_FAILURE_TITLE = "[async delegation task failed"
_SINGLE_FAILURE_GOAL = re.compile(r"^Task:\s*(?P<goal>.+)$", re.M)
_SINGLE_FAILURE_ERROR = re.compile(r"^Error:\s*.+$", re.M)


def _is_delegation_outcome_text(text: str) -> bool:
    lowered = (text or "").lstrip().casefold()
    return (
        lowered.startswith("[async delegation batch complete")
        or lowered.startswith("[async delegation complete")
        or lowered.startswith(_SINGLE_FAILURE_TITLE)
    )


def _single_failure_block(text: str) -> Tuple[Tuple[str, str], ...]:
    """``(goal, reason)`` for the early single-child failure notice, if that is it."""
    if not (text or "").lstrip().casefold().startswith(_SINGLE_FAILURE_TITLE):
        return ()
    goal = _SINGLE_FAILURE_GOAL.search(text)
    error = _SINGLE_FAILURE_ERROR.search(text)
    if error is None:
        return ()
    return ((str(goal.group("goal")).strip() if goal else "", error.group(0)),)


def _failed_delegation_blocks(text: str) -> Tuple[Tuple[str, str], ...]:
    """``(goal, block)`` for every task the envelope reports as not done."""
    headers = list(_TASK_HEADER.finditer(text))
    blocks = []
    for position, match in enumerate(headers):
        end = headers[position + 1].start() if position + 1 < len(headers) else len(text)
        block = text[match.end():end]
        # A zero-token admission stop is a completed model response in Hermes,
        # so the host's ✓ status cannot be treated as proof that work ran.
        stopped = _router_stopped_summary(block)
        if match.group("icon") in {"✗", "⚠"} or stopped:
            blocks.append((str(match.group("goal") or "").strip(), block))
    return tuple(blocks)


def _router_stopped_summary(block: str) -> bool:
    # Batch headers leave their status suffix on the first line. A completed
    # worker's first result line is the router's reserved marker.
    result = block.split("\n", 1)[-1] if "\n" in block else block
    for candidate in (block.lstrip(), result.lstrip()):
        if (candidate.startswith("[ROUTER WORKER STOPPED]")
                and "No work was performed by this call." in candidate[:500]):
            return True
    return False


def _single_stopped_block(text: str) -> Tuple[Tuple[str, str], ...]:
    if not (text or "").lstrip().casefold().startswith("[async delegation complete"):
        return ()
    goal = re.search(r"^Original goal:\s*(.*)$", text, re.M)
    result = text.split("--- RESULT ---", 1)
    if not goal or len(result) != 2:
        return ()
    summary = result[1].lstrip()
    if not _router_stopped_summary(summary):
        return ()
    return ((goal.group(1).strip(), summary),)


def _kind_for_goal(goal: str, cfg: Dict[str, Any]) -> str:
    """The work kind of a leaf goal, from the same classifier every route uses.

    Reusing the classifier rather than adding a lookup is the point: a second,
    private notion of what kind of work a goal is would drift from the one that
    decided the route, and then the retry advice would name a chain the operator
    never associated with it.
    """
    # A route label describes the stopped worker, not the semantic work kind.
    goal = re.sub(r"^\s*\[(?:luna|spark|terra|sol|opus5|sonnet5|haiku|qwen|grok)(?::xhigh)?\]\s*",
                  "", goal, flags=re.I)
    if not goal.strip():
        return "default"
    try:
        decision = classify_request(
            {"messages": [{"role": "user", "content": goal}]},
            api_call_count=1,
            config=cfg,
            allow_plan_label_over_design=True,
        )
    except Exception:
        return "default"
    return decision.kind or "default"


def _chain_entries(kind: str, cfg: Dict[str, Any], *, model_param: bool = True) -> Tuple[str, ...]:
    """The kind's order, restricted to routes this host can actually express."""
    targets = set(_delegation_target_names())
    switches = cfg.get("callable") or {}
    return tuple(
        name for name in _preference_list(kind, cfg)
        if switches.get(name) is True and _target_is_offered(name, cfg)
        and (name in targets if model_param else
             (name in claude_delegation.TIER_FOR_TARGET and claude_delegation.is_active())
             or (name in (cfg.get("models") or {}) and
                 _account_of(name, cfg) in ("", str(cfg.get("provider", "openai-codex")))))
    )


def _retry_chain(kind: str, cfg: Dict[str, Any], *, model_param: bool = True) -> Tuple[str, ...]:
    readings = _guarded_readings(cfg)
    states = _account_states(cfg, readings)
    if claude_delegation.is_active():
        offered = set(_chain_entries(kind, cfg, model_param=model_param))
        names, _reason = _advised_chain(kind, cfg, states, offered, readings)
    else:
        names = _chain_entries(kind, cfg, model_param=model_param)
    return tuple(name for name in names if states.get(_account_of(name, cfg)) != "closed")


def _next_available_entry(kind: str, cfg: Dict[str, Any], *, model_param: bool = True) -> Optional[str]:
    return next(
        (name for name in _retry_chain(kind, cfg, model_param=model_param)
         if _tier_cooldown_remaining(name, cfg) <= 0),
        None,
    )


def _earliest_free_entry(kind: str, cfg: Dict[str, Any], *, model_param: bool = True) -> Optional[Tuple[str, int]]:
    """The earliest known cooldown among accounts still admitting workers."""
    waiting = [(name, remaining) for name in _retry_chain(kind, cfg, model_param=model_param)
               if (remaining := _tier_cooldown_remaining(name, cfg)) > 0]
    if not waiting:
        return None
    name, remaining = min(waiting, key=lambda item: item[1])
    return name, int(remaining // 60) + 1


# Hermes's own wording when ``delegate_task`` cannot resolve the route it was
# given. It names the *configured default* provider, because an unknown or absent
# target degrades to the default rather than failing -- so a call that meant to
# reach Claude reports a Codex problem, and the parent reads it as "Claude is
# unavailable" when Claude was never asked.
_DISPATCH_PROVIDER_FAILURE = re.compile(r"cannot resolve delegation provider", re.I)
_DISPATCH_NOTICE_LIMIT = 4000


def _item_text(item: Any) -> str:
    """Flatten one request item to searchable text, whatever its wire shape."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        try:
            return json.dumps(item, ensure_ascii=False, default=str)
        except Exception:
            return str(item)
    return str(item)


def _available_delegation_targets(cfg: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(
        name for name in _delegation_target_names()
        if _target_is_offered(name, cfg) and _tier_cooldown_remaining(name, cfg) <= 0
    )


def _dispatch_failure_instruction(request: Any, cfg: Dict[str, Any]) -> str:
    """Answer a delegate_task that failed before any worker started.

    The quota notice next to this one reads a delegation *outcome*: a worker that
    ran and died. This case never gets that far -- the tool returns an error
    inline, no child exists, and no envelope is ever delivered. Observed with the
    Codex account exhausted: the parent believed it was delegating to opus5, the
    call named no target, so it resolved the configured default and came back
    saying Codex was out of quota. The parent then reasoned, correctly from what
    it was shown and wrongly in fact, that its Claude delegation had failed.
    """
    items = _request_items(request)
    if not any(
        _is_tool_item(item) and _DISPATCH_PROVIDER_FAILURE.search(_item_text(item)[:_DISPATCH_NOTICE_LIMIT])
        for item in items[-12:]
    ):
        return ""
    available = _available_delegation_targets(cfg)
    return (
        "\n\n[ROUTER — DELEGATION IS BLOCKED AT THE HOST, NOT AT THAT TARGET]\n"
        "This is not a fact about the account named in the error. delegate_task resolves the "
        "configured default delegation provider once, for the whole call, before it reads the "
        "tasks at all -- so when that provider is unavailable every delegation fails, including "
        "a task that names a target on a healthy account. Its model: value is never reached.\n"
        "Retrying with a different model: will fail identically. "
        + (
            "Targets that are themselves fine right now: " + ", ".join(available) + " -- "
            "unreachable only because the default route is down. "
            if available else ""
        )
        + "Either point delegation.provider and delegation.model at a route that works, or do "
        "the work in this turn, or wait for the default route to recover. Say which of those you "
        "chose rather than re-issuing the same call.\n"
    )


def _misdispatch_instruction(target: str, running_on: str) -> str:
    """Tell a misdispatched leaf to report the correction instead of working.

    Addressed to the leaf because that is who this router can still reach: the
    parent has already dispatched and will not be consulted again until the leaf
    reports. Making the leaf's one answer *be* the correction turns a wasted
    branch into the message the parent needs.
    """
    return (
        f"\n\n[ROUTER — WRONG DISPATCH MECHANISM]\n"
        f"This goal opens with [{target}], which is not a route. A goal-text prefix only "
        f"renames the model inside the default provider, so this leaf is running on "
        f"{running_on}, not on {target}, and its tool use has been switched off.\n"
        f"Do not begin the work and do not plan it. Reply with exactly this line and "
        f"nothing else:\n"
        f"MISDISPATCHED: this goal names {target} in its text, which is not a route. "
        f"Re-dispatch it unchanged with delegate_task(model=\"{target}\").\n"
    )


def _quota_redispatch_instruction(request: Any, cfg: Dict[str, Any]) -> str:
    """Turn a leaf that died on an account limit into an actionable re-dispatch.

    The envelope already carries the goal and the provider's own error, so the
    conductor can see that something failed -- but not that the account, rather
    than the task, is what stopped; not which target its configured order says
    to use next; and not that the work already committed is worth continuing
    from. Those three are the difference between a re-dispatch and a re-plan.

    Deliberately advice-shaped in one respect only: the router names the target
    and does not re-dispatch. Choosing what to do with a stopped leaf -- retry,
    narrow, wait, drop -- is the conductor's, and a router that silently respawned
    work would be the hardcoded selection this design exists to avoid.
    """
    text, _index = _last_user_text_and_index(_request_items(request))
    if not text:
        return ""
    candidates = (_failed_delegation_blocks(text) or _single_failure_block(text)
                  or _single_stopped_block(text))
    stopped = [
        (goal, block) for goal, block in candidates
        if (_router_stopped_summary(block)
            or _QUOTA_STOP_WORDING.search(_delegation_failure_reason(block)))
    ]
    if not stopped:
        return ""
    lines = []
    model_param = _host_delegate_has_model(request)
    for goal, _block in stopped:
        kind = _kind_for_goal(goal, cfg)
        label = (goal[:120] + "…") if len(goal) > 120 else (goal or "the stopped leaf")
        target = _next_available_entry(kind, cfg, model_param=model_param)
        if target:
            call = (f'delegate_claude(tier="{claude_delegation.TIER_FOR_TARGET[target]}")'
                    if claude_delegation.is_active() and target in claude_delegation.TIER_FOR_TARGET
                    else f"delegate_task(model=\"{target}\")" if model_param
                    else f'delegate_task(tasks=[{{"goal": "[{target}] <original objective>"}}])')
            lines.append(f"- {label}\n  {kind} work -> re-dispatch with {call}")
            continue
        waiting = _earliest_free_entry(kind, cfg, model_param=model_param)
        if waiting:
            name, minutes = waiting
            lines.append(
                f"- {label}\n  {kind} work -> every configured target is cooling; "
                f"{name} frees up first, in about {minutes} min"
            )
        else:
            lines.append(
                f"- {label}\n  {kind} work -> no configured target is currently available; "
                f"wait for fresh usage data or account recovery"
            )
    return (
        "\n\n[ROUTER — A WORKER STOPPED ON AN ACCOUNT LIMIT]\n"
        "The account refused the call; the task itself did not fail and its plan is still "
        "valid. Sending the same goal to the same target again will fail the same way while "
        "it is cooling.\n"
        + "\n".join(lines)
        + "\nRe-dispatch each one with the call named above. Keep its objective and context, "
          "replacing any prior route prefix with the target's prefix, and tell the retry to "
        + "continue from what the stopped worker already committed in its worktree instead of "
        "starting over. Do not re-plan or narrow the goal: only the account changed.\n"
    )


def _is_tool_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("role") in {"tool", "function"}:
        return True
    if item.get("type") in {"function_call", "function_call_output", "tool_result"}:
        return True
    if item.get("tool_calls"):
        return True
    content = item.get("content")
    if isinstance(content, list):
        return any(_is_tool_item(part) for part in content)
    return False


def _current_turn_has_tool_activity(items: list, last_user_index: int) -> bool:
    if last_user_index < 0:
        return False
    return any(_is_tool_item(item) for item in items[last_user_index + 1 :])


# Default effort per tier. ``.get`` rather than ``[]``: a preference list may name a
# tier this map never anticipated, and an unknown tier must not raise inside routing.
_DEFAULT_EFFORT = {"luna": "low", "spark": "low", "terra": "medium", "sol": "high", "qwen": "medium",
                   "grok": "medium"}


def _decision(
    tier: str,
    reason: str,
    cfg: Dict[str, Any],
    *,
    explicit: bool = False,
    effort_key: Optional[str] = None,
    mandatory: bool = False,
    kind: str = "",
) -> RouteDecision:
    efforts = cfg.get("effort") or {}
    if effort_key is None:
        # The explicit-label bump used to be Sol's alone, so ``explicit_<tier>``
        # was unreadable config for every other tier. It is consulted for all of
        # them now, but only where the operator actually wrote the key: an absent
        # one must leave that tier exactly as it routed before.
        #
        # Sol keeps its unconditional form. There the missing key deliberately
        # falls through to the built-in tier default rather than the configured
        # ``sol`` value -- an explicit Sol label is an escalation, and the tests
        # pin that behaviour.
        explicit_key = f"explicit_{tier}"
        use_explicit = explicit and (tier == "sol" or explicit_key in efforts)
        effort_key = explicit_key if use_explicit else tier
    # An escalation degrades to the tier's explicit key, never to its floor: a
    # missing ``explicit_<tier>_xhigh`` means "no escalation configured", and
    # answering that with the plain tier value would silently cap [sol:xhigh].
    candidates = [effort_key]
    if effort_key.endswith("_xhigh"):
        candidates.append(effort_key[: -len("_xhigh")])
    effort = next(
        (str(efforts[key]).casefold() for key in candidates if efforts.get(key)),
        _DEFAULT_EFFORT.get(tier, "medium"),
    )
    return RouteDecision(
        tier=tier, model=cfg["models"][tier], reason=reason, effort=effort,
        mandatory=mandatory, kind=kind,
    )


def _eligible_tiers(
    user_text: str,
    *,
    has_image_attachment: bool,
    api_call_count: int,
    is_tool_loop: bool,
) -> Tuple[str, ...]:
    """Tiers this request independently qualifies for, ignoring gate precedence.

    Deliberately position-independent: it answers "was this tier ever a candidate"
    rather than "which gate won". Comparing it against the chosen tier is what
    turns a silent never-fires route into a visible preempted one.
    """
    text = _normalise(user_text)
    # Terra is deliberately absent: it is the unconditional default, so listing it
    # would mark every non-Terra decision as preempting it and carry no signal.
    eligible = []

    if _is_design_request(user_text):
        eligible.append("sol")

    # Spark is text-only, first-call-only, and restricted to non-design read-only
    # work. Anything else can only reach it through an explicit label.
    if (
        not has_image_attachment
        and api_call_count == 1
        and not is_tool_loop
        and _is_spark_read_only_request(user_text)
        and not _is_consequential_spark_request(user_text)
    ):
        eligible.append("spark")

    if _is_acknowledgement_only(user_text) or re.match(
        r"^(szia|hello|hi|hey|jo reggelt|jo estet|koszonom|koszi|thanks|thank you)[!. ]*$", text
    ):
        eligible.append("luna")

    return tuple(dict.fromkeys(eligible))


def classify_request(
    request: Dict[str, Any],
    api_call_count: int = 1,
    config: Optional[Dict[str, Any]] = None,
    *,
    allow_plan_label_over_design: bool = False,
) -> RouteDecision:
    """Classify one provider request, annotated with the tiers it lost out on."""
    decision = _classify_request(
        request,
        api_call_count,
        config,
        allow_plan_label_over_design=allow_plan_label_over_design,
    )
    items = _request_items(request)
    user_text, user_index = _last_user_text_and_index(items)
    user_text = _without_host_injected_context(_without_router_contract(user_text))
    eligible = _eligible_tiers(
        user_text,
        has_image_attachment=_request_has_image_attachment(request),
        api_call_count=api_call_count,
        is_tool_loop=_current_turn_has_tool_activity(items, user_index),
    )
    vetoed = tuple(tier for tier in eligible if tier != decision.tier)
    if vetoed:
        decision = replace(decision, vetoed_by=vetoed)
    # Last, so a preference is applied to whatever the gates decided rather than
    # competing with them, and so the veto list still records the built-in view.
    return _apply_preferences(decision, config or _load_config())


# Text this plugin injects itself. A delegated leaf carries the routing contract in
# its own first message, and the contract necessarily talks about applying, implementing
# and committing — so classifying the leaf from that text made every [spark] leaf look
# like mutating work. Measured: goal alone reads read-only, goal + contract does not,
# on the word "apply" from the contract. Stripped before classification only; the
# preview and the log still show what was actually sent.
_ROUTE_CHOICE_OPENING = "Route choice for delegated workers"
_ROUTER_CONTRACT_MARKERS = (
    "Set the delegate_task 'model' parameter",
    _ROUTE_CHOICE_OPENING,
    "[ROUTER] This turn classifies as",
    "planning conductor.",
    "[INTERNAL ORCHESTRATOR PREFLIGHT]",
)


def _without_router_contract(text: str) -> str:
    """Drop the router's own injected contract from the end of a message."""
    if not text:
        return text
    cut = min(
        (pos for pos in (text.find(marker) for marker in _ROUTER_CONTRACT_MARKERS) if pos != -1),
        default=-1,
    )
    return text[:cut].rstrip() if cut > 0 else text


# Hermes appends host/plugin context after the operator's turn. A matching tag is
# not provenance: an operator can paste the same markup. Strip only blocks whose
# first content line is the host's exact signature. Keep this table so a future
# injected context type needs one tag/signature entry and no generic tag matcher.
_HOST_INJECTED_CONTEXT_TAGS = {
    "memory-context": "[System note: The following is recalled memory context, NOT new user input.",
    "EXTREMELY_IMPORTANT": "superpowers:using-superpowers bootstrap for hermes",
}
_HOST_INJECTED_CONTEXT_BLOCK = re.compile(
    r"(?P<prefix>^|\r?\n[ \t]*\r?\n)[ \t]*(?:"
    + "|".join(
        rf"<{re.escape(tag)}>\r?\n{re.escape(signature)}"
        + (r"[^\r\n]*" if tag == "memory-context" else "")
        + rf"(?:\r?\n|$)(?:.*?</{re.escape(tag)}>|.*\Z)"
        for tag, signature in _HOST_INJECTED_CONTEXT_TAGS.items()
    )
    + r")",
    re.DOTALL,
)


def _without_host_injected_context(text: str) -> str:
    """Drop exact host/plugin context blocks appended after the operator text."""
    if not text:
        return text
    return _HOST_INJECTED_CONTEXT_BLOCK.sub(lambda match: match.group("prefix"), text).rstrip()


def _classify_request(
    request: Dict[str, Any],
    api_call_count: int = 1,
    config: Optional[Dict[str, Any]] = None,
    *,
    allow_plan_label_over_design: bool = False,
) -> RouteDecision:
    """Classify one provider request. Terra is the normal durable default."""
    cfg = config or _load_config()
    has_image_attachment = _request_has_image_attachment(request)
    items = _request_items(request)
    user_text, user_index = _last_user_text_and_index(items)
    # Classify what the operator asked for, not context the host or this plugin attached.
    user_text = _without_host_injected_context(_without_router_contract(user_text))
    text = _normalise(user_text)

    # A closed praise/approval follow-up has no implementation objective.  Check
    # it before the design boundary so a mention of UI/CSS does not by itself
    # create an unnecessary Sol delegation.
    if _is_acknowledgement_only(user_text):
        return _decision("luna", "acknowledgement-only follow-up", cfg, kind="chat")

    # Role separation is a hard policy boundary: only Sol performs visual/product
    # design analysis or design implementation. Terra may coordinate and approve
    # the resulting evidence, while Spark may inspect only non-design read-only
    # facts. This check intentionally precedes manual/benchmark overrides so a
    # label cannot route design work to another tier.
    # The gate precedes the manual override so a *user* label cannot route design
    # work off Sol. For a delegated conductor the keyword test misfires: it cannot
    # tell "coordinate work that includes design" from "do design", so any
    # UI-adjacent objective pinned the planner to Sol, which then owned both the
    # conducting and the [sol] leaf it was meant to delegate -- 15 of 16 routing
    # decisions on one turn. Only the conductor label is exempt; a [spark] or
    # [sol] leaf still faces the gate, and Sol still owns every [sol] leaf.
    if _is_design_request(user_text) and not (
        allow_plan_label_over_design and _is_plan_labelled_worker(text)
    ):
        return _decision("sol", "design analysis or implementation is Sol-only", cfg, mandatory=True, kind="design")

    benchmark_force = _normalise(os.getenv("MODEL_ROUTER_BENCHMARK_FORCE_MODEL", ""))
    if benchmark_force in ("luna", "spark", "terra", "sol"):
        if benchmark_force == "spark":
            if has_image_attachment:
                return _decision("terra", "image attachment requires a vision-capable route", cfg, mandatory=True)
            if not _is_spark_read_only_request(user_text):
                return _decision("terra", "Spark is restricted to non-design read-only subtasks", cfg)
        return _decision(benchmark_force, "benchmark environment force override", cfg, explicit=True)

    override = re.match(r"^\s*\[(luna|spark|terra|sol)(?::(xhigh))?\](?:\s|$)", text)
    if override:
        tier = override.group(1)
        if tier == "spark" and has_image_attachment:
            return _decision("terra", "image attachment requires a vision-capable route", cfg, mandatory=True)
        # A delegated leaf carries a conductor's declaration, so the router
        # looks only for contradiction. A root [spark] is a label someone typed
        # with nothing behind it, and still has to show its read-only intent.
        label_read_only = (
            _is_spark_read_only_work(user_text)
            if allow_plan_label_over_design
            else _is_spark_read_only_request(user_text)
        )
        # Luna faces the same test as Spark, and for the same reason: it is a
        # bounded, low-effort tier whose label carried no capability check at
        # all, so a conductor could hand it "stabilize, correct, test and commit"
        # and the router obeyed. The two escape hatches stay tier-specific in
        # wording because downstream branches match these reasons verbatim.
        if tier in ("spark", "luna") and not label_read_only:
            if _is_consequential_spark_request(user_text):
                return _decision("sol", f"consequential {tier.capitalize()} task requires Sol", cfg, mandatory=True)
            if _is_design_request(user_text):
                return _decision("sol", "design analysis or implementation is Sol-only", cfg, mandatory=True, kind="design")
            return _decision("terra", f"{tier.capitalize()} is restricted to non-design read-only subtasks", cfg)
        requested_effort = override.group(2)
        # Escalation is not Sol's alone. The ``:xhigh`` suffix parsed for every
        # tier and was then discarded for all but Sol, so [luna:xhigh] silently
        # ran at Luna's floor effort with no way to say otherwise.
        effort_key = f"explicit_{tier}_xhigh" if requested_effort == "xhigh" else None
        label = f"{tier}:{requested_effort}" if requested_effort else tier
        return _decision(tier, f"explicit [{label}] override", cfg, explicit=True, effort_key=effort_key)

    # The durable completion of a [terra] orchestrator includes the complete
    # reviewed evidence and can be much longer than the normal Sol threshold.
    # Keep the final hand-back with Terra so the supervisor's acceptance gate,
    # integration ownership, and final answer are not silently reassigned.
    if "[async delegation batch complete" in text and "role: orchestrator" in text and "[terra]" in text:
        return _decision("terra", "completed Terra supervisor review", cfg)

    sol_min_chars = int(cfg.get("thresholds", {}).get("sol_min_chars", 3500))
    if len(user_text) >= sol_min_chars:
        return _decision("sol", f"long request ({len(user_text)} characters)", cfg, effort_key="sol_long", kind="long")

    # Consequential domains are biased toward Sol even when the prompt is short.
    sensitive = re.compile(
        r"\b(security|biztonsag|vulnerability|sebezhetoseg|malware|"
        r"auth|oauth|authentication|authorization|jogosultsag|"
        r"credential|credentials|belepesi\s+adat\w*|jelsz\w*|password|passwd|"
        r"payment|fizetes|billing|szamlazas|webhook|jogi|legal|"
        r"orvosi|medical|diagnos|gyogyszer|befektetes|investment|adozas|tax)\b"
    )
    if sensitive.search(text):
        return _decision("sol", "sensitive or consequential domain", cfg, mandatory=True, kind="sensitive")

    # High-consequence engineering stays on Sol. Ordinary repository debugging,
    # refactoring and test execution stay on Terra: the implementation benchmark
    # showed that Terra is the safer integration owner, while bounded Spark work
    # remains available through explicit/delegated workers.
    critical_work = re.compile(
        r"\b(migrate|migr(?:al|ald)|deploy(?:ol|old)?|telepitsd|install|production|prod(?:ra|on|ban|ba|ot)?|"
        r"deep research|mely kutatas|kutass reszletesen)\b"
    )
    if critical_work.search(text):
        return _decision("sol", "consequential engineering or research task", cfg, mandatory=True, kind="critical")

    action = re.compile(
        r"\b(modositsd|konfigurald|configure|restart|ujraindit|torol(?:d|j)|delete|remove|"
        r"upload|toltsd fel|publish|send|kuldd|execute|futtasd|javitsd|fix|connect|"
        r"lepj be|allitsd be|create|hozd letre)\b"
    )
    consequential_system = re.compile(
        r"\b(ssh|sudo|server|szerver|firewall|dns|database|adatbazis|"
        r"hosting|gateway|docker|kubernetes|systemd|config|konfiguracio)\b"
    )
    if action.search(text) and consequential_system.search(text):
        return _decision("sol", "consequential system action", cfg, mandatory=True, kind="critical")

    # Spark is text-only. This hard route sits after the higher-priority Sol
    # safety routes and before every Spark classifier, so an image cannot be
    # routed to Spark by a primary, override, benchmark, or tool-loop branch.
    if has_image_attachment:
        return _decision("terra", "image attachment requires a vision-capable route", cfg, mandatory=True)

    # Unlabelled work is never sent directly to Spark. Terra plans and owns the
    # task first; only its explicit [spark] leaf goals may use Spark.  Repeated
    # calls merely preserve Terra ownership rather than reclassifying from
    # prompt keywords.
    if api_call_count > 1:
        if _current_turn_has_tool_activity(items, user_index):
            return _decision("terra", "ordinary current-turn tool loop", cfg)
        return _decision("terra", "repeated current-turn call safety promotion", cfg)

    # Review and exploration are recognised so a preference list can reach them.
    # Both deliberately keep Terra as their built-in route: naming the category
    # must not change any behaviour on its own, only make it addressable.
    review_request = re.compile(
        r"\b(review|reviewold|nezd at|nezd meg a kodot|code review|atnezes|"
        r"velemenyezd|critique|audit|ellenorizd a kodot)\b"
    )
    if review_request.search(text):
        return _decision("terra", "code review or critique", cfg, kind="review")

    if _is_spark_read_only_request(user_text) and not _is_consequential_spark_request(user_text):
        return _decision("terra", "read-only inspection", cfg, kind="explore")

    repo_implementation = re.compile(
        r"\b(debug|debugold|hibakeres|traceback|stack trace|root cause|"
        r"refactor|teszteld|run the tests|futtasd a teszt|javitsd|fix|"
        r"implement|repo|kod|code)\b"
    )
    if repo_implementation.search(text):
        return _decision("terra", "normal repository implementation owner", cfg, kind="code")

    if re.search(r"\b(csinald meg|hajtsd vegre|do it|make the changes|folytasd)\b", text):
        return _decision("terra", "context-dependent action owned by Terra", cfg, kind="code")



    luna_max_chars = int(cfg.get("thresholds", {}).get("luna_max_chars", 700))
    if len(user_text) <= luna_max_chars:
        greeting = re.compile(
            r"^(szia|hello|hi|hey|jo reggelt|jo estet|koszonom|koszi|thanks|thank you)[!. ]*$"
        )
        simple_transform = re.compile(
            r"^(forditsd|fordits|translate|ird at|fogalmazd at|rewrite|javitsd a helyesirast|"
            r"helyesiras|roviditsd|shorten)\b"
        )
        simple_definition = re.compile(
            r"^(mi az|mit jelent|what is|what does|ki az|who is)\b[^?\n]{0,160}\??$"
        )
        brief_chat = re.compile(
            r"^(ez|az|hat ez|oke|ok|rendben|ertem|furcsa|szomoru|kar|igazad van)\b[^\n]{0,180}$"
        )
        short_explanation = re.compile(
            r"^(miert|hogyhogy|why)\b[^\n]{0,240}\??$"
        )
        technical = re.compile(
            r"\b(implement|kod|code|api|ssh|server|szerver|database|adatbazis|deploy|"
            r"production|config|konfiguracio|debug|teszt|test|security|biztonsag)\b"
        )
        if greeting.search(text):
            return _decision("luna", "greeting or acknowledgement", cfg, kind="chat")
        if simple_transform.search(text) and "```" not in user_text:
            return _decision("luna", "simple language transformation", cfg, kind="chat")
        if simple_definition.search(text):
            return _decision("luna", "short definition request", cfg, kind="chat")
        if brief_chat.search(text):
            return _decision("luna", "brief non-actionable conversation", cfg, kind="chat")
        if short_explanation.search(text) and not technical.search(text):
            return _decision("luna", "short low-risk explanation", cfg, kind="chat")

    return _decision(
        str(cfg.get("default_model", "terra")),
        "default general-purpose route",
        cfg,
        kind="default",
    )


def _log_decision(decision: RouteDecision, kwargs: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    log_cfg = cfg.get("logging", {})
    if not log_cfg.get("enabled", True):
        return
    # No invented default: a config that does not name the audit log does not
    # get written to it. Defaulting here meant every caller holding a partial
    # config -- the test suite above all -- appended to the real log, and those
    # entries then show up as real traffic to anything that reads it back.
    configured = str(log_cfg.get("path") or "").strip()
    if not configured:
        return
    path = hermes_path(configured)
    request = kwargs.get("request")
    event_kind = _lifecycle_event_kind(request)
    delegation_id = _completion_delegation_id(request) if event_kind == "async_delegation_completion" else ""
    entry = {
        # Routing decisions do not need sub-second precision. Keeping this at
        # whole seconds makes the JSONL easier to scan and group.
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "pid": os.getpid(),
        "turn_id": kwargs.get("turn_id", ""),
        "api_call_count": kwargs.get("api_call_count", 1),
        "tier": decision.tier,
        "model": decision.model,
        "effort": decision.effort,
        "reason": decision.reason,
        # Completion envelopes can contain child summaries and must not become
        # a second uncontrolled raw-prompt store.  Persist only a stable
        # correlation id; the viewer resolves a bounded redacted description.
        "prompt_preview": "Delegált feladat befejezési eseménye" if delegation_id else _logged_prompt_preview(request, log_cfg),
    }
    if decision.vetoed_by:
        # Tiers this request qualified for but did not get. Absence over a large
        # sample means the tier's preconditions never hold; presence means the
        # tier is reachable and something ahead of it keeps winning.
        entry["vetoed_by"] = list(decision.vetoed_by)
    if event_kind:
        # The consumer resolves any origin task from durable state.  Do not put
        # the synthetic completion envelope (which can contain a full result)
        # into a second provenance field.
        entry["event_kind"] = event_kind
    if delegation_id:
        entry["delegation_id"] = delegation_id
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _shadow_path(cfg: Dict[str, Any]) -> Path:
    shadow = cfg.get("shadow") or {}
    return hermes_path(shadow.get("path", "~/.hermes/logs/spark-shadow-benchmark.jsonl"))


def _shadow_event(cfg: Dict[str, Any], event: Dict[str, Any]) -> None:
    """Persist local-only benchmark progress without delaying the parent turn."""
    path = _shadow_path(cfg)
    cycle_id = str((cfg.get("shadow") or {}).get("cycle_id") or "").strip()
    event = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        **({"cycle_id": cycle_id} if cycle_id else {}),
        **event,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _SHADOW_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def _benchmark_id(parent_turn_id: str) -> str:
    """Stable local identifier for one forced parent shadow attempt."""
    digest = hashlib.sha256(parent_turn_id.encode("utf-8")).hexdigest()[:16]
    return f"shadow-{digest}"


def _read_shadow_events(cfg: Dict[str, Any]) -> list[Dict[str, Any]]:
    path = _shadow_path(cfg)
    if not path.exists():
        return []
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []


def _shadow_forced_event(cfg: Dict[str, Any], parent_turn_id: str) -> Optional[Dict[str, Any]]:
    cycle_id = str((cfg.get("shadow") or {}).get("cycle_id") or "").strip()
    for event in reversed(_read_shadow_events(cfg)):
        if (
            event.get("event") == "delegation_forced"
            and event.get("turn_id") == parent_turn_id
            and (not cycle_id or str(event.get("cycle_id") or "") == cycle_id)
        ):
            return event
    return None


def _completed_shadow_benchmark_count(events: Iterable[Dict[str, Any]]) -> int:
    """Count only benchmark IDs with an auditable, fully closed parent/child lifecycle."""
    required_events = {"delegation_forced", "child_started", "child_completed", "parent_completed"}
    lifecycle_by_benchmark: Dict[str, set[str]] = {}
    for event in events:
        benchmark_id = event.get("benchmark_id")
        if not benchmark_id:
            continue
        lifecycle_by_benchmark.setdefault(str(benchmark_id), set()).add(str(event.get("event", "")))
    return sum(required_events <= lifecycle for lifecycle in lifecycle_by_benchmark.values())


def _completed_actual_spark_benchmark_count(events: Iterable[Dict[str, Any]], cfg: Dict[str, Any]) -> int:
    """Count closed shadow pairs only when their child was actually routed to Spark.

    A completed lifecycle with no correlatable router rows is counted conservatively so
    a broken log cannot create unbounded new shadow work. A child with correlated
    non-Spark rows is explicitly excluded and may be replaced by another sample.
    """
    required_events = {"delegation_forced", "child_started", "child_completed", "parent_completed"}
    cycle_id = str((cfg.get("shadow") or {}).get("cycle_id") or "").strip()
    if cycle_id:
        events = [event for event in events if str(event.get("cycle_id") or "") == cycle_id]
    lifecycle_by_benchmark: Dict[str, set[str]] = {}
    child_session_by_benchmark: Dict[str, str] = {}
    for event in events:
        benchmark_id = event.get("benchmark_id")
        if not benchmark_id:
            continue
        benchmark_id = str(benchmark_id)
        lifecycle_by_benchmark.setdefault(benchmark_id, set()).add(str(event.get("event", "")))
        if event.get("event") == "child_completed" and event.get("child_session_id"):
            child_session_by_benchmark[benchmark_id] = str(event["child_session_id"])

    route_path = hermes_path((cfg.get("logging") or {}).get("path", _DEFAULT_CONFIG["logging"]["path"]))
    try:
        route_events = [json.loads(line) for line in route_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return _completed_shadow_benchmark_count(events)

    spark_model = str((cfg.get("models") or {}).get("spark", ""))
    completed = 0
    for benchmark_id, lifecycle in lifecycle_by_benchmark.items():
        if not required_events <= lifecycle:
            continue
        child_session_id = child_session_by_benchmark.get(benchmark_id)
        matching_routes = [
            route for route in route_events
            if child_session_id and child_session_id in str(route.get("turn_id", ""))
        ]
        if not matching_routes or all(str(route.get("model", "")) == spark_model for route in matching_routes):
            completed += 1
    return completed


def _summary_digest(summary: Any) -> str:
    return hashlib.sha256(str(summary or "").encode("utf-8")).hexdigest()


def _prepare_shadow_delegation(request: Dict[str, Any], benchmark_id: str) -> Dict[str, Any]:
    """Force the parent to create one safe Spark worker through Hermes' own tool loop."""
    shadow = deepcopy(request)
    instruction = (
        "\n\n[INTERNAL SPARK MEDIUM SHADOW BENCHMARK]\n"
        f"Benchmark ID: {benchmark_id}. Call delegate_task now with exactly one fully specified child task. "
        "Its goal is to independently analyze the same user objective in read-only mode; its context must prohibit "
        "edits, commands, external messages, deploys, credentials, database/payment operations, and destructive "
        "actions. Request a concise evidence-based plan, risks, test/review checklist, and proposed answer.\n"
    )
    _append_user_instruction(shadow, instruction)
    delegate_tool = _find_delegate_tool(shadow)
    if delegate_tool is None:
        return shadow
    # Codex Responses has historically treated a named function choice as a
    # best-effort hint. A one-tool, required call is deterministic and leaves
    # the normal complete toolset untouched on the following parent iteration.
    shadow["tools"] = [delegate_tool]
    shadow["tool_choice"] = "required"
    shadow["parallel_tool_calls"] = False
    return shadow


def _append_user_instruction(request: Dict[str, Any], instruction: str) -> None:
    """Append an internal instruction to the latest user turn in the request's
    own wire shape.

    ``input_text`` is a Responses-API part type.  Hermes converts to the
    provider wire format in ``build_api_kwargs`` *before* llm_request
    middleware runs, so an ``anthropic_messages`` route (the TokenPlan Qwen
    endpoint) reaches this code already Anthropic-shaped, where ``input_text``
    is not a valid content block.  Emitting it there either 400s the call or
    gets the block dropped — which is how a "forced" preflight can arrive at
    the model with its entire instruction missing.
    """
    items = _request_items(request)
    _, index = _last_user_text_and_index(items)
    if index < 0 or not isinstance(items[index], dict):
        return
    block_type = (
        "input_text"
        if not isinstance(request.get("messages"), list) and isinstance(request.get("input"), list)
        else "text"
    )
    content = items[index].get("content")
    if isinstance(content, str):
        items[index]["content"] = content + instruction
    elif isinstance(content, list):
        items[index]["content"] = [*content, {"type": block_type, "text": instruction}]


def _tool_names(request: Any) -> list:
    """Tool names in a request, in either wire shape; [] when there are none."""
    if not isinstance(request, dict):
        return []
    names = []
    for tool in request.get("tools") or []:
        if isinstance(tool, dict):
            name = tool.get("name") or (tool.get("function") or {}).get("name")
            if name:
                names.append(str(name))
    return names


# Anthropic OAuth requests are normalised for Claude Code compatibility, which
# prefixes every tool name with ``mcp__``. Matching the bare name there found
# nothing, so a parent on a Claude account was told it had no delegate_task tool
# and skipped its preflight — the one path where spreading work matters most.
_DELEGATE_TOOL_NAMES = frozenset({"delegate_task", "mcp__delegate_task"})


def _is_delegate_tool_name(name: Any) -> bool:
    return isinstance(name, str) and name in _DELEGATE_TOOL_NAMES


def _find_delegate_tool(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the delegate_task tool definition in any wire shape or naming."""
    for tool in request.get("tools") or []:
        if isinstance(tool, dict) and (
            _is_delegate_tool_name(tool.get("name"))
            or _is_delegate_tool_name((tool.get("function") or {}).get("name"))
        ):
            return tool
    return None


def _tool_schema_slot(tool: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Return the (owner, key) pair holding a tool's JSON schema.

    OpenAI/Codex tools keep it at ``function.parameters``; Anthropic Messages
    tools keep it at ``input_schema``.  Reading only ``parameters`` silently
    skipped every schema-hardening step on the Anthropic path, so the
    ``role="orchestrator"`` constraint never reached a Qwen planner.
    """
    if isinstance(tool.get("function"), dict):
        return tool["function"], "parameters"
    if isinstance(tool.get("input_schema"), dict):
        return tool, "input_schema"
    return tool, "parameters"


def _host_delegate_has_model(request: Any) -> bool:
    """Whether this host's delegate_task really takes a ``model`` argument.

    Hermes v0.21.3 has none. Telling a conductor to set one sends every
    cross-account leaf to the default route while it believes otherwise.
    """
    if not isinstance(request, dict):
        return False
    tool = _find_delegate_tool(request)
    if tool is None:
        return False
    owner, key = _tool_schema_slot(tool)
    schema = owner.get(key)
    return isinstance(schema, dict) and "model" in (schema.get("properties") or {})


def _dispatch_phrase(target: str) -> str:
    """How a conductor reaches a target: the delegate_claude tool for Claude, else model:<name>.

    While Claude delegation is active, a non-Claude target is no longer phrased as
    ``model:<name>`` either -- that reads as the same delegate_task 'model'
    parameter this host does not have. It is named as the goal-prefix route
    instead; the Claude route through ``delegate_claude`` is unaffected.
    """
    if not claude_delegation.is_active():
        return f"model:{target}"
    tier = claude_delegation.TIER_FOR_TARGET.get(target)
    if tier:
        return f'delegate_claude(tier="{tier}")'
    return f"delegate_task (goal prefix [{target}])"


def _is_anthropic_shaped(request: Dict[str, Any]) -> bool:
    """True when the request already carries the Anthropic Messages shape."""
    if not isinstance(request.get("messages"), list):
        return False
    return any(
        isinstance(tool, dict) and isinstance(tool.get("input_schema"), dict)
        for tool in request.get("tools") or []
    )


def _supports_forced_tool_choice(kwargs: Dict[str, Any], decision: RouteDecision) -> bool:
    """Whether this route can be made to call a tool by protocol.

    TokenPlan's Anthropic-compatible Qwen endpoint rejects ``tool_choice``
    outright — including ``{"type": "auto"}`` — so Hermes' Anthropic adapter
    omits the field there.  A preflight on that route is therefore a prompt
    contract, never an enforced one.  This matters because the preflight also
    amputates the toolset to a single tool: without the matching
    ``tool_choice`` the parent is left with one optional tool and no way to do
    anything else, which is strictly worse than not preflighting at all.
    """
    if "qwen" in str(decision.model).casefold():
        return False
    if str(kwargs.get("provider") or "").casefold() == "qwen-token":
        return False
    return "token-plan." not in str(kwargs.get("base_url") or "").casefold()


_HERMES_CONFIG_PATH = hermes_path("~/.hermes/config.yaml")


def _hermes_delegation_target_names() -> Tuple[str, ...]:
    """Targets the host will actually accept in ``delegate_task(model=...)``.

    Read from Hermes's own ``delegation.targets`` rather than this plugin's
    config: that map builds the tool's ``model`` enum, and the host silently
    drops any target with an empty model. Naming one here that the host has
    dropped would point the planner at a route that cannot spawn.
    """
    if yaml is None:
        return ()
    try:
        raw = yaml.safe_load(_HERMES_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        targets = (raw.get("delegation") or {}).get("targets") or {}
        return tuple(sorted(
            str(name).strip().casefold()
            for name, spec in targets.items()
            if isinstance(spec, dict) and str(spec.get("model") or "").strip()
        ))
    except Exception:
        return ()


def _delegation_target_names() -> Tuple[str, ...]:
    """Every delegation target the conductor may be offered.

    Hermes's ``delegation.targets`` plus, while ``delegate_claude`` is registered,
    Claude delegation's targets. It reaches them through its own tool rather
    than ``delegate_task(model=...)``, so they need no entry in Hermes's config --
    and without this, ``haiku`` could never appear in a recommendation.
    """
    names = set(_hermes_delegation_target_names())
    if claude_delegation.is_active():
        names |= set(claude_delegation.target_names(_load_config()))
    return tuple(sorted(names))


def _recent_account_load(cfg: Dict[str, Any], window_seconds: int) -> Dict[str, int]:
    """Calls per account over the recent window, read from this router's own log.

    Call counts, not quota readings: the runtime does not report tokens or cost
    to the route log, so anything phrased as "83% used" would be invented. A
    relative load figure is what the data supports, and it is enough to tell an
    idle account from a busy one.

    Only the tail of the log is parsed. It reaches tens of megabytes, and this
    runs on the preflight path where a full scan would be felt.
    """
    log_cfg = cfg.get("logging") or {}
    path = hermes_path(log_cfg.get("path") or "")
    tier_providers = cfg.get("tier_providers") or {}
    if not str(path) or not tier_providers:
        return {}
    cutoff = datetime.now(timezone.utc).timestamp() - max(60, int(window_seconds))
    counts: Dict[str, int] = {}
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - _USAGE_TAIL_BYTES))
            if size > _USAGE_TAIL_BYTES:
                handle.readline()  # discard the partial line the seek landed in
            for raw in handle:
                try:
                    entry = json.loads(raw)
                    observed = datetime.fromisoformat(
                        str(entry.get("timestamp") or "").replace("Z", "+00:00")
                    )
                except Exception:
                    continue
                if observed.tzinfo is None:
                    observed = observed.replace(tzinfo=timezone.utc)
                if observed.timestamp() < cutoff:
                    continue
                account = tier_providers.get(str(entry.get("tier") or ""))
                if account:
                    counts[account] = counts.get(account, 0) + 1
    except OSError:
        return {}
    return counts


def _peers_for(name: str, cfg: Dict[str, Any]) -> Tuple[str, ...]:
    """Targets of comparable strength that can take this one's work."""
    for members in (cfg.get("peer_groups") or {}).values():
        if isinstance(members, list) and name in members:
            return tuple(peer for peer in members if peer != name)
    return ()


def _account_of(name: str, cfg: Dict[str, Any]) -> str:
    return str((cfg.get("tier_providers") or {}).get(name) or "")


def _account_states(cfg: Dict[str, Any], readings: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """State per guarded account, from the non-blocking cached reading.

    Fails open: a malformed ``usage_guard`` block (a bad percent, a broken
    accounts map, ...) must not take routing down with it, so any exception
    here is swallowed and reported as "no guarded accounts" instead.
    """
    try:
        accounts = (usage_guard.guard_config(cfg).get("accounts") or {})
        return {
            account: usage_guard.state(
                account, cfg, readings[account] if readings is not None else usage_guard.peek(account, cfg)
            )
            for account in accounts if usage_guard.guarded(account, cfg)
        }
    except Exception:
        _logger.warning("_account_states failed; reporting no guarded accounts", exc_info=True)
        return {}


def _account_mark(name: str, cfg: Dict[str, Any], states: Dict[str, str]) -> str:
    try:
        account = _account_of(name, cfg)
        state = states.get(account)
        if state == "soft":
            return f" [{usage_guard.account_label(account)} soft limit]"
        if state == "closed":
            return f" [{usage_guard.account_label(account)} closed]"
        return ""
    except Exception:
        _logger.warning("_account_mark failed for %r; no mark added", name, exc_info=True)
        return ""


def _target_availability(
    names: Iterable[str], cfg: Dict[str, Any], states: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Per-target cooldown note, empty when the target is available.

    Cooling targets are annotated rather than dropped. LiteLLM excludes a
    deployment that would exceed its limit, but its deployments are
    interchangeable and ours are not: hiding a cooling Sol would invite the
    planner to send design work somewhere it is not allowed, which the
    classifier then refuses outright. Saying "unavailable, and for how long"
    lets the conductor wait or narrow the objective instead.
    """
    offered = set(names)
    states = _account_states(cfg) if states is None else states
    notes = {}
    for name in names:
        remaining = _tier_cooldown_remaining(name, cfg)
        if not remaining:
            notes[name] = _account_mark(name, cfg, states)
            continue
        alive = [
            peer for peer in _peers_for(name, cfg)
            if peer in offered and not _tier_cooldown_remaining(peer, cfg)
        ]
        instead = f"; use {' or '.join(alive)} instead" if alive else ""
        notes[name] = f" [unavailable for another {int(remaining // 60) + 1} min{instead}]" + _account_mark(name, cfg, states)
    return notes


def _delegation_targets_detail() -> Dict[str, Dict[str, str]]:
    """``{name: {provider, model}}`` from Hermes's own ``delegation.targets``."""
    if yaml is None:
        return {}
    try:
        raw = yaml.safe_load(_HERMES_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        targets = (raw.get("delegation") or {}).get("targets") or {}
        return {
            str(name).strip().casefold(): {
                "provider": str(spec.get("provider") or "").strip().casefold(),
                "model": str(spec.get("model") or "").strip(),
            }
            for name, spec in targets.items()
            if isinstance(spec, dict) and str(spec.get("model") or "").strip()
        }
    except Exception:
        return {}


def _external_target_for_model(model: str) -> Optional[str]:
    """Target name for a model this router cannot route but should still record.

    A child on another provider is invisible here by design -- the middleware
    cannot move a call across providers, so it returns None. But invisible to
    the router became invisible to the operator too: a Claude worker produced no
    card, no count and no line in the per-account load, so the one account whose
    usage most needed watching was the one nothing reported on.
    """
    if not model:
        return None
    for name, spec in _delegation_targets_detail().items():
        if spec.get("model") == model:
            return name
    # Observing an existing worker must not depend on whether this request can spawn one.
    return claude_delegation.target_for_model(model, _load_config())


def _target_is_offered(name: str, cfg: Dict[str, Any]) -> bool:
    """Whether a delegation target should be put in front of the conductor.

    A target with a callability switch obeys it even when it is not a routable
    tier: the dashboard toggle otherwise reads as if it governed Claude while
    changing nothing.
    """
    switches = cfg.get("callable") or {}
    # Deliberately not _is_callable_tier: that folds in the cooldown, and a
    # cooling target must stay visible. Hiding it invites the planner to route
    # work somewhere it is not allowed -- a cooling Sol does not make design work
    # someone else's job -- and it hides the fact that waiting is an option.
    return switches.get(name) is True if name in switches else True


def _read_only_leaf_tier(cfg: Optional[Dict[str, Any]] = None) -> str:
    """The tier a bounded read-only evidence leaf may actually be labelled with.

    The conductor contract named Spark unconditionally. With ``callable.spark``
    switched off Spark is not offered at all, so that sentence pointed the
    conductor at a tier it cannot spawn and left read-only discovery with no
    labelled home -- and an unlabelled goal is the one case the design gate
    still decides from raw keywords.
    """
    cfg = cfg if isinstance(cfg, dict) else _load_config()
    for tier in ("spark", "luna"):
        if _target_is_offered(tier, cfg):
            return tier
    return ""


def _read_only_leaf_sentence(cfg: Optional[Dict[str, Any]] = None) -> str:
    """What the bounded read-only worker is for, named after a tier that exists."""
    tier = _read_only_leaf_tier(cfg)
    if not tier:
        return ""
    return (
        f"Use [{tier}] with model:{tier} for bounded low-risk read-only source/component discovery, "
        "logs, test-case design, isolated patch proposals, or research. Reading and mapping source "
        "that happens to contain UI is such a leaf, not design work. "
    )


def _read_only_delegation_clause(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Name the read-only worker's route in the conductor's own contract."""
    tier = _read_only_leaf_tier(cfg)
    if not tier:
        return ""
    return (
        f"Delegate bounded, self-contained low-risk non-design read-only evidence loops to "
        f"{tier.capitalize()} with a goal beginning [{tier}] and model:{tier}. "
    )


def _leaf_label_contract(cfg: Optional[Dict[str, Any]] = None) -> str:
    """The rule that every leaf goal opens with its own tier label.

    Stating the prefix per tier -- "prefix a Spark leaf with [spark]", "design
    must be prefixed [sol]" -- only ever covered the tiers it named, so a leaf
    the conductor did not file under one of them went out with no label. That is
    not neutral: the design gate runs ahead of every label check and is waived
    only for a labelled worker, so an unlabelled goal is classified from its own
    words, where one mention of UI, UX, CSS or layout reads as design work and
    pins the leaf to Sol however read-only it is. Measured on 2026-09-18: two
    read-only discovery leaves ("identify ... UI components and existing
    route/UI tests", "the STAFF receipt issuer self-service UI") both opened on
    Sol from their first call, on the bare word "ui".
    """
    cfg = cfg if isinstance(cfg, dict) else _load_config()
    labels = [
        f"[{tier}]" for tier in ("luna", "spark", "terra", "sol")
        if _target_is_offered(tier, cfg)
    ]
    if not labels:
        return ""
    return (
        f"Begin every worker goal with that leaf's own tier label in square brackets "
        f"({', '.join(labels)}), matching the chosen worker route. This is not optional "
        "and not only for design or read-only leaves: a goal that starts with no label is "
        "re-classified from its own text, and there a single mention of UI, UX, CSS, layout or "
        "styling is read as design work and forces the leaf onto Sol no matter how read-only it "
        "is -- naming UI files to grep is enough to trigger it. The label is what tells the router "
        "the tier was already decided by a planner that saw the objective. "
    )


def _model_param_contract(
    orchestrator_tier: str, cfg: Optional[Dict[str, Any]] = None, *, model_param: bool = True
) -> str:
    """The sentence that makes route choice expressible instead of implied.

    A ``[sol]``/``[spark]`` goal prefix is only a model rename inside the
    default provider, so it can never reach a target that lives on a separate
    account. Without this, every leaf inherits the default route: the delegation
    registry shows every child ever spawned running on the default model, even
    ones whose goal was explicitly prefixed for another target.
    """
    cfg = cfg if isinstance(cfg, dict) else _load_config()
    # A target whose tier is switched off is not a route: the cross-provider guard
    # raises for it mid-session, so offering it produces a leaf that never runs.
    names = [
        name for name in _delegation_target_names()
        if name != orchestrator_tier and _target_is_offered(name, cfg)
    ]
    # The operator's order is the one place the orchestrator's own tier belongs:
    # excluding it from ``names`` is right for "other targets to spread across",
    # but a chain like code: opus5 > terra > qwen then rendered as "code: opus5"
    # and lost its next step -- exactly the entry that has to take over when the
    # first one runs out. A leaf on the conductor's own account is allowed; it is
    # merely not a way to spread load.
    preference_names = [
        name for name in _delegation_target_names() if _target_is_offered(name, cfg)
    ]
    if not model_param:
        reachable = lambda n: (_account_of(n, cfg) == cfg.get("provider", "openai-codex")
                               or (claude_delegation.is_active() and n in claude_delegation.TIER_FOR_TARGET))
        names = [n for n in names if reachable(n)]
        preference_names = [n for n in preference_names if reachable(n)]
    notes = _target_availability(names, cfg)
    scope = (
        f" (targets: {', '.join(name + notes.get(name, '') for name in names)})" if names else ""
    )
    if model_param:
        opening = (
            f"Set the delegate_task 'model' parameter on every worker to choose its route{scope}. "
            "A goal-text prefix only renames the model inside the default provider and cannot reach a "
            "target on a separate account, so a leaf intended for one must carry model:<name>. "
        )
    else:
        opening = (
            f"{_ROUTE_CHOICE_OPENING}{scope}: this host's delegate_task has no model parameter, so a "
            "goal-text prefix picks a tier inside the default provider and cannot reach a target on a "
            "separate account. "
        )
    if claude_delegation.is_active():
        claude_rule = (
            "Concretely: [opus5] and [sonnet5] are not labels, and neither is a model parameter. A goal "
            "beginning with one is not routed to Claude; the prefix is inert, the goal is classified on its "
            "remaining text, and the leaf runs on this provider -- so it is stopped at its first call and "
            "returned for re-dispatch. Claude targets are reached only by calling delegate_claude with tier "
            "\"haiku\", \"sonnet\" or \"opus\". "
        ) + _DEFERRED_DELEGATION_HINT
    elif model_param:
        # The general rule was already here and lost anyway, seven goals running.
        # It shares a paragraph with [spark]/[sol], which *are* prefixes, so
        # [opus5] is the obvious blend of the two mechanisms -- and it silently
        # became a Sol leaf. Naming the mistake beats restating the rule.
        claude_rule = (
            "Concretely: [opus5] and [sonnet5] are not labels. A goal beginning with one is not "
            "routed to Claude; the prefix is inert, the goal is classified on its remaining text, "
            "and the leaf runs on this provider -- so it is stopped at its first call and returned "
            "for re-dispatch. Name those targets only in the model parameter. "
        )
    else:
        claude_rule = "Cross-account targets cannot be reached through delegate_task on this host. "
    return (
        opening
        + claude_rule
        + "Prefer spreading genuinely independent leaves across different targets so separate accounts and "
        "quotas absorb the work in parallel; never split work merely to use more targets. "
        f"{_peer_group_sentence(names, cfg)}"
        f"{_claude_target_sentence(names, cfg)}"
        # preference_names, not names: this rule is about the account a leaf runs
        # on, not about spreading load, and `names` drops the conductor's own tier.
        # With the operator's code chain making opus5 the conductor, that dropped
        # opus5 -- the one leaf the rule was written from -- out of its own rule.
        f"{_recon_before_expensive_target_sentence(preference_names, cfg)}"
        f"{_preference_sentence(preference_names, cfg, model_param=model_param)}"
        f"{_goal_orientation_sentence()}"
        f"{_account_load_sentence(cfg)}"
    )


def _preference_sentence(names: Iterable[str], cfg: Dict[str, Any], *, model_param: bool = True) -> str:
    """State the operator's per-work-kind target order to the conductor.

    The router cannot route across providers, so a preference naming an external
    account is only ever realisable here: the conductor is the one that picks a
    delegate_task target. Only offered targets are mentioned — advising a leaf onto
    a switched-off account would produce a child that never runs.

    Two things were wrong with stating only the winner, as advice.

    The chain vanished exactly when it mattered. Only the single first available
    target was named, and availability folds in the cooldown, so a cooling opus5
    erased ``code`` from the contract altogether -- indistinguishable from a kind
    the operator never configured. The conductor could not advance to the next
    entry because it was never told there was one. The whole order is stated now,
    with the cooling entries annotated rather than dropped, for the same reason
    ``_target_availability`` annotates them.

    And it was phrased as a request while the ``[spark]``/``[sol]`` rules in the
    same paragraph are imperatives, so the two contradicted each other and the
    imperative won every time. The operator's configuration is not weaker than a
    built-in rule; it is the one thing here that was chosen deliberately.
    """
    # Local tiers are reachable by label even when the host has no named model
    # targets. Use the same candidate and ordering logic as root advice.
    offered = set(names) | {
        name for name in (cfg.get("models") or {})
        if _account_of(name, cfg) == str(cfg.get("provider", "openai-codex"))
        and _target_is_offered(name, cfg)
    }
    readings = _guarded_readings(cfg)
    states = _account_states(cfg, readings)
    notes = _target_availability(offered, cfg, states=states)
    chains = []
    for kind in WORK_KINDS:
        if claude_delegation.is_active():
            chain, _ = _advised_chain(kind, cfg, states, offered, readings)
        else:
            chain = [name for name in _preference_list(kind, cfg) if name in offered]
        if not chain:
            continue
        chains.append(f"{kind}: " + " > ".join(name + notes.get(name, "") for name in chain))
    if not chains:
        return ""
    decides = (
        "and it decides which call carries the leaf: a Claude target goes through delegate_claude with its "
        "tier, any other target through delegate_task. "
        if claude_delegation.is_active() else
        "and it decides the leaf's model: parameter. " if model_param else
        "and it decides the leaf's goal-prefix route inside the configured provider. "
    )
    return (
        "The operator's target order per kind of work, highest priority first -- "
        + "; ".join(chains)
        + ". A leaf of one of these kinds must take the first target in that kind's order, "
        "and when an entry is marked unavailable must move to the next entry in the same "
        "order rather than choosing freely. This is the operator's configuration, not a "
        "suggestion, "
        + decides
    )


def _worker_order_note(request: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """Refresh a conductor's capacity advice without reclassifying its goal."""
    names = set(_tool_names(request))
    direct_claude = {claude_delegation.TOOL_NAME, f"mcp__{claude_delegation.TOOL_NAME}"}
    if _find_delegate_tool(request) is None and not names & direct_claude:
        return ""
    order = _preference_sentence(_delegation_target_names(), cfg, model_param=_host_delegate_has_model(request))
    return ("\n\n[ROUTER] Current worker order replaces earlier capacity advice. " + order) if order else ""


def _goal_orientation_sentence() -> str:
    """Require the goal to carry what the worker cannot see for itself.

    Every delegated worker starts at ``history=0``. It does not share this
    conversation on any target, so a fact the conductor knows and does not write
    down is a fact the worker must spend iterations rediscovering -- against a
    budget that ``delegate_task`` cannot raise per leaf, because the host treats
    ``delegation.max_iterations`` as authoritative and ignores the argument.

    Measured, on one Opus leaf: sixteen iterations, twenty tool calls, an input
    context grown from 20k to 56k, and not one edit -- the entire budget spent
    reconstructing a repository the goal never described. The leaf whose goal
    carried its own state and asked for a single artefact finished in nine with
    one write. The difference was the goal, not the model and not the budget.
    """
    return (
        "A worker sees its goal and nothing else: it does not share this conversation, so a "
        "fact you leave out is one it must spend iterations rediscovering, and its iteration "
        "budget is fixed and cannot be raised per leaf. Every goal therefore states the "
        "absolute worktree path, the branch and the commit it builds on, what already exists "
        "there, which files or modules are in scope, and how the result is verified. Give one "
        "worker one finishable artefact rather than a feature to implement: a goal phrased as "
        "a product requirement has no boundary, and it is spent on orientation before the "
        "first edit. "
    )


_CONTEXT_CONTRACT_MARKER = "worktree it runs in"
_CONTEXT_CONTRACT_DESCRIPTION = (
    "Required. The orientation this child cannot see for itself: the absolute path of the "
    "worktree it runs in, the branch and the commit it builds on, what already exists there, "
    "which files or modules are in scope, and how its result is verified. Each child sees only "
    "its own context, so repeat shared background in every task that needs it. Write \"none\" "
    "only when the goal genuinely depends on no repository state."
)
_GOAL_CONTRACT_MARKER = "absolute worktree path"
_GOAL_CONTRACT_CLAUSE = (
    " Every fact it needs must be here: the absolute worktree path, the branch and the "
    "commit it builds on, what already exists there, which files or modules are in scope, "
    "and how the result is verified. Give it one finishable artefact, not a feature to "
    "implement -- its iteration budget is fixed and cannot be raised per task, so a goal "
    "with no boundary is spent on orientation before the first edit. A task on an external "
    "Claude target (model: opus5 or sonnet5) is the most expensive place in the fleet to "
    "discover any of this: dispatch a cheap read-only recon task first and carry its findings "
    "in this task's context, rather than letting the Claude leaf do the reading itself."
)


def _with_goal_contract(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Carry the goal requirements on the delegate_task schema itself.

    These rules only ever travelled inside the forced preflight, so a turn that
    skipped it delegated with nobody having been told what a goal has to carry.
    A twelve-character root prompt falls under ``orchestration.min_chars``, no
    conductor is created, and the parent dispatches straight from its own toolset
    -- which is how a whole-feature goal went out with no worktree, branch or base
    commit in it, and the leaf spent all sixteen iterations rediscovering them.

    On the schema rather than appended to the message because a middleware edit
    does not persist into the conversation -- that is why the preflight needs a
    rescue pass at all. The parent may delegate on any call of the turn, so an
    appended sentence would have to be repeated on every one of them; a tool
    description is read once, exactly where the goal is written.

    Returns None when there is nothing to change, so the caller can tell a
    no-op from a rewrite.
    """
    tool = _find_delegate_tool(request)
    if tool is None:
        return None
    routed = deepcopy(request)
    tool = _find_delegate_tool(routed)
    schema_owner, schema_key = _tool_schema_slot(tool)
    schema = schema_owner.get(schema_key)
    if not isinstance(schema, dict):
        return None
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None
    # The advertised batch shape, plus the legacy single-goal one the handler
    # still accepts: a request carrying either must not slip through unannotated.
    objects = [(properties.get("tasks") or {}).get("items"), schema]
    changed = False
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        slot = obj.get("properties")
        if not isinstance(slot, dict):
            continue
        goal = slot.get("goal")
        if isinstance(goal, dict):
            description = str(goal.get("description") or "")
            if _GOAL_CONTRACT_MARKER not in description:
                goal["description"] = description + _GOAL_CONTRACT_CLAUSE
                changed = True
        context = slot.get("context")
        if isinstance(context, dict):
            if _CONTEXT_CONTRACT_MARKER not in str(context.get("description") or ""):
                context["description"] = _CONTEXT_CONTRACT_DESCRIPTION
                changed = True
            # The forcing step. A description is advice and lost three times over
            # -- against the built-in tie-breaker, against the [spark]/[sol]
            # vocabulary, and here against nothing at all. Requiring the field
            # makes a context-free call invalid rather than merely discouraged,
            # and on a `strict` tool the provider is the one enforcing it.
            required = obj.get("required")
            if isinstance(required, list) and "context" not in required:
                required.append("context")
                changed = True
    return routed if changed else None


def _peer_group_sentence(names: Iterable[str], cfg: Dict[str, Any]) -> str:
    """Name the substitutions, so a loaded target is a choice rather than a wait.

    Dropping an unavailable target from the list told the planner only that it
    was gone. Knowing what replaces it is what turns one account's exhaustion
    into work continuing somewhere else.
    """
    offered = set(names)
    groups = [
        [name for name in members if name in offered]
        for members in (cfg.get("peer_groups") or {}).values()
        if isinstance(members, list)
    ]
    usable = [group for group in groups if len(group) > 1]
    if not usable:
        return ""
    listed = "; ".join(" / ".join(group) for group in usable)
    return (
        f"Comparable in strength and on different accounts, so they substitute for each other "
        f"when one is loaded or unavailable: {listed}. Substituting is for capacity only -- the "
        f"rules each label carries still apply, so a Spark leaf must still be read-only and design "
        f"work still belongs to Sol. "
    )


# The targets that draw on the Claude subscription. One owner, because two rules
# now turn on "is this leaf on the expensive account" and they must not drift.
_EXPENSIVE_TARGETS = frozenset({"opus5", "sonnet5"})


def _claude_target_sentence(names: Iterable[str], cfg: Optional[Dict[str, Any]] = None) -> str:
    """What the Claude targets are for, once they are offered at all.

    A bare name in a list tells the conductor nothing about when to reach for it,
    and these are the two that draw on a different subscription entirely -- the
    reason the target list exists.

    The built-in "sonnet5 by default" tie-breaker only speaks where the operator
    has not. A per-kind preference naming a Claude target answers the same
    question, and emitting both put two contradictory instructions in one
    paragraph: the unconditional default beat the hedged preference sentence
    every time, so ``code -> model:opus5`` never once decided a leaf.
    """
    claude_names = _EXPENSIVE_TARGETS | ({"haiku"} if claude_delegation.is_active() else frozenset())
    claude = [name for name in names if name in claude_names]
    if not claude:
        return ""
    both = "opus5" in claude and "sonnet5" in claude
    # Read the configured lists, not the currently-available winner: a cooling
    # opus5 would otherwise revive the built-in default mid-session, which is the
    # one moment the operator's own order needs to be the thing that speaks.
    operator_chose = bool(cfg) and any(
        name in claude
        for kind in WORK_KINDS
        for name in _preference_list(kind, cfg)
    )
    reach = (
        'Reach them with delegate_claude(tier="haiku"|"sonnet"|"opus"), never with delegate_task. '
        if claude_delegation.is_active() else ""
    )
    return (
        f"{' and '.join(claude)} run on Claude, a different subscription from every other "
        f"target, so they are the strongest way to keep independent work off a single quota. "
        f"They are ordinary workers with the usual tools: give them implementation or deep "
        f"review, not just reading. "
        + reach
        + ("Use sonnet5 by default and reserve opus5 for consequential or hard work. "
           if both and not operator_chose else "")
    )


def _recon_target_names(names: Iterable[str], cfg: Dict[str, Any]) -> list:
    """The offered targets a read-only recon leaf should run on, cheapest first.

    The operator's ``explore`` order owns this when it is set; the light peer
    group answers the same question when it is not. Both are read rather than
    hardcoded because "cheap" is an account fact, not a property of a name.
    """
    offered = [name for name in names if name not in _EXPENSIVE_TARGETS]
    chain = [name for name in _preference_list("explore", cfg) if name in offered]
    if chain:
        return chain[:2]
    for members in (cfg.get("peer_groups") or {}).values():
        if not isinstance(members, list) or _EXPENSIVE_TARGETS.intersection(members):
            continue
        light = [name for name in members if name in offered]
        if light:
            return light[:2]
    return []


def _recon_before_expensive_target_sentence(
    names: Iterable[str], cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """Keep an external Claude leaf off its own orientation.

    A worker on a separate subscription is the most expensive place in the fleet
    to read a repository, and its iteration budget is the one the host refuses to
    raise per leaf -- so orientation spent there is spent at the highest price and
    buys no edit. Measured, on the leaf this rule is written from: sixteen
    iterations, twenty tool calls, every one of them a read, and a final summary
    that said "I hit the tool-call iteration limit during the codebase-
    understanding phase, before writing any tests or implementation."

    Stated as a dispatch order to the conductor rather than as advice to the leaf,
    because the leaf cannot act on it from inside. It had ``delegate_task`` and the
    depth to use it; by the time it knew enough to hand the reading away it had
    already paid for the context it would have been handing away.
    """
    cfg = cfg if isinstance(cfg, dict) else {}
    claude = [name for name in names if name in _EXPENSIVE_TARGETS]
    recon = _recon_target_names(names, cfg)
    # No cheap target on offer means the reading has nowhere else to go; an
    # instruction to move it would only cost the leaf a refused dispatch.
    if not claude or not recon:
        return ""
    return (
        f"{' and '.join(claude)} must not spend a budget on orientation: it is fixed, it cannot "
        f"be raised per leaf, and reading is the one thing every other target does for less. "
        f"Dispatch a read-only recon leaf on {' or '.join(recon)} first -- the files and symbols "
        f"in scope, what already exists there, how the result is verified -- and carry its "
        f"findings in the `context` of the {' / '.join(claude)} leaf. Send that leaf only once "
        f"its goal can name what it will change. A goal that begins with discovery is one whose "
        f"budget is gone before the first edit. "
    )


def _account_load_sentence(cfg: Dict[str, Any]) -> str:
    """Tell the conductor where the traffic has actually been going.

    Spreading work was previously an instruction with nothing behind it: the
    conductor was told to use separate accounts but had no way to see that one
    of them had taken every call for the last hour and another had taken none.
    """
    policy = cfg.get("usage_report") or {}
    if not policy.get("enabled", True):
        return ""
    window = int(policy.get("window_seconds", 3600) or 3600)
    counts = _recent_account_load(cfg, window)
    if not counts:
        return ""
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    summary = ", ".join(f"{account} {calls}" for account, calls in ordered)
    idle = [
        account
        for account in sorted({str(v) for v in (cfg.get("tier_providers") or {}).values()})
        if counts.get(account, 0) == 0
    ]
    # "No calls" reads as spare capacity, but it is equally what an exhausted
    # account looks like -- which is exactly what Qwen was when this sentence
    # last recommended it. State the fact and leave the inference alone.
    tail = f" {', '.join(idle)} has taken none in this window." if idle else ""
    return (
        f"Recent load over the last {window // 60} minutes, in calls per account: {summary}. "
        f"These are call counts from this router's own log, not quota readings -- read them as "
        f"relative load.{tail}"
    )


def _conductor_tier(
    cfg: Optional[Dict[str, Any]],
    request: Optional[Dict[str, Any]] = None,
    parent: str = "",
) -> str:
    """The tier the forced conductor child should run on.

    ``orchestration.conductor`` when the operator has pinned one, else the
    parent's own tier (``parent``, while ``orchestration.conductor_follows_parent``
    is on), else ``default_model``, else the first callable tier its ``fallbacks`` chain
    reaches. Every step is skipped when its tier cannot be called: pinning the
    conductor to a configured default is what made an exhausted account fail the
    whole preflight -- the parent had already moved to a working account, and its
    planner was still being sent to the one that had run out.

    This used to read ``preferences.code`` on the reasoning that planning and
    coordination are code work. They are not the same question. ``code`` says
    where an implementation *leaf* belongs, and the moment an operator answered
    that with ``[opus5, terra, qwen]`` -- a deliberate choice about who writes the
    code -- it silently also moved every conductor onto the Claude subscription,
    where coordination then paid external-account prices for planning. One key
    answering two unrelated questions cannot be set correctly for both, so the
    conductor now has its own.

    The parent step exists because the default made Codex the only planner: a
    session switched to Grok (2026-09-25) handed every plan back to Terra, or --
    before any tier could orchestrate -- worked alone. Following the parent
    keeps planning on the account the operator chose for the session.
    """
    cfg = cfg or {}
    model_param = _host_delegate_has_model(request)
    targets = _delegation_targets_detail() if request is not None and model_param else {}

    def reachable(tier: str) -> bool:
        if not _is_callable_tier(tier, cfg) or not _target_is_offered(tier, cfg):
            return False
        if request is None:
            return True
        # A goal label can only change the model within the delegate_task
        # provider. An off-provider planner requires a named host target and
        # an actual model field in this request's schema.
        account = _account_of(tier, cfg) or targets.get(tier, {}).get("provider")
        if account and account != cfg.get("provider", "openai-codex"):
            return model_param and tier in targets
        if tier in (cfg.get("models") or {}):
            return True
        return model_param and tier in targets

    pinned = str((cfg.get("orchestration") or {}).get("conductor") or "").strip().casefold()
    if pinned and reachable(pinned):
        return pinned
    parent = str(parent or "").strip().casefold()
    follows_parent = bool((cfg.get("orchestration") or {}).get("conductor_follows_parent", True))
    # Spark is a leaf-only tier: a root Spark label is already deferred to the default.
    if follows_parent and parent and parent != "spark" and reachable(parent):
        return parent
    default = str(cfg.get("default_model", "terra"))
    if reachable(default):
        return default
    chain = cfg.get("fallbacks") or {}
    seen, current = {default}, default
    for _ in range(3):
        nxt = str(chain.get(current) or "")
        if not nxt or nxt in seen:
            break
        if reachable(nxt):
            return nxt
        seen.add(nxt)
        current = nxt
    for tier in dict.fromkeys((*((cfg.get("models") or {}).keys()), *targets.keys())):
        if reachable(tier):
            return tier
    return default if request is None else ""


def _prepare_orchestration_delegation(
    request: Dict[str, Any],
    plan_id: str,
    max_tasks: int,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    force_tools: bool = True,
    claude_choice: Tuple[str, str, str] = ("", "", ""),
    parent_tier: str = "",
) -> Dict[str, Any]:
    """Force one real conductor-supervised dispatch before parent execution.

    This is an operational delegation checkpoint, never a benchmark: the
    parent must use returned evidence, explicitly accept/reject it, and retain
    integration ownership.

    ``claude_choice`` is (kind, target, balanced); target is set when the turn's
    first choice is a Claude tier, and balanced explains a load-balancing reorder. The call then offers delegate_claude next to delegate_task and still
    requires one of them -- forcing delegate_task alone made that preference
    unreachable (a review turn ran on Terra instead of Sonnet, 2026-09-19).
    """
    orchestrator_tier = _conductor_tier(cfg, request, parent_tier)

    routed = deepcopy(request)
    claude_kind, claude_target, balanced = claude_choice
    claude_tool, via_bridge = _claude_route_tool(request) if claude_target else (None, False)
    if claude_tool is None:
        claude_target = ""
    model_param = _host_delegate_has_model(request)
    original_tool = _find_delegate_tool(request)
    owner, key = _tool_schema_slot(original_tool) if original_tool else ({}, "")
    batch_shape = "tasks" in ((owner.get(key) or {}).get("properties") or {})
    planner_call = 'delegate_task with one tasks entry' if batch_shape else 'delegate_task with role="orchestrator"'
    claude_hint = (
        "For real work prefer a native Claude worker through delegate_claude when one is offered. "
        + _DEFERRED_DELEGATION_HINT
        if claude_delegation.is_active() else
        "For real work use a reachable worker route. "
    )
    if claude_target:
        claude_tier = claude_delegation.TIER_FOR_TARGET[claude_target]
        lead = (
            f"Plan ID: {plan_id}. Before any normal tool action, delegate exactly once, by one of two calls. "
            f"CLAUDE ROUTE (preferred): this turn classifies as {claude_kind}, and its "
            + (f"load-balanced first choice is {claude_target} (Balanced: {balanced}), " if balanced else
               f"configured first choice is {claude_target}, ")
            + f"so give the whole objective to one Claude worker with delegate_claude(tier=\"{claude_tier}\"). "
            + ('It is a deferred tool: call it through tool_call with name "delegate_claude" and its arguments '
               '(tier, tasks). ' if via_bridge else "")
            + "Take this route unless the work needs several workers; on it there is no conductor, and you accept "
            "or reject the Claude worker's result yourself. "
            f"CONDUCTOR ROUTE: otherwise call {planner_call} and a goal beginning with "
            f"[{orchestrator_tier}]. "
        )
    else:
        lead = (
            f"Plan ID: {plan_id}. Before any normal tool action, call {planner_call} exactly once "
            f"and a goal beginning with [{orchestrator_tier}]. "
            + (f"Balanced: {balanced}, so this turn's delegation stays off Claude. " if balanced else "")
        )
    instruction = (
        f"\n\n[INTERNAL ORCHESTRATOR PREFLIGHT]\n"
        f"{lead}This creates a dedicated {orchestrator_tier} planner and conductor, not a benchmark worker. "
        "Give that conductor the full current objective as a self-contained goal: it does not see this conversation, "
        "so a short user turn such as 'do it' stands for the plan discussed before it, written out in full. "
        "If the turn needs no real work (an acknowledgement, a direct answer), write a one-line goal instead: "
        "the router then returns control to you without spawning anything. It must first inspect any current image itself and Create a structured dispatch plan "
        f"before any implementation. The plan may contain zero to {max_tasks} independent workers; do not invent work merely to fill slots. "
        f"{orchestrator_tier} chooses the decomposition from the actual task. {_leaf_label_contract(cfg)}"
        f"{_read_only_leaf_sentence(cfg)}"
        "Any visual/product/UI/UX/CSS/layout/design-system analysis or implementation is Sol-only and must be prefixed [sol]; prefix a consequential "
        f"worker goal with [sol] only for security/auth/credentials/payment/migration/production analysis. Workers receive a "
        "self-contained textual scope, never the original image. A read-only leaf must stay read-only: prohibit edits, commands with side effects, "
        "external messages, deploys, credentials, database/auth/payment operations, and destructive actions. "
        f"{_model_param_contract(orchestrator_tier, cfg, model_param=model_param)} "
        "A [sonnet-review] or [opus-review] leaf takes no 'model', because its route is its label; it is read-only "
        "and replaces a single call rather than running an agent. "
        f"{claude_hint}"
        "Write the goal as objective and acceptance criteria only: what must change, where, and how it is verified. "
        "Do not restate this routing policy inside the goal. The conductor already receives it verbatim as an immutable "
        "contract in the required `context` field, and the goal is re-read as a description of the work -- routing "
        "vocabulary repeated there is classified as the work itself and re-routes the conductor away from "
        f"{orchestrator_tier}. "
        f"{orchestrator_tier} keeps coordination, acceptance/rejection, shared-file integration, and final verification; Sol owns all design analysis and design implementation. "
        "It waits for delegated evidence, explicitly records SUPERVISOR DECISION: ACCEPT or REJECT for every worker, then completes the owned work. "
        f"The current parent must not perform normal implementation; the {orchestrator_tier} conductor owns the reviewed result.\n"
    )
    _append_user_instruction(routed, instruction)
    # Third copy of the same lookup, and the one that raised rather than skipping when
    # the name did not match. All three now go through _find_delegate_tool.
    delegate_tool = _find_delegate_tool(routed)
    if delegate_tool is None:
        return routed
    # Do not merely ask the parent to create an orchestrator: constrain the
    # one permitted tool schema so the runtime receives an actual
    # ``role=orchestrator`` child. Natural-language instructions alone are not
    # a reliable control plane, as models can otherwise emit the default leaf
    # role and skip the planner layer altogether.
    planner_tool = deepcopy(delegate_tool)
    schema_owner, schema_key = _tool_schema_slot(planner_tool)
    schema = schema_owner.get(schema_key)
    if isinstance(schema, dict):
        properties = schema.setdefault("properties", {})
        if batch_shape:
            tasks_schema = properties["tasks"]
            tasks_schema["minItems"] = tasks_schema["maxItems"] = 1
            schema["required"] = list(dict.fromkeys([*(schema.get("required") or []), "tasks"]))
            schema = tasks_schema["items"]
            properties = schema.setdefault("properties", {})
        properties.setdefault("goal", {"type": "string"})
        if not batch_shape:
            properties["role"] = {
                "type": "string",
                "enum": ["orchestrator"],
                "description": f"Required fixed role for the {orchestrator_tier} planning child.",
            }
        # This travels in the actual delegated child's system prompt. It avoids
        # relying on the parent model to faithfully copy the planner contract
        # into a free-form goal/context field.
        properties["context"] = {
            "type": "string",
            "enum": [
                f"You are the {orchestrator_tier} planning conductor. Do not perform design analysis or design implementation. {_leaf_label_contract(cfg)}Route every visual/product/UI/UX/CSS/layout/design-system task to Sol with a goal beginning [sol] {'and model:sol' if model_param else 'inside the configured provider'}. {_read_only_delegation_clause(cfg)}Read-only does not make a design question non-design: judging visual hierarchy, appearance, spacing or styling is Sol's work even when nothing is written. That worker receives source discovery, tests, logs and research -- questions with a factual answer, including ones whose answer lives in UI source files. {_model_param_contract(orchestrator_tier, cfg, model_param=model_param)} A purely read-only review leaf may instead be labelled [sonnet-review] or [opus-review], which runs it through the Claude Code CLI on a separate subscription. Use [sonnet-review] for routine checks and [opus-review] for consequential ones. Such a leaf takes no 'model' -- its route is its label -- must name the repository by its absolute path, must carry every fact it needs in the goal, and must never be asked to edit, run commands, or implement. Write every leaf goal as objective and acceptance criteria only: never restate this routing policy inside a leaf goal, because a leaf is re-classified from its own goal text and routing vocabulary repeated there is read as the work itself. The orchestrator retains coordination, evidence acceptance/rejection, integration, and final approval. Use zero leaves only when the objective genuinely has no independently useful non-design text-only investigation, test, source-discovery, or research subtask."
            ],
            "description": f"Required immutable routing contract for the {orchestrator_tier} planner.",
        }
        # The conductor's tier is a *route*, not a prefix. A goal beginning
        # "[qwen]" only renames a model inside the default provider, so with no
        # `model` parameter the planner was created on the delegation default --
        # 24 calls of a Qwen conductor on Terra, the account the delegation
        # exists to spare. The prose contract cannot cover this on its own: it
        # deliberately lists the *other* targets to spread across, so the
        # conductor's own tier is the one name it never offers. Pin it the same
        # deterministic way `role` and `context` are pinned.
        contract_cfg = cfg if isinstance(cfg, dict) else _load_config()
        if "model" in properties and _target_is_offered(orchestrator_tier, contract_cfg):
            properties["model"] = {
                **(properties.get("model") or {}),
                "enum": [orchestrator_tier],
                "description": f"Required route for the {orchestrator_tier} planning conductor.",
            }
            pinned_model = True
        else:
            pinned_model = False
        required = list(schema.get("required") or [])
        for name in ("goal", "context") + (() if batch_shape else ("role",)) + (("model",) if pinned_model else ()):
            if name not in required:
                required.append(name)
        schema["required"] = required
    if force_tools and claude_target:
        # A deferred Claude tool must be described before tool_call can invoke it.
        # Keep that discovery step available during the restricted first call.
        describe_tool = next((tool for tool in request.get("tools") or []
                              if isinstance(tool, dict) and
                              (tool.get("name") or (tool.get("function") or {}).get("name"))
                              in {"tool_describe", "mcp__tool_describe"}), None) if via_bridge else None
        routed["tools"] = [planner_tool] + ([describe_tool] if describe_tool else []) + [claude_tool]
        if _is_anthropic_shaped(routed):
            routed["tool_choice"] = {"type": "any"}
        else:
            routed["tool_choice"] = "required"
            routed["parallel_tool_calls"] = False
    elif force_tools:
        # One tool plus a required choice is deterministic, and leaves the
        # normal complete toolset untouched on the parent's next iteration.
        routed["tools"] = [planner_tool]
        if _is_anthropic_shaped(routed):
            # Name the tool exactly as offered: Anthropic OAuth requests carry it as
            # mcp__delegate_task, and forcing the bare name is a 400 that drops the
            # whole turn onto the fallback account.
            forced_name = (planner_tool.get("name")
                           or (planner_tool.get("function") or {}).get("name")
                           or "delegate_task")
            routed["tool_choice"] = {"type": "tool", "name": forced_name}
        else:
            routed["tool_choice"] = "required"
            routed["parallel_tool_calls"] = False
    else:
        # No protocol-level forcing on this route (TokenPlan Qwen). Keep the
        # parent's full toolset — a lone optional tool would leave it unable to
        # act — and swap in the hardened delegate_task schema so that *if* it
        # delegates, it can only produce a role="orchestrator" planner child.
        routed["tools"] = [
            planner_tool if tool is delegate_tool else tool
            for tool in (routed.get("tools") or [])
        ]
    return routed


def _sol_opus5_preflight_enabled(cfg: Dict[str, Any]) -> bool:
    """Return true only for the explicitly configured, currently callable bridge."""
    policy = cfg.get("sol_opus5_preflight") or {}
    return (
        _is_callable_tier("sol", cfg)
        and _is_callable_tier("opus5", cfg)
        and bool(policy.get("enabled"))
        and str(policy.get("owner", "")).casefold() == "sol"
        and str(policy.get("bridge_model", "")).casefold() == "claude-opus-5-5"
        and bool(policy.get("require_successful_auth_probe"))
    )


def _prepare_sol_opus5_preflight(request: Dict[str, Any], plan_id: str) -> Dict[str, Any]:
    """Force a Sol-owned preflight without routing it through Spark or Terra.

    The normal Sol request remains OpenAI-compatible; this records the required
    read-only external Claude Code Opus 5.5 review contract rather than attempting
    an unsafe provider/model-string substitution.
    """
    routed = deepcopy(request)
    instruction = (
        "\n\n[INTERNAL SOL + CLAUDE OPUS 5 PREFLIGHT]\n"
        f"Plan ID: {plan_id}. Before normal implementation, call delegate_task exactly once with a goal beginning [sol]. "
        "This is a Sol-owned visual/product/UI/UX/CSS/layout/design-system preflight. The delegated Sol reviewer must first inspect any "
        "current image itself, create a structured dispatch/review plan, and perform the configured read-only Claude Code bridge review "
        "with requested alias `opus`. Accept the review only if the bridge reports canonical effective model `claude-opus-5-5`; otherwise "
        "stop and report the unavailable bridge without falling back to Spark or Terra. Do not expose credentials or make writes, deploys, "
        "payments, or production changes during preflight. Spark and Terra are not preflight targets for this request.\n"
    )
    _append_user_instruction(routed, instruction)
    delegate_tool = _find_delegate_tool(routed)
    if delegate_tool is None:
        return routed
    routed["tools"] = [delegate_tool]
    routed["tool_choice"] = "required"
    routed["parallel_tool_calls"] = False
    return routed


def _orchestration_path(cfg: Dict[str, Any]) -> Path:
    orchestration = cfg.get("orchestration") or {}
    return hermes_path(orchestration.get("path", "~/.hermes/logs/terra-spark-orchestration.jsonl"))


def _orchestration_event(cfg: Dict[str, Any], event: Dict[str, Any]) -> None:
    path = _orchestration_path(cfg)
    payload = {"timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), **event}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _SHADOW_LOCK:
            # Hooks may be replayed by the gateway/desktop lifecycle.  A child
            # session identifies one real lifecycle transition, so emitting it
            # twice turns accounting noise into an apparent extra worker.
            # A replay has the same lifecycle *phase*.  Different phases for the
            # same turn (for example initial preflight and later rescue) are real
            # transitions and must remain observable.
            identity_keys = ("event", "phase", "plan_id", "turn_id", "child_session_id")
            identity = tuple(event.get(key) for key in identity_keys)
            if path.exists() and any(identity == tuple(record.get(key) for key in identity_keys)
                                     for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())):
                return
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def _orchestration_forced_event(cfg: Dict[str, Any], parent_turn_id: str) -> Optional[Dict[str, Any]]:
    path = _orchestration_path(cfg)
    if not path.exists():
        return None
    try:
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return None
    for event in reversed(events):
        if event.get("event") == "preflight_forced" and event.get("turn_id") == parent_turn_id:
            return event
    return None


def _host_delegation_limits() -> Dict[str, Any]:
    """Read effective host limits; unknown capabilities never justify fan-out."""
    try:
        from tools.delegate_tool_config import (
            _get_max_spawn_depth, _get_max_concurrent_children,
            _get_orchestrator_enabled, _load_config as host_config,
        )
        from tools.delegate_tool import DEFAULT_MAX_ITERATIONS
        depth = _get_max_spawn_depth()
        return {"max_spawn_depth": depth,
                "max_concurrent_children": _get_max_concurrent_children(),
                "max_iterations": host_config().get("max_iterations", DEFAULT_MAX_ITERATIONS),
                "conductor_available": _get_orchestrator_enabled() and depth >= 2}
    except (ImportError, AttributeError, TypeError, ValueError):
        return {"conductor_available": False}


def runtime_diagnostic(
    request: Any = None, cfg: Optional[Dict[str, Any]] = None,
    topology: str = runtime_capabilities.DEFAULT_TOPOLOGY, transport: str = "hermes_codex",
) -> Dict[str, Any]:
    """Read-only S01 view: configured feature vs runtime capability vs topology.

    Not called from the routing hot path; it never dispatches, probes a model,
    writes config, or raises the host spawn depth. Fresh admission is still
    required before any execution.
    """
    cfg = _load_config() if cfg is None else cfg
    snap = runtime_capabilities.snapshot(request, cfg)
    return runtime_capabilities.diagnostic(snap, topology, transport)


def _read_host_config() -> Dict[str, Any]:
    """The host's config.yaml as a dict, read-only; ``{}`` when absent or unreadable."""
    if yaml is None:
        return {}
    try:
        raw = yaml.safe_load(_HERMES_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def identity_diagnostic(
    cfg: Optional[Dict[str, Any]] = None,
    observed: Optional[Dict[str, Any]] = None,
    request: Any = None,
) -> Dict[str, Any]:
    """Read-only S02 view: which config owner names which Claude model, per tier.

    Reads router config and the host config file; no network, subprocess, model
    call or write, and it is not on the routing hot path.
    """
    cfg = _load_config() if cfg is None else cfg
    snap = runtime_capabilities.snapshot(request, cfg)
    return target_identity.drift_diagnostic(cfg, _read_host_config(), observed, snap)


def _orchestration_skip_reason(
    kwargs: Dict[str, Any], cfg: Dict[str, Any], decision: RouteDecision
) -> Optional[str]:
    """Name the gate that rejected this turn, or None when it is eligible.

    The route log records the winning route, not the dispatch that never
    happened, so a turn that runs twenty parent calls with no worker looks
    identical whether orchestration was ineligible, deduped, or never reached.
    Returning the reason lets the caller record it.
    """
    policy = cfg.get("orchestration") or {}
    request = kwargs.get("request")
    sol_preflight = decision.tier == "sol" and _sol_opus5_preflight_enabled(cfg)
    # Every tier the router knows can orchestrate. This used to be Sol plus
    # default_model only, so a session switched to Grok (2026-09-25) never saw a
    # preflight and worked alone. An external parent (a fallback account)
    # orchestrates exactly like a local one: only the model rewrite is
    # provider-bound, the delegation contract is not.
    orchestration_tiers = (
        {"sol"} | {str(cfg.get("default_model", "terra"))}
        | set(cfg.get("models") or {}) | set(_delegation_target_names())
    )
    if not policy.get("enabled"):
        return "orchestration_disabled"
    if decision.tier not in orchestration_tiers:
        return f"tier_not_orchestrator:{decision.tier}"
    if not isinstance(request, dict):
        return "request_not_a_dict"
    api_call_count = int(kwargs.get("api_call_count", 1) or 1)
    # Normal path: dispatch on the first Terra call. Recovery path: if that
    # process missed its initial checkpoint, rescue a genuinely long tool loop
    # once instead of letting it remain 20-30 Terra calls with no Spark work.
    rescue_min_calls = max(2, int(policy.get("rescue_min_calls", 6) or 6))
    is_rescue = api_call_count >= rescue_min_calls
    if api_call_count != 1 and not is_rescue:
        return f"mid_loop_call:{api_call_count}"
    if str(kwargs.get("platform", "")).casefold() == "subagent" or ":sa-" in str(kwargs.get("turn_id", "")):
        return "subagent_turn"
    # One owner for "is delegate_task on offer": this duplicated the check inline and
    # the two copies drifted the moment Anthropic's mcp__ prefix appeared.
    if _find_delegate_tool(request) is None:
        return "no_delegate_task_tool"
    items = _request_items(request)
    user_text, user_index = _last_user_text_and_index(items)
    if not user_text:
        return "no_user_text"
    # Read what the operator actually asked for, not host/plugin context appended
    # after it -- the superpowers bootstrap names delegate_task in its own text, so
    # reading raw text here made every bootstrapped root turn look like an explicit
    # delegation choice and lose its forced preflight.
    operator_text = _without_host_injected_context(_without_router_contract(user_text))
    # The operator explicitly named a delegation tool for this turn -- the forced
    # delegate_task planning preflight would override that choice. `user_text` is
    # the incoming request's own latest user turn, read before this same call adds
    # the router's routing note or preflight contract to it (those are appended to
    # a deep copy further down the pipeline), so this cannot fire on our own text.
    if re.search(r"\bdelegate_(?:claude|task)\b", operator_text):
        return "explicit_delegation_tool"
    # A task too short to decompose is not worth a planner round trip plus up to
    # max_tasks bounded workers. Without this gate every actionable Terra turn
    # forced a fan-out dispatch, the dominant source of perceived latency. That
    # is a first-call latency argument, so it must not gate the rescue: a turn
    # already deep in a tool loop has spent far more than a planner round trip,
    # and a short prompt ("csinald meg") routinely opens the longest loops.
    # Router-owned tiers only: an external parent (a Claude account the session
    # fell back to) has always been preflighted regardless of length.
    #
    # A short follow-up is not judged by its own length either: "csinald meg"
    # after a discussed plan stands for that whole plan. The parent then writes
    # the self-contained objective, and on_pre_tool_call measures *that*
    # (orchestration.min_goal_chars) -- a small objective returns to the parent
    # without spawning anything.
    if decision.tier in (cfg.get("models") or {}) and not sol_preflight and not is_rescue:
        min_chars = max(0, int(policy.get("min_chars", 180) or 0))
        if len(operator_text) < min_chars and not _has_prior_exchange(items, user_index):
            return f"prompt_shorter_than_min_chars:{len(operator_text)}<{min_chars}"
    # Explicit bounded UI requests authorised for the verified bridge are a
    # single-hop exception: Sol retains policy ownership, but no Sol/Terra/Spark
    # planner call is created before Opus execution middleware handles the turn.
    if decision.tier == "sol" and _is_explicit_bounded_opus_ui_request(operator_text, cfg):
        return "explicit_bounded_opus_ui_request"
    # A normal first-call preflight must be before parent tool work. The rescue
    # path intentionally runs inside an existing tool loop.
    if not is_rescue and _current_turn_has_tool_activity(items, user_index):
        return "tool_activity_before_first_call"
    text = _normalise(user_text)
    # Completion delivery is the Terra supervisor's reviewed hand-back, never a
    # fresh task to fan out again. Without this guard it can recursively create
    # another orchestration cycle merely because the consolidated evidence is long.
    if "[async delegation batch complete" in text:
        return "delegation_completion_delivery"
    # A compaction envelope can precede the real current user request in the
    # same message. Route from the suffix after its end marker; treating the
    # whole envelope as an internal continuation silently suppresses dispatch.
    if text.startswith("[context compaction"):
        end_marker = re.search(r"\[end of context summary[^\]]*\]", text)
        if not end_marker:
            return "compaction_envelope_without_end_marker"
        text = text[end_marker.end():].strip()
        if not text:
            return "compaction_envelope_with_empty_suffix"
    internal_prefixes = (
        "review the conversation above and consider saving to memory",
        "what do you see in this image?",
    )
    if text.startswith(internal_prefixes):
        return "internal_prompt_prefix"
    # Terra is the planner for every actionable Terra turn. Do not try to infer
    # task decomposability from keyword lists: real work is often introduced by
    # terse contextual requests, screenshots, or a tool loop whose prompt has
    # none of the old multi-step marker words. The planner itself decides whether
    # zero, one, or several bounded workers are useful; host guards still
    # constrain their scopes and hard-risk requests route to Sol first.
    if not text:
        return "empty_normalised_text"
    if not sol_preflight and not _host_delegation_limits()["conductor_available"]:
        return "host_has_no_conductor_depth; parent_delegates_direct_workers"
    if not _conductor_tier(cfg, request, decision.tier):
        return "no_reachable_conductor_route; parent_delegates_direct_workers"
    return None


def _orchestration_eligible(kwargs: Dict[str, Any], cfg: Dict[str, Any], decision: RouteDecision) -> bool:
    return _orchestration_skip_reason(kwargs, cfg, decision) is None


def _force_terra_supervisor_preflight(
    kwargs: Dict[str, Any], cfg: Dict[str, Any], decision: RouteDecision
) -> Optional[Dict[str, Any]]:
    turn_id = str(kwargs.get("turn_id", ""))
    api_call_count = int(kwargs.get("api_call_count", 1) or 1)
    phase = "rescue" if api_call_count > 1 else "preflight"
    skip_reason = _orchestration_skip_reason(kwargs, cfg, decision)
    if skip_reason is not None:
        # Record only the two calls where a dispatch was actually due. Logging
        # every mid-loop call would bury the signal under one line per parent
        # call, which is the noise the api_call_count gate already rejects.
        policy = cfg.get("orchestration") or {}
        rescue_min_calls = max(2, int(policy.get("rescue_min_calls", 6) or 6))
        # An operator who turned orchestration off has nothing to diagnose, and
        # a config that omits `path` falls back to the shared production log --
        # so logging this case makes every unrelated caller write to it.
        if policy.get("enabled") and (api_call_count == 1 or api_call_count == rescue_min_calls):
            with _SHADOW_LOCK:
                _orchestration_event(
                    cfg,
                    {
                        "event": "preflight_skipped",
                        "phase": phase,
                        "api_call_count": api_call_count,
                        "turn_id": turn_id,
                        "parent_model": decision.tier,
                        "skip_reason": skip_reason,
                        # Names only, no schemas: "no delegate_task tool" is otherwise
                        # indistinguishable from "no tools at all" or "a different wire
                        # shape", and those need different fixes.
                        "tools_seen": _tool_names(kwargs.get("request")),
                    },
                )
        return None
    # Sol's own design preflight only when its bridge is configured; without it a
    # Sol parent orchestrates through the same conductor contract as any tier.
    sol_preflight = decision.tier == "sol" and _sol_opus5_preflight_enabled(cfg)
    claude_choice = (
        _claude_first_choice(kwargs["request"], cfg, decision) if not sol_preflight else ("", "", "")
    )
    with _SHADOW_LOCK:
        if _orchestration_forced_event(cfg, turn_id):
            return None
        plan_id = f"plan-{hashlib.sha256(turn_id.encode('utf-8')).hexdigest()[:16]}"
        _orchestration_event(
            cfg,
            {
                "event": "preflight_forced",
                "phase": phase,
                "api_call_count": api_call_count,
                "plan_id": plan_id,
                "turn_id": turn_id,
                "parent_model": decision.tier,
                "preflight_owner": (
                    "sol" if sol_preflight else _conductor_tier(cfg, kwargs["request"], decision.tier)
                ),
                "preflight_bridge_model": "claude-opus-5-5" if sol_preflight else None,
                "max_tasks": min(3, max(1, int((cfg.get("orchestration") or {}).get("max_tasks", 3)))),
                "parent_prompt_preview": _prompt_preview(kwargs.get("request") or {}),
                **({"balanced": claude_choice[2]} if claude_choice[2] else {}),
            },
        )
    if sol_preflight:
        return _prepare_sol_opus5_preflight(kwargs["request"], plan_id)
    return _prepare_orchestration_delegation(
        kwargs["request"],
        plan_id,
        min(3, max(1, int((cfg.get("orchestration") or {}).get("max_tasks", 3)))),
        cfg=cfg,
        force_tools=_supports_forced_tool_choice(kwargs, decision),
        claude_choice=claude_choice,
        parent_tier=decision.tier,
    )


def _shadow_eligible(kwargs: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    shadow = cfg.get("shadow") or {}
    request = kwargs.get("request")
    if not shadow.get("enabled") or not isinstance(request, dict):
        return False
    if _request_has_image_attachment(request):
        return False
    if int(kwargs.get("api_call_count", 1) or 1) != 1:
        return False
    if str(kwargs.get("platform", "")).casefold() == "subagent" or ":sa-" in str(kwargs.get("turn_id", "")):
        return False
    if str(request.get("model", "")) != str((cfg.get("models") or {}).get("terra", "")):
        return False
    items = _request_items(request)
    text, user_index = _last_user_text_and_index(items)
    if not text or _current_turn_has_tool_activity(items, user_index):
        return False
    if "[image attached" in _normalise(text) or "[screenshot]" in _normalise(text):
        return False
    has_delegate_tool = _find_delegate_tool(request) is not None
    if not has_delegate_tool:
        return False
    return classify_request(request, api_call_count=1, config=cfg).tier == "terra"


def _force_shadow_delegation_if_eligible(kwargs: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not _shadow_eligible(kwargs, cfg):
        return None
    with _SHADOW_LOCK:
        path = _shadow_path(cfg)
        used = 0
        if path.exists():
            try:
                prior_events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                # Earlier named-choice attempts were logged as
                # ``delegation_requested`` but never reached the child tool.
                # Only a required-tool attempt suppresses a repeat for this
                # turn, so already-active Terra turns receive one recovery try.
                cycle_id = str((cfg.get("shadow") or {}).get("cycle_id") or "").strip()
                if any(
                    event.get("event") == "delegation_forced"
                    and event.get("turn_id") == str(kwargs.get("turn_id", ""))
                    and (not cycle_id or str(event.get("cycle_id") or "") == cycle_id)
                    for event in prior_events
                ):
                    return None
                used = _completed_actual_spark_benchmark_count(prior_events, cfg)
            except Exception:
                used = 0
        if used >= int((cfg.get("shadow") or {}).get("limit", 10)):
            return None
        turn_id = str(kwargs.get("turn_id", ""))
        benchmark_id = _benchmark_id(turn_id)
        _shadow_event(
            cfg,
            {
                "event": "delegation_forced",
                "turn_id": turn_id,
                "benchmark_id": benchmark_id,
                "parent_model": "terra",
                "shadow_model": "spark",
                "effort": "medium",
                "parent_prompt_preview": _prompt_preview(kwargs.get("request") or {}),
            },
        )
    return _prepare_shadow_delegation(kwargs["request"], benchmark_id)


_NOTE_TOOL_NAMES = frozenset({
    "delegate_task", "mcp__delegate_task", "delegate_claude", "mcp__delegate_claude",
    "tool_call", "mcp__tool_call",
})
# Hermes's Tool Search defers every plugin tool by default: the parent sees only
# tool_search/tool_describe/tool_call plus a catalog stub, and a direct call to
# delegate_claude is rejected as unknown until it is loaded once. Every place
# that tells a parent to call delegate_claude while Claude delegation is active must also
# say how to reach it.
_DEFERRED_DELEGATION_HINT = (
    'If delegate_claude is not in your tool list it is a deferred tool: load it once with tool_describe, '
    'then call it through tool_call with name "delegate_claude". '
)
_CLAUDE_DELEGATION_AVAILABLE_LINE = (
    'Claude delegation is available through delegate_claude: tier "haiku" for quick lookups, "sonnet" as '
    'the default worker, "opus" for hard or consequential work.'
)


def _note_names(kind: str, cfg: Dict[str, Any], states: Dict[str, str], claude_offered: set) -> list:
    """The kind's preference chain, limited to what can be offered, in advice order."""
    names = []
    for name in _preference_list(kind, cfg):
        if name in claude_delegation.TIER_FOR_TARGET:
            if name in claude_offered and _target_is_offered(name, cfg):
                names.append(name)
        elif _is_routable_tier(name, cfg) and _target_is_offered(name, cfg):
            names.append(name)
    held = {account for account, state in states.items() if state in ("soft", "closed")}
    if held:
        # An account at its limit still runs, but it is no longer the first thing to reach for.
        names = ([n for n in names if _account_of(n, cfg) not in held]
                 + [n for n in names if _account_of(n, cfg) in held])
    # Unavailable entries remain visible for explanations, behind usable ones.
    return sorted(names, key=lambda n: (
        states.get(_account_of(n, cfg)) == "closed" or _tier_cooldown_remaining(n, cfg) > 0,
    ))


def _advised_chain(
    kind: str, cfg: Dict[str, Any], states: Dict[str, str], claude_offered: set, readings: Dict[str, Any]
) -> Tuple[list, str]:
    """(chain, reason): the advice chain, load-balanced between accounts under Claude delegation.

    After the soft/hard guard has ordered the chain, compare its first account with
    the next account the chain lists, on ``window`` (5-hour by default; the
    parent's share counts on its own account). When the first one is at
    least ``busy_percent`` and the other is at least ``margin_percent`` points
    freer, that account's first entry moves to the front. Only listed targets
    move, the parent never does, and a missing or stale reading changes nothing.
    ``reason`` is "" unless the order changed.
    """
    names = _note_names(kind, cfg, states, claude_offered)
    policy = usage_guard.balance_config(cfg)
    if not policy["enabled"] or not claude_delegation.is_active() or len(names) < 2:
        return names, ""
    first_account = _account_of(names[0], cfg)
    other = next((n for n in names
                  if _account_of(n, cfg) not in ("", first_account)
                  and states.get(_account_of(n, cfg)) not in ("soft", "closed")
                  and _tier_cooldown_remaining(n, cfg) <= 0), "")
    if not first_account or not other:
        return names, ""
    other_account = _account_of(other, cfg)
    busy_reading, free_reading = readings.get(first_account), readings.get(other_account)
    if not (usage_guard.fresh(busy_reading, cfg) and usage_guard.fresh(free_reading, cfg)):
        return names, ""
    busy = usage_guard.load(busy_reading, policy["window"])
    free = usage_guard.load(free_reading, policy["window"])
    if busy is None or free is None:
        return names, ""
    if busy[0] < policy["busy_percent"] or busy[0] - free[0] < policy["margin_percent"]:
        return names, ""
    reason = (f"{kind} → {other} first ({usage_guard.account_label(first_account)} {busy[1]} {busy[0]:.0f}% vs "
              f"{usage_guard.account_label(other_account)} {free[0]:.0f}%)")
    return [other] + [n for n in names if n != other], reason


def _note_label(name: str, notes: Dict[str, str], *, as_call: bool) -> str:
    if not as_call:
        return f"{name}{notes.get(name, '')}"
    tier = claude_delegation.TIER_FOR_TARGET.get(name)
    call = f'delegate_claude(tier="{tier}")' if tier else f"delegate_task (goal prefix [{name}])"
    return f"{name} → {call}{notes.get(name, '')}"


def _guarded_readings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """One cached usage reading per guarded account; {} when the guard is unreadable."""
    try:
        guarded_accounts = [a for a in (usage_guard.guard_config(cfg).get("accounts") or {})
                            if usage_guard.guarded(a, cfg)]
        return {account: usage_guard.peek(account, cfg) for account in guarded_accounts}
    except Exception:
        return {}


def _claude_first_choice(
    request: Dict[str, Any], cfg: Dict[str, Any], decision: RouteDecision
) -> Tuple[str, str, str]:
    """(kind, claude_target, balanced) for this turn's forced call.

    ``claude_target`` is set when the first advised choice is a Claude tier;
    ``balanced`` is the balancing reason when load balancing reordered the chain,
    whichever way it went. The same chain the routing note advises from, so the
    forced call and the note never disagree. An external parent's decision
    carries no kind, so the turn is classified here the way the note classifies it.
    """
    if not claude_delegation.is_active():
        return "", "", ""
    kind = decision.kind if decision.kind in WORK_KINDS else ""
    if not kind:
        try:
            kind = classify_request(request, api_call_count=1, config=cfg).kind or ""
        except Exception:
            return "", "", ""
    readings = _guarded_readings(cfg)
    states = _account_states(cfg, readings)
    chain, balanced = _advised_chain(kind, cfg, states,
                                     set(_delegation_target_names()), readings)
    first = next((name for name in chain if states.get(_account_of(name, cfg)) != "closed"
                  and _tier_cooldown_remaining(name, cfg) <= 0), "")
    return kind, (first if first in claude_delegation.TIER_FOR_TARGET else ""), balanced


def _claude_route_tool(request: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], bool]:
    """(tool, via_bridge): delegate_claude itself when listed, else the tool_call bridge, else None."""
    bridge = None
    for tool in request.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name") or (tool.get("function") or {}).get("name")
        if name in (claude_delegation.TOOL_NAME, f"mcp__{claude_delegation.TOOL_NAME}"):
            return tool, False
        if name in claude_delegation._BRIDGE_CALL_NAMES and bridge is None:
            bridge = tool
    return (bridge, True) if bridge is not None else (None, False)


def _routing_note(request: Dict[str, Any], kwargs: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """Advice for a root parent: which account and tier, per kind of work.

    With orchestration off, a parent on an external account got no routing advice
    at all; the preference chains only ever reached a forced conductor. This is
    that advice, once per turn, and advisory: the parent may overrule it.
    """
    if not claude_delegation.is_active():
        return ""
    if int(kwargs.get("api_call_count", 1) or 1) != 1:
        return ""
    if str(kwargs.get("platform", "")).casefold() == "subagent" or ":sa-" in str(kwargs.get("turn_id", "")):
        return ""
    if not set(_tool_names(request)) & _NOTE_TOOL_NAMES:
        return ""
    text, _index = _last_user_text_and_index(_request_items(request))
    if not text or _is_delegation_outcome_text(text) or _lifecycle_event_kind(request):
        return ""
    try:
        kind = classify_request(request, api_call_count=1, config=cfg).kind or "default"
    except Exception:
        kind = "default"
    # One peek per guarded account for the whole note: the chain, the "other
    # kinds" summary and the usage line all read the same snapshot instead of
    # each re-peeking (and each risking a different answer mid-note).
    readings = _guarded_readings(cfg)
    states = _account_states(cfg, readings)
    claude_offered = set(_delegation_target_names())

    chain, balanced = _advised_chain(kind, cfg, states, claude_offered, readings)
    notes = _target_availability(chain, cfg, states=states)
    lines = [f"[ROUTER] This turn classifies as: {kind}."]
    if balanced:
        lines.append(f"Balanced: {balanced}; the busier account is spared, not closed.")
    if chain:
        lines.append(f"If you delegate {kind} work: "
                     + " > ".join(_note_label(n, notes, as_call=True) for n in chain) + ".")
    else:
        lines.append(f"No preference is configured for {kind} work; delegate_task keeps its built-in route.")
    others = []
    for other in WORK_KINDS:
        if other == kind:
            continue
        names, _balanced = _advised_chain(other, cfg, states, claude_offered, readings)
        if names:
            other_notes = _target_availability(names, cfg, states=states)
            others.append(f"{other}: " + " > ".join(_note_label(n, other_notes, as_call=False)
                                                   for n in names))
    if others:
        lines.append("Other kinds: " + "; ".join(others) + ".")
    if not any(name in claude_delegation.TIER_FOR_TARGET for k in WORK_KINDS for name in _preference_list(k, cfg)):
        lines.append(_CLAUDE_DELEGATION_AVAILABLE_LINE)
    usage = []
    for account in sorted(states, key=lambda a: usage_guard.account_label(a) != "Claude"):
        limits = usage_guard.account_limits(account, cfg)
        reading = readings.get(account)
        value = "unknown" if reading is None or reading.weekly is None else f"{reading.weekly:.0f}%"
        usage.append(f"{usage_guard.account_label(account)} weekly {value} "
                     f"(soft {limits['soft_percent']:.0f}%, hard {limits['hard_percent']:.0f}%)")
    if usage:
        lines.append("Usage: " + "; ".join(usage) + ".")
    lines.append(_DEFERRED_DELEGATION_HINT.rstrip())
    lines.append("Advisory: if you route differently, say why in one line.")
    return "\n\n" + "\n".join(lines) + "\n"


def route_llm_request(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """Hermes llm_request middleware entrypoint.

    Claude delegation is scoped to this request: offered only while a Claude
    model is switched on in ``callable`` and the request itself carries
    delegate_claude. A session whose tool list predates a Claude switch flip is
    then never told to call a tool it lacks, nor steered to a Claude that is off.
    """
    available = claude_delegation.availability_block(_load_config()) == ""
    claude_delegation.note_availability(available)
    active = available and claude_delegation.is_active() and claude_delegation.offered(
        _tool_names(kwargs.get("request"))
    )
    with claude_delegation.request_scope(active):
        return _route_llm_request(**kwargs)


def on_pre_gateway_dispatch(**kwargs: Any) -> None:
    """Notice a Claude switch flip before the gateway builds a new session's agent."""
    try:
        claude_delegation.note_availability(claude_delegation.availability_block(_load_config()) == "")
    except Exception as exc:
        _logger.debug("pre_gateway_dispatch: Claude availability check skipped: %s", exc)
    return None


def _route_llm_request(**kwargs: Any) -> Optional[Dict[str, Any]]:
    cfg = _load_config()
    disabled = os.environ.get("HERMES_MODEL_ROUTER_DISABLE", "").casefold() in {
        "1", "true", "yes", "on"
    }
    if disabled or not cfg.get("enabled", True):
        return None

    active_model = str(kwargs.get("model", ""))
    supported_models = set(cfg.get("models", {}).values())
    request = kwargs.get("request")
    # Accept requests from any provider that has supported models
    external_parent = ""
    if active_model not in supported_models:
        external = _external_target_for_model(active_model)
        if not external:
            return None
        # Observed, not routed: the model stays the parent's, because this middleware
        # can only rewrite within one provider. Orchestration is not provider-bound
        # though, and returning here meant a parent on a fallback account silently
        # lost its delegation contract and worked alone.
        _log_decision(
            RouteDecision(external, active_model, "external delegation target", "external"),
            kwargs,
            cfg,
        )
        external_parent = external
    if not isinstance(request, dict):
        return None

    subagent_marker = (
        str(kwargs.get("platform", "")).casefold() == "subagent"
        or ":sa-" in str(kwargs.get("turn_id", ""))
    )

    if external_parent:
        # No classification and no rewrite: the route is not ours to choose. The only
        # thing owed here is the preflight that turns a lone parent into a conductor.
        decision = RouteDecision(
            external_parent, active_model, "external delegation target", "external",
        )
        forced = _force_terra_supervisor_preflight(kwargs, cfg, decision)
        # A conductor on Claude reaches this branch and returns early, so the
        # stopped-worker notice would never have been attached for exactly the
        # setup that needs it most: the `code` chain puts the conductor on an
        # external account precisely so the leaves can go elsewhere.
        redispatch = (
            _quota_redispatch_instruction(request, cfg)
            if isinstance(request, dict)
            and _is_delegation_outcome_text(
                _last_user_text_and_index(_request_items(request))[0]
            )
            else ""
        )
        # The forced preflight already carries the full contract; the note is for
        # the turns it leaves alone.
        note = ""
        if forced is None:
            try:
                note = (_worker_order_note(request, cfg) if subagent_marker
                        else _routing_note(request, kwargs, cfg))
            except Exception:
                note = ""
        if forced is None and not redispatch and not note:
            return None
        forced = deepcopy(forced if forced is not None else request)
        if redispatch:
            _append_user_instruction(forced, redispatch)
        if note:
            _append_user_instruction(forced, note)
        # Same envelope as the normal path, minus any model/provider change: the
        # request carries the added contract, the route stays exactly as it arrived.
        return {
            "request": forced,
            "source": "model-router",
            "reason": decision.reason,
            "metadata": {
                "tier": decision.tier,
                "model": active_model,
                "effort": decision.effort,
                "provider": kwargs.get("provider"),
            },
        }

    decision = classify_request(
        request,
        api_call_count=int(kwargs.get("api_call_count", 1) or 1),
        config=cfg,
        # Only a delegated worker carries a label this router itself emitted; a
        # root turn's label is whatever the user typed, so the design gate keeps
        # precedence there.
        allow_plan_label_over_design=subagent_marker,
    )

    decision = _require_callable(decision, cfg)
    # Preserve one durable user-facing owner for the whole root session.  Sol,
    # Spark, Qwen and Opus are available as explicitly requested or delegated
    # workers, not invisible replacements for the person talking to the user.
    active_tier = next(
        (tier for tier, model in (cfg.get("models") or {}).items() if model == active_model),
        None,
    )
    session_policy = cfg.get("session_policy") or {}
    latest_user_text, _ = _last_user_text_and_index(_request_items(request))
    latest_user_text = _without_host_injected_context(_without_router_contract(latest_user_text))
    explicit_root_override = bool(
        re.match(r"^\s*\[(?:luna|spark|terra|sol)(?::xhigh)?\](?:\s|$)", _normalise(latest_user_text))
    )
    if (
        not subagent_marker
        and bool(session_policy.get("pin_root_parent", False))
        and not explicit_root_override
        and active_tier
        and _is_callable_tier(active_tier, cfg)
        and decision.tier != active_tier
    ):
        decision = _decision(
            active_tier,
            f"root session parent pinned; {decision.tier} reserved for an explicit or delegated worker",
            cfg,
        )
    # [spark] is an internal leaf label emitted by a Terra plan, not a public
    # root-route escape hatch. A user/root turn must still be assessed by Terra.
    if decision.tier == "spark" and not subagent_marker:
        decision = _decision(
            str(cfg.get("default_model", "terra")),
            "root Spark label deferred to default orchestrator",
            cfg,
        )
    is_spark_subagent = (
        subagent_marker
        and active_model == str((cfg.get("models") or {}).get("spark", ""))
        and bool((cfg.get("delegation") or {}).get("preserve_spark_subagents", True))
        and _is_callable_tier("spark", cfg)
    )
    design_only = decision.reason == "design analysis or implementation is Sol-only"
    if is_spark_subagent and not _request_has_image_attachment(request) and not design_only:
        # The parent chooses whether a task is eligible for delegation. Preserve
        # the explicitly pinned Spark worker across its bounded tool loop so the
        # delegated task can actually complete. Only an explicit Sol request or
        # a hard safety signal overrides the worker; Sol being the normal parent
        # default must not silently promote every safe child.
        user_text, _ = _last_user_text_and_index(_request_items(request))
        user_text = _without_host_injected_context(_without_router_contract(user_text))
        if decision.tier != "sol" and not _is_spark_read_only_request(user_text):
            decision = _decision("terra", "Spark is restricted to non-design read-only subtasks", cfg)
        explicit = decision.reason.startswith("explicit [")
        hard_sol_reasons = {
            "sensitive or consequential domain",
            "consequential engineering or research task",
            "consequential system action",
        }
        if decision.reason == "Spark is restricted to non-design read-only subtasks":
            pass
        elif not explicit and decision.reason in hard_sol_reasons:
            affirmative_text = _without_negated_safety_constraints(user_text)
            if affirmative_text != user_text:
                affirmative_decision = classify_request(
                    {"model": active_model, "messages": [{"role": "user", "content": affirmative_text}]},
                    api_call_count=1,
                    config=cfg,
                )
                if affirmative_decision.reason not in hard_sol_reasons:
                    decision = _decision("spark", "eligible delegated Spark subtask", cfg)
        elif not explicit:
            decision = _decision("spark", "eligible delegated Spark subtask", cfg)
    elif (
        subagent_marker
        and active_model == str((cfg.get("models") or {}).get("terra", ""))
        and not design_only
        # Only *unlabelled* child work belongs to the planner tier. A labelled
        # leaf has already been decided -- accepted, or rejected for a stated
        # reason -- and this branch used to overwrite both. It tested the label
        # by string-matching "explicit [" at the front of the reason, which any
        # later rewrite erases: a [spark] leaf becomes "fallback from disabled
        # spark" the moment Spark is not callable, so every legitimate Spark
        # leaf was demoted to the planner tier and its Luna fallback lost.
        # A rejected leaf fared worse -- "consequential Spark task requires Sol"
        # was demoted to Terra, turning a deliberate escalation into the exact
        # tier the escalation existed to avoid.
        and not _PLAN_LABEL.match(_normalise(latest_user_text))
    ):
        # The delegation default is Terra so the forced first child is a real
        # planner. Unlabelled child work remains with the default_model rather
        # than being silently demoted before it can decompose the task.
        default_model_tier = str(cfg.get("default_model", "terra"))
        if not decision.reason.startswith("explicit ["):
            decision = _decision(default_model_tier, f"{default_model_tier.capitalize()} planner or integration subagent", cfg)
    turn_id = str(kwargs.get("turn_id") or "")
    if decision.tier == "spark" and _spark_quota_exhausted_for_turn(turn_id):
        fallback_model = _quota_fallback_model(decision.model, cfg)
        if fallback_model:
            fallback_tier = next(
                (tier for tier, model in (cfg.get("models") or {}).items() if model == fallback_model),
                "luna",
            )
            decision = RouteDecision(
                fallback_tier,
                fallback_model,
                "Spark quota already exhausted for this turn",
                _quota_fallback_effort(cfg, fallback_tier),
            )
    # Rules above may intentionally rewrite the tier (root labels, subagents,
    # quota). Re-validate the final destination immediately before dispatch.
    decision = _require_callable(decision, cfg)
    # Any parent tier may orchestrate; _orchestration_skip_reason owns the gates.
    forced_preflight_request = _force_terra_supervisor_preflight(kwargs, cfg, decision)
    forced_shadow_request = (
        _force_shadow_delegation_if_eligible(kwargs, cfg)
        if decision.tier == str(cfg.get("default_model", "terra")) and forced_preflight_request is None
        else None
    )
    # The orchestration gates above must see the tier this request actually
    # classified to -- an already-stepped-down decision would offer Terra's
    # preflight to a Sol request that never got Sol's, and would skip the
    # Sol/Opus preflight the request was actually entitled to. Usage only
    # touches the *dispatched* tier, once those gates have already run.
    # Account savings apply to workers. A root's durable identity survives every
    # later rewrite, including an explicit root tier selection.
    if subagent_marker:
        decision = _usage_step_down(decision, cfg)
    # Hermes middleware cannot switch the underlying provider/transport. If a
    # rule selects a tier owned by another provider, changing only `model`
    # produces invalid calls such as `gpt-5.6-terra` at the Qwen Anthropic
    # endpoint. Preserve the explicitly active tier instead; an orchestrator
    # change across providers must happen through the persisted model config
    # and a fresh session.
    active_tier = next(
        (tier for tier, model in (cfg.get("models") or {}).items() if model == active_model),
        None,
    )
    tier_providers = cfg.get("tier_providers", {})
    if (
        active_tier
        and decision.tier != active_tier
        and tier_providers.get(decision.tier) != tier_providers.get(active_tier)
    ):
        if not _is_callable_tier(active_tier, cfg):
            raise RuntimeError(
                f"Active ModelRouter tier '{active_tier}' is disabled and cannot be "
                f"moved to provider '{tier_providers.get(decision.tier)}' mid-session"
            )
        decision = _decision(
            active_tier,
            "cross-provider route preserved for the active session",
            cfg,
        )

    # A delegated review leaf reaches Claude only through the CLI bridge. Explain
    # every static decline here, before execution, instead of letting its label
    # silently fall through to an ordinary worker.
    if subagent_marker and _CLAUDE_REVIEW_LABEL.match(_normalise(latest_user_text)):
        if str(kwargs.get("provider", "")).casefold() != str(cfg.get("provider", "")).casefold():
            reason = f"provider '{kwargs.get('provider')}' is unavailable for the Claude CLI bridge"
        else:
            source_request = kwargs.get("original_request")
            if not isinstance(source_request, dict):
                source_request = request
            _route, reason = _delegated_claude_review_status(latest_user_text, cfg, request=source_request)
        if reason:
            decision = replace(
                decision,
                reason=f"Claude review not taken ({reason}); ran as an ordinary {decision.tier} worker",
            )

    routed = forced_preflight_request or forced_shadow_request or dict(request)
    # A worker that died on an account limit is the one failure the conductor
    # cannot act on from the envelope alone. Attached here, after the forced
    # requests, so it survives whichever of them produced ``routed``; deep-copied
    # first because the plain path is a shallow copy whose message dicts are the
    # caller's, and appending to those would mutate the conversation itself.
    if _is_delegation_outcome_text(latest_user_text):
        redispatch = _quota_redispatch_instruction(request, cfg)
        if redispatch:
            routed = deepcopy(routed)
            _append_user_instruction(routed, redispatch)
    # A dispatch that failed before any worker existed. The notice above reads a
    # delegation outcome and there is none here: no child ran, so nothing will
    # ever be delivered to explain it.
    dispatch_failure = _dispatch_failure_instruction(request, cfg)
    if dispatch_failure:
        routed = deepcopy(routed)
        _append_user_instruction(routed, dispatch_failure)
    # A goal that names an external target in its text is a dispatch error, and
    # the expensive thing about it was that nothing said so: the label is inert,
    # the classifier reads the rest of the goal, and the leaf works to completion
    # on the account the dispatcher was trying to spare -- fourteen Sol calls in
    # the case that prompted this. Stop it at its first call instead and let it
    # report the correction, which reaches the parent as the leaf's own summary.
    #
    # Not raised: a middleware exception is fail-open here. Hermes logs it and
    # sends the request unrouted, which is worse than the silence it replaces.
    # ``tool_choice: none`` is the lever that actually bounds the leaf -- and the
    # toolset is left intact deliberately, because an emptied ``tools`` array
    # alongside it is the combination providers are most likely to reject, and a
    # 400 here would trade a quiet waste for a noisy crash.
    misdispatched = _misdispatched_external_label(latest_user_text, active_model, cfg)
    if misdispatched and subagent_marker:
        routed = deepcopy(routed)
        _append_user_instruction(routed, _misdispatch_instruction(misdispatched, decision.model))
        routed["tool_choice"] = "none"
        routed.pop("parallel_tool_calls", None)
        decision = replace(
            decision,
            reason=(
                f"goal names '{misdispatched}' in its text, which is not a route; "
                f"leaf stopped for re-dispatch"
            ),
        )
    elif forced_preflight_request is None:
        # The preflight carries these rules already, inside the conductor's
        # contract; this is the path where no conductor was created at all.
        with_goal_contract = _with_goal_contract(routed)
        if with_goal_contract is not None:
            routed = with_goal_contract
        if subagent_marker:
            order = _worker_order_note(request, cfg)
            if order:
                routed = deepcopy(routed)
                _append_user_instruction(routed, order)
        if forced_shadow_request is None and not subagent_marker:
            note = _routing_note(request, kwargs, cfg)
            if note:
                routed = deepcopy(routed)
                _append_user_instruction(routed, note)
    # TokenPlan's Anthropic-compatible Qwen endpoint rejects OpenAI/Codex
    # control fields. Sanitize the final request after every orchestration,
    # shadow, fallback, and cross-provider rewrite has run.
    if "qwen" in str(decision.model).casefold():
        routed.pop("tool_choice", None)
        routed.pop("parallel_tool_calls", None)
        routed.pop("reasoning", None)
    if decision.tier in {"luna", "spark"}:
        routed = _strip_historical_image_attachments(routed)

    # Determine the target provider for this tier (for logging only)
    tier_providers = cfg.get("tier_providers", {})
    target_provider = tier_providers.get(decision.tier, cfg.get("provider", "openai-codex"))

    routed["model"] = decision.model
    # NE váltson providert middleware-ben! A Hermes a config.yaml-ból veszi a providert.
    # A middleware csak a modelt és reasoning effort-ot módosítsa.
    # Only set reasoning effort for OpenAI-compatible providers.
    # Anthropic-transport providers (e.g. qwen-token) do not support the
    # `reasoning` keyword and will reject the request with a TypeError.
    # A provider váltás a config.yaml-ban történik, nem middleware-ben.
    # Check if the target model is Qwen (Anthropic Messages API)
    is_qwen_model = "qwen" in decision.model.lower()
    if not is_qwen_model:
        reasoning = dict(routed.get("reasoning") or {})
        reasoning["effort"] = decision.effort
        routed["reasoning"] = reasoning
    _log_decision(decision, kwargs, cfg)

    return {
        "request": routed,
        "source": "model-router",
        "reason": decision.reason,
        "metadata": {
            "tier": decision.tier,
            "model": decision.model,
            "effort": decision.effort,
            "provider": target_provider,
        },
    }


def _is_transient_provider_failure(error: BaseException) -> bool:
    """Return true only for failures that are safe to retry on another model."""
    text = str(error).casefold()
    return any(
        marker in text
        for marker in (
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "internal server error",
            "upstream connect error",
            "connection termination",
            "disconnect/reset before headers",
            "connection reset",
            "gateway timeout",
        )
    )


def _is_quota_exhaustion(error: BaseException) -> bool:
    """Recognize persistent provider-account exhaustion, not ordinary 429 pacing.

    A short provider rate limit should remain an error for the caller to retry;
    switching models for it would waste the fallback.  Weekly/account quotas do
    not recover during the current task, so Spark can safely hand that task to
    Terra instead.
    """
    text = str(error).casefold()
    return "429" in text and any(
        marker in text
        for marker in (
            "weekly limit",
            "weekly quota",
            "quota exhausted",
            "quota has been exhausted",
            "usage limit",
            "usage quota",
            "account quota",
            "credit balance is too low",
            "insufficient quota",
        )
    )


def _spark_quota_exhausted_for_turn(turn_id: str) -> bool:
    if not turn_id:
        return False
    with _QUOTA_LOCK:
        return turn_id in _SPARK_QUOTA_EXHAUSTED_TURNS


def _remember_spark_quota_exhaustion(turn_id: str) -> None:
    if not turn_id:
        return
    with _QUOTA_LOCK:
        if len(_SPARK_QUOTA_EXHAUSTED_TURNS) >= _MAX_REMEMBERED_QUOTA_TURNS:
            _SPARK_QUOTA_EXHAUSTED_TURNS.pop()
        _SPARK_QUOTA_EXHAUSTED_TURNS.add(turn_id)


def _quota_fallback_model(active_model: str, cfg: Dict[str, Any]) -> Optional[str]:
    """Resolve the configured one-time fallback for a depleted Spark quota."""
    models = cfg.get("models") or {}
    if active_model != models.get("spark"):
        return None
    configured = (cfg.get("quota_fallbacks") or {}).get("spark") or "luna"
    fallback_tier = str(configured.get("model") if isinstance(configured, dict) else configured).casefold()
    candidate = models.get(fallback_tier)
    if not _is_callable_tier(fallback_tier, cfg):
        return None
    return str(candidate) if candidate and candidate != active_model else None


def _quota_fallback_effort(cfg: Dict[str, Any], fallback_tier: str) -> str:
    """Use a quota-recovery-specific effort without altering normal tier routes."""
    configured = (cfg.get("quota_fallbacks") or {}).get("spark") or {}
    if isinstance(configured, dict) and configured.get("effort"):
        return str(configured["effort"])
    return str((cfg.get("effort") or {}).get(fallback_tier) or "medium")


def _is_model_unavailable(error: BaseException) -> bool:
    """A model this account cannot use at all, as opposed to one that is busy.

    Distinct from quota (recovers) and from a 5xx (a blip): the provider is saying
    the model does not exist for these credentials, so retrying it later in the
    same session is pointless. Matched on the message rather than the status code
    because the same refusal arrives as 400 and as 404 depending on the endpoint.
    """
    text = str(error).casefold()
    if not any(code in text for code in ("400", "404")):
        return False
    return any(
        marker in text
        for marker in (
            "is not supported when using",
            "model is not supported",
            "model not supported",
            "does not exist or you do not have access",
            "model not found",
            "no access to model",
            "is not available for your",
        )
    )


def _durable_fallback_model(
    active_model: str, cfg: Dict[str, Any], request: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Walk the CONFIGURED ``fallbacks`` chain for a model this account cannot use.

    Deliberately not ``_transient_fallback_model``: that one holds a hardcoded map
    for provider blips, while this case is the operator's own substitution policy —
    if the config says ``spark: luna``, a Spark that does not exist here belongs on
    Luna and nowhere else. Design and image guards still apply, because an
    unavailable model is no reason to violate a routing policy.
    """
    models = cfg.get("models") or {}
    active_tier = next((tier for tier, model in models.items() if model == active_model), "")
    if not active_tier:
        return None
    if (
        active_tier == "sol"
        and isinstance(request, dict)
        and _is_design_request(_last_user_text_and_index(_request_items(request))[0])
    ):
        return None
    chain = cfg.get("fallbacks") or {}
    visited = {active_tier}
    current = active_tier
    for _ in range(3):
        nxt = str(chain.get(current) or "")
        if not nxt or nxt in visited:
            return None
        visited.add(nxt)
        candidate = models.get(nxt)
        if candidate and candidate != active_model and _is_callable_tier(nxt, cfg):
            if (
                nxt == "spark"
                and isinstance(request, dict)
                and _request_has_image_attachment(request)
            ):
                current = nxt
                continue
            return str(candidate)
        current = nxt
    return None


def _transient_fallback_model(active_model: str, cfg: Dict[str, Any], request: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Choose one fallback without violating the Sol-only design boundary."""
    models = cfg.get("models") or {}
    if (
        active_model == models.get("sol")
        and isinstance(request, dict)
        and _is_design_request(_last_user_text_and_index(_request_items(request))[0])
    ):
        # A provider outage must not downgrade design analysis/implementation
        # to Terra or Spark. Preserve the original error for a later Sol retry.
        return None
    fallback_tier = {
        "luna": "spark",
        "spark": "sol",
        "terra": "sol",
        "sol": "spark",
    }.get(next((tier for tier, model in models.items() if model == active_model), ""))
    candidate = models.get(fallback_tier) if fallback_tier else None
    if fallback_tier and not _is_callable_tier(fallback_tier, cfg):
        candidate = None
    if candidate == models.get("spark") and isinstance(request, dict) and _request_has_image_attachment(request):
        image_fallback_tier = "terra" if active_model != models.get("terra") else "sol"
        candidate = models.get(image_fallback_tier) if _is_callable_tier(image_fallback_tier, cfg) else None
    return str(candidate) if candidate and candidate != active_model else None


def _opus5_write_intent(text: str) -> bool:
    """Return whether the coding request explicitly asks for repo mutation."""
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().casefold()
    return bool(re.search(
        r"\b(implement|fix|change|modify|edit|write|add|remove|refactor|migrate|"
        r"javits|javitas|modosits|modositas|implemental|keszits|hozzaad|torol|ird at)\b",
        normalized,
    ))


def _recent_verified_opus5_route(cfg: Dict[str, Any]) -> bool:
    """Use local route evidence as a cheap, credential-free availability gate."""
    coding_cfg = cfg.get("coding_agent") or {}
    policy = coding_cfg.get("explicit_ui") or {}
    ttl = max(1, int(policy.get("require_recent_verified_probe_seconds", 86400) or 86400))
    path = hermes_path((cfg.get("logging") or {}).get("path", _DEFAULT_CONFIG["logging"]["path"]))
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return False
    now = datetime.now(timezone.utc)
    for line in reversed(lines[-5000:]):
        try:
            event = json.loads(line)
            if event.get("tier") != "opus5" or event.get("model") != "claude-opus-5-5":
                continue
            observed = datetime.fromisoformat(str(event.get("timestamp") or "").replace("Z", "+00:00"))
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            return (now - observed).total_seconds() <= ttl
        except Exception:
            continue
    return False


_GOAL_ABSOLUTE_PATH = re.compile(r"(?:(?<!\S)|(?<=[`(\"']))(?:/|~/)\S+")
_WORKSPACE_PATH_BLOCK = re.compile(r"(?im)^WORKSPACE PATH:\s*\r?\n\s*([^\r\n]+)")
_PATH_TRAILING_PUNCTUATION = ".,;:!?)]}\"'`"
_REPO_DIRECTORY_CACHE_TTL_SECONDS = 60
_MAX_REPO_DIRECTORY_CACHE_ENTRIES = 512
_DISPATCH_REVIEW_REPOSITORY_TTL_SECONDS = 60 * 60
_MAX_DISPATCH_REVIEW_REPOSITORIES = 256
_REPO_DIRECTORY_CACHE: OrderedDict[Tuple[str, bool], Tuple[float, Optional[Path]]] = OrderedDict()
_DISPATCH_REVIEW_REPOSITORIES: OrderedDict[str, Tuple[float, object]] = OrderedDict()
_AMBIGUOUS_DISPATCH_REVIEW_REPOSITORY = object()
_REPOSITORY_CACHE_LOCK = threading.RLock()


def _bounded_cache_get(cache: OrderedDict, key: Any, ttl_seconds: float) -> Tuple[bool, Any]:
    with _REPOSITORY_CACHE_LOCK:
        entry = cache.get(key)
        if entry is None:
            return False, None
        recorded_at, value = entry
        if time.monotonic() - recorded_at >= ttl_seconds:
            cache.pop(key, None)
            return False, None
        cache.move_to_end(key)
        return True, value


def _bounded_cache_put(cache: OrderedDict, key: Any, value: Any, *, max_entries: int) -> None:
    with _REPOSITORY_CACHE_LOCK:
        cache[key] = (time.monotonic(), value)
        cache.move_to_end(key)
        while len(cache) > max_entries:
            cache.popitem(last=False)


def _existing_directory(path: Path) -> Optional[Path]:
    """``path`` resolved when it is a directory, else None -- never raises.

    The path often comes from a task's text, so a name the OS refuses (a
    component over NAME_MAX raises ENAMETOOLONG, not False) must read as "no
    directory" rather than escape into the admission guard, which the host
    then skips.
    """
    try:
        return path.resolve() if path.is_dir() else None
    except (OSError, ValueError, RuntimeError):
        return None


def _repo_directory(value: str, *, git_top_level: bool = False) -> Optional[Path]:
    """Return an existing directory, collapsing a Git child to its work-tree root."""
    try:
        candidate = Path(str(value).strip()).expanduser()
    except RuntimeError:
        return None
    candidate = _existing_directory(candidate)
    if candidate is None:
        return None
    cache_key = (str(candidate), git_top_level)
    found, cached = _bounded_cache_get(
        _REPO_DIRECTORY_CACHE, cache_key, _REPO_DIRECTORY_CACHE_TTL_SECONDS,
    )
    if found:
        return cached
    try:
        completed = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
            check=False, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        completed = None
    if completed is not None and completed.returncode == 0:
        result = _existing_directory(Path(completed.stdout.strip()))
    else:
        result = None if git_top_level else candidate
    _bounded_cache_put(
        _REPO_DIRECTORY_CACHE, cache_key, result,
        max_entries=_MAX_REPO_DIRECTORY_CACHE_ENTRIES,
    )
    return result


def _goal_repository(text: str) -> Optional[Path]:
    for token in _GOAL_ABSOLUTE_PATH.findall(text or ""):
        candidate = _repo_directory(token.rstrip(_PATH_TRAILING_PUNCTUATION), git_top_level=True)
        if candidate is not None:
            return candidate
    return None


def _workspace_repository(request: Optional[Dict[str, Any]]) -> Optional[Path]:
    if not isinstance(request, dict):
        return None
    texts = [_text_from_content(request.get("instructions"))]
    texts.extend(
        _text_from_content(item.get("content"))
        for item in _request_items(request)
        if isinstance(item, dict) and str(item.get("role") or "").casefold() == "system"
    )
    for text in texts:
        matches = list(_WORKSPACE_PATH_BLOCK.finditer(text or ""))
        if matches:
            candidate = _repo_directory(matches[-1].group(1).strip())
            if candidate is not None:
                return candidate
    return None


def _remember_dispatch_review_repository(text: str, repository: Path) -> None:
    key = _normalise(text)
    resolved = repository.resolve()
    with _REPOSITORY_CACHE_LOCK:
        found, remembered = _bounded_cache_get(
            _DISPATCH_REVIEW_REPOSITORIES, key, _DISPATCH_REVIEW_REPOSITORY_TTL_SECONDS,
        )
        if found and remembered is not _AMBIGUOUS_DISPATCH_REVIEW_REPOSITORY and remembered != resolved:
            resolved = _AMBIGUOUS_DISPATCH_REVIEW_REPOSITORY
        elif found:
            resolved = remembered
        _bounded_cache_put(
            _DISPATCH_REVIEW_REPOSITORIES, key, resolved,
            max_entries=_MAX_DISPATCH_REVIEW_REPOSITORIES,
        )


def _dispatch_review_repository_match(normalised: str, goal: str) -> bool:
    if len(goal) < 20 or not normalised.startswith(goal):
        return False
    return len(normalised) == len(goal) or normalised[len(goal)] == " "


def _remembered_dispatch_review_repository(text: str) -> Optional[Path]:
    normalised = _normalise(text)
    with _REPOSITORY_CACHE_LOCK:
        found, repository = _bounded_cache_get(
            _DISPATCH_REVIEW_REPOSITORIES, normalised,
            _DISPATCH_REVIEW_REPOSITORY_TTL_SECONDS,
        )
        if not found:
            matching_entry: Optional[Tuple[str, object]] = None
            for goal, (recorded_at, candidate) in list(_DISPATCH_REVIEW_REPOSITORIES.items()):
                if time.monotonic() - recorded_at >= _DISPATCH_REVIEW_REPOSITORY_TTL_SECONDS:
                    _DISPATCH_REVIEW_REPOSITORIES.pop(goal, None)
                elif _dispatch_review_repository_match(normalised, goal):
                    if matching_entry is None or len(goal) > len(matching_entry[0]):
                        matching_entry = (goal, candidate)
            if matching_entry is None:
                return None
            goal, repository = matching_entry
            _DISPATCH_REVIEW_REPOSITORIES.move_to_end(goal)
        # Copy the candidate under the lock, then stat it (Path.is_dir, a
        # filesystem call) only after releasing the lock: a slow or stalled
        # stat must not block every other route holding this shared lock.
        candidate = repository
    return _existing_remembered_directory(candidate)


def _existing_remembered_directory(repository: object) -> Optional[Path]:
    if not isinstance(repository, Path):
        return None
    return repository if _existing_directory(repository) is not None else None


def _delegated_review_repository(text: str, cfg: Dict[str, Any], *, request: Optional[Dict[str, Any]] = None,
                                 dispatch_cwd: Optional[Path] = None,
                                 at_dispatch: bool = False) -> Optional[Path]:
    """Resolve a labelled review leaf's repository without reading the process cwd.

    The child request carries its workspace path. Dispatch has no child request, so
    it may supply the parent's cwd, but only an existing Git work tree is usable.
    ``at_dispatch`` is the caller's own explicit signal for "this is a dispatch, not
    an execution" -- it must not be inferred from ``dispatch_cwd is None``, since a
    dispatch whose parent has no resolvable workspace hint also passes ``None``. The
    remembered dispatch->repo map is consulted only at execution (``not at_dispatch``);
    at dispatch the task's own sources (goal, aliases, request, cwd, default) decide.
    """
    goal_repo = _goal_repository(text)
    if goal_repo is not None:
        return goal_repo
    coding_cfg = cfg.get("coding_agent") or {}
    normalised = _normalise(text)
    for alias, value in (coding_cfg.get("repo_aliases") or {}).items():
        if _normalise(str(alias)) in normalised:
            candidate = _repo_directory(str(value))
            if candidate is not None:
                return candidate
    if not at_dispatch:
        dispatched_repo = _remembered_dispatch_review_repository(text)
        if dispatched_repo is not None:
            return dispatched_repo
    workspace_repo = _workspace_repository(request)
    if workspace_repo is not None:
        return workspace_repo
    if dispatch_cwd is not None:
        cwd_repo = _repo_directory(str(dispatch_cwd), git_top_level=True)
        if cwd_repo is not None:
            return cwd_repo
    value = str(coding_cfg.get("default_repo") or "").strip()
    return _repo_directory(value) if value else None


def _opus5_repo_for_request(text: str, cfg: Dict[str, Any]) -> Optional[Path]:
    """Resolve only configured local roots; never discover or read credentials."""
    coding_cfg = cfg.get("coding_agent") or {}
    normalised = _normalise(text)
    aliases = coding_cfg.get("repo_aliases") or {}
    for alias, value in aliases.items():
        if _normalise(str(alias)) in normalised:
            candidate = Path(str(value)).expanduser()
            return candidate.resolve() if candidate.is_dir() else None
    value = str(coding_cfg.get("default_repo") or "").strip()
    candidate = Path(value).expanduser() if value else None
    return candidate.resolve() if candidate is not None and candidate.is_dir() else None


def _verified_explicit_opus5_ui_repo(text: str, cfg: Dict[str, Any]) -> Optional[Path]:
    """Return a repo only after all cheap local bridge eligibility checks pass."""
    coding_cfg = cfg.get("coding_agent") or {}
    canonical = str(coding_cfg.get("canonical_model") or coding_cfg.get("model") or "")
    if (
        not coding_cfg.get("enabled")
        or canonical != "claude-opus-5-5"
        or not _is_explicit_bounded_opus_ui_request(text, cfg)
        or shutil.which("claude") is None
        or not _recent_verified_opus5_route(cfg)
    ):
        return None
    return _opus5_repo_for_request(text, cfg)


def _verified_explicit_opus5_review_repo(text: str, cfg: Dict[str, Any]) -> Optional[Path]:
    """Permit an explicit, bounded Opus review with no write capability."""
    coding_cfg = cfg.get("coding_agent") or {}
    reviewer_cfg = coding_cfg.get("reviewer") or {}
    canonical = str(coding_cfg.get("canonical_model") or coding_cfg.get("model") or "")
    if (
        not coding_cfg.get("enabled")
        or not reviewer_cfg.get("enabled")
        or canonical != "claude-opus-5-5"
        or len(text or "") > int(reviewer_cfg.get("max_chars", 8000) or 8000)
        or shutil.which("claude") is None
        or not _recent_verified_opus5_route(cfg)
    ):
        return None
    from .claude_opus_bridge import classify_review_dispatch

    eligible, _reason = classify_review_dispatch(text)
    return _opus5_repo_for_request(text, cfg) if eligible else None


def _delegated_claude_review_status(text: str, cfg: Dict[str, Any], *, request: Optional[Dict[str, Any]] = None,
                                    dispatch_cwd: Optional[Path] = None,
                                    requested_model: Optional[str] = None,
                                    at_dispatch: bool = False) -> Tuple[Optional[Tuple[Path, str]], str]:
    """Return the static delegated-review route or the reason it cannot be taken.

    ``at_dispatch`` is the caller's explicit "this is a dispatch" signal, passed
    straight through to ``_delegated_review_repository``; a hint-less dispatch
    (``dispatch_cwd is None``) must not be mistaken for an execution just because
    it lacks a workspace hint.
    """
    coding_cfg = cfg.get("coding_agent") or {}
    policy = coding_cfg.get("delegated_review") or {}
    if not policy.get("enabled"):
        return None, "delegated-review policy is disabled"
    if len(text or "") > int(policy.get("max_chars", 8000) or 8000):
        return None, "review goal exceeds the delegated-review character limit"
    if shutil.which("claude") is None:
        return None, "Claude CLI is unavailable"
    from .claude_opus_bridge import CLAUDE_REVIEW_MODELS, review_model_alias

    alias = review_model_alias(text)
    if alias is None:
        return None, "goal has no supported Claude review label"
    named_model = str(requested_model or "").strip()
    if named_model:
        expected_model = CLAUDE_REVIEW_MODELS.get(alias, "")
        target_model = (_delegation_targets_detail().get(named_model.casefold()) or {}).get("model", "")
        if named_model.casefold() != expected_model.casefold() and str(target_model).casefold() != expected_model.casefold():
            return None, f"task names model {named_model}"
    allowed = policy.get("models")
    if isinstance(allowed, list) and alias not in [str(name).casefold() for name in allowed]:
        return None, f"Claude {alias} review tier is not allowed"
    if alias not in CLAUDE_REVIEW_MODELS:
        return None, f"Claude {alias} review tier is unknown"
    target = {"opus": "opus5", "sonnet": "sonnet5"}[alias]
    if not _is_callable_tier(target, cfg):
        return None, f"Claude {alias} review tier is switched off"
    repo = _delegated_review_repository(text, cfg, request=request, dispatch_cwd=dispatch_cwd,
                                        at_dispatch=at_dispatch)
    if repo is None:
        return None, "no repository could be resolved"
    if dispatch_cwd is not None:
        _remember_dispatch_review_repository(text, repo)
    return (repo, alias), ""


def _verified_delegated_claude_review(text: str, cfg: Dict[str, Any], *, request: Optional[Dict[str, Any]] = None) -> Optional[Tuple[Path, str]]:
    """Resolve a delegated, read-only Claude review leaf to (repo, Claude tier), or None.

    Thin wrapper over ``_delegated_claude_review_status``, which now also checks
    the dispatch-time ``requested_model`` against the review label and returns the
    reason a route was refused; this wrapper drops both and keeps only the
    (repo, tier) result, for callers that just need to know whether a route exists.

    Deliberately independent of ``coding_agent.enabled``. That switch also arms
    the conservative coding classifier, which fires with no explicit label and
    would capture the first call of a coding turn -- the reason the whole bridge
    is off. This path needs none of that: it requires an explicit review label
    the planner had to write, and it is the caller's job to admit only delegated
    workers, so a root turn can never be diverted into a subprocess.
    """
    return _delegated_claude_review_status(text, cfg, request=request)[0]


def _run_opus5_bridge(*, repo: str, task: str, write: bool, review: bool = False, cfg: Dict[str, Any],
                      model: Optional[str] = None, requested_alias: Optional[str] = None,
                      adjustment: str = "", identity: Any = None, **context: Any) -> Dict[str, Any]:
    """Middleware CLI entry; keep admission at the existing policy call site.

    Maps router config onto the public ``claude_opus_bridge.dispatch``, which
    owns the one normalised CLI boundary (shared with the standalone ``main()``
    and direct Python callers) and calls only its private raw subprocess
    operation. Not wrapped again here, so each run yields one receipt. No
    concurrency cap: the pre-S05 bridge had none. A typed ClaudeBridgeFailure is
    a terminal outcome and is recorded as such, never held as unknown.
    """
    from .claude_opus_bridge import dispatch

    coding_cfg = cfg.get("coding_agent") or {}
    return dispatch(
        task,
        Path(repo),
        write=write,
        review=review,
        model=model,
        requested_alias=requested_alias,
        adjustment=adjustment,
        identity=identity,
        timeout=int(coding_cfg.get("timeout_seconds", 300)),
        max_turns=(coding_cfg.get("delegated_review") or {}).get("max_turns") if review else coding_cfg.get("max_turns"),
        max_budget_usd=coding_cfg.get("max_budget_usd", 5.0),
        parent_session_id=str(context.get("parent_session_id") or context.get("session_id")
                              or str(context.get("turn_id") or "").split(":", 1)[0]),
        parent_turn_id=str(context.get("parent_turn_id") or context.get("turn_id") or ""),
        lifecycle_path=hermes_path(coding_cfg["lifecycle_path"]) if coding_cfg.get("lifecycle_path") else None,
    )


_SUBSTITUTION_PROVENANCE_PREFIX = "[ROUTER SUBSTITUTION PROVENANCE v1] "


def _provenance_text(value: Any, fallback: str = "unknown") -> str:
    return value[:128] if isinstance(value, str) and value else fallback


def _provenance_fact(value: Any) -> Dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    return {
        "value": _provenance_text(value.get("value")),
        "source": _provenance_text(value.get("source"), "not_observed"),
        "canonical": bool(value.get("canonical", True)),
    }


def _provenance_identity(value: Any) -> Dict[str, Any]:
    if hasattr(value, "as_dict"):
        value = value.as_dict()
    value = value if isinstance(value, dict) else {}
    effort = value.get("effort") if isinstance(value.get("effort"), dict) else {}
    return {
        "provider": _provenance_text(value.get("provider")),
        "account": _provenance_text(value.get("account")),
        "transport": _provenance_text(value.get("transport")),
        "alias": _provenance_text(value.get("alias")),
        "selection_mode": _provenance_text(value.get("selection_mode"), "unknown"),
        "requested": _provenance_fact(value.get("requested")),
        "resolved": _provenance_fact(value.get("resolved")),
        "observed": _provenance_fact(value.get("observed")),
        "effort": {
            "requested": _provenance_text(effort.get("requested"), "unknown"),
            "applied": _provenance_text(effort.get("applied"), "unknown"),
            "source": _provenance_text(effort.get("source"), "not_observed"),
        },
    }


def _provenance_mapping(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key)[:64]: _provenance_text(item) for key, item in sorted(value.items())[:16]
            if isinstance(key, str) and isinstance(item, str)}


def _substitution_provenance(kind: str, *, identity: Any, substitution: Any,
                             replacement: Any = None, executed: Any = None) -> Dict[str, Any]:
    marker = {
        "kind": kind,
        "identity": _provenance_identity(identity),
        "substitution": _provenance_mapping(substitution),
        # A replacement or a same-provider step-down is not an independent,
        # cross-provider review merely because its output contains this evidence.
        "satisfies_cross_provider_review": False,
    }
    if isinstance(replacement, dict):
        marker.update({
            "policy": _provenance_text(replacement.get("policy")),
            "reason": _provenance_text(replacement.get("reason")),
            "failure_kind": _provenance_text(replacement.get("failure_kind")),
            "requested_review": _provenance_mapping(replacement.get("requested_review")),
            "planned": _provenance_mapping(replacement.get("planned")),
        })
    if isinstance(executed, dict):
        marker["executed"] = _provenance_mapping(executed)
    return marker


def _provenance_annotation(marker: Dict[str, Any]) -> str:
    encoded = json.dumps(marker, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _SUBSTITUTION_PROVENANCE_PREFIX + encoded


class _AnnotatedResponseOverlay:
    """Read-through view of a response that cannot take the provenance carrier.

    ``output`` and ``output_text`` are the overlay's own values; every other
    attribute (usage, status, incomplete details, error, id, ...) is read from
    the untouched original, so nothing the host reads is dropped.
    """

    def __init__(self, original: Any, output: list, output_text: str) -> None:
        self._original = original
        self.output = output
        self.output_text = output_text

    def __getattr__(self, name: str) -> Any:
        original = self.__dict__.get("_original")
        if original is None:
            raise AttributeError(name)
        if isinstance(original, dict):
            try:
                return original[name]
            except KeyError as exc:
                raise AttributeError(name) from exc
        return getattr(original, name)


_PROVENANCE_TOOL_ITEM_TYPES = frozenset({"function_call", "custom_tool_call"})
_PROVENANCE_COMMENTARY_PHASES = frozenset({"commentary", "analysis"})
_PROVENANCE_FINAL_PHASES = frozenset({"final_answer", "final"})
# Mirrors the host normalizer (agent/codex_responses_adapter.py): these statuses
# make it continue the turn, except on server-side tool items, which xAI leaves
# ``in_progress`` inside completed responses.
_PROVENANCE_INCOMPLETE_STATUSES = frozenset({"queued", "in_progress", "incomplete"})
_PROVENANCE_SERVER_TOOL_ITEM_TYPES = frozenset({
    "web_search_call", "file_search_call", "code_interpreter_call",
    "image_generation_call", "computer_call", "local_shell_call", "mcp_call",
})


def _provenance_field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _provenance_item_text(item: Any) -> str:
    """Assistant text of one Responses message item, as the host reads it.

    A ``refusal`` part carries its text in ``refusal``; the host treats it as
    final assistant text, so it counts as final text here too.
    """
    if _provenance_field(item, "type") != "message":
        return ""
    content = _provenance_field(item, "content")
    chunks = []
    for part in (content if isinstance(content, list) else ()):
        kind = _provenance_field(part, "type")
        text = (_provenance_field(part, "refusal") if kind == "refusal"
                else _provenance_field(part, "text") if kind in ("output_text", "text") else None)
        if isinstance(text, str):
            chunks.append(text)
    return "".join(chunks)


def _provenance_shape(result: Any) -> Optional[Dict[str, Any]]:
    """What the host normalizer will make of a Responses result, or None."""
    output = _provenance_field(result, "output")
    if not isinstance(output, list):
        return None
    aggregate = _provenance_field(result, "output_text")
    if not isinstance(aggregate, str):
        aggregate = "\n".join(text for text in map(_provenance_item_text, output) if text)
    phases = [str(_provenance_field(item, "phase") or "").casefold() for item in output
              if _provenance_field(item, "type") == "message"]
    commentary = any(phase in _PROVENANCE_COMMENTARY_PHASES for phase in phases)
    final_phase = any(phase in _PROVENANCE_FINAL_PHASES for phase in phases)
    has_tool_call = any(_provenance_field(item, "type") in _PROVENANCE_TOOL_ITEM_TYPES for item in output)
    has_item_text = any(_provenance_item_text(item).strip()
                        and str(_provenance_field(item, "phase") or "").casefold()
                        not in _PROVENANCE_COMMENTARY_PHASES
                        for item in output)
    # The host falls back to the aggregate when no item carries final text
    # (stream-delivered answers), unless the text is commentary-only.
    aggregate_only = (not has_item_text and bool(aggregate.strip())
                      and (final_phase or not commentary))
    # Same precedence as the host's ``_normalize_codex_response``: a queued or
    # in-progress response, or an incomplete non-server item, always continues;
    # a top-level ``incomplete`` (e.g. max_output_tokens) or commentary
    # continues only without a final-phase message; an ``incomplete`` response
    # whose reason is ``content_filter`` is a refusal, never a final ``stop``.
    # The final-output hook (``on_transform_llm_output``) is the backstop when
    # this mirror and the host ever disagree.
    status = str(_provenance_field(result, "status") or "").casefold()
    details = _provenance_field(result, "incomplete_details")
    reason = str(_provenance_field(details, "reason") or "").strip().casefold() if details is not None else ""
    item_incomplete = any(
        str(_provenance_field(item, "status") or "").casefold() in _PROVENANCE_INCOMPLETE_STATUSES
        and _provenance_field(item, "type") not in _PROVENANCE_SERVER_TOOL_ITEM_TYPES
        for item in output)
    incomplete = (
        (status == "incomplete" and reason == "content_filter")
        or status in ("queued", "in_progress")
        or item_incomplete
        or ((status == "incomplete" or commentary) and not final_phase)
    )
    return {
        "output": output, "aggregate": aggregate, "phases": phases,
        "has_tool_call": has_tool_call, "has_final_text": has_item_text or aggregate_only,
        "aggregate_only": aggregate_only,
        # A reply the host ends the turn with: final text, no tool call, nothing
        # that makes the normalizer report ``incomplete``.
        "terminal": (has_item_text or aggregate_only) and not has_tool_call and not incomplete,
    }


def _annotated_response(result: Any, marker: Dict[str, Any]) -> Any:
    """Add one parent-visible provenance carrier without changing execution semantics.

    The carrier is a separate assistant message appended after every original
    output item, holding only the marker line. The host normalizer joins message
    texts with a newline, so the marker lands on its own line after all useful
    final text (multipart text cannot run into the JSON), and a final-phase
    carrier stays in the projected final answer when an earlier commentary
    message is routed to reasoning. No original item is mutated, removed or
    reordered: tool calls, reasoning, statuses and usage survive, and the
    finish reason the host derives is unchanged.

    Never partially applied: an SDK model (computed read-only ``output_text``)
    gets a copy with the extra item; a mutable response gets the aggregate
    written first (the only step that can fail) and then the item appended; any
    other response gets a read-through overlay. A response with neither final
    text nor a tool call (reasoning- or commentary-only, which the host
    continues) is returned unannotated: marker-only text would turn it into a
    final ``stop``.
    """
    annotation = _provenance_annotation(marker)

    shape = _provenance_shape(result)
    if shape is None:
        # A scalar/non-Responses legacy value has no host-supported content
        # carrier. Preserve its byte-for-byte contract rather than replacing it.
        return result
    output, aggregate, phases = shape["output"], shape["aggregate"], shape["phases"]
    if not (shape["has_final_text"] or shape["has_tool_call"]):
        return result
    if any(_SUBSTITUTION_PROVENANCE_PREFIX in _provenance_item_text(item) for item in output):
        return result  # already carries one; never add a second marker
    # Stream-delivered answers can arrive as ``output_text`` with no item final
    # text; the host reads that aggregate, so carry it as an item too (once an
    # item carries text the host stops reading the aggregate).
    leading = [("text", aggregate.strip())] if shape["aggregate_only"] else []
    # Mirror the response's own phase convention: final_answer only when the
    # response already declares a final phase. Adding one to a phase-less or
    # commentary-only response would change the host's finish reason.
    phase = "final_answer" if any(p in _PROVENANCE_FINAL_PHASES for p in phases) else None
    annotated_text = aggregate.rstrip() + ("\n\n" if aggregate.strip() else "") + annotation
    # The leading newlines keep a computed SDK ``output_text`` (parts joined with
    # no separator) parseable too; the normalizer strips them from content.
    carrier_text = "\n\n" + annotation

    model_copy = getattr(result, "model_copy", None)
    if callable(model_copy) and not isinstance(result, dict):
        try:
            from openai.types.responses import ResponseOutputMessage, ResponseOutputText

            def sdk_message(text: str, item_phase: Optional[str]) -> Any:
                return ResponseOutputMessage.model_construct(
                    id="", type="message", role="assistant", status="completed", phase=item_phase,
                    content=[ResponseOutputText.model_construct(type="output_text", text=text,
                                                                annotations=[])])
            extra = [sdk_message(text, None) for _, text in leading]
            copied = model_copy(update={"output": [*output, *extra, sdk_message(carrier_text, phase)]})
            if isinstance(getattr(copied, "output", None), list):
                return copied
        except Exception:
            pass  # fall through to the overlay; the original is untouched

    def message(text: str, item_phase: Optional[str]) -> Any:
        return SimpleNamespace(type="message", role="assistant", status="completed", phase=item_phase,
                               content=[SimpleNamespace(type="output_text", text=text)])
    added = [message(text, None) for _, text in leading] + [message(carrier_text, phase)]
    try:
        if isinstance(result, dict):
            result["output_text"] = annotated_text
        else:
            setattr(result, "output_text", annotated_text)
    except Exception:
        return _AnnotatedResponseOverlay(result, [*output, *added], annotated_text)
    output.extend(added)
    return result


def _opus5_response(result: Dict[str, Any]) -> Any:
    """Adapt a verified Claude Code result to Hermes' Codex Responses contract."""
    from .claude_opus_bridge import CLAUDE_REVIEW_MODELS

    text = str(result.get("result") or "").strip()
    model = str(result.get("effective_model") or result.get("model") or "")
    if not text or model not in set(CLAUDE_REVIEW_MODELS.values()):
        raise RuntimeError(f"Claude bridge returned no verified result; effective model was {model or 'missing'}")
    raw_usage = result.get("usage") or result.get("model_usage") or {}
    input_tokens = int(raw_usage.get("input_tokens", raw_usage.get("inputTokens", 0)) or 0)
    output_tokens = int(raw_usage.get("output_tokens", raw_usage.get("outputTokens", 0)) or 0)
    response = SimpleNamespace(
        output=[SimpleNamespace(
            type="message",
            status="completed",
            content=[SimpleNamespace(type="output_text", text=text)],
        )],
        output_text=text,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        status="completed",
        model=model,
    )
    substitution = result.get("substitution")
    if isinstance(substitution, dict):
        return _annotated_response(response, _substitution_provenance(
            "claude_cli_step_down", identity=result.get("identity"), substitution=substitution,
        ))
    return response


def _record_ordinary_replacement_result(result: Any, replacement: Dict[str, Any]) -> Any:
    """Carry final ordinary-provider evidence in normalizer-surviving content."""
    def field(name: str) -> Any:
        return result.get(name) if isinstance(result, dict) else getattr(result, name, None)

    model = str(field("model") or "").strip()
    provider = str(field("provider") or "").strip()
    executed = {
        "model": model or "unknown",
        "model_source": "ordinary_provider.response.model" if model else "not_observed",
        "provider": provider or "unknown",
        "provider_source": "ordinary_provider.response.provider" if provider else "not_observed",
        "status": "observed" if model or provider else "unknown",
    }
    provenance = {"policy": replacement["policy"], "requested_review": replacement["requested_review"],
                  "failure_kind": replacement["failure_kind"], "planned": replacement["planned"],
                  "executed": executed}
    route_reason = (
        "Claude CLI review did not complete; actual replacement executed on "
        f"{executed['model']} via {executed['provider']} "
        f"(model evidence: {executed['model_source']}; provider evidence: {executed['provider_source']})."
    )
    annotated = _annotated_response(result, _substitution_provenance(
        "ordinary_replacement", identity=replacement.get("identity"), substitution={},
        replacement=replacement, executed=executed,
    ))
    if isinstance(annotated, dict):
        annotated["replacement_provenance"] = provenance
        annotated["route_reason"] = route_reason
    else:
        try:
            setattr(annotated, "replacement_provenance", provenance)
            setattr(annotated, "route_reason", route_reason)
        except Exception:
            pass
    return annotated


# Replacement provenance retained across the API calls of one child turn (N1).
# A failed review's ordinary replacement can end its first call on reasoning,
# commentary or a tool call; the host then continues the same turn, and only the
# eventual final reply reaches the parent. The entry is keyed on the turn id
# the host passes to every middleware call of that turn (and to post_llm_call),
# kept while the host may still continue the turn, removed when the turn ends
# (post_llm_call), and bounded by a TTL and a size cap so an abandoned turn
# cannot leak.
# Each entry is (monotonic stamp, replacement, marker lines the middleware has
# already put on this turn's replies); the final-output hook uses the last one
# to avoid marking a reply twice.
_PENDING_REPLACEMENTS: "OrderedDict[str, Tuple[float, Dict[str, Any], set]]" = OrderedDict()
_PENDING_REPLACEMENT_LOCK = threading.Lock()
_PENDING_REPLACEMENT_MAX = 256
_PENDING_REPLACEMENT_TTL_SECONDS = 6 * 3600.0
_PENDING_REPLACEMENT_MARKERS_MAX = 8
_pending_replacement_clock = time.monotonic


def _pending_replacement_key(context: Dict[str, Any]) -> Optional[str]:
    # The host mints one turn id per turn (``<session>:<task>:<uuid>``, kept for
    # every call of the turn) and passes it to the middleware, to
    # transform_llm_output and to post_llm_call. The session id is not part of
    # the key: compression can rotate it in the middle of a turn, which would
    # orphan the entry.
    turn_id = str(context.get("turn_id") or "")
    return turn_id or None  # without a turn identity the state could not be scoped


def _pending_replacement_prune(now: float) -> None:
    """Drop expired entries, then the oldest beyond the cap. Caller holds the lock."""
    for key in [key for key, entry in _PENDING_REPLACEMENTS.items()
                if now - entry[0] > _PENDING_REPLACEMENT_TTL_SECONDS]:
        del _PENDING_REPLACEMENTS[key]
    while len(_PENDING_REPLACEMENTS) > max(0, int(_PENDING_REPLACEMENT_MAX)):
        _PENDING_REPLACEMENTS.popitem(last=False)


def _pending_replacement_get(key: Optional[str]) -> Optional[Dict[str, Any]]:
    entry = _pending_replacement_entry(key)
    return entry[0] if entry else None


def _pending_replacement_entry(key: Optional[str]) -> Optional[Tuple[Dict[str, Any], Tuple[str, ...]]]:
    """(replacement, marker lines already delivered on this turn's replies), or None."""
    if key is None:
        return None
    with _PENDING_REPLACEMENT_LOCK:
        _pending_replacement_prune(_pending_replacement_clock())
        entry = _PENDING_REPLACEMENTS.get(key)
        return (entry[1], tuple(entry[2])) if entry else None


def _pending_replacement_put(key: Optional[str], replacement: Dict[str, Any],
                             marker_lines: Tuple[str, ...] = ()) -> None:
    if key is None:
        return
    with _PENDING_REPLACEMENT_LOCK:
        now = _pending_replacement_clock()
        previous = _PENDING_REPLACEMENTS.pop(key, None)
        delivered = [*(previous[2] if previous else ()), *marker_lines]
        _PENDING_REPLACEMENTS[key] = (now, replacement,
                                      tuple(delivered[-max(1, int(_PENDING_REPLACEMENT_MARKERS_MAX)):]))
        _pending_replacement_prune(now)


def _pending_replacement_discard(key: Optional[str]) -> None:
    if key is None:
        return
    with _PENDING_REPLACEMENT_LOCK:
        _PENDING_REPLACEMENTS.pop(key, None)


def _deliver_replacement_result(result: Any, replacement: Dict[str, Any],
                                key: Optional[str], *, first_call: bool) -> Any:
    """Annotate a replacement reply and keep its provenance for the rest of the turn.

    The reply that failed over keeps the round-1 behaviour (a final or tool-call
    reply carries the marker). A later reply of the same turn is annotated only
    when it would end the turn, so interim reasoning, commentary and tool calls
    stay exactly as the provider sent them; a reasoning- or commentary-only
    reply is never turned into a marker-only answer. The entry is kept even
    after a terminal reply, because the host can still continue the turn (ack,
    stall or degenerate-final nudges) and only its last reply reaches the
    parent; it is removed when the turn ends (``on_post_llm_call``) or by the
    TTL/size bounds.
    """
    shape = _provenance_shape(result)
    if shape is None:
        return _record_ordinary_replacement_result(result, replacement) if first_call else result
    if first_call or shape["terminal"]:
        annotated = _record_ordinary_replacement_result(result, replacement)
        _pending_replacement_put(key, replacement, _provenance_marker_lines(annotated))
        return annotated
    _pending_replacement_put(key, replacement)
    return result


def _provenance_marker_lines(result: Any) -> Tuple[str, ...]:
    """Marker lines carried by a Responses result's message items."""
    output = _provenance_field(result, "output")
    lines = []
    for item in (output if isinstance(output, list) else ()):
        lines += [line.strip() for line in _provenance_item_text(item).splitlines()
                  if line.strip().startswith(_SUBSTITUTION_PROVENANCE_PREFIX.strip())]
    return tuple(lines)


def on_transform_llm_output(response_text: Any = None, turn_id: Any = None, **_: Any) -> Optional[str]:
    """Final-delivery backstop for a turn whose review was substituted (N3/N4).

    The host fires ``transform_llm_output`` once per turn with the text that
    becomes the turn's final response (and so a child's ``summary`` for its
    parent), including text the execution middleware never saw: the
    iteration-limit summary is a direct provider call, and a reply whose shape
    the middleware's terminal mirror misjudged is passed through unmarked. When
    this turn has retained replacement provenance and the final text does not
    already carry a marker line this router delivered on the turn, one marker
    line is appended. A turn with no retained entry gets ``None`` (text
    unchanged). The executed provider/model of the final text is not observed
    here, so it is reported ``unknown``/``not_observed``.
    """
    key = _pending_replacement_key({"turn_id": turn_id})
    entry = _pending_replacement_entry(key)
    if entry is None or not isinstance(response_text, str) or not response_text.strip():
        return None
    replacement, delivered = entry
    present = {line.strip() for line in response_text.splitlines()}
    if any(line in present for line in delivered):
        return None  # the middleware already marked the reply that ends the turn
    unobserved = {"model": "unknown", "model_source": "not_observed",
                  "provider": "unknown", "provider_source": "not_observed", "status": "unknown"}
    marker = _substitution_provenance("ordinary_replacement", identity=replacement.get("identity"),
                                      substitution={}, replacement=replacement, executed=unobserved)
    marker["carrier"] = "final_output"
    return response_text.rstrip() + "\n\n" + _provenance_annotation(marker)


def _maybe_run_opus5(request: Dict[str, Any], cfg: Dict[str, Any], **kwargs: Any) -> Optional[Any]:
    """Execute the first safe, non-design coding call through Claude Code OAuth."""
    coding_cfg = cfg.get("coding_agent") or {}
    if not coding_cfg.get("enabled") and not (coding_cfg.get("delegated_review") or {}).get("enabled"):
        return None
    if int(kwargs.get("api_call_count") or 1) != 1:
        return None
    if str(kwargs.get("api_mode") or "") != "codex_responses":
        return None

    source_request = kwargs.get("original_request")
    if not isinstance(source_request, dict):
        source_request = request
    text = _last_user_text_and_index(_request_items(source_request))[0]
    routing_text = _without_host_injected_context(_without_router_contract(text))

    from .claude_opus_bridge import review_model_alias
    requested_alias = review_model_alias(routing_text) or "opus"
    adjustment = ""
    admission_refusal = ""

    def bridge_failure_kind(error: BaseException) -> str:
        message = str(error).casefold()
        return str(getattr(error, "failure_kind", "")) or (
            "max-turn" if "max turn" in message else "budget" if "budget" in message
            else "timeout" if "timed out" in message else "malformed-json" if "json" in message
            else "nonzero-exit" if "exit " in message else "execution-error"
        )

    def ordinary_replacement(error: BaseException, identity: Any = None) -> Dict[str, Any]:
        failure_kind = bridge_failure_kind(error)
        model = str(request.get("model") or "")
        tier = next((name for name, candidate in (cfg.get("models") or {}).items()
                     if str(candidate) == model), "ordinary")
        planned = {
            "tier": tier,
            "model": model or "unknown",
            "provider": str(kwargs.get("provider") or cfg.get("provider") or "unknown"),
            "source": "ordinary_provider.request_configuration",
        }
        identity_data = identity.as_dict() if hasattr(identity, "as_dict") else identity
        return {
            "policy": "legacy_ordinary_provider_route",
            "requested_review": {"tier": requested_alias, "transport": "claude_cli"},
            "tier": planned["tier"], "model": planned["model"], "provider": planned["provider"],
            "planned": planned,
            "reason": "attempted Claude CLI review failed; an ordinary provider replacement is planned, not observed",
            "failure_kind": failure_kind,
            **({"identity": identity_data} if isinstance(identity_data, dict) else {}),
        }

    def audit_refusal(message: str) -> None:
        turn_id = str(kwargs.get("parent_turn_id") or kwargs.get("turn_id") or "")
        session_id = str(kwargs.get("parent_session_id") or kwargs.get("session_id")
                         or turn_id.split(":", 1)[0])
        claude_delegation._log(cfg, {
            "event": "bridge_claude", "tier_requested": requested_alias,
            "tier_used": "none", "outcome": "refused", "message": message,
            "session_id": session_id, "turn_id": turn_id,
        })

    def admitted_alias() -> Optional[str]:
        """Check Claude only after a real bridge route has been established."""
        nonlocal adjustment, admission_refusal
        target = {"opus": "opus5", "sonnet": "sonnet5"}[requested_alias]
        if not _is_callable_tier(target, cfg):
            admission_refusal = f"Claude CLI tier {requested_alias} is disabled or cooling."
            audit_refusal(admission_refusal)
            return None
        if not usage_guard.guarded("anthropic", cfg):
            return requested_alias
        outcome = usage_guard.apply("anthropic", target, cfg, usage_guard.read("anthropic", cfg))
        if outcome.refused:
            admission_refusal = outcome.refused
            audit_refusal(admission_refusal)
            return None
        if not _is_callable_tier(outcome.tier, cfg):
            admission_refusal = f"Claude CLI substitution {outcome.tier} is disabled or cooling."
            audit_refusal(admission_refusal)
            return None
        adjustment = outcome.adjusted
        return {"opus5": "opus", "sonnet5": "sonnet"}.get(outcome.tier)

    def bridge_response(*, review: bool, **bridge_kwargs: Any) -> Any:
        """Return a bridge response and audit only a review that reached the bridge."""
        try:
            return _opus5_response(_run_opus5_bridge(review=review, **bridge_kwargs))
        except Exception as error:
            if review:
                message = str(error).strip()[:300] or type(error).__name__
                identity = bridge_kwargs.get("identity")
                selection_mode = str(getattr(identity, "selection_mode", ""))
                failure_kind = bridge_failure_kind(error)
                failure = {"message": message, "failure_kind": failure_kind,
                           "identity": getattr(error, "identity", None),
                           "selection_mode": selection_mode}
                # The bridge can fail before it attaches an identity (for example
                # process start, invalid repository or malformed response). Carry
                # the selection decision made before the attempt, not an optional
                # exception attribute, across the middleware boundary.
                setattr(error, "_bridge_selection_mode", selection_mode)
                setattr(error, "_bridge_review_failure", failure)
                audit_event = {
                    "event": "bridge_claude", "outcome": "error", "message": message,
                    "failure_kind": failure_kind,
                    "failure_class": str(getattr(error, "failure_class", "execution-error")),
                    "tier_requested": requested_alias, "tier_used": requested_alias,
                    "session_id": str(kwargs.get("session_id") or ""),
                    "turn_id": str(kwargs.get("turn_id") or ""),
                }
                if selection_mode != "exact":
                    replacement = ordinary_replacement(
                        error, getattr(error, "identity", None) or bridge_kwargs.get("identity"),
                    )
                    failure["replacement"] = replacement
                    audit_event["substitution"] = replacement
                claude_delegation._log(cfg, audit_event)
            raise

    # A delegated review leaf runs on Claude and returns its verdict as the
    # leaf's answer. Restricted to delegated workers: the documented hazard of
    # this bridge is that it captures the first call of a turn, which matters
    # only for the parent that still has to plan.
    if (
        str(kwargs.get("platform", "")).casefold() == "subagent"
        or ":sa-" in str(kwargs.get("turn_id", ""))
    ):
        delegated = _verified_delegated_claude_review(routing_text, cfg, request=source_request)
        policy = coding_cfg.get("delegated_review") or {}
        selection_mode = str(policy.get("selection_mode") or "profile_preferred")
        requested_model = str(policy.get("requested_model") or "").strip() or None
        if delegated is None and selection_mode == "exact" and review_model_alias(routing_text) is not None:
            try:
                _route, route_reason = _delegated_claude_review_status(
                    routing_text, cfg, request=source_request,
                )
            except Exception:
                route_reason = "the Claude CLI review route could not be established"
            from . import target_identity
            try:
                resolution = target_identity.resolve_target(
                    requested_alias, transport="claude_cli", selection_mode=selection_mode,
                    requested_model=requested_model, cfg=cfg,
                )
                identity = resolution.identity
            except Exception:
                # Exact mode is mandatory: an unresolvable identity still stops.
                identity = None
            refusal = "exact Claude CLI review route was refused before bridge selection: " + route_reason
            audit_refusal(refusal)
            from .claude_opus_bridge import ClaudeBridgeFailure
            raise ClaudeBridgeFailure(refusal, "capability", identity=identity, refused=True)
        if delegated is not None:
            repo, _requested_alias = delegated
            from . import target_identity
            resolution = target_identity.resolve_target(
                requested_alias, transport="claude_cli", selection_mode=selection_mode,
                requested_model=requested_model, cfg=cfg,
            )
            if resolution.status != target_identity.RESOLVED:
                refusal = "Claude CLI review route is unsupported: " + "; ".join(resolution.reasons)
                audit_refusal(refusal)
                from .claude_opus_bridge import ClaudeBridgeFailure
                raise ClaudeBridgeFailure(refusal, "capability", identity=resolution.identity, refused=True)
            identity = resolution.identity
            alias = admitted_alias()
            if alias is None:
                if identity.selection_mode == "exact":
                    refusal = admission_refusal or "exact Claude CLI review admission was refused"
                    from .claude_opus_bridge import ClaudeBridgeFailure
                    raise ClaudeBridgeFailure(refusal, "capability", identity=identity, refused=True)
                return None
            if identity.selection_mode == "exact" and alias != requested_alias:
                refusal = "exact Claude CLI review cannot substitute the requested tier"
                audit_refusal(refusal)
                from .claude_opus_bridge import ClaudeBridgeFailure
                raise ClaudeBridgeFailure(refusal, "capability", identity=identity, refused=True)
            allowed = (coding_cfg.get("delegated_review") or {}).get("models")
            if isinstance(allowed, list) and alias not in allowed:
                if identity.selection_mode == "exact":
                    refusal = f"exact Claude CLI review tier {alias} is not enabled by delegated_review.models"
                    audit_refusal(refusal)
                    from .claude_opus_bridge import ClaudeBridgeFailure
                    raise ClaudeBridgeFailure(refusal, "capability", identity=identity, refused=True)
                return None
            # From here on, ``resolved`` names the alias admission actually put on
            # argv (requested stays the original ask), so success, pre-launch
            # failure and replacement provenance all describe the real invocation.
            from .claude_opus_bridge import admitted_invocation_identity
            identity = admitted_invocation_identity(identity, alias)
            return bridge_response(
                review=True,
                repo=str(repo),
                task=text,
                write=False,
                model=alias,
                requested_alias=requested_alias,
                adjustment=adjustment,
                identity=identity,
                cfg=cfg,
                turn_id=str(kwargs.get("turn_id") or ""),
                parent_session_id=kwargs.get("parent_session_id") or kwargs.get("session_id"),
                parent_turn_id=kwargs.get("parent_turn_id"),
                provider=str(kwargs.get("provider") or ""),
            )

    if not coding_cfg.get("enabled"):
        return None
    review_repo = _verified_explicit_opus5_review_repo(routing_text, cfg)
    if review_repo is not None:
        alias = admitted_alias()
        if alias is None:
            return None
        return bridge_response(
            review=True,
            repo=str(review_repo),
            task=text,
            write=False,
            model=alias,
            requested_alias=requested_alias,
            adjustment=adjustment,
            cfg=cfg,
            turn_id=str(kwargs.get("turn_id") or ""),
            parent_session_id=kwargs.get("parent_session_id") or kwargs.get("session_id"),
            parent_turn_id=kwargs.get("parent_turn_id"),
            provider=str(kwargs.get("provider") or ""),
        )
    explicit_ui_repo = _verified_explicit_opus5_ui_repo(routing_text, cfg)
    if explicit_ui_repo is not None:
        alias = admitted_alias()
        if alias is None:
            return None
        result = _run_opus5_bridge(
            repo=str(explicit_ui_repo),
            task=f"[opus5] {text}",
            write=_opus5_write_intent(routing_text),
            model=alias,
            requested_alias=requested_alias,
            adjustment=adjustment,
            cfg=cfg,
            turn_id=str(kwargs.get("turn_id") or ""),
            parent_session_id=kwargs.get("parent_session_id") or kwargs.get("session_id"),
            parent_turn_id=kwargs.get("parent_turn_id"),
            provider=str(kwargs.get("provider") or ""),
        )
        return _opus5_response(result)

    decision = classify_request(source_request, api_call_count=1)
    if decision.tier != "terra":
        return None

    from .claude_opus_bridge import classify_coding_dispatch

    eligible, _reason = classify_coding_dispatch(text)
    if not eligible:
        return None
    repo = str(coding_cfg.get("default_repo") or "").strip()
    repo_path = Path(repo).expanduser() if repo else None
    if repo_path is None or not repo_path.is_dir():
        return None
    alias = admitted_alias()
    if alias is None:
        return None

    result = _run_opus5_bridge(
        repo=str(repo_path.resolve()),
        task=text,
        write=_opus5_write_intent(routing_text),
        model=alias,
        requested_alias=requested_alias,
        adjustment=adjustment,
        cfg=cfg,
        turn_id=str(kwargs.get("turn_id") or ""),
        parent_session_id=kwargs.get("parent_session_id") or kwargs.get("session_id"),
        parent_turn_id=kwargs.get("parent_turn_id"),
        api_request_id=str(kwargs.get("api_request_id") or ""),
        provider=str(kwargs.get("provider") or ""),
    )
    return _opus5_response(result)


def run_llm_with_transient_failover(**kwargs: Any) -> Any:
    """Retry one transient provider failure with a different routed model.

    This is execution middleware, so it runs around the actual provider call;
    request middleware alone cannot recover an exception after the request was
    sent. The original exception is preserved for auth, validation and other
    non-transient failures.
    """
    request = kwargs.get("request")
    next_call = kwargs.get("next_call")
    # Current Hermes exposes an LLM-only retry callback so providers can be
    # retried without violating the single-use downstream ``next_call`` contract.
    # Keep the direct callable fallback for unit-level and older-host compatibility.
    retry_call = kwargs.get("retry_call") or next_call
    cfg = _load_config()
    provider = str(kwargs.get("provider", "")).casefold()
    configured_provider = str(cfg.get("provider", "openai-codex")).casefold()
    worker = (cfg.get("enabled", True) and isinstance(request, dict)
              and (str(kwargs.get("platform", "")).casefold() == "subagent"
                   or ":sa-" in str(kwargs.get("turn_id", ""))))

    def stop_if_closed() -> Optional[Any]:
        if worker:
            refused = worker_admission.refusal(provider, str(request.get("model", "")), cfg)
            if refused:
                return worker_admission.stopped_response(refused, request.get("model", ""))
        return None

    if not isinstance(request, dict) or not callable(next_call) or provider != configured_provider:
        if not isinstance(request, dict) or not callable(next_call):
            return next_call(request)
        stopped = stop_if_closed()
        if stopped is not None:
            return stopped
        # The guard below exists because this middleware rewrites request["model"]
        # within one provider. Noticing that an account just refused a call needs
        # none of that, and skipping it here is why a Qwen weekly-quota 429 left
        # no cooldown -- the conductor was still being told that account was idle.
        try:
            response = next_call(request)
        except Exception as error:
            if _is_quota_exhaustion(error) or _is_transient_provider_failure(error):
                failing_model = str(request.get("model", ""))
                _record_tier_failure(
                    next(
                        (tier for tier, model in (cfg.get("models") or {}).items()
                         if model == failing_model),
                        "",
                    ) or (_external_target_for_model(failing_model) or ""),
                    cfg,
                    quota=_is_quota_exhaustion(error),
                    error=error,
                )
            raise
        # A turn whose review failed on the configured provider can end on the
        # host's own fallback provider (e.g. after a reasoning-only stall); its
        # final reply still carries the retained provenance.
        pending_key = _pending_replacement_key(kwargs)
        pending = _pending_replacement_get(pending_key)
        if pending:
            return _deliver_replacement_result(response, pending, pending_key, first_call=False)
        return response

    def stopped_exact_response(error: BaseException) -> Optional[Any]:
        """Translate mandatory-exact bridge failures at the host callback boundary."""
        identity = getattr(error, "identity", None)
        selection_mode = str(getattr(error, "_bridge_selection_mode", "")
                             or getattr(identity, "selection_mode", ""))
        if getattr(error, "refused", False) or selection_mode == "exact":
            message = str(getattr(error, "route_reason", "") or error).strip() or "exact Claude CLI review was refused"
            return worker_admission.stopped_response(
                "Exact Claude CLI review was not completed: " + message,
                str(request.get("model", "")),
            )
        return None

    opus_context = {key: value for key, value in kwargs.items() if key not in {"request", "next_call", "retry_call"}}
    replacement_provenance = None
    pending_key = _pending_replacement_key(kwargs)
    first_replacement_call = False
    try:
        opus_response = _maybe_run_opus5(request, cfg, **opus_context)
    except Exception as error:
        stopped = stopped_exact_response(error)
        if stopped is not None:
            return stopped
        # Failures before an attempt are not attributed to the CLI; attempted
        # reviews write a separate audit with the concrete reason.
        _logger.warning("Claude bridge did not complete; falling back to the normal route", exc_info=True)
        opus_response = None
        review_failure = getattr(error, "_bridge_review_failure", None)
        if review_failure:
            replacement_provenance = review_failure["replacement"]
            first_replacement_call = True
            request = deepcopy(request)
            _append_user_instruction(
                request,
                "Claude CLI review did not complete "
                f"({review_failure['failure_kind']}: {review_failure['message']}). "
                f"This response is an explicit legacy substitution attempt planned for {replacement_provenance['tier']} "
                f"({replacement_provenance['model']}); it is not the requested Claude review. "
                "Its actual provider/model, if observed in the response, is recorded as replacement provenance.",
            )
    if opus_response is not None:
        return opus_response
    if replacement_provenance is None:
        # A later call of a turn whose review failed earlier: the provenance
        # retained for this turn goes onto its final reply.
        replacement_provenance = _pending_replacement_get(pending_key)

    def delivered(response: Any) -> Any:
        if not replacement_provenance:
            return response
        return _deliver_replacement_result(response, replacement_provenance, pending_key,
                                           first_call=first_replacement_call)

    stopped = stop_if_closed()
    if stopped is not None:
        return stopped

    try:
        return delivered(next_call(request))
    except Exception as error:
        active_model = str(request.get("model", ""))
        quota_exhausted = _is_quota_exhaustion(error)
        # A model this account cannot use at all. Neither a quota (which recovers)
        # nor a blip (which is worth retrying): switched on in the dashboard but
        # refused by the provider, it would otherwise abort every leaf routed to it.
        unavailable = not quota_exhausted and _is_model_unavailable(error)
        active_tier = next(
            (tier for tier, model in (cfg.get("models") or {}).items() if model == active_model), "",
        )
        # Record before deciding what to do about it. The gap this closes is the
        # case with no fallback configured, where the old code re-raised and left
        # nothing behind -- the next call walked into the same exhausted account.
        if quota_exhausted or _is_transient_provider_failure(error):
            _record_tier_failure(cfg=cfg, tier=active_tier, quota=quota_exhausted, error=error)
        elif unavailable and active_tier:
            # Long, and stated plainly: nothing about this recovers by waiting, so
            # the point of the cooldown is to stop offering the tier this session.
            _enter_cooldown(
                active_tier, cfg,
                seconds=float((cfg.get("cooldown") or {}).get("unavailable_seconds", 21600) or 21600),
                reason="model unavailable on this account",
            )
        if quota_exhausted:
            fallback_model = _quota_fallback_model(active_model, cfg)
        elif unavailable:
            fallback_model = _durable_fallback_model(active_model, cfg, request)
        else:
            fallback_model = _transient_fallback_model(active_model, cfg, request)
        if not fallback_model or not (
            quota_exhausted or unavailable or _is_transient_provider_failure(error)
        ):
            raise
        if quota_exhausted:
            _remember_spark_quota_exhaustion(str(kwargs.get("turn_id") or ""))

        fallback_request = dict(request)
        fallback_request["model"] = fallback_model
        fallback_tier = next(
            (tier for tier, model in (cfg.get("models") or {}).items() if model == fallback_model),
            "fallback",
        )
        reasoning = dict(fallback_request.get("reasoning") or {})
        reasoning["effort"] = (
            _quota_fallback_effort(cfg, fallback_tier)
            if quota_exhausted
            else str((cfg.get("effort") or {}).get(fallback_tier) or "medium")
        )
        fallback_request["reasoning"] = reasoning
        _log_decision(
            RouteDecision(
                fallback_tier,
                fallback_model,
                (
                    f"Spark quota exhausted; failover from {active_model}"
                    if quota_exhausted
                    else f"{active_model} unavailable on this account; configured failover"
                    if unavailable
                    else f"transient failover from {active_model}"
                ),
            ),
            {**kwargs, "request": fallback_request},
            cfg,
        )
        retry_response = retry_call(fallback_request)
        return delivered(retry_response)


def on_post_llm_call(**kwargs: Any) -> None:
    """Persist supervisor checkpoints and the legacy disabled benchmark lifecycle."""
    # The turn has ended: provenance retained for it can no longer reach a reply.
    _pending_replacement_discard(_pending_replacement_key(kwargs))
    cfg = _load_config()
    turn_id = str(kwargs.get("turn_id") or "")
    orchestrated = _orchestration_forced_event(cfg, turn_id)
    if orchestrated:
        response = kwargs.get("assistant_response") or ""
        _orchestration_event(
            cfg,
            {
                "event": "terra_checkpoint_completed",
                "plan_id": orchestrated.get("plan_id"),
                "turn_id": turn_id,
                "parent_model": kwargs.get("model") or orchestrated.get("parent_model", "terra"),
                "supervisor_decision_present": "supervisor decision:" in str(response).casefold(),
                "response_sha256": _summary_digest(response),
            },
        )
    forced = _shadow_forced_event(cfg, turn_id)
    if not forced:
        return
    response = kwargs.get("assistant_response") or ""
    _shadow_event(
        cfg,
        {
            "event": "parent_completed",
            "benchmark_id": forced.get("benchmark_id") or _benchmark_id(turn_id),
            "turn_id": turn_id,
            "parent_model": kwargs.get("model") or forced.get("parent_model", "terra"),
            "parent_response_chars": len(str(response)),
            "parent_response_sha256": _summary_digest(response),
        },
    )


def on_subagent_start(**kwargs: Any) -> None:
    """Attach child lifecycle records to real supervisor work and legacy shadows."""
    cfg = _load_config()
    parent_turn_id = str(kwargs.get("parent_turn_id") or "")
    orchestrated = _orchestration_forced_event(cfg, parent_turn_id)
    if orchestrated:
        _orchestration_event(
            cfg,
            {
                "event": "spark_child_started",
                "plan_id": orchestrated.get("plan_id"),
                "turn_id": parent_turn_id,
                "child_session_id": kwargs.get("child_session_id"),
                "child_subagent_id": kwargs.get("child_subagent_id"),
                "child_goal_preview": _redacted_preview(kwargs.get("child_goal") or "", limit=240),
            },
        )
    forced = _shadow_forced_event(cfg, parent_turn_id)
    if not forced:
        return
    _shadow_event(
        cfg,
        {
            "event": "child_started",
            "benchmark_id": forced.get("benchmark_id") or _benchmark_id(parent_turn_id),
            "turn_id": parent_turn_id,
            "parent_model": forced.get("parent_model", "terra"),
            "shadow_model": forced.get("shadow_model", "spark"),
            "child_session_id": kwargs.get("child_session_id"),
            "child_subagent_id": kwargs.get("child_subagent_id"),
            "child_role": kwargs.get("child_role"),
            "child_goal_preview": _redacted_preview(kwargs.get("child_goal") or "", limit=240),
        },
    )


def on_subagent_stop(**kwargs: Any) -> None:
    """Persist real supervisor child outcomes and legacy benchmark outcomes."""
    cfg = _load_config()
    parent_turn_id = str(kwargs.get("parent_turn_id") or "")
    child_session_id = kwargs.get("child_session_id")
    summary = kwargs.get("child_summary") or ""
    orchestrated = _orchestration_forced_event(cfg, parent_turn_id)
    if orchestrated:
        _orchestration_event(
            cfg,
            {
                "event": "spark_child_completed",
                "plan_id": orchestrated.get("plan_id"),
                "turn_id": parent_turn_id,
                "child_session_id": child_session_id,
                "child_status": kwargs.get("child_status"),
                "child_model": kwargs.get("child_model"),
                "child_api_calls": kwargs.get("child_api_calls"),
                "summary_sha256": _summary_digest(summary),
            },
        )
    forced = _shadow_forced_event(cfg, parent_turn_id)
    if not forced:
        return
    _shadow_event(
        cfg,
        {
            "event": "child_completed",
            "benchmark_id": forced.get("benchmark_id") or _benchmark_id(parent_turn_id),
            "turn_id": parent_turn_id,
            "child_session_id": child_session_id,
            "child_status": kwargs.get("child_status"),
            "duration_ms": kwargs.get("duration_ms"),
            "child_model": kwargs.get("child_model"),
            "child_api_calls": kwargs.get("child_api_calls"),
            "input_tokens": kwargs.get("input_tokens"),
            "output_tokens": kwargs.get("output_tokens"),
            "cost_usd": kwargs.get("cost_usd"),
            "exit_reason": kwargs.get("exit_reason"),
            "summary_chars": len(str(summary)),
            "summary_sha256": _summary_digest(summary),
        },
    )


def _requested_delegation_targets(args: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[str, ...]:
    """Router tiers a ``delegate_task`` call names: the call-level ``model`` and each task's.

    A full model name (``qwen3.7-plus``) is mapped back to its tier. A name the
    router does not know is left to Hermes: this gate only enforces the router's
    own switches, not Hermes's target list.
    """
    by_model = {str(model): str(tier) for tier, model in (cfg.get("models") or {}).items()}
    known = set(cfg.get("callable") or {}) | set(cfg.get("models") or {})
    raw = [args.get("model")]
    tasks = args.get("tasks")
    if isinstance(tasks, list):
        raw += [task.get("model") for task in tasks if isinstance(task, dict)]
    names = []
    for value in raw:
        name = str(value or "").strip()
        name = by_model.get(name, name.casefold())
        if name in known and name not in names:
            names.append(name)
    return tuple(names)


def _hermes_worker_fallback_configured() -> bool:
    """Whether Hermes's ``delegation.fallback_providers`` names any route."""
    if yaml is None:
        return False
    try:
        raw = yaml.safe_load(_HERMES_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        chain = (raw.get("delegation") or {}).get("fallback_providers")
        return isinstance(chain, list) and bool(chain)
    except Exception:
        return False


_CONDUCTOR_CONTRACT = re.compile(r"^You are the \S+ planning conductor\.")


def _task_budget_path(cfg: Dict[str, Any]) -> Optional[Path]:
    configured = str((cfg.get("task_budget") or {}).get("path") or "").strip()
    return hermes_path(configured) if configured else None


def _bounded_low_risk_dispatch(args: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    """Whether this dispatch is small enough for the one-worker safety budget."""
    policy = cfg.get("task_budget") or {}
    if not policy.get("enabled"):
        return False
    tasks = args.get("tasks") if isinstance(args.get("tasks"), list) else [args]
    if len(tasks) != 1 or not isinstance(tasks[0], dict):
        return False
    goal = str(tasks[0].get("goal") or args.get("goal") or "").strip()
    if not goal or len(goal) > max(1, int(policy.get("low_risk_max_chars", 1200) or 1200)):
        return False
    affirmative = _without_negated_safety_constraints(goal)
    # "production" alone is not a risk signal: the pricing-copy incident was a
    # small copy fix on a production page, and exempting the word let it fan out
    # into a conductor plus two workers.  Consequential *actions* stay exempt.
    return not (
        _is_design_request(goal)
        or _is_consequential_spark_request(goal)
        or re.search(r"\b(deploy|migration|security|credential|password|payment|database|ssh|sudo)\b", _normalise(affirmative))
    )


def _bounded_dispatch_block(args: Dict[str, Any], cfg: Dict[str, Any], turn_id: str) -> str:
    """Persist and enforce a one-worker, one-depth budget for a small clear task."""
    if not _bounded_low_risk_dispatch(args, cfg):
        return ""
    policy = cfg.get("task_budget") or {}
    max_depth = max(1, int(policy.get("max_depth", 1) or 1))
    # The budget belongs to the user's root turn, not to each child turn: a
    # child worker's own spawns must draw on the same allowance.
    root_turn = str(turn_id).split(":sa-", 1)[0]
    tasks = args.get("tasks") if isinstance(args.get("tasks"), list) else [args]
    evidence = str((tasks[0] if tasks and isinstance(tasks[0], dict) else {}).get("escalation_evidence") or args.get("escalation_evidence") or "").strip()
    # Hermes encodes each child boundary as :sa-N:.  A low-risk leaf may not
    # create another child: that is the recursive escalation the budget exists to stop.
    if str(turn_id).count(":sa-") >= max_depth:
        return "Low-risk task depth budget reached; complete this bounded task directly instead of delegating again. Nothing was spawned."
    path = _task_budget_path(cfg)
    if path is None:
        return ""
    max_decisions = max(1, int(policy.get("max_routing_decisions", 1) or 1))
    try:
        with _SHADOW_LOCK:
            records = []
            if path.exists():
                records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            prior = [record for record in records if record.get("turn_id") == root_turn]
            allowed = max_decisions + (1 if evidence and policy.get("require_evidence_for_second_worker", True) else 0)
            if len(prior) >= allowed:
                return (
                    "Low-risk task routing budget reached: a second worker or cross-provider handoff "
                    "requires an `escalation_evidence` field naming a failed test or concrete blocker, "
                    "and this bounded task permits at most one such escalation. Nothing was spawned."
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    "event": "bounded_worker_admitted", "turn_id": root_turn,
                    "escalated": bool(evidence),
                    "max_routing_decisions": max_decisions, "max_depth": max_depth,
                }) + "\n")
    except Exception as exc:
        # Observability persistence cannot turn a safe direct task into an outage.
        _logger.warning("task budget state unavailable; leaving delegation unchanged: %s", exc)
    return ""


def _declined_conductor_goal(args: Dict[str, Any], cfg: Dict[str, Any], turn_id: str = "") -> str:
    """Refusal text when a forced conductor's composed goal is too small to plan.

    Whether a turn deserves a conductor is decided from the objective the parent
    wrote, not from the user's message: "csinald meg" after a long discussion is
    a large task, and the parent is the one that can expand it. Only the router's
    own forced planner call is measured -- recognised by its pinned contract --
    never a delegation the parent chose on its own.
    """
    policy = cfg.get("orchestration") or {}
    min_goal = max(0, int(policy.get("min_goal_chars", 500) or 0))
    if not min_goal or not policy.get("enabled"):
        return ""
    entries = args.get("tasks") if isinstance(args.get("tasks"), list) else [args]
    if len(entries) != 1 or not isinstance(entries[0], dict):
        return ""
    entry = entries[0]
    if not _CONDUCTOR_CONTRACT.match(str(entry.get("context") or args.get("context") or "")):
        return ""
    goal_chars = len(str(entry.get("goal") or "").strip())
    if goal_chars >= min_goal:
        return ""
    with _SHADOW_LOCK:
        _orchestration_event(cfg, {
            "event": "preflight_declined",
            "turn_id": turn_id,
            "goal_chars": goal_chars,
            "min_goal_chars": min_goal,
        })
    return (
        f"No conductor for this turn: the objective you wrote is {goal_chars} characters, "
        f"below orchestration.min_goal_chars ({min_goal}), so it is small enough to do directly. "
        "Continue with your normal tools; you may still delegate an independent part to a worker. "
        "Nothing was spawned."
    )


def on_pre_tool_call(tool_name: str = "", args: Optional[Dict[str, Any]] = None, **_: Any) -> Optional[Dict[str, str]]:
    """Apply the worker budget and refuse unavailable delegation targets.

    ``delegate_task`` offers every entry of Hermes's ``delegation.targets``, so a
    conductor can name ``model: "qwen"`` while the dashboard has Qwen off. The
    ``llm_request`` middleware does notice -- but Hermes logs a middleware error and
    sends the request unchanged, so the child still ran on the disabled account
    (live 2026-09-25: a Terra worker spawned a Qwen reconnaissance child that
    died on a 403). Blocking here stops the spawn before it happens and hands the
    agent a working route instead. Fails open: a broken check never blocks work.

    A target switched off is always refused: that is the operator's instruction.
    A target merely cooling down is refused only while the worker fallback chain
    is empty -- with one, Hermes can still move the child to another account,
    which the router's own failover (one provider only) cannot.
    """
    # ``delegate_claude`` is a worker spawn too.  Leaving it outside this hook
    # would let a small task evade its one-worker budget by changing provider.
    if tool_name not in {"delegate_task", claude_delegation.TOOL_NAME} or not isinstance(args, dict):
        return None
    try:
        cfg = _load_config()
        bounded = _bounded_dispatch_block(args, cfg, str(_.get("turn_id") or ""))
        if bounded:
            return {"action": "block", "message": bounded}
        declined = _declined_conductor_goal(args, cfg, str(_.get("turn_id") or ""))
        if declined:
            return {"action": "block", "message": declined}
        switched_on = cfg.get("callable") or {}
        rescued = None
        blocked = []
        # Claude's tool resolves its own tier; availability is enforced by its
        # implementation.  The generic target check is only meaningful for
        # delegate_task's model argument.
        if tool_name == claude_delegation.TOOL_NAME:
            return None
        for name in _requested_delegation_targets(args, cfg):
            if _is_callable_tier(name, cfg):
                continue
            if switched_on.get(name) is True:  # only cooling down
                if rescued is None:
                    rescued = _hermes_worker_fallback_configured()
                if rescued:
                    continue
            blocked.append(name)
        if not blocked:
            return None
        claude_ok = claude_delegation.is_active()
        parts = []
        for name in blocked:
            why = claude_delegation._unavailable(name, cfg) or "unavailable"
            options = []
            codex = claude_delegation.next_codex_route(name, cfg)
            if codex:
                options.append(f'model "{codex}"')
            if claude_ok:
                for peer in (name, *_peers_for(name, cfg)):
                    tier = claude_delegation.TIER_FOR_TARGET.get(peer)
                    if tier and _is_callable_tier(peer, cfg):
                        options.append(f'delegate_claude with tier "{tier}"')
                        break
            hint = f" Use {' or '.join(options)} instead." if options else ""
            parts.append(f'Delegation target "{name}" is {why}.{hint}')
        _logger.warning("model_router blocked delegate_task to %s", ", ".join(blocked))
        return {"action": "block", "message": " ".join(parts) + " Nothing was spawned."}
    except Exception as exc:  # pragma: no cover - defensive
        _logger.debug("delegate_task gate skipped: %s", exc)
        return None


def register(ctx: Any) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_middleware("llm_request", route_llm_request)
    ctx.register_middleware("llm_execution", run_llm_with_transient_failover)
    from .execution_adapters import guard_legacy_tool_execution
    ctx.register_middleware("tool_execution", guard_legacy_tool_execution)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("transform_llm_output", on_transform_llm_output)
    ctx.register_hook("subagent_start", on_subagent_start)
    ctx.register_hook("subagent_stop", on_subagent_stop)
    ctx.register_hook("pre_gateway_dispatch", on_pre_gateway_dispatch)
    claude_delegation.register(ctx)
