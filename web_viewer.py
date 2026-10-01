#!/usr/bin/env python3
"""Local browser dashboard for the Hermes model-router JSONL log."""

from __future__ import annotations

import argparse
import hashlib
import io
import tempfile
import threading
import traceback
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    import yaml
except ImportError:
    yaml = None

from agent_activity import load_agent_activity
from hermes_paths import hermes_path
from view_log import DEFAULT_LOG, load_entries

DEFAULT_STATE_DB = hermes_path("~/.hermes/state.db")
DEFAULT_AGENT_LOG = hermes_path("~/.hermes/logs/agent.log")
DEFAULT_BRIDGE_LIFECYCLE = hermes_path("~/.hermes/logs/claude-code-bridge.jsonl")
DEFAULT_ROOT_LIMIT = 10
RAW_HISTORY_LIMIT = 10000
# Beside this file, exactly like the router's own _CONFIG_PATH: `hermes plugins
# install` names the directory after the manifest (model-router), so a hardcoded
# ~/.hermes/plugins/model_router path read nothing on a normal install and every
# save raised.
CONFIG_PATH = Path(__file__).resolve().parent / "router_config.yaml"


HERMES_CONFIG_PATH = hermes_path("~/.hermes/config.yaml")
_CONFIG_LOCK = threading.RLock()


def _read_hermes_config() -> dict:
    """The Hermes config as a dict, or {} when unreadable."""
    if yaml is None or not HERMES_CONFIG_PATH.exists():
        return {}
    try:
        with open(HERMES_CONFIG_PATH, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _hermes_unreadable_message() -> str:
    """Why a save refuses to touch the Hermes config, with the parse error when there is one."""
    message = "The Hermes config could not be read; refusing to overwrite it"
    if yaml is None or not HERMES_CONFIG_PATH.exists():
        return message
    try:
        with open(HERMES_CONFIG_PATH, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except Exception as exc:
        detail = " ".join(str(exc).split())
        return f"{message}: {HERMES_CONFIG_PATH} is not valid YAML ({detail}). Fix it and reload settings."
    if loaded is not None and not isinstance(loaded, dict):
        return f"{message}: {HERMES_CONFIG_PATH} is not a YAML mapping."
    return message


class HermesConfigChanged(RuntimeError):
    """~/.hermes/config.yaml changed between a save's read and its write."""


def _hermes_stamp():
    """(mtime_ns, size) of the Hermes config, or None when it is missing."""
    try:
        stat = HERMES_CONFIG_PATH.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _read_hermes_snapshot():
    """(stamp, config) for one read-modify-write. The stamp is taken first, so a
    change that lands during the read makes the write refuse rather than slip by."""
    stamp = _hermes_stamp()
    return stamp, _read_hermes_config()


def _write_hermes_config(config: dict, stamp=None) -> None:
    """Write the Hermes config, keeping a timestamped copy of what was there.

    This file is not ours. It carries the user's providers, approvals and command
    allowlist, and a dashboard save that damaged it would be hard to reconstruct —
    so every write leaves a restore point beside it first. With ``stamp`` (from
    ``_read_hermes_snapshot``), a file that changed since that read is left alone.
    """
    import shutil
    from datetime import datetime

    if yaml is None:
        raise RuntimeError("yaml not available")
    if stamp is not None and _hermes_stamp() != stamp:
        raise HermesConfigChanged(
            "~/.hermes/config.yaml changed while this save was being prepared; nothing was written. "
            "Reload the page and save again."
        )
    if HERMES_CONFIG_PATH.exists():
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        shutil.copy2(HERMES_CONFIG_PATH, HERMES_CONFIG_PATH.with_name(f"config.yaml.bak-router-{stamp}"))
    _atomic_write(HERMES_CONFIG_PATH, yaml.dump(
        config, default_flow_style=False, allow_unicode=True, sort_keys=False).encode("utf-8"))


def _atomic_write(path: Path, content: bytes) -> None:
    """Readers see either the previous complete document or the new one."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            if path.exists():
                os.fchmod(handle.fileno(), path.stat().st_mode & 0o777)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _config_revision() -> str:
    digest = hashlib.sha256()
    for path in (CONFIG_PATH, _local_config_path(), HERMES_CONFIG_PATH):
        content = path.read_bytes() if path.exists() else None
        digest.update(repr((str(path), content)).encode("utf-8"))
    return digest.hexdigest()


def _hermes_chain(*path: str) -> list:
    """The fallback chain stored at ``path`` in the Hermes config, as a plain list."""
    node = _read_hermes_config()
    for key in path:
        node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            return []
    return [
        {"provider": str(e.get("provider") or ""), "model": str(e.get("model") or "")}
        for e in node if isinstance(e, dict) and e.get("provider") and e.get("model")
    ] if isinstance(node, list) else []


def _save_hermes_fallback(payload, router_cfg: dict, *, hermes=None, persist=True):
    """Persist the orchestrator and delegated-child chains; an error string, or None.

    Absent keys are left alone, so saving one chain never clears the other.

    The Settings page shows the main agent as ONE chain: the model Hermes starts
    on, then its fallbacks. The primary is therefore dropped from its own
    fallback list -- it used to sit there as a second entry, a step that can
    never help, because when the primary's provider is out so is that entry.
    """
    if not isinstance(payload, dict):
        return "hermes_fallback must be an object"
    options = _fallback_chain_options(router_cfg)
    stamp, config = _read_hermes_snapshot() if hermes is None else (None, hermes)
    if not config:
        return _hermes_unreadable_message()
    # Whatever is already saved is always accepted, even when it names a route the
    # picker no longer (or never did) offer -- a chain the operator already has
    # configured, elsewhere, must never be the reason a save 400s.
    already_saved = {
        "orchestrator": _hermes_chain("fallback_providers"),
        "children": _hermes_chain("delegation", "fallback_providers"),
    }
    chains = {}
    for key, label in (("orchestrator", "orchestrator"), ("children", "delegated children")):
        if key not in payload:
            continue
        allowed = options + [
            {"key": "", "provider": e["provider"], "model": e["model"]} for e in already_saved[key]
        ]
        chain, error = _clean_fallback_chain(payload[key], allowed)
        if error:
            return f"{label}: {error}"
        # Hermes fails over without consulting the router's switches: a route
        # switched off in the dashboard is dropped rather than kept as a
        # fallback Hermes would still try (Qwen answered those with a 403).
        off = {(o["provider"], o["model"]) for o in options
               if o.get("key") and (router_cfg.get("callable") or {}).get(o["key"]) is False}
        chain = [e for e in chain if (e["provider"], e["model"]) not in off]
        if key == "orchestrator":
            primary = _hermes_parent(config)
            chain = [e for e in chain
                     if (e["provider"], e["model"]) != (primary["provider"], primary["model"])]
        chains[key] = chain
    if not chains:
        return None
    if "orchestrator" in chains:
        config["fallback_providers"] = chains["orchestrator"]
    if "children" in chains:
        delegation = config.get("delegation")
        if not isinstance(delegation, dict):
            delegation = {}
            config["delegation"] = delegation
        delegation["fallback_providers"] = chains["children"]
    try:
        if persist:
            _write_hermes_config(config, stamp)
    except Exception as exc:
        return f"Could not write the Hermes config: {exc}"
    return None


def _sync_hermes_default_model(tier: str, config: dict, hermes: dict | None = None, stamp=None, *, persist=True) -> str | None:
    """Point Hermes's own ``model`` block at this router tier.

    A missing or unreadable Hermes config is skipped, not an error: there is no
    parent to move. Once the parent is known to follow this tier, a write that
    fails -- or a file that changed since ``hermes`` was read -- is returned as
    an error, so the save stores nothing rather than reporting a switch Hermes
    never got. ``_write_hermes_config`` leaves a restore point before touching a
    file that also carries providers, approvals and the command allowlist.
    """
    if yaml is None:
        return None
    if hermes is None:
        stamp, hermes = _read_hermes_snapshot()
    if not hermes:
        return None
    model_name = str((config.get("models") or {}).get(tier) or "")
    provider = _tier_provider(tier, config)
    if not model_name:
        return None
    model = hermes.get("model")
    if not isinstance(model, dict):
        model = {}
        hermes["model"] = model
    model["default"] = model_name
    model["provider"] = provider

    provider_config = (hermes.get("providers") or {}).get(provider) or {}
    transport = provider_config.get("transport", "")
    # xAI's OAuth route speaks the Codex Responses API; its base URL comes from
    # Hermes's own provider overlay, so nothing endpoint-specific is written.
    if transport == "anthropic_messages" or provider not in {"openai-codex", "xai-oauth"}:
        model["api_mode"] = "anthropic_messages" if transport == "anthropic_messages" else "chat_completions"
        base_url = provider_config.get("base_url") or provider_config.get("api")
        if base_url:
            model["base_url"] = base_url
        if provider_config.get("api_key"):
            model["api_key"] = provider_config["api_key"]
    else:
        model["api_mode"] = "codex_responses"
        model.pop("base_url", None)
        model.pop("api_key", None)

    # The delegation block is a separate child runtime and is left alone beyond
    # registering this tier as a reachable target.
    delegation = hermes.get("delegation")
    if not isinstance(delegation, dict):
        delegation = {}
        hermes["delegation"] = delegation
    targets = delegation.get("targets")
    if not isinstance(targets, dict):
        targets = {}
        delegation["targets"] = targets
    targets[tier] = {"provider": provider, "model": model_name}
    try:
        if persist:
            _write_hermes_config(hermes, stamp)
    except HermesConfigChanged as exc:
        return str(exc)
    except Exception as exc:
        return f"Could not move Hermes's parent to {model_name}: {exc}"
    return None


def _save_default_model(requested: str, config: dict, *, hermes=None, persist=True) -> str | None:
    """Store the default model, syncing Hermes only when it actually changed.

    ``default_model`` is the one router setting that also decides the model
    Hermes itself launches with, and the Settings page posts it on *every* save
    -- a callable toggle, a preference chain, a fallback edit. Writing it through
    unconditionally is what let an unrelated save move a Claude parent back onto
    this provider's tier, silently. So the Hermes write is gated on a real
    change; an error string means nothing was stored.
    """
    previous = str(config.get("default_model") or "")
    callable_tiers = config.get("callable") or {}
    fallbacks = config.get("fallbacks") or {}
    candidate = str(requested)
    visited = {candidate}
    while not callable_tiers.get(candidate, True):
        candidate = str(fallbacks.get(candidate) or "")
        if not candidate or candidate in visited:
            return f"No enabled fallback for default model '{requested}'"
        visited.add(candidate)
    # A delegation target is not a startable model. It has no entry in `models`,
    # so writing it through blanked Hermes's model.default -- and the router's own
    # _decision raises KeyError for a tier it cannot route.
    if not str((config.get("models") or {}).get(candidate) or ""):
        return (
            f"'{candidate}' is a delegation target, not a tier this router can start "
            f"Hermes on; reach it with a delegated worker instead"
        )
    # Only a parent this router serves itself follows its default tier. A parent on
    # another account (Claude) is set in Hermes's own config: writing the tier
    # through would silently drag the whole conversation onto this provider --
    # measured live when switching Qwen off moved default_model to terra and
    # replaced the Opus parent at the next Hermes start. One read serves both the
    # check and the write, so the file cannot change between them unnoticed.
    if candidate != previous:
        stamp, hermes = _read_hermes_snapshot() if hermes is None else (None, hermes)
        if _parent_is_router_model(config, hermes):
            error = _sync_hermes_default_model(candidate, config, hermes, stamp, persist=persist)
            if error:
                return error
    config["default_model"] = candidate
    return None


def _claude_parent_options(config: dict) -> list:
    """Claude models Hermes itself may start on: {key, provider, model} per switched-on tier.

    A Claude tier is no router tier (no ``models`` entry), so it can never be the
    router's ``default_model``; it can still be the model Hermes launches with, on
    Hermes's own ``anthropic`` provider. A switched-off tier is not offered.
    """
    tiers = (config.get("claude_delegation") or {}).get("tiers") or {}
    callable_tiers = config.get("callable") or {}
    options = []
    for tier, target in _CLAUDE_TARGET_FOR_TIER.items():
        model = str(tiers.get(tier) or "")
        if model and callable_tiers.get(target) is True:
            options.append({"key": target, "provider": "anthropic", "model": model})
    return options


def _save_main_parent(key, config: dict, *, hermes=None, persist=True) -> str | None:
    """Move the model Hermes starts on to ``key``: a router tier or a Claude target.

    Posted only when the operator picks a parent, never on unrelated saves, so it
    may move a parent that is on another account (``_save_default_model`` never
    does). A router tier also becomes ``default_model``; a Claude parent leaves the
    router's default route and conductor where they are. An error string, or None.
    """
    key = str(key or "")
    stamp, hermes = _read_hermes_snapshot() if hermes is None else (None, hermes)
    if not hermes:
        return _hermes_unreadable_message()
    if (config.get("callable") or {}).get(key) is False:
        return f"'{key}' is switched off; switch it on before starting Hermes on it"
    claude = next((o for o in _claude_parent_options(config) if o["key"] == key), None)
    if claude is None:
        if not str((config.get("models") or {}).get(key) or ""):
            return f"'{key}' is not a model Hermes can start on"
        if _hermes_parent(hermes) != {"provider": _tier_provider(key, config),
                                      "model": str(config["models"][key])}:
            error = _sync_hermes_default_model(key, config, hermes, stamp, persist=persist)
            if error:
                return error
        config["default_model"] = key
        return None
    if _hermes_parent(hermes) == {"provider": claude["provider"], "model": claude["model"]}:
        return None
    model = hermes.get("model")
    if not isinstance(model, dict):
        model = {}
        hermes["model"] = model
    model["default"], model["provider"] = claude["model"], claude["provider"]
    # What Hermes's own `hermes model` anthropic flow writes: the anthropic runtime
    # fixes its endpoint itself, and a Codex api_mode or a stale base URL / key left
    # behind would send the Claude parent down the wrong transport.
    for stale in ("api_mode", "base_url", "api_key"):
        model.pop(stale, None)
    if persist:
        try:
            _write_hermes_config(hermes, stamp)
        except HermesConfigChanged as exc:
            return str(exc)
        except Exception as exc:
            return f"Could not move Hermes's parent to {claude['model']}: {exc}"
    return None


def _tier_provider(tier: str, config: dict) -> str:
    return str((config.get("tier_providers") or {}).get(tier) or "openai-codex")


def _hermes_parent(hermes: dict) -> dict:
    """The model Hermes itself starts on, as {provider, model}."""
    model = hermes.get("model") if isinstance(hermes, dict) else None
    model = model if isinstance(model, dict) else {}
    return {"provider": str(model.get("provider") or ""), "model": str(model.get("default") or "")}


def _router_tier_for(provider: str, model: str, config: dict) -> str:
    """The router tier serving exactly this provider/model, or ''."""
    for tier, name in (config.get("models") or {}).items():
        if name == model and (not provider or _tier_provider(tier, config) == provider):
            return str(tier)
    return ""


def _worker_model_options(config: dict) -> list:
    """Tiers a ``delegate_task`` worker may default to: the Codex account's own.

    The Hermes delegation block carries only provider and model -- no base URL or
    key -- so only the provider Hermes already authenticates natively is offered.
    """
    return [str(t) for t in (config.get("models") or {})
            if _tier_provider(str(t), config) == "openai-codex"]


def _worker_model_status(config: dict, hermes: dict | None = None) -> dict:
    hermes = hermes if hermes is not None else _read_hermes_config()
    delegation = hermes.get("delegation") if isinstance(hermes, dict) else None
    delegation = delegation if isinstance(delegation, dict) else {}
    provider, model = str(delegation.get("provider") or ""), str(delegation.get("model") or "")
    return {"provider": provider, "model": model, "tier": _router_tier_for(provider, model, config),
            "options": _worker_model_options(config)}


def _save_worker_model(tier, config: dict, *, hermes=None, persist=True):
    """Point Hermes's ``delegation`` block (the ``delegate_task`` default) at a tier.

    It used to be invisible on the page, so switching the default model left every
    Codex worker on the old one. An error string, or None.
    """
    tier = str(tier or "")
    if tier not in _worker_model_options(config):
        return f"'{tier}' cannot be the delegate_task worker model"
    stamp, hermes = _read_hermes_snapshot() if hermes is None else (None, hermes)
    if not hermes:
        return _hermes_unreadable_message()
    delegation = hermes.get("delegation")
    if not isinstance(delegation, dict):
        delegation = {}
        hermes["delegation"] = delegation
    model = str((config.get("models") or {}).get(tier) or "")
    provider = _tier_provider(tier, config)
    if (delegation.get("provider"), delegation.get("model")) == (provider, model):
        return None
    delegation["provider"], delegation["model"] = provider, model
    if persist:
        try:
            _write_hermes_config(hermes, stamp)
        except Exception as exc:
            return f"Could not write the Hermes config: {exc}"
    return None


def _parent_is_router_model(config: dict, hermes: dict | None = None) -> bool:
    """Whether Hermes currently starts on one of this router's own tiers.

    Name and provider both have to match a tier: the same model name served by a
    different provider is someone else's parent. A model block with no provider
    is judged by name, as before.
    """
    model = (hermes if hermes is not None else _read_hermes_config()).get("model")
    if not isinstance(model, dict):
        return False
    parent = str(model.get("default") or "")
    provider = str(model.get("provider") or "")
    return bool(parent) and any(
        name == parent and (not provider or _tier_provider(tier, config) == provider)
        for tier, name in (config.get("models") or {}).items()
    )


# Router target names stay what the config, cooldowns and dashboard already use;
# claude_delegation.py speaks in short tier names ("haiku"/"sonnet"/"opus"). Mirrored
# here rather than imported so this file keeps working as a standalone script.
_CLAUDE_TARGET_FOR_TIER = {"haiku": "haiku", "sonnet": "sonnet5", "opus": "opus5"}


def _fallback_chain_options(router_cfg: dict) -> list:
    """Every route a fallback entry may name: this router's own models, its Claude
    delegation tiers, and Hermes's own ``delegation.targets``.

    Restricting this to Hermes delegation targets alone rejected a chain that named
    a route this installation plainly has -- the router's own account, or a Claude
    tier -- because it was never registered as a Hermes delegation target. Offering
    every route the installation actually has, from every source that knows about
    one, is what makes the picker (and a save) match what is really configured.
    """
    options = []
    seen = set()

    def add(key: str, provider: str, model: str) -> None:
        provider, model = provider.strip(), model.strip()
        if not provider or not model or (provider, model) in seen:
            return
        seen.add((provider, model))
        options.append({"key": key, "provider": provider, "model": model})

    tier_providers = router_cfg.get("tier_providers") or {}
    for tier, model in (router_cfg.get("models") or {}).items():
        provider = tier_providers.get(tier)
        if provider:
            add(str(tier), str(provider), str(model))

    for tier, model in ((router_cfg.get("claude_delegation") or {}).get("tiers") or {}).items():
        target = _CLAUDE_TARGET_FOR_TIER.get(str(tier))
        if target and model:
            add(target, "anthropic", str(model))

    targets = ((_read_hermes_config().get("delegation") or {}).get("targets") or {})
    for name, spec in targets.items():
        if not isinstance(spec, dict):
            continue
        add(str(name), str(spec.get("provider") or ""), str(spec.get("model") or ""))

    return sorted(options, key=lambda o: o["key"])


def _clean_fallback_chain(raw, options: list):
    """``(chain, None)`` for a valid list of {provider, model}, ``(None, error)`` otherwise.

    An empty list is preserved rather than dropped: for a delegated child ``[]`` means
    "no fallback", which is a different instruction from "inherit the parent's".
    """
    if raw is None:
        return None, None
    if not isinstance(raw, list):
        return None, "A fallback chain must be a list of {provider, model} entries"
    allowed = {(o["provider"], o["model"]) for o in options}
    chain = []
    for entry in raw:
        if not isinstance(entry, dict):
            return None, f"Fallback entry must be an object, got {type(entry).__name__}"
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            return None, "Each fallback entry needs a provider and a model"
        if allowed and (provider, model) not in allowed:
            return None, f"Unknown route '{provider}/{model}' — not a configured delegation target"
        pair = {"provider": provider, "model": model}
        if pair not in chain:
            chain.append(pair)
    return chain, None


def _router_module():
    """Import the router package from this standalone script, or None.

    The dashboard runs as a plain file with its own directory as cwd, so the
    package only becomes importable once its PARENT is on sys.path -- but that
    only works when the plugin directory is literally named ``model_router``.
    The repo is ``hermes-model-router`` and the installed plugin directory is
    ``model-router`` (hyphenated, after the manifest name), so a normal launch
    (``~/.hermes/hermes-agent/venv/bin/python
    ~/.hermes/plugins/model-router/web_viewer.py``) never has a ``model_router``
    package to find that way, and the first attempt returns None. The fallback
    loads the package from this script's own directory under the name
    ``model_router`` regardless of what the directory is actually called.
    """
    try:
        import sys

        parent = str(Path(__file__).resolve().parent.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        import model_router

        return model_router
    except Exception:
        pass
    try:
        import importlib.util
        import sys

        here = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location(
            "model_router", here / "__init__.py", submodule_search_locations=[str(here)]
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules["model_router"] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        sys.modules.pop("model_router", None)
        return None


def _known_tier_names(config: dict) -> set:
    """Tier names a preference may legally contain: routable models plus delegation targets."""
    names = set(config.get("models") or {})
    names |= set(config.get("callable") or {})
    router = _router_module()
    if router is not None:
        try:
            names |= set(router._delegation_target_names())
        except Exception:
            pass
    return {str(n).strip().casefold() for n in names if str(n).strip()}


def _clean_preferences(raw, config: dict):
    """``(cleaned, None)`` for a valid preferences map, ``(None, error)`` otherwise.

    Rejects rather than silently drops: a typo'd tier that vanished on save would
    look like the setting simply did not work.
    """
    router = _router_module()
    WORK_KINDS = tuple(getattr(router, "WORK_KINDS", ())) if router else ()
    if not isinstance(raw, dict):
        return None, "preferences must be an object of kind -> ordered list"
    known = _known_tier_names(config)
    cleaned = {}
    for kind, value in raw.items():
        key = str(kind).strip().casefold()
        if WORK_KINDS and key not in WORK_KINDS:
            return None, f"Unknown work kind '{kind}'"
        if not isinstance(value, list):
            return None, f"Preference for '{key}' must be a list"
        order = []
        for item in value:
            name = str(item or "").strip().casefold()
            if not name:
                continue
            if name not in known:
                return None, f"Unknown model '{item}' in '{key}'"
            if name not in order:
                order.append(name)
        # An empty list means "no preference": store nothing rather than a shape
        # the router would have to special-case.
        if order:
            cleaned[key] = order
    return cleaned, None


def _router_status() -> dict:
    """Cooldown state and per-account load, read through the router's own code.

    Imported lazily and defensively: the dashboard is a standalone script and
    must keep serving the log even if the router package cannot be imported.
    Recomputing either of these here instead would let the panel and the routing
    decision disagree about which tiers are available.
    """
    empty = {"cooldowns": {}, "load": {}, "window_minutes": 0, "routable": []}
    router = _router_module()
    if router is None:
        return empty
    try:
        _load_config = router._load_config
        _read_cooldown_state = router._read_cooldown_state
        _recent_account_load = router._recent_account_load
        _tier_cooldown_remaining = router._tier_cooldown_remaining
    except AttributeError:
        return empty
    try:
        config = _load_config()
        window = int((config.get("usage_report") or {}).get("window_seconds") or 3600)
        recorded = _read_cooldown_state(config).get("tiers") or {}
        cooldowns = {}
        # Every tier the router knows, from the router's own config rather than a
        # list repeated here: sonnet5 was missing from the hardcoded tuple, so a
        # cooling Sonnet reported no cooldown at all and the dashboard showed the
        # account as merely idle. The `routable` field below already learned this
        # lesson; the loop two lines up had not.
        tracked = sorted(set(config.get("models") or {}) | set(config.get("callable") or {}))
        for tier in tracked:
            remaining = _tier_cooldown_remaining(tier, config)
            if remaining > 0:
                cooldowns[tier] = {
                    "seconds": int(remaining),
                    "reason": str((recorded.get(tier) or {}).get("reason") or ""),
                }
        return {
            "cooldowns": cooldowns,
            "load": _recent_account_load(config, window),
            "window_minutes": max(1, window // 60),
            # Which tiers can actually hold the orchestrator role. A tier the
            # router has no model entry for cannot: picking it raises a KeyError
            # on the first routing decision. Served rather than hardcoded so the
            # dropdown cannot drift from the router's own tier map again.
            "routable": sorted(config.get("models") or {}),
        }
    except Exception:
        return empty


def _tier_accounts(config: dict) -> dict:
    return {str(t): str(a) for t, a in (config.get("tier_providers") or {}).items()}


def _delegation_log_path(config: dict):
    configured = str((config.get("claude_delegation") or {}).get("log_path") or "").strip()
    return hermes_path(configured) if configured else None


_DELEGATION_LOG_TAIL_BYTES = 1_000_000


def _read_delegation_log(config: dict, limit: int = 500):
    """(audit lines, latest registration line) from claude-delegation.jsonl; malformed lines skipped.

    Only the tail of the log is read: it is never rotated, so a poll every few
    seconds that loaded the whole file would get slower forever. Same pattern
    as the router's own ``_recent_account_load`` in ``__init__.py``.
    """
    path = _delegation_log_path(config)
    audits, registration = [], None
    if path is None or not path.exists():
        return audits, registration
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - _DELEGATION_LOG_TAIL_BYTES))
            if size > _DELEGATION_LOG_TAIL_BYTES:
                handle.readline()  # discard the partial line the seek landed in
            lines = handle.readlines()
    except OSError:
        return audits, registration
    # The registration line is looked for across the WHOLE tail (already
    # bounded by _DELEGATION_LOG_TAIL_BYTES above), independent of `limit`:
    # windowing it down further to the last `limit*2` lines could drop a
    # registration that was still well within what got read, just older than
    # that second, audit-sized window.
    for raw in lines:
        try:
            entry = json.loads(raw.decode("utf-8"))
        except Exception:
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("event") == "registration":
            registration = entry
        elif entry.get("event") in {"delegate_claude", "bridge_claude"}:
            audits.append(entry)
    return audits[-limit:], registration


def _accounts_status(config: dict) -> dict:
    """One object per account that has a callable tier, in the same shape for every account."""
    router = _router_module()
    guard = getattr(router, "usage_guard", None) if router else None
    delegation = getattr(router, "claude_delegation", None) if router else None
    # Every tier with a `callable` entry, on or off: a switched-off tier (the
    # shipped config ships spark: false) still needs its switch so it can be
    # turned back on, not just the ones currently enabled.
    known_tiers = config.get("callable") or {}
    tiers_by_account: dict = {}
    for tier, account in _tier_accounts(config).items():
        if tier in known_tiers:
            tiers_by_account.setdefault(account, []).append(tier)
    _audits, registration = _read_delegation_log(config)
    # M10: served so the dashboard can grey out a stale reading at 2x this,
    # instead of assuming a fixed staleness window the guard doesn't use.
    cache_seconds = float(guard.guard_config(config).get("cache_seconds", 300)) if guard else 300.0
    now = time.time()
    accounts = {}
    for account, tiers in sorted(tiers_by_account.items()):
        limits = guard.account_limits(account, config) if guard else None
        reading = guard.cached(account, config) if guard else None
        state = guard.state(account, config, reading) if guard else "unknown"
        label = guard.account_label(account) if guard else account
        item = {
            "label": label,
            "tiers": sorted(tiers),
            "state": state,
            "usage": None if reading is None else {
                "weekly": reading.weekly, "session": reading.session,
                "weekly_resets_at": reading.weekly_resets_at, "session_resets_at": reading.session_resets_at,
            },
            "usage_age_seconds": None if reading is None else max(0, int(now - reading.fetched_at)),
            "has_usage_source": bool(guard and guard.has_fetcher(account)),
            "guard": limits is not None,
            "soft_percent": limits["soft_percent"] if limits else None,
            "hard_percent": limits["hard_percent"] if limits else None,
            "step_down": limits["step_down"] if limits else {},
            "cache_seconds": cache_seconds,
        }
        if account == "anthropic":
            block = config.get("claude_delegation") or {}
            # delegate_claude is offered exactly while a Claude model is switched on.
            if delegation is not None and hasattr(delegation, "availability_block"):
                enabled = delegation.availability_block(config) == ""
            else:
                enabled = any(known_tiers.get(tier) is True for tier in tiers)
            registered = bool(registration and registration.get("registered"))
            item["delegation"] = {
                "tool": "delegate_claude", "enabled": enabled, "registered": registered,
                # The tool is registered whatever the switches say, and offered per session
                # by its check_fn, so only an unregistered tool still needs a restart.
                "restart_needed": enabled and not registered,
                "default_tier": str(block.get("default_tier") or "sonnet"),
                "tiers": list(getattr(delegation, "TIERS", ("haiku", "sonnet", "opus"))),
            }
        else:
            item["delegation"] = {"tool": "delegate_task", "always_on": True}
        accounts[account] = item
    return accounts


def _save_usage_limits(raw, config: dict):
    if not isinstance(raw, dict):
        return "usage_limits must be an object of account -> {soft_percent, hard_percent}"
    if not raw:
        # Nothing to save: leave an absent usage_guard block absent rather than
        # setdefault()-ing an empty one into existence on every settings save.
        return None
    accounts = ((config.setdefault("usage_guard", {})).setdefault("accounts", {}))
    for account, values in raw.items():
        if account not in accounts:
            return f"No usage guard is configured for account '{account}'"
        try:
            soft, hard = float(values["soft_percent"]), float(values["hard_percent"])
        except Exception:
            return f"Limits for '{account}' need numeric soft_percent and hard_percent"
        if not (0 < soft < hard <= 100):
            return f"Limits for '{account}' must satisfy 0 < soft < hard <= 100"
        # 80, not 80.0: a whole percentage is written the way the file spells it.
        accounts[account]["soft_percent"], accounts[account]["hard_percent"] = (
            int(v) if v.is_integer() else v for v in (soft, hard))
    return None


MANAGED_EFFORT_TIERS = ("luna", "spark", "terra", "sol", "grok")
EFFORT_LEVELS = ("low", "medium", "high", "xhigh")
_DEFAULT_MANAGED_EFFORT = {"luna": "low", "spark": "low", "terra": "medium", "sol": "high", "grok": "medium"}


def _effective_managed_effort_defaults() -> dict:
    """The effective default effort level for every managed tier: the router's own
    shipped defaults layered over ``_DEFAULT_MANAGED_EFFORT`` (the fallback used
    when the router module itself is not importable)."""
    router = _router_module()
    router_defaults = getattr(router, "_DEFAULT_CONFIG", {}) if router is not None else {}
    effective_defaults = dict(_DEFAULT_MANAGED_EFFORT)
    if isinstance(router_defaults, dict) and isinstance(router_defaults.get("effort"), dict):
        effective_defaults.update(router_defaults["effort"])
    return effective_defaults


def _effort_status(config: dict) -> dict:
    """GET's ``effort`` payload: a string for every managed tier, never ``null``.

    A managed tier absent from the merged config (e.g. an older shipped file
    without ``effort.grok``) falls back to its effective default -- the router's
    own shipped default when the router is importable, else ``_DEFAULT_MANAGED_EFFORT``
    (which now covers every managed tier) -- so this never reports ``None``.
    """
    existing = config.get("effort") or {}
    defaults = _effective_managed_effort_defaults()
    return {
        tier: existing.get(tier, defaults.get(tier))
        for tier in MANAGED_EFFORT_TIERS
    }


def _save_effort(raw, config: dict):
    """Update only dashboard-managed routed-tier effort levels from Settings.

    Every submitted key and value is checked before ``config`` changes, so a
    partial payload cannot leave an earlier tier changed after a later one fails.
    A ``None`` value for a managed tier is ignored rather than rejected, as a
    belt-and-braces guard: GET (``_effort_status``) never reports ``None`` for a
    managed tier, but the frontend still posts the whole visible ``effort`` block
    on every save, so a value must round-trip without erroring the entire payload.
    """
    if not isinstance(raw, dict):
        return "effort must be an object of tier -> low, medium, high, or xhigh"
    cleaned = {}
    for tier, value in raw.items():
        if tier not in MANAGED_EFFORT_TIERS:
            return f"The dashboard cannot edit effort key '{tier}'"
        if value is None:
            continue
        if not isinstance(value, str):
            return f"effort.{tier} must be one of: low, medium, high, xhigh"
        normalized = value.strip().casefold()
        if normalized not in EFFORT_LEVELS:
            return f"effort.{tier} must be one of: low, medium, high, xhigh"
        cleaned[tier] = normalized
    if not cleaned:
        # Leave an absent effort block absent when a client posts no changes.
        return None
    existing_effort = config.get("effort")
    existing_effort = existing_effort if isinstance(existing_effort, dict) else {}
    effective_defaults = _effective_managed_effort_defaults()
    to_write = {}
    for tier, value in cleaned.items():
        if tier not in existing_effort and value == effective_defaults.get(tier):
            # The frontend posts every visible level. Do not pin an effective
            # default that an older shipped config did not explicitly contain.
            continue
        to_write[tier] = value
    if not to_write:
        return None
    effort = config.get("effort")
    if not isinstance(effort, dict):
        effort = config["effort"] = {}
    for tier, value in to_write.items():
        effort[tier] = value
    return None


def _assign_in_place(container: dict, key: str, value) -> None:
    """``container[key] = value``, but an existing mapping or list is updated in place.

    The page posts whole blocks (``callable``, ``preferences``) on every save.
    Replacing a round-trip-loaded block with a plain dict drops every comment
    attached to it -- including the one above the next key, which the loader
    attaches to this block's last entry -- and turns ``[a, b]`` into a block list.
    """
    current = container.get(key)
    if isinstance(current, dict) and isinstance(value, dict):
        for stale in [k for k in current if k not in value]:
            del current[stale]
        for k, v in value.items():
            _assign_in_place(current, k, v)
    elif isinstance(current, list) and isinstance(value, list):
        del current[:]
        current.extend(value)
    else:
        container[key] = value


def _balance_status(config: dict) -> dict:
    """``usage_guard.balance`` with the guard's own defaults filled in."""
    router = _router_module()
    guard = getattr(router, "usage_guard", None) if router else None
    if guard is not None:
        settings = guard.balance_config(config)
        return {key: settings[key] for key in ("enabled", "busy_percent", "margin_percent", "window")}
    raw = ((config.get("usage_guard") or {}).get("balance") or {})
    return {"enabled": raw.get("enabled") is True, "busy_percent": float(raw.get("busy_percent", 20)),
            "margin_percent": float(raw.get("margin_percent", 10)), "window": str(raw.get("window") or "5-hour")}


def _save_balance(raw, config: dict):
    """Switch load balancing on or off and, when given, set its two thresholds.

    Validated before anything is written, so a bad threshold leaves the block as it was.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("enabled"), bool):
        return "balance.enabled must be true or false"
    thresholds = {}
    for key, low in (("busy_percent", 0.0), ("margin_percent", 1.0)):
        if key not in raw:
            continue
        try:
            value = float(raw[key])
        except (TypeError, ValueError):
            return f"balance.{key} must be a number"
        if not (low <= value <= 100):
            return f"balance.{key} must be between {low:.0f} and 100"
        thresholds[key] = int(value) if value.is_integer() else value
    guard = config.get("usage_guard")
    if not isinstance(guard, dict):
        guard = config["usage_guard"] = {}
    block = guard.get("balance")
    if not isinstance(block, dict):
        block = guard["balance"] = {}
    block["enabled"] = raw["enabled"]
    block.update(thresholds)
    return None


LOCAL_CONFIG_HEADER = (
    "# Your own router settings, layered over router_config.yaml. Git-ignored:\n"
    "# the dashboard saves here, keeping only what differs from the shipped file.\n"
)


def _local_config_path() -> Path:
    """The operator's own settings, beside the shipped router_config.yaml (the router's rule too)."""
    return CONFIG_PATH.with_name("router_config.local.yaml")


def _merge(base: dict, override: dict) -> dict:
    """``override`` over a deep copy of ``base``, mapping by mapping -- the router's layering."""
    import copy

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _overlay(base: dict, target: dict) -> dict:
    """The smallest mapping that, merged over ``base``, gives ``target``.

    A key ``target`` drops cannot be expressed as an override; the dashboard
    never drops one the shipped file sets.
    """
    out = {}
    for key, value in target.items():
        current = base.get(key) if isinstance(base, dict) else None
        if isinstance(value, dict) and isinstance(current, dict):
            nested = _overlay(current, value)
            if nested:
                out[key] = nested
        elif not (isinstance(base, dict) and key in base) or current != value:
            out[key] = value
    return out


def _load_yaml_mapping(path: Path) -> dict:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


CLAUDE_SWITCHES = ("opus5", "sonnet5", "haiku")


def _has_legacy_claude_keys(local: dict) -> bool:
    """Whether router_config.local.yaml still carries ``workflow`` or ``claude_delegation.enabled``.

    Both are retired: the router turns them into Claude ``callable`` switches at
    load time, and the dashboard's next save writes those switches instead.
    """
    block = local.get("claude_delegation") if isinstance(local, dict) else None
    return isinstance(local, dict) and ("workflow" in local or (isinstance(block, dict) and "enabled" in block))


def _effective_router_config(shipped: dict, local: dict) -> dict:
    """``local`` over ``shipped``, with the retired keys translated exactly as the router does.

    The translation is the router's own ``_apply_legacy_claude_switches``, not a
    copy of it. When the local file carries a legacy key and the router cannot be
    imported, this raises rather than showing Claude switched off while the
    router treats it as on (and a save then writing that off state).
    """
    merged = _merge(shipped, local)
    if not _has_legacy_claude_keys(local):
        return merged
    router = _router_module()
    translate = getattr(router, "_apply_legacy_claude_switches", None) if router is not None else None
    if translate is None:
        raise RuntimeError(
            "The router package could not be imported in this dashboard process, so the retired "
            "'workflow' / 'claude_delegation.enabled' keys in router_config.local.yaml cannot be "
            "translated into Claude switches. Settings are not shown or saved until it can be.")
    return translate(merged, local)


def _read_router_config() -> dict:
    """The shipped router_config.yaml with router_config.local.yaml over it, as the router reads it."""
    shipped = _load_yaml_mapping(CONFIG_PATH)
    try:
        local = _load_yaml_mapping(_local_config_path())
    except Exception:
        local = {}
    return _effective_router_config(shipped, local)


def _read_router_config_for_update(path: Path | None = None):
    """(config, dump) for a read-modify-write that keeps the file's comments and layout.

    ruamel.yaml round-trips comments; PyYAML is the fallback and strips them. A
    missing file reads as an empty mapping.
    """
    path = CONFIG_PATH if path is None else path
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    try:
        from ruamel.yaml import YAML
    except ImportError:
        YAML = None
    if YAML is not None:
        round_trip = YAML()
        round_trip.preserve_quotes = True
        round_trip.width = 4096
        round_trip.indent(mapping=2, sequence=2, offset=0)
        config = round_trip.load(text) or {}

        def dump(data, handle):
            round_trip.dump(data, handle)
        return config, dump

    def dump(data, handle):
        yaml.dump(data, handle, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return yaml.safe_load(text) or {}, dump


def _save_claude_delegation(raw, config: dict):
    """Set ``claude_delegation.default_tier``.

    ``enabled`` is retired (the Claude switches decide), so a payload from an
    old open tab that still carries it is accepted and the key is ignored.
    """
    if not isinstance(raw, dict):
        return "claude_delegation must be an object"
    if "default_tier" in raw:
        tier = str(raw["default_tier"])
        if tier not in ("haiku", "sonnet", "opus"):
            return f"Unknown Claude tier '{tier}'"
        config.setdefault("claude_delegation", {})["default_tier"] = tier
    return None


def _claude_delegation_module():
    """The router's ``claude_delegation`` submodule, or None when not importable.

    Reached through ``_router_module()`` (the same standalone/plugin import
    fallback every other dashboard reader uses) rather than importing the
    module directly, so a dashboard run outside the router package degrades
    to the unavailable status below instead of raising.
    """
    router = _router_module()
    return getattr(router, "claude_delegation", None) if router else None


CLAUDE_REASONING_TIERS = ("sonnet", "opus")
_CLAUDE_REASONING_DEFAULT_LEVELS = {"sonnet": "medium", "opus": "medium"}
_CLAUDE_REASONING_TRACEBACKS: set[tuple[type[BaseException], str]] = set()
_CLAUDE_REASONING_TRACEBACK_LOCK = threading.Lock()
_CLAUDE_REASONING_TRACEBACKS_MAX = 64


def _claude_reasoning_defaults(delegation) -> dict:
    defaults = getattr(delegation, "DEFAULT_REASONING_EFFORT", None) if delegation is not None else None
    if not isinstance(defaults, dict):
        defaults = _CLAUDE_REASONING_DEFAULT_LEVELS
    return {tier: str(defaults.get(tier, _CLAUDE_REASONING_DEFAULT_LEVELS[tier])) for tier in CLAUDE_REASONING_TIERS}


def _claude_reasoning_pinned(config: dict) -> dict:
    block = config.get("claude_delegation")
    effort = block.get("reasoning_effort") if isinstance(block, dict) else None
    return {tier: isinstance(effort, dict) and tier in effort for tier in CLAUDE_REASONING_TIERS}


def _claude_reasoning_fallback_levels(config: dict, defaults: dict) -> dict:
    block = config.get("claude_delegation")
    effort = block.get("reasoning_effort") if isinstance(block, dict) else None
    effort = effort if isinstance(effort, dict) else {}
    levels = {}
    for tier in CLAUDE_REASONING_TIERS:
        value = effort.get(tier)
        normalized = value.strip().casefold() if isinstance(value, str) else ""
        levels[tier] = normalized if normalized in EFFORT_LEVELS else defaults[tier]
    return levels


def _haiku_reasoning_supported(delegation) -> bool:
    """Whether Haiku is one of this router's editable reasoning-effort tiers.

    Derived from ``claude_delegation.EDITABLE_REASONING_TIERS`` rather than a
    hardcoded literal, so a future tier change is reflected here automatically
    instead of silently drifting. False whenever the router module (or the
    tuple on it) isn't available -- the safe default the fallback branches
    already use.
    """
    tiers = getattr(delegation, "EDITABLE_REASONING_TIERS", ()) if delegation is not None else ()
    return "haiku" in tiers


def _claude_reasoning_status(config: dict) -> dict:
    """``claude_reasoning_effort`` for ``GET /api/config``: levels plus bridge availability.

    Falls back to the router's own defaults, unavailable, with a nonempty
    reason, whenever the router package (or the reasoning-effort API on it)
    cannot be imported -- the dashboard must keep serving even without it.
    """
    delegation = _claude_delegation_module()
    defaults = _claude_reasoning_defaults(delegation)
    pinned = _claude_reasoning_pinned(config)
    if delegation is None:
        return {
            "available": False,
            "reason": "Router package not importable in this dashboard process",
            "levels": _claude_reasoning_fallback_levels(config, defaults),
            "defaults": defaults,
            "pinned": pinned,
            "haiku_supported": False,
        }
    try:
        levels = delegation.reasoning_effort_config(config)
        available, reason = delegation.reasoning_bridge_status()
    except Exception as exc:
        signature = (type(exc), str(exc))
        should_print = False
        with _CLAUDE_REASONING_TRACEBACK_LOCK:
            if signature not in _CLAUDE_REASONING_TRACEBACKS and len(_CLAUDE_REASONING_TRACEBACKS) < _CLAUDE_REASONING_TRACEBACKS_MAX:
                _CLAUDE_REASONING_TRACEBACKS.add(signature)
                should_print = True
        if should_print:
            traceback.print_exc()
        return {
            "available": False,
            "reason": f"claude_delegation reasoning-effort API not usable ({type(exc).__name__}: {exc})",
            "levels": _claude_reasoning_fallback_levels(config, defaults),
            "defaults": defaults,
            "pinned": pinned,
            "haiku_supported": _haiku_reasoning_supported(delegation),
        }
    return {
        "available": bool(available),
        "reason": str(reason or ""),
        "levels": {tier: levels[tier] for tier in CLAUDE_REASONING_TIERS},
        "defaults": defaults,
        "pinned": pinned,
        "haiku_supported": _haiku_reasoning_supported(delegation),
    }


def _save_claude_reasoning_effort(raw, config: dict):
    """Update only ``claude_delegation.reasoning_effort``'s two editable tiers.

    Every submitted key and value is checked into a temporary ``cleaned``
    mapping before ``config`` changes, matching ``_save_effort``'s rule that a
    partial payload cannot leave an earlier tier changed after a later one
    fails.
    """
    if not isinstance(raw, dict):
        return "claude_reasoning_effort must be an object of tier -> low, medium, high, or xhigh"
    cleaned = {}
    for tier, value in raw.items():
        if tier not in CLAUDE_REASONING_TIERS:
            return f"The dashboard cannot edit Claude reasoning-effort key '{tier}'"
        if value is None or value == "":
            cleaned[tier] = None
            continue
        if not isinstance(value, str):
            return f"claude_reasoning_effort.{tier} must be one of: low, medium, high, xhigh"
        normalized = value.strip().casefold()
        if normalized not in EFFORT_LEVELS:
            return f"claude_reasoning_effort.{tier} must be one of: low, medium, high, xhigh"
        cleaned[tier] = normalized
    if not cleaned:
        # Leave an absent reasoning_effort block absent when a client posts no changes.
        return None
    existing_block = config.get("claude_delegation")
    existing_effort = existing_block.get("reasoning_effort") if isinstance(existing_block, dict) else None
    existing_effort = existing_effort if isinstance(existing_effort, dict) else {}
    delegation = _claude_delegation_module()
    defaults = _claude_reasoning_defaults(delegation)
    to_write = {}
    to_remove = set()
    for tier, value in cleaned.items():
        if value is None:
            if tier in existing_effort:
                to_remove.add(tier)
        elif tier not in existing_effort and value == defaults[tier]:
            # No live override exists and the posted value already matches the
            # effective default: writing it would pin today's default forever
            # (a future default change would never reach this dashboard).
            continue
        else:
            to_write[tier] = value
    if not to_write and not to_remove:
        return None
    block = config.get("claude_delegation")
    if not isinstance(block, dict):
        block = config["claude_delegation"] = {}
    reasoning_effort = block.get("reasoning_effort")
    if not isinstance(reasoning_effort, dict):
        reasoning_effort = block["reasoning_effort"] = {}
    for tier in to_remove:
        reasoning_effort.pop(tier, None)
    for tier, value in to_write.items():
        reasoning_effort[tier] = value
    if not reasoning_effort:
        block.pop("reasoning_effort", None)
    return None


HTML = r'''<!doctype html>
<html lang="hu">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Home Lab</title>
<style>
:root{color-scheme:dark;--bg:#080b12;--panel:#111723;--border:#253044;--text:#e9eef8;--muted:#8997ad;--luna:#61dafb;--spark:#f3f6fb;--terra:#88e36f;--sol:#ffb454;--opus5:#d695ff;--sonnet5:#9fb4ff;--haiku:#ff9ecf;--qwen:#38d9a9;--grok:#fde047;--accent:#a78bfa}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#17203a 0,transparent 35%),var(--bg);color:var(--text);font:14px/1.5 Inter,Segoe UI,system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:28px}h1{font-size:26px;margin:0}.sub{color:var(--muted);margin:4px 0 22px}.toolbar,.cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}.toolbar{align-items:end;background:rgba(17,23,35,.86);padding:14px;border:1px solid var(--border);border-radius:14px;backdrop-filter:blur(10px)}
label{display:grid;gap:5px;color:var(--muted);font-size:12px}input,select,button{background:#0b111c;color:var(--text);border:1px solid var(--border);border-radius:8px;padding:9px 11px;font:inherit}input[type=search]{min-width:260px}button{cursor:pointer}button:hover{border-color:var(--accent)}button:disabled{cursor:wait;opacity:.72}.status.refreshing{color:#c4b5fd}.check{display:flex;align-items:center;gap:7px;padding:9px 2px}.check input{accent-color:var(--accent)}
.card{min-width:135px;flex:1;background:linear-gradient(145deg,#151d2c,#0e1420);border:1px solid var(--border);border-radius:14px;padding:15px}.card .n{font-size:25px;font-weight:750}.card .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}.luna .n{color:var(--luna)}.spark .n{color:var(--spark)}.terra .n{color:var(--terra)}.sol .n{color:var(--sol)}.opus5 .n{color:var(--opus5)}.sonnet5 .n{color:var(--sonnet5)}.haiku .n{color:var(--haiku)}.qwen .n{color:var(--qwen)}.grok .n{color:var(--grok)}
.table-wrap{overflow:auto;border:1px solid var(--border);border-radius:14px;background:rgba(13,18,29,.92)}table{border-collapse:collapse;table-layout:fixed;width:1405px;min-width:100%}th{position:sticky;top:0;background:#161e2c;color:var(--muted);font-size:11px;letter-spacing:.07em;text-align:left;text-transform:uppercase;user-select:none}th,td{padding:11px 13px;border-bottom:1px solid #1e2838;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.prompt-text{min-width:0}.word-wrap .prompt-text{white-space:normal;overflow:visible;text-overflow:clip;overflow-wrap:anywhere}.resizer{position:absolute;z-index:2;top:0;right:-4px;width:9px;height:100%;cursor:col-resize;touch-action:none}.resizer::after{content:'';position:absolute;left:4px;top:20%;width:1px;height:60%;background:#3b4a62}.resizer:hover::after,.resizer.dragging::after{width:2px;background:var(--accent)}body.resizing{cursor:col-resize;user-select:none}tbody tr:hover{background:#151d2b}code{color:#b9c5d8}.pill{display:inline-block;border:1px solid currentColor;border-radius:99px;padding:2px 8px;font-size:12px;font-weight:700;margin-right:4px}.pill.luna{color:var(--luna)}.pill.spark{color:var(--spark)}.pill.terra{color:var(--terra)}.pill.sol{color:var(--sol)}.pill.opus5{color:var(--opus5)}.pill.sonnet5{color:var(--sonnet5)}.pill.haiku{color:var(--haiku)}.pill.qwen{color:var(--qwen)}.pill.grok{color:var(--grok)}.route{white-space:nowrap}.reason{color:#c2ccdb}.running-agent-pill{display:inline-block;margin-left:8px;padding:2px 7px;border:1px solid #88e36f;border-radius:99px;color:#88e36f;font-size:10px;font-weight:800;letter-spacing:.06em;vertical-align:middle;animation:agentPulse 1.4s ease-in-out infinite}@keyframes agentPulse{50%{box-shadow:0 0 11px rgba(136,227,111,.7)}}.play-indicator{display:inline-flex;align-items:center;justify-content:center;width:19px;height:19px;margin-left:8px;border-radius:50%;background:#54d66a;color:#07110b;font-size:10px;font-weight:900;vertical-align:middle;box-shadow:0 0 12px rgba(84,214,106,.75);animation:agentPulse 1.4s ease-in-out infinite}.active-router-row{background:rgba(84,214,106,.055)}.calls,.expand{text-align:center}.prompt-toggle{border:0;background:transparent;padding:0;color:var(--accent);font-size:15px;line-height:1}.prompt-toggle:hover{border:0;color:#c4b5fd}.detail>td{padding:10px 13px 14px;background:#0b111c;overflow:visible}.details-table{width:calc(100% - 28px);min-width:0;margin-left:28px;border:1px solid #253044;border-radius:8px;table-layout:fixed}.details-table th{position:static;background:#111927}.details-table th,.details-table td{padding:8px 10px;font-size:12px}.details-table tbody tr:last-child td{border-bottom:0}.status{margin-left:auto;color:var(--muted);padding:9px 4px}.empty{text-align:center;color:#8997ad;padding:40px}.error{color:#ff6b7a} @media(max-width:700px){main{padding:16px}input[type=search]{min-width:180px}.status{width:100%;margin:0}}
.account-cards{display:grid;gap:14px}.account-card{background:linear-gradient(145deg,#151d2c,#0e1420);border:1px solid var(--border);border-radius:14px;padding:15px}.account-card.anthropic{border-color:var(--opus5)}.account-card.openai-codex{border-color:var(--terra)}.account-card.qwen-token{border-color:var(--qwen)}.account-card.xai-oauth{border-color:var(--grok)}.account-row{display:grid;grid-template-columns:110px 1fr;gap:10px;padding:8px 0;border-top:1px solid #1e2838}.pref-chip.primary select{padding:3px 6px;margin-right:4px}.panel-heading .settings-status{font-size:12px;font-weight:400;margin-left:10px}.account-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));column-gap:22px;margin-top:8px}.account-grid .account-row{display:flex;flex-direction:column;gap:6px;min-width:0}.account-grid .account-row>span{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}.account-grid .account-row.usage{grid-column:1/-1}.account-card .model-list{display:grid;grid-template-columns:44px max-content 190px;column-gap:8px}.account-card .model-list.no-effort{grid-template-columns:44px max-content}.account-card .model-switch{display:grid;grid-column:1/-1;grid-template-columns:subgrid;align-items:center;min-height:50px}.account-card .model-switch select{width:190px;min-width:0;line-height:20px;padding:7px 8px;height:auto}.account-card .model-switch select.effort-none{opacity:.45;background:#1a2130;color:#8997ad;border-style:dashed;cursor:not-allowed}.account-card .model-switch .cooldown-pill{grid-column:1/-1;justify-self:start;width:100%;min-width:0;contain:inline-size}.account-card .model-switch.disabled>span{color:var(--muted)}.account-card .limits label{display:inline-flex;align-items:center;gap:6px;margin-right:10px}.account-card .limits input{width:64px;padding:5px 7px}.account-card .stepdown{margin-top:6px;font-size:12px;color:var(--muted)}.account-card .usage-line{display:grid;grid-template-columns:60px minmax(0,1fr) 42px 150px;align-items:center;gap:10px;padding:2px 0}.account-card .usage-value{text-align:right}.account-card .usage-reset{font-size:12px;color:var(--muted)}.account-card .usage-age{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:6px;font-size:12px;color:var(--muted)}.account-card .usage-load{margin-left:auto}.account-card .usage-error{color:#ff6b7a;flex-basis:100%}@media (max-width:760px){.account-grid{grid-template-columns:1fr}.account-card .usage-line{grid-template-columns:50px minmax(0,1fr) 40px}.account-card .usage-reset{grid-column:2/-1}}.usage-bar{position:relative;height:10px;border-radius:99px;background:#1e2838;overflow:hidden}.usage-fill{height:100%}.usage-tick{position:absolute;top:0;width:2px;height:100%;background:#8997ad}.state-badge.open{color:#88e36f}.state-badge.soft{color:#ffb454}.state-badge.closed{color:#ff6b7a}.state-badge.unknown{color:#8997ad}.account-card.stale .usage-bar{opacity:.45}
.account-groups{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}.account-group{flex:1;min-width:280px;display:flex;flex-direction:column;border:1px solid var(--border);border-radius:14px;padding:10px}.account-group.anthropic{border-color:var(--opus5)}.account-group.openai-codex{border-color:var(--terra)}.account-group.qwen-token{border-color:var(--qwen)}.account-group.xai-oauth{border-color:var(--grok)}.account-group-head{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}.account-group-cards{display:flex;gap:8px;flex-wrap:wrap}.account-group-usage{margin-top:auto;padding-top:8px}.account-group-usage .usage-reset{margin-left:8px}.account-usage{font-size:12px}.account-usage.none{color:var(--muted)}.account-usage.stale .usage-bar{opacity:.45}
.delegation-chips{display:inline-flex;flex-wrap:wrap;gap:4px;margin-left:8px;vertical-align:middle}.delegation-chip{border:1px solid currentColor;border-radius:99px;padding:1px 7px;font-size:11px}.delegation-chip.openai-codex{color:var(--terra)}.delegation-chip.anthropic{color:var(--opus5)}.delegation-chip.qwen-token{color:var(--qwen)}.delegation-chip.xai-oauth{color:var(--grok)}.delegation-chip.marked{font-weight:700}
</style><style>.agents{margin-top:22px;padding:18px;border:1px solid var(--border);border-radius:14px;background:linear-gradient(145deg,#151d2c,#0e1420)}.agents h2{margin:0;font-size:18px}.agent-parent{margin-top:12px;border-top:1px solid #253044;padding-top:12px}.agent-session{color:#c4b5fd;font-size:12px;letter-spacing:.06em}.agent-child{display:grid;grid-template-columns:10px 1fr auto;gap:10px;align-items:center;margin-top:9px;padding:10px 12px;border-radius:10px;background:#0b111c}.agent-dot{width:9px;height:9px;border-radius:50%;background:#8997ad}.agent-dot.running{background:#88e36f;box-shadow:0 0 12px #88e36f}.agent-goal{font-weight:700}.agent-activity{color:#a78bfa;font-size:12px;margin-top:2px}.agent-meta{color:var(--muted);font-size:12px;text-align:right}.agent-empty{color:var(--muted);padding:12px 0}</style><style>.lab-header{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:18px}.lab-kicker{color:#a78bfa;font-size:11px;font-weight:800;letter-spacing:.14em}.lab-title{font-size:30px;font-weight:800;letter-spacing:-.04em}.tabs{display:flex;gap:6px;padding:5px;border:1px solid var(--border);border-radius:12px;background:#0b111c}.tab{border:0;background:transparent;color:var(--muted);font-weight:700}.tab.active{background:#252039;color:#e9ddff}.panel[hidden]{display:none}.panel-heading{font-size:18px;font-weight:750;margin:0 0 4px}</style><style>.console{margin:10px 0 4px 19px;border:1px solid #2c3951;border-radius:10px;background:#080d16}.console summary,.agent-history summary{cursor:pointer;padding:9px 11px;color:#c4b5fd;font-weight:700}.console-event{border-top:1px solid #1e2838}.console-event.compact{padding:6px 11px;color:#c6d0df;font:12px ui-monospace,SFMono-Regular,Consolas,monospace}.console-label{padding:7px 11px;color:#88e36f;font:12px ui-monospace,SFMono-Regular,Consolas,monospace}.console pre{margin:0;padding:0 11px 11px;max-height:220px;overflow:auto;white-space:pre-wrap;word-break:break-word;color:#c6d0df;font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}.console-empty{padding:11px;color:var(--muted)}.agent-history{margin-top:20px;border-top:1px solid #253044}.history-item{padding:7px 12px;color:var(--muted);border-top:1px solid #1e2838}</style><style>.settings{margin-top:22px;display:flex;flex-direction:column;gap:18px}.settings-section{padding:18px;border:1px solid var(--border);border-radius:14px;background:linear-gradient(145deg,#151d2c,#0e1420)}.settings-section h3{margin:0 0 14px;font-size:16px;color:var(--accent);text-transform:uppercase;letter-spacing:.1em}.toggle-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}.toggle-item{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:14px;border:1px solid var(--border);border-radius:10px;background:#0b111c}.toggle-item.disabled{opacity:.5;border-color:#1a2033}.toggle-label{display:flex;flex-direction:column;gap:2px;min-width:0;flex:1 1 auto}.toggle-name{font-weight:700;font-size:14px}.toggle-desc{font-size:11px;color:var(--muted)}.switch{position:relative;width:44px;height:24px;flex:0 0 44px}.switch input{opacity:0;width:0;height:0}.switch .slider{position:absolute;cursor:pointer;top:0;left:0;right:0;bottom:0;background:#253044;border-radius:24px;transition:.2s}.switch .slider::before{position:absolute;content:'';height:18px;width:18px;left:3px;bottom:3px;background:#8997ad;border-radius:50%;transition:.2s}.switch input:checked+.slider{background:#88e36f}.switch input:checked+.slider::before{transform:translateX(20px);background:#07110b}.default-model-row{display:flex;align-items:end;gap:12px;padding:14px;border:1px solid var(--border);border-radius:10px;background:#0b111c}.default-model-row label{flex:0 0 auto}.default-model-row select{min-width:200px}.save-settings{align-self:flex-end;padding:10px 20px;background:var(--accent);color:#fff;border:0;border-radius:8px;font-weight:700;cursor:pointer}.save-settings:hover{background:#8b72f0}.save-settings:disabled{opacity:.6;cursor:wait}.settings-status{margin-left:auto;font-size:12px;color:var(--muted)}.cooldown-pill{display:inline-block;margin-top:4px;padding:2px 7px;border-radius:999px;background:#3a2418;border:1px solid #7c4a25;color:#ffbe8a;font-size:10px;font-weight:700;letter-spacing:.04em;white-space:normal;overflow-wrap:anywhere;max-width:100%}.toggle-item.cooling{border-color:#7c4a25}.account-load{margin-top:14px;padding:12px 14px;border:1px solid var(--border);border-radius:10px;background:#0b111c;font-size:12px;color:var(--muted)}.account-load b{color:#e6edf6;font-weight:700}.account-load .idle{color:#88e36f}</style><style>.pref-kinds{display:flex;flex-direction:column;gap:10px}.pref-kind{padding:12px 14px;border:1px solid var(--border);border-radius:10px;background:#0b111c}.pref-kind-head{display:flex;align-items:center;justify-content:space-between;gap:12px}.pref-kind-name{font-weight:700;font-size:14px}.pref-kind-desc{font-size:11px;color:var(--muted);margin-top:2px}.pref-chain{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px;align-items:center}.pref-chip{display:inline-flex;align-items:center;gap:6px;padding:4px 8px;border-radius:999px;background:#1b2437;border:1px solid #2c3951;font-size:12px;font-weight:700}.pref-chip.external{border-color:#4b3a7a;background:#241d3a;color:#c9b8ff}.pref-chip.anthropic{border-color:var(--opus5)}.pref-chip.openai-codex{border-color:var(--terra)}.pref-chip.qwen-token{border-color:var(--qwen)}.pref-chip.xai-oauth{border-color:var(--grok)}.chain-account{color:var(--muted);font-weight:600}.chain-account-mark{color:#ffb454;font-weight:700}.pref-chip button{border:0;background:transparent;color:var(--muted);cursor:pointer;padding:0 2px;font-size:12px}.pref-chip button:hover{color:#e6edf6}.pref-chip .rank{color:var(--muted);font-weight:600}.pref-add{min-width:150px}.pref-empty{color:var(--muted);font-size:12px}.pref-note{margin-top:10px;font-size:11px;color:var(--muted)}.balance-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.balance-row .pref-note{margin-top:0;flex-basis:100%}.balance-row .balance-field{display:inline-flex;align-items:center;gap:6px;margin-left:6px}.balance-field input{width:64px;padding:5px 7px}.fb-file{display:block;margin-top:6px;color:#ffbe8a;font-size:11px;font-weight:700}</style><style>.command-frame{display:block;width:100%;height:calc(100vh - 180px);min-height:680px;border:1px solid var(--border);border-radius:14px;background:#111723}</style>
</head>
<body><main>
<div class="lab-header"><div><div class="lab-kicker" data-i18n="lab.kicker">HERMES · LOCAL OBSERVABILITY</div><div class="lab-title" data-i18n="lab.title">AI Home Lab</div><div class="sub" data-i18n="lab.sub">Modellek, háttéragentek és élő munkafolyamatok egy helyen</div></div><nav class="tabs" aria-label="AI Home Lab nézetek"><button class="tab active" data-tab="router"><span data-i18n="tab.router">Model Router</span></button><button class="tab" data-tab="settings"><span data-i18n="tab.settings">Beállítások</span></button><button class="tab" data-tab="command"><span data-i18n="tab.command">Hermes Command Center</span></button></nav></div>
<section id="router-panel" class="panel" hidden><h2 class="panel-heading"><span data-i18n="router.heading">Model Router</span></h2><div class="sub" data-i18n="router.sub">Élő JSONL routing napló · automatikus frissítés 3 másodpercenként</div>
<div class="toolbar">
<label><span data-i18n="router.tier.label">Modell</span><select id="tier"><option value="" data-i18n="router.tier.all">Mind</option></select></label>
<label>Keresés<input id="search" type="search" data-i18n-placeholder="router.search.placeholder"></label>
<label><span data-i18n="router.last.label">Utolsó root promptok</span><select id="last"><option>1</option><option>5</option><option selected>10</option></select></label>
<label class="check"><input id="grouped" type="checkbox" checked> <span data-i18n="router.grouped">Promptonként összevonva</span></label>
<label class="check"><input id="word-wrap" type="checkbox"> <span data-i18n="router.wordwrap">Sortörés</span></label>
<label class="check"><input id="auto" type="checkbox" checked> <span data-i18n="router.auto">Automatikus frissítés</span></label>
<button id="refresh" data-i18n="router.refresh">Frissítés</button><span id="status" class="status" data-i18n="router.loading">Betöltés…</span>
</div>
<div class="cards"><div class="card"><div class="n" id="total">0</div><div class="k" data-i18n="card.total">Összes routing döntés</div></div><div class="card luna"><div class="n" id="luna">0</div><div class="k" data-i18n="card.luna">GPT-6 Luna</div></div><div class="card spark"><div class="n" id="spark">0</div><div class="k" data-i18n="card.spark">GPT-5.3 Spark</div></div><div class="card terra"><div class="n" id="terra">0</div><div class="k" data-i18n="card.terra">GPT-5.6 Terra</div></div><div class="card sol"><div class="n" id="sol">0</div><div class="k" data-i18n="card.sol">GPT-6 Sol</div></div><div class="card opus5"><div class="n" id="opus5">0</div><div class="k" data-i18n="card.opus5">Claude Opus 5.5</div></div><div class="card sonnet5"><div class="n" id="sonnet5">0</div><div class="k" data-i18n="card.sonnet5">Claude Sonnet 5</div></div><div class="card haiku"><div class="n" id="haiku">0</div><div class="k" data-i18n="card.haiku">Claude Haiku 4.5</div></div><div class="card qwen"><div class="n" id="qwen">0</div><div class="k" data-i18n="card.qwen">Qwen 3.7 Plus</div></div><div class="card grok"><div class="n" id="grok">0</div><div class="k" data-i18n="card.grok">Grok 4.7</div></div></div>
<div class="account-groups" id="account-groups"></div>
<div id="runs" class="router-runs" aria-live="polite"></div><div class="table-wrap" hidden><table id="log-table"><colgroup><col style="width:55px"><col style="width:110px"><col style="width:90px"><col style="width:350px"><col style="width:90px"><col style="width:230px"><col style="width:480px"></colgroup><thead><tr><th class="expand"></th><th data-i18n="th.date">Dátum</th><th data-i18n="th.time">Idő (CET/CEST)</th><th data-i18n="th.prompt">Prompt</th><th class="calls" data-i18n="th.calls">Hívások</th><th data-i18n="th.route">Útvonal</th><th data-i18n="th.reason">Indok</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section id="settings-panel" class="panel" hidden><h2 class="panel-heading"><span data-i18n="settings.heading">Beállítások</span> <span class="settings-status" id="settings-status"></span></h2><div class="sub" data-i18n="settings.sub">Modellek hívhatósága és alapértelmezett modell</div><div class="settings"><div class="settings-section"><h3 data-i18n="settings.accounts.heading">Fiókok</h3><div class="account-cards" id="account-cards"></div></div><div class="settings-section" id="balance-section" hidden><h3 data-i18n="settings.balance.heading">Terheléselosztás</h3><div id="balance-settings"></div></div><div class="settings-section"><h3 data-i18n="settings.main.heading">Fő agent</h3><div class="sub"><span data-i18n="settings.main.sub"></span> <b class="fb-file" data-i18n="settings.fb.file"></b></div><div class="pref-kinds" id="main-chain"></div></div><div class="settings-section"><h3 data-i18n="settings.workers.heading">Workerek</h3><div class="sub" data-i18n="settings.workers.sub"></div><div class="pref-kinds" id="worker-settings"></div></div><div class="settings-section"><h3 data-i18n="settings.prefs.heading">Preferált modellek munkatípusonként</h3><div class="sub" data-i18n="settings.prefs.sub">Sorrendben, a legjobb elöl. A router az első hívható elemet választja.</div><div class="pref-kinds" id="pref-kinds"></div></div><div class="settings-section"><h3 data-i18n="settings.language">Nyelv</h3><div class="default-model-row"><label><span data-i18n="settings.language">Nyelv</span><select id="language-select"><option value="en" data-i18n="settings.lang.en">English</option><option value="hu" data-i18n="settings.lang.hu">Magyar</option></select></label></div></div></div></section>
<section id="command-panel" class="panel" hidden><h2 class="panel-heading"><span data-i18n="tab.command">Hermes Command Center</span></h2><div class="sub">A Hermes hivatalos helyi kezelőfelülete</div><iframe class="command-frame" title="Hermes Command Center" src="http://127.0.0.1:9119/"></iframe></section>
</main>
<script>
const $=id=>document.getElementById(id);
// ── i18n ──────────────────────────────────────────────────────────
const I18N = {
  en: {
    'settings.worker.limits': 'Effective worker limits: depth {depth}, concurrent children {children}, iterations {iterations}.',
    // Header
    'lab.kicker': 'HERMES · LOCAL OBSERVABILITY',
    'lab.title': 'AI Home Lab',
    'lab.sub': 'Models, background agents and live workflows in one place',
    // Tabs
    'tab.router': 'Model Router',
    'tab.settings': 'Settings',
    'tab.command': 'Hermes Command Center',
    // Router panel
    'router.heading': 'Model Router',
    'router.sub': 'Live JSONL routing log · auto-refresh every 3 seconds',
    'router.tier.label': 'Model',
    'router.tier.all': 'All',
    'router.search.placeholder': 'prompt, model or reason…',
    'router.last.label': 'Last root prompts',
    'router.grouped': 'Grouped by prompt',
    'router.wordwrap': 'Word wrap',
    'router.auto': 'Auto refresh',
    'router.refresh': 'Refresh',
    'router.loading': 'Loading…',
    // Cards
    'card.total': 'Total routing decisions',
    'card.luna': 'GPT-6 Luna',
    'card.spark': 'GPT-5.3 Spark',
    'card.terra': 'GPT-5.6 Terra',
    'card.sol': 'GPT-6 Sol',
    'card.opus5': 'Claude Opus 5.5',
    'card.sonnet5': 'Claude Sonnet 5.5',
    'card.haiku': 'Claude Haiku 4.5',
    'card.qwen': 'Qwen 3.7 Plus',
    'card.grok': 'Grok 4.7',
    // Table headers
    'th.date': 'Date',
    'th.time': 'Time (CET/CEST)',
    'th.prompt': 'Prompt',
    'th.calls': 'Calls',
    'th.route': 'Route',
    'th.reason': 'Reason',
    // Misc router
    'no.entries': 'No matching entries.',
    'not.recoverable': 'Not recoverable',
    'details.close': 'Close details',
    'details.open': 'Open API calls',
    'related.missing': 'Related earlier message not found.',
    'subagent.close': 'Close sub-agents',
    'subagent.open': 'Open sub-agents',
    'subagent.running': 'Sub-agent running',
    'subagent.done': 'Finished',
    'main.agent': 'Main agent',
    'root.continuation': 'INTERNAL ROUTER CONTINUATIONS',
    'root.continuation.title': 'Not a delegated worker: internal continuations and routing decisions of the main thread between calls.',
    'close.router.steps': 'Close internal router steps',
    'open.router.steps': 'Open internal router steps',
    // Agent panel
    'agents.heading': 'Delegated sub-agents',
    'agents.running.done.error': 'running · completed · errors',
    'agents.recent': 'recent runs',
    'agents.none': 'No delegated sub-agents to display.',
    'agents.main.thread': 'MAIN THREAD',
    'agents.prev.task': 'Previous main task',
    'agents.reason.default': 'Independent subtask',
    'agents.calls': 'calls',
    'agents.close.last': 'Last',
    'agents.close.steps': 'tool events',
    'agents.no.events': 'No stored tool events for this agent yet.',
    'agents.recently.done': 'Recently completed',
    'agents.main.task': 'Main task',
    'agents.open.tasks': 'Open subtasks',
    'agents.close.tasks': 'Close subtasks',
    'agents.subtasks': 'subtasks',
    'agents.running': 'Running',
    'agents.done': 'Done',
    'agents.task.desc': 'Task description',
    'agents.no.desc': 'No saved task description.',
    'agents.close.inner': 'Close inner tasks',
    'agents.open.inner': 'Open inner tasks',
    'agents.below.root': 'Below root',
    'agents.requested.ro': 'Requested READ-ONLY',
    'agents.requested.ro.title': 'The main agent explicitly requested this sub-agent for read-only/verification only.',
    // Settings panel
    'settings.heading': 'Settings',
    'settings.sub': 'Accounts, load balancing, main agent, workers and routing',
    'settings.main.heading': 'Main agent',
    'settings.main.sub': 'Hermes starts on the first model; when its provider is out, the rest take over in order. The first one also takes the default route and coordinates (conductor).',
    'settings.main.primary': 'starts here',
    'settings.main.external': 'Hermes currently starts on {model}, set outside the router: the first entry here only sets the default route and the conductor.',
    'settings.main.claude': 'Hermes starts on Claude. The router\'s default route and conductor stay on {tier}.',
    'settings.workers.heading': 'Workers',
    'settings.workers.sub': 'Delegated agents: which model they get when the call names none, and where they go when it is out.',
    'settings.workers.codex': 'Codex workers',
    'settings.workers.codex.desc': 'delegate_task, when the goal names no tier',
    'settings.workers.claude': 'Claude workers',
    'settings.workers.claude.desc': 'delegate_claude, when the call names no tier',
    'settings.workers.fallback': 'Worker fallback',
    'settings.workers.fallback.desc': 'A leaf pinned to a target never inherits the main agent chain',
    'settings.accounts.heading': 'Accounts',
    'settings.balance.heading': 'Load balancing',
    'settings.balance.label': 'Load balancing',
    'settings.balance.busy': 'from',
    'settings.balance.window.5-hour': '5-hour',
    'settings.balance.window.tighter': 'tighter (weekly or 5-hour)',
    'settings.balance.margin': 'gap',
    'settings.balance.desc': 'Evens out the {window} windows: when the preferred account is at {busy}% or more and the other is at least {margin} points freer, the freer one goes first. The parent counts on its own account. The soft/hard limits still apply.',
    'account.state.open': 'open',
    'account.state.soft': 'soft limit',
    'account.state.closed': 'closed',
    'account.state.unknown': 'unknown',
    'account.models': 'Models',
    'account.usage': 'Usage',
    'account.usage.week': 'week',
    'account.usage.session': '5-hour',
    'account.usage.resets': 'resets',
    'account.usage.age': 'read {n} ago',
    'account.usage.none': 'no usage data for this account',
    'account.usage.refresh': 'Refresh',
    'main.usage.none': 'no usage data',
    'account.limits': 'Limits',
    'account.limits.soft': 'soft',
    'account.limits.hard': 'hard',
    'account.limits.stepdown': 'step down',
    'account.delegation': 'Delegation',
    'account.delegation.via': 'via {tool}',
    'account.delegation.always': 'always on (Hermes built-in route)',
    'account.delegation.default_tier': 'default tier',
    'account.delegation.live': 'delegate_claude live',
    'account.delegation.restart': 'restart Hermes to apply',
    'account.delegation.off': 'off — every Claude model is switched off',
    'account.load': 'Load',
    'account.load.calls': '{n} calls in the last {m} min',
    'settings.cooling': 'cooling down',
    'settings.load.title': 'Recent load per account',
    'settings.load.window': 'last {n} min',
    'settings.load.idle': 'no calls — prefer it for an independent leaf',
    'settings.load.counts': 'call counts from the route log, not quota readings',
    'settings.load.empty': 'No routed calls in the window.',
    'settings.default.heading': 'Default model (Orchestrator)',
    'settings.default.desc': 'The model Hermes itself starts on, and the tier that counts as the orchestrator. A preference chain below can re-route a turn, but not change this.',
    'settings.effort.low': 'low',
    'settings.effort.medium': 'medium',
    'settings.effort.high': 'high',
    'settings.effort.xhigh': 'xhigh',
    'settings.claude_effort.haiku_no_reasoning': 'no reasoning allowed',
    'settings.claude_effort.default': 'Default ({level})',
    'settings.prefs.heading': 'Routing: preferred models per kind of work',
    'settings.prefs.sub': 'In order, best first. The router takes the first callable entry. This re-routes a turn; it does not change the model Hermes starts on.',
    'settings.prefs.add': 'add model...',
    'settings.prefs.empty': 'No preference — the built-in route applies.',
    'settings.prefs.note': 'A grey chip is routed by the router itself. A purple one is on another account, so it is passed to the conductor as a delegation recommendation.',
    'settings.prefs.up': 'move up',
    'settings.prefs.down': 'move down',
    'settings.prefs.remove': 'remove',
    'kind.design': 'UI and visual design',
    'kind.code': 'Writing code',
    'kind.explore': 'Exploring, read-only inspection',
    'kind.review': 'Review, critique',
    'kind.sensitive': 'Security sensitive (auth, payment)',
    'kind.critical': 'Critical (deploy, migration, production)',
    'kind.long': 'Long requests',
    'kind.chat': 'Short conversation',
    'kind.default': 'Everything else',
    'settings.fb.file': 'This writes ~/.hermes/config.yaml, not the router config — a copy is kept before every save.',
    'settings.fb.orchestrator': 'Orchestrator',
    'settings.fb.orchestrator.desc': 'The main agent, when its own provider is out',
    'settings.fb.children': 'Delegated workers',
    'settings.fb.children.desc': 'Applies to delegate_task workers. Claude workers from delegate_claude return failures to the parent for re-dispatch; they do not use this fallback chain.',
    'settings.fb.empty': 'No fallback — the turn fails when the primary is out.',
    'settings.fb.inherit': 'Empty for workers means no fallback at all, not "inherit".',
    'settings.language': 'Language',
    'settings.lang.en': 'English',
    'settings.lang.hu': 'Magyar',
    // Model descriptions
    'model.desc.luna': 'Fast, simple tasks',
    'model.desc.spark': 'Read-only delegated work',
    'model.desc.terra': 'General orchestrator',
    'model.desc.sol': 'Security-critical, design',
    'model.desc.opus5': 'Claude account — consequential work',
    'model.desc.sonnet5': 'Claude account — default choice',
    'model.desc.haiku': 'Claude account — quick lookups',
    'model.desc.qwen': 'Alternative model',
    'model.desc.grok': 'SuperGrok account — heavy implementation',
    // Settings messages
    'settings.reload': 'Reload settings before saving again.',
    'settings.saving': 'Saving...',
    'settings.saved': 'Saved ✓',
    'settings.error.unknown': 'Unknown error',
    'settings.error.prefix': 'Error: ',
    'settings.error.load': 'Failed to load settings:',
    // Status
    'status.refreshing': 'Refreshing…',
    'status.loading': 'Loading…',
    'status.error.prefix': 'Error: ',
    'status.error.kept': ' · previous list kept',
    // Execution tree
    'exec.below.root': 'Below root',
    'exec.open.subtasks': 'Open subtasks',
    'exec.close.subtasks': 'Close subtasks',
    'exec.no.desc': 'No saved task description.',
    'exec.opus.review': 'Opus review',
    'exec.internal.step': 'Internal router step',
    'exec.read.only': 'READ-ONLY',
    'exec.requested.ro': 'REQUESTED READ-ONLY',
    // Misc
    'internal.router.step': 'Internal router step',
    'resizer.title': 'Drag to resize column · double-click: reset',
    'play.running.agent': 'Currently running main agent',
    'play.running.sub': 'AGENTS RUNNING NOW',
    'status.refresh.short': 'Refresh…',
    'status.kept': ' · previous list shown',
    'status.rootprompts': 'root prompts',
    'status.locale': 'en-GB',
    'total.routing.decisions': 'TOTAL ROUTING DECISIONS',
    'run.worker.routing': 'worker-routing',
    'agents.sum.running': 'running',
    'agents.sum.completed': 'completed',
    'agents.sum.failed': 'errors',
    'agents.main.agent': 'MAIN AGENT',
    'agents.none.active': 'No active background agents.',
    'agents.pill.running': 'RUNNING NOW',
    'agents.pill.agent': 'AGENT',
    'state.running.short': 'RUNNING',
    'state.done.short': 'DONE',
    'state.success.short': 'SUCCESS',
    'state.error.short': 'ERROR',
    'duration.h': 'h',
    'duration.m': 'm',
    'duration.s': 's',
    'exec.no.route': 'no stored route data',
    'exec.external': 'EXTERNAL',
    'exec.own': 'own',
    'exec.total': 'total',
    'root.continuation.reason': 'System continuation · routing decisions between calls',
    'settings.error.save': 'Failed to save settings:',
    'agents.unknown.model': 'unknown model',
    'play.running.subagent': 'Currently running sub-agent',
    'label.model.short': 'model',
    'label.active.turn': 'Active turn: ',
    'exec.tree.label': 'Execution tree',
    'th.time.short': 'Time',
  },
  hu: {
    'settings.worker.limits': 'Érvényes munkáskorlátok: mélység {depth}, párhuzamos gyermekek {children}, iterációk {iterations}.',
    'lab.kicker': 'HERMES · LOCAL OBSERVABILITY',
    'lab.title': 'AI Home Lab',
    'lab.sub': 'Modellek, háttéragentek és élő munkafolyamatok egy helyen',
    'tab.router': 'Model Router',
    'tab.settings': 'Beállítások',
    'tab.command': 'Hermes Command Center',
    'router.heading': 'Model Router',
    'router.sub': 'Élő JSONL routing napló · automatikus frissítés 3 másodpercenként',
    'router.tier.label': 'Modell',
    'router.tier.all': 'Mind',
    'router.search.placeholder': 'prompt, modell vagy indok…',
    'router.last.label': 'Utolsó root promptok',
    'router.grouped': 'Promptonként összevonva',
    'router.wordwrap': 'Sortörés',
    'router.auto': 'Automatikus frissítés',
    'router.refresh': 'Frissítés',
    'router.loading': 'Betöltés…',
    'card.total': 'Összes routing döntés',
    'card.luna': 'GPT-6 Luna',
    'card.spark': 'GPT-5.3 Spark',
    'card.terra': 'GPT-5.6 Terra',
    'card.sol': 'GPT-6 Sol',
    'card.opus5': 'Claude Opus 5.5',
    'card.sonnet5': 'Claude Sonnet 5.5',
    'card.haiku': 'Claude Haiku 4.5',
    'card.qwen': 'Qwen 3.7 Plus',
    'card.grok': 'Grok 4.7',
    'th.date': 'Dátum',
    'th.time': 'Idő (CET/CEST)',
    'th.prompt': 'Prompt',
    'th.calls': 'Hívások',
    'th.route': 'Útvonal',
    'th.reason': 'Indok',
    'no.entries': 'Nincs a szűrésnek megfelelő bejegyzés.',
    'not.recoverable': 'Nem visszakereshető',
    'details.close': 'Részletek bezárása',
    'details.open': 'API-hívások megnyitása',
    'related.missing': 'A kapcsolódó korábbi üzenet nem található.',
    'subagent.close': 'Mellékagentek bezárása',
    'subagent.open': 'Mellékagentek megnyitása',
    'subagent.running': 'Mellékagent fut',
    'subagent.done': 'Kész',
    'main.agent': 'Fő agent',
    'root.continuation': 'BELSŐ ROUTER FOLYTATÁSOK',
    'root.continuation.title': 'Nem delegált worker: a főszál belső, hívások közötti folytatásai és routing-döntései.',
    'close.router.steps': 'Belső router-lépések bezárása',
    'open.router.steps': 'Belső router-lépések megnyitása',
    'agents.heading': 'Delegált mellékszálak',
    'agents.running.done.error': 'fut · kész · hiba',
    'agents.recent': 'legutóbbi futás',
    'agents.none': 'Nincs megjeleníthető delegált mellékszál.',
    'agents.main.thread': 'FŐSZÁL',
    'agents.prev.task': 'Korábbi fő feladat',
    'agents.reason.default': 'Önálló részfeladat',
    'agents.calls': 'hívás',
    'agents.close.last': 'Utolsó',
    'agents.close.steps': 'lépés',
    'agents.no.events': 'Még nincs tárolt tool-esemény ehhez az agenthez.',
    'agents.recently.done': 'Legutóbb befejezett munkák',
    'agents.main.task': 'Fő feladat',
    'agents.open.tasks': 'Alfeladatok megnyitása',
    'agents.close.tasks': 'Alfeladatok bezárása',
    'agents.subtasks': 'mellékszál',
    'agents.running': 'Fut',
    'agents.done': 'Kész',
    'agents.task.desc': 'Feladatleírás',
    'agents.no.desc': 'Nincs megőrzött feladatleírás.',
    'agents.close.inner': 'Belső feladatok bezárása',
    'agents.open.inner': 'Belső feladatok megnyitása',
    'agents.below.root': 'Gyökér alatt',
    'agents.requested.ro': 'KÉRT READ-ONLY',
    'agents.requested.ro.title': 'A fő agent kifejezetten csak olvasási/ellenőrzési feladatra kérte ezt a mellékszálat.',
    'settings.heading': 'Beállítások',
    'settings.sub': 'Fiókok, terheléselosztás, fő agent, workerek és útválasztás',
    'settings.main.heading': 'Fő agent',
    'settings.main.sub': 'A Hermes az első modellen indul; ha annak a szolgáltatója kiesik, a többi sorban átveszi. Az első viszi az alapértelmezett routingot és a koordinálást (conductor) is.',
    'settings.main.primary': 'itt indul',
    'settings.main.external': 'A Hermes most a(z) {model} modellen indul, amit a routeren kívül állítottak be: az első elem itt csak az alapértelmezett routingot és a conductort állítja.',
    'settings.main.claude': 'A Hermes Claude-dal indul. A router alapértelmezett routingja és conductora marad: {tier}.',
    'settings.workers.heading': 'Workerek',
    'settings.workers.sub': 'Delegált agentek: melyik modellt kapják, ha a hívás nem nevez meg egyet, és hová mennek, ha az kiesik.',
    'settings.workers.codex': 'Codex workerek',
    'settings.workers.codex.desc': 'delegate_task, ha a cél nem nevez meg tiert',
    'settings.workers.claude': 'Claude workerek',
    'settings.workers.claude.desc': 'delegate_claude, ha a hívás nem nevez meg tiert',
    'settings.workers.fallback': 'Worker tartaléklánc',
    'settings.workers.fallback.desc': 'A célhoz rögzített levél soha nem örökli a fő agent láncát',
    'settings.accounts.heading': 'Fiókok',
    'settings.balance.heading': 'Terheléselosztás',
    'settings.balance.label': 'Terheléselosztás',
    'settings.balance.busy': 'ettől',
    'settings.balance.window.5-hour': '5 órás',
    'settings.balance.window.tighter': 'szűkebb (heti vagy 5 órás)',
    'settings.balance.margin': 'különbség',
    'settings.balance.desc': 'Kiegyenlíti a(z) {window} kereteket: ha a preferált fiók legalább {busy}%-on áll, és a másik legalább {margin} ponttal szabadabb, a szabadabb kerül előre. A szülő a saját fiókját terheli. A soft/hard limitek továbbra is érvényesek.',
    'account.state.open': 'nyitva',
    'account.state.soft': 'lágy korlát',
    'account.state.closed': 'lezárva',
    'account.state.unknown': 'ismeretlen',
    'account.models': 'Modellek',
    'account.usage': 'Használat',
    'account.usage.week': 'hét',
    'account.usage.session': '5 órás',
    'account.usage.resets': 'nullázódik',
    'account.usage.age': '{n} ezelőtt olvasva',
    'account.usage.none': 'ehhez a fiókhoz nincs használati adat',
    'account.usage.refresh': 'Frissítés',
    'main.usage.none': 'nincs használati adat',
    'account.limits': 'Korlátok',
    'account.limits.soft': 'lágy',
    'account.limits.hard': 'kemény',
    'account.limits.stepdown': 'visszalépés',
    'account.delegation': 'Delegálás',
    'account.delegation.via': '{tool} eszközzel',
    'account.delegation.always': 'mindig aktív (Hermes beépített útvonal)',
    'account.delegation.default_tier': 'alapértelmezett szint',
    'account.delegation.live': 'delegate_claude aktív',
    'account.delegation.restart': 'a Hermes újraindítása szükséges',
    'account.delegation.off': 'kikapcsolva — minden Claude-modell ki van kapcsolva',
    'account.load': 'Terhelés',
    'account.load.calls': '{n} hívás az utolsó {m} percben',
    'settings.cooling': 'hűl',
    'settings.load.title': 'Fogyás accountonként',
    'settings.load.window': 'utolsó {n} perc',
    'settings.load.idle': 'nincs hívás — ide érdemes önálló leafet adni',
    'settings.load.counts': 'hívásszám a route logból, nem kvótaadat',
    'settings.load.empty': 'Nincs routolt hívás az ablakban.',
    'settings.default.heading': 'Alapértelmezett modell (Orchestrator)',
    'settings.default.desc': 'Ezzel a modellel indul maga a Hermes, és ez számít orchestratornak. Az alábbi preferencia-lánc egy fordulót átirányíthat, ezt viszont nem írja felül.',
    'settings.effort.low': 'low',
    'settings.effort.medium': 'medium',
    'settings.effort.high': 'high',
    'settings.effort.xhigh': 'xhigh',
    'settings.claude_effort.haiku_no_reasoning': 'nincs gondolkodás',
    'settings.claude_effort.default': 'Alapértelmezett ({level})',
    'settings.prefs.heading': 'Útválasztás: preferált modellek munkatípusonként',
    'settings.prefs.sub': 'Sorrendben, a legjobb elöl. A router az első hívható elemet választja. Ez egy fordulót irányít át; a Hermes indulási modelljét nem változtatja meg.',
    'settings.prefs.add': 'modell hozzáadása...',
    'settings.prefs.empty': 'Nincs preferencia — a beépített útvonal érvényes.',
    'settings.prefs.note': 'A szürke elemet maga a router irányítja. A lila másik fiókon van, ezért delegációs ajánlásként kerül a karmesterhez.',
    'settings.prefs.up': 'előrébb',
    'settings.prefs.down': 'hátrébb',
    'settings.prefs.remove': 'eltávolítás',
    'kind.design': 'UI és vizuális tervezés',
    'kind.code': 'Kódírás',
    'kind.explore': 'Feltárás, csak olvasó vizsgálat',
    'kind.review': 'Review, véleményezés',
    'kind.sensitive': 'Biztonságérzékeny (auth, fizetés)',
    'kind.critical': 'Kritikus (deploy, migráció, produkció)',
    'kind.long': 'Hosszú kérések',
    'kind.chat': 'Rövid beszélgetés',
    'kind.default': 'Minden más',
    'settings.fb.file': 'Ez a ~/.hermes/config.yaml fájlt írja, nem a routerét — minden mentés előtt másolat készül róla.',
    'settings.fb.orchestrator': 'Orchestrator',
    'settings.fb.orchestrator.desc': 'A fő ügynök, amikor a saját szolgáltatója kifogyott',
    'settings.fb.children': 'Delegált munkások',
    'settings.fb.children.desc': 'A delegate_task munkásaira érvényes. A delegate_claude Claude-munkásai hiba esetén új delegálást kérnek a szülőtől; ezt a fallback-láncot nem használják.',
    'settings.fb.empty': 'Nincs tartalék — az elsődleges kifogyásakor a forduló elhal.',
    'settings.fb.inherit': 'A munkásoknál az üres lánc azt jelenti: nincs tartalék, nem azt, hogy örökli.',
    'settings.language': 'Nyelv',
    'settings.lang.en': 'English',
    'settings.lang.hu': 'Magyar',
    'model.desc.luna': 'Gyors, egyszerű feladatok',
    'model.desc.spark': 'Read-only delegált munkák',
    'model.desc.terra': 'Általános orchestrator',
    'model.desc.sol': 'Biztonságkritikus, design',
    'model.desc.opus5': 'Claude account — súlyosabb munka',
    'model.desc.sonnet5': 'Claude account — alapértelmezett',
    'model.desc.haiku': 'Claude account — gyors keresések',
    'model.desc.qwen': 'Alternatív modell',
    'model.desc.grok': 'SuperGrok fiók — nehéz implementáció',
    'settings.reload': 'Mentés előtt töltsd újra a beállításokat.',
    'settings.saving': 'Mentés...',
    'settings.saved': 'Mentve ✓',
    'settings.error.unknown': 'Ismeretlen hiba',
    'settings.error.prefix': 'Hiba: ',
    'settings.error.load': 'Nem sikerült betölteni:',
    'status.refreshing': 'Frissítés…',
    'status.loading': 'Betöltés…',
    'status.error.prefix': 'Hiba: ',
    'status.error.kept': ' · a korábbi lista megmaradt',
    'exec.below.root': 'Gyökér alatt',
    'exec.open.subtasks': 'Alfeladatok megnyitása',
    'exec.close.subtasks': 'Alfeladatok bezárása',
    'exec.no.desc': 'Nincs megőrzött feladatleírás.',
    'exec.opus.review': 'Opus review',
    'exec.internal.step': 'Belső router-lépés',
    'exec.read.only': 'READ-ONLY',
    'exec.requested.ro': 'KÉRT READ-ONLY',
    'internal.router.step': 'Belső router-lépés',
    'resizer.title': 'Húzd az oszlop szélességének módosításához · dupla kattintás: alaphelyzet',
    'play.running.agent': 'Éppen futó fő agent',
    'play.running.sub': 'ÉPPEN FUT ·',
    'status.refresh.short': 'Frissítés…',
    'status.kept': ' · a korábbi lista látszik',
    'status.rootprompts': 'root prompt',
    'status.locale': 'hu-HU',
    'total.routing.decisions': 'ÖSSZES ROUTING DÖNTÉS',
    'run.worker.routing': 'worker-routing',
    'agents.sum.running': 'fut',
    'agents.sum.completed': 'kész',
    'agents.sum.failed': 'hiba',
    'agents.main.agent': 'FŐ AGENT',
    'agents.none.active': 'Nincs aktív háttéragent.',
    'agents.pill.running': 'ÉPPEN FUT',
    'agents.pill.agent': 'AGENT',
    'state.running.short': 'FUT',
    'state.done.short': 'KÉSZ',
    'state.success.short': 'SIKER',
    'state.error.short': 'HIBA',
    'duration.h': 'ó',
    'duration.m': 'p',
    'duration.s': 'mp',
    'exec.no.route': 'nincs tárolt route-adat',
    'exec.external': 'KÜLSŐ',
    'exec.own': 'saját',
    'exec.total': 'összesített',
    'root.continuation.reason': 'Rendszerfolytatás · hívások közötti routing-döntések',
    'settings.error.save': 'Nem sikerült menteni:',
    'agents.unknown.model': 'ismeretlen modell',
    'play.running.subagent': 'Éppen futó mellékagent',
    'label.model.short': 'modell',
    'label.active.turn': 'Aktív turn: ',
    'exec.tree.label': 'Végrehajtási fa',
    'th.time.short': 'Idő',
  }
};
let currentLang = localStorage.getItem('model-router-lang') || 'en';
function t(key) { return (I18N[currentLang] && I18N[currentLang][key]) || (I18N.en[key]) || key; }
function applyLanguage() {
  document.documentElement.lang = currentLang === 'hu' ? 'hu' : 'en';
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    const attr = el.getAttribute('data-i18n-attr');
    const val = t(key);
    if (attr) { el.setAttribute(attr, val); }
    else { el.textContent = val; }
  });
  // Update placeholders separately
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    el.placeholder = t(el.getAttribute('data-i18n-placeholder'));
  });
  // Update titles
  document.querySelectorAll('[data-i18n-title]').forEach(el => {
    el.title = t(el.getAttribute('data-i18n-title'));
  });
  // Re-render settings if visible
  if (currentConfig) renderSettings();
  // Re-render dynamic content
  if (typeof render === 'function') { try { render(); } catch(e){} }
  if (typeof renderAgents === 'function') { try { renderAgents(agentActivity); } catch(e){} }
}
let entries=[];let selectedTab=localStorage.getItem('ai-home-lab-tab')||null;const expandedPrompts=new Set(),promptExpansionKey='model-router-expanded-prompts-v1';function persistedPromptExpansions(){try{return new Set(JSON.parse(localStorage.getItem(promptExpansionKey)||'[]'))}catch(e){return new Set()}}function isPromptExpanded(key){return expandedPrompts.has(key)||persistedPromptExpansions().has(key)}function setPromptExpanded(key,open){const persisted=persistedPromptExpansions();open?persisted.add(key):persisted.delete(key);localStorage.setItem(promptExpansionKey,JSON.stringify([...persisted]));open?expandedPrompts.add(key):expandedPrompts.delete(key)}const descriptionExpansionKey='model-router-description-expansion-v1';function descriptionExpansions(){try{return JSON.parse(localStorage.getItem(descriptionExpansionKey)||'{}')}catch(e){return {}}}function createDescriptionDetails(key,text){const node=document.createElement('details'),state=descriptionExpansions();node.className='agent-worker-goal-details';node.open=!!state[key];const summary=document.createElement('summary');summary.textContent=t('agents.task.desc');const body=document.createElement('div');body.className='agent-worker-goal';body.textContent=text||t('agents.no.desc');node.append(summary,body);node.addEventListener('toggle',()=>{const next=descriptionExpansions();next[key]=node.open;localStorage.setItem(descriptionExpansionKey,JSON.stringify(next))});return node}
function setTab(name,remember=true){if(name==='agents')name='router';selectedTab=name;if(remember)localStorage.setItem('ai-home-lab-tab',name);$('router-panel').hidden=name!=='router';$('settings-panel').hidden=name!=='settings';$('command-panel').hidden=name!=='command';document.querySelectorAll('.tab').forEach(tab=>tab.classList.toggle('active',tab.dataset.tab===name));if(name==='settings')loadSettings();}

// Settings management
let currentConfig=null;
let settingsSaveQueue=Promise.resolve(),settingsPending=0,settingsSaveFailed=false,settingsLoadGeneration=0;
async function loadSettings(){
  if(settingsPending)return;
  const generation=++settingsLoadGeneration;
  try{
    const response=await fetch('/api/config',{cache:'no-store'});
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const loaded=await response.json();
    if(generation!==settingsLoadGeneration||settingsPending)return;
    currentConfig=loaded;settingsSaveFailed=false;
    renderSettings();
  }catch(error){
    console.error(t('settings.error.load'),error);
    $('settings-status').textContent=t('settings.error.prefix')+error.message;
    $('settings-status').style.color='#ff6b7a';
  }
}

function renderSettings(){
  if(!currentConfig)return;
  const models=['luna','spark','terra','sol','opus5','sonnet5','haiku','qwen','grok'];
  const modelLabels={luna:t('card.luna'),spark:t('card.spark'),terra:t('card.terra'),sol:t('card.sol'),opus5:t('card.opus5'),sonnet5:t('card.sonnet5'),haiku:t('card.haiku'),qwen:t('card.qwen'),grok:t('card.grok')};
  const modelDescriptions={luna:t('model.desc.luna'),spark:t('model.desc.spark'),terra:t('model.desc.terra'),sol:t('model.desc.sol'),opus5:t('model.desc.opus5'),sonnet5:t('model.desc.sonnet5'),haiku:t('model.desc.haiku'),qwen:t('model.desc.qwen'),grok:t('model.desc.grok')};
  renderAccounts();
  renderBalance();
  renderPreferences(modelLabels);
  renderMainChain(modelLabels,models);
  renderWorkers(modelLabels);
  if(currentConfig.delegation_limits){const limits=currentConfig.delegation_limits,note=document.createElement('div');note.className='pref-note';note.textContent=t('settings.worker.limits').replace('{depth}',limits.max_spawn_depth??'—').replace('{children}',limits.max_concurrent_children??'—').replace('{iterations}',limits.max_iterations??'—');$('worker-settings').append(note)}
}
function ageText(seconds){if(seconds==null)return '';const m=Math.round(seconds/60);return t('account.usage.age').replace('{n}',m<60?`${m} min`:`${Math.round(m/60)} h`)}
function resetText(iso){if(!iso)return '';const d=new Date(iso);return Number.isNaN(d.getTime())?'':`${t('account.usage.resets')} ${d.toLocaleString(t('status.locale'),{weekday:'short',hour:'2-digit',minute:'2-digit'})}`}
function usageRow(labelKey,percent,resetsAt,soft,hard,overrideColor){const known=typeof percent==='number',width=known?Math.min(100,Math.max(0,percent)):0,color=overrideColor||(!known?'#8997ad':percent>=hard?'#ff6b7a':percent>=soft?'#ffb454':'#88e36f');return `<div class="usage-line"><span class="usage-label">${t(labelKey)}</span><div class="usage-bar"><div class="usage-fill" style="width:${width}%;background:${color}"></div>${soft?`<span class="usage-tick" style="left:${soft}%"></span>`:''}${hard?`<span class="usage-tick" style="left:${hard}%"></span>`:''}</div><span class="usage-value">${known?Math.round(percent)+'%':'—'}</span><span class="usage-reset">${resetText(resetsAt)}</span></div>`}
const ACCOUNT_ORDER=['openai-codex','anthropic','xai-oauth','qwen-token'],TIER_ORDER=['luna','spark','terra','sol','haiku','sonnet5','opus5','grok','qwen'];
// A tier switched off in Settings leaves the overview, and an account with none
// left leaves with it; its card in Settings stays, since that is where it comes back on.
function accountGroupsFor(tierAccounts,accounts,callable={}){const groups=new Map();for(const tier of TIER_ORDER){const a=tierAccounts[tier];if(!a||callable[tier]===false)continue;if(!groups.has(a))groups.set(a,[]);groups.get(a).push(tier)}return [...groups.entries()].sort((x,y)=>(ACCOUNT_ORDER.indexOf(x[0])+99)%99-(ACCOUNT_ORDER.indexOf(y[0])+99)%99).map(([account,tiers])=>({account,label:(accounts[account]||{}).label||account,tiers}))}
function compactUsage(info){if(!info||!info.has_usage_source)return `<div class="account-usage none">${t('main.usage.none')}</div>`;const u=info.usage||{};
  // M10: the bar's colour follows the account's actual state (which also
  // reacts to the 5-hour session window), not a colour recomputed from the
  // weekly percent alone -- those two can disagree.
  const stateColor={open:'#88e36f',soft:'#ffb454',closed:'#ff6b7a'}[info.state]||'#8997ad';
  const staleAfter=2*(info.cache_seconds||300),stale=info.usage_age_seconds!=null&&info.usage_age_seconds>staleAfter;
  return `<div class="account-usage ${info.state}${stale?' stale':''}" title="${t('account.usage.session')} ${u.session==null?'—':Math.round(u.session)+'%'} · ${ageText(info.usage_age_seconds)}">${usageRow('account.usage.week',u.weekly,u.weekly_resets_at,info.soft_percent,info.hard_percent,stateColor)}</div>`}
function tierFilterOptions(groups){return groups.map(g=>`<optgroup label="${g.label}">${g.tiers.map(x=>`<option>${x}</option>`).join('')}</optgroup>`).join('')}
let accountsState={accounts:{},tier_accounts:{},callable:{}};
function groupModelCards(){const host=$('account-groups');if(!host)return;const callable=accountsState.callable||{},groups=accountGroupsFor(accountsState.tier_accounts,accountsState.accounts,callable);for(const tier of TIER_ORDER){const card=document.querySelector(`.card.${tier}`);if(card)card.hidden=callable[tier]===false}const shown=new Set(groups.map(g=>g.account));for(const box of host.querySelectorAll('.account-group'))box.hidden=!shown.has(box.dataset.account);for(const g of groups){let box=host.querySelector(`.account-group[data-account="${g.account}"]`);if(!box){box=document.createElement('div');box.className=`account-group ${g.account}`;box.dataset.account=g.account;box.innerHTML=`<div class="account-group-head">${g.label}</div><div class="account-group-cards"></div><div class="account-group-usage"></div>`;host.append(box)}const cards=box.querySelector('.account-group-cards');for(const tier of g.tiers){const card=document.querySelector(`.cards .card.${tier}`)||host.querySelector(`.card.${tier}`);if(card&&card.parentElement!==cards)cards.append(card)}box.querySelector('.account-group-usage').innerHTML=compactUsage(accountsState.accounts[g.account])}}
function renderTierFilter(){const select=$('tier'),current=select.value,groups=accountGroupsFor(accountsState.tier_accounts,accountsState.accounts,accountsState.callable);select.innerHTML=`<option value="">${t('router.tier.all')}</option>`+tierFilterOptions(groups);select.value=current}
async function loadAccounts(){try{const r=await fetch('/api/config',{cache:'no-store'});if(!r.ok)return;const body=await r.json();accountsState={accounts:body.accounts||{},tier_accounts:body.tier_accounts||{},callable:body.callable||{}};groupModelCards();renderTierFilter()}catch(e){}}
// The card header already names the vendor, so the switches drop the repeated
// "Claude "/"GPT-" prefix: "GPT-5.6 Luna" -> "Luna 5.6", "Claude Opus 5.5" -> "Opus 5.5".
function shortModelName(label){const gpt=/^GPT-(\S+)\s+(.+)$/.exec(label);return gpt?`${gpt[2]} ${gpt[1]}`:label.replace(/^Claude\s+/,'')}
function escapeHtml(value){return String(value??'').replace(/[&<>'"]/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]))}
const ROUTER_EFFORT_TIERS=['luna','spark','terra','sol','grok'];
function renderEffort(tiers,off=false){const effort=currentConfig.effort||{},levels=['low','medium','high','xhigh'];
  return tiers.filter(tier=>ROUTER_EFFORT_TIERS.includes(tier)).map(tier=>`<select data-effort="${tier}"${off?' disabled':''}>${levels.map(level=>`<option value="${level}"${effort[tier]===level?' selected':''}>${t('settings.effort.'+level)}</option>`).join('')}</select>`).join('')}
function renderClaudeReasoningEffort(tiers,off=false){const state=currentConfig.claude_reasoning_effort||{available:false,levels:{}},levels=state.levels||{},defaults=state.defaults||{},pinned=state.pinned||{},options=['low','medium','high','xhigh'],disabled=state.available&&!off?'':' disabled',title=state.available?'':` title="${escapeHtml(state.reason||'')}"`;
  return tiers.map(model=>{const tier={sonnet5:'sonnet',opus5:'opus'}[model];
    if(!tier)return model==='haiku'?`<select class="effort-none" disabled><option>${t('settings.claude_effort.haiku_no_reasoning')}</option></select>`:'';
    const isPinned=!!pinned[tier],defaultLabel=t('settings.claude_effort.default').replace('{level}',t('settings.effort.'+(defaults[tier]||levels[tier]||'medium')));
    return `<select data-claude-effort="${tier}"${disabled}${title}><option value=""${isPinned?'':' selected'}>${defaultLabel}</option>${options.map(level=>`<option value="${level}"${isPinned&&levels[tier]===level?' selected':''}>${t('settings.effort.'+level)}</option>`).join('')}</select>`;
  }).join('')}
function usageError(account){const message=((currentConfig||{}).usage_errors||{})[account];return message?`<span class="usage-error">${escapeHtml(message)}</span>`:''}
function accountCard(account,info){const callable=currentConfig.callable||{},cooldowns=currentConfig.cooldowns||{},load=(currentConfig.load||{})[account]||0,d=info.delegation||{};
  const loadText=`<span class="usage-load">${t('account.load')}: ${t('account.load.calls').replace('{n}',load).replace('{m}',currentConfig.window_minutes||60)}</span>`;
  const hasEffort=info.tiers.some(m=>ROUTER_EFFORT_TIERS.includes(m))||account==='anthropic';
  const models=info.tiers.map(m=>{const on=callable[m]!==false,cool=cooldowns[m],effort=ROUTER_EFFORT_TIERS.includes(m)?renderEffort([m],!on):account==='anthropic'?renderClaudeReasoningEffort([m],!on):'';return `<div class="model-switch${on?'':' disabled'}${cool?' cooling':''}"><label class="switch"><input type="checkbox" data-model="${m}" ${on?'checked':''}><span class="slider"></span></label><span title="${t('card.'+m)}">${shortModelName(t('card.'+m))}</span>${effort}${cool?`<span class="cooldown-pill">${t('settings.cooling')} · ${Math.ceil(cool.seconds/60)}m${cool.reason?` · ${escapeHtml(cool.reason)}`:''}</span>`:''}</div>`}).join('');
  const modelControls=`<div class="account-row models"><span>${t('account.models')}</span><div class="model-list${hasEffort?'':' no-effort'}">${models}</div></div>`;
  const u=info.usage,usage=info.has_usage_source?(u?usageRow('account.usage.week',u.weekly,u.weekly_resets_at,info.soft_percent,info.hard_percent)+usageRow('account.usage.session',u.session,u.session_resets_at,info.soft_percent,info.hard_percent)+`<div class="usage-age">${ageText(info.usage_age_seconds)} <button type="button" data-refresh-usage="${account}">${t('account.usage.refresh')}</button>${usageError(account)}${loadText}</div>`:`<div class="usage-age">${t('account.state.unknown')} <button type="button" data-refresh-usage="${account}">${t('account.usage.refresh')}</button>${usageError(account)}${loadText}</div>`):`<div class="usage-none">${t('account.usage.none')}</div><div class="usage-age">${loadText}</div>`;
  const off=info.guard?'':'disabled',step=Object.entries(info.step_down||{}).map(([a,b])=>`${a} → ${b}`).join(', ');
  const limits=`<label>${t('account.limits.soft')} <input type="number" min="1" max="99" data-limit="soft" data-account="${account}" value="${info.soft_percent??''}" ${off}>%</label> <label>${t('account.limits.hard')} <input type="number" min="2" max="100" data-limit="hard" data-account="${account}" value="${info.hard_percent??''}" ${off}>%</label>${step?`<div class="stepdown">${t('account.limits.stepdown')}: ${step}</div>`:''}`;
  // delegate_claude is on while a Claude model is switched on; the row only reports it.
  // The default tier is chosen under Workers, next to the other worker defaults.
  const delegation=d.tool==='delegate_claude'?`${d.enabled?t('account.delegation.via').replace('{tool}','delegate_claude'):t('account.delegation.off')}`:`${t('account.delegation.via').replace('{tool}',d.tool||'delegate_task')} · ${t('account.delegation.always')}`;
  // M7: the live/restart badge moved out of the Delegation row and into the
  // card header, next to the state badge -- the other at-a-glance status.
  const liveBadge=d.tool!=='delegate_claude'?'':d.restart_needed?` <span class="restart">${t('account.delegation.restart')}</span>`:(d.enabled&&d.registered?` <span class="live">${t('account.delegation.live')}</span>`:'');
  // M10: staleness follows 2x whatever cache_seconds the accounts payload
  // actually served, not a hardcoded ten minutes.
  const stale=info.usage_age_seconds!=null&&info.usage_age_seconds>2*(info.cache_seconds||300);
  return `<div class="account-card ${account}${stale?' stale':''}"><div class="account-head"><b>${info.label}</b> <span class="state-badge ${info.state}">${t('account.state.'+info.state)}</span>${liveBadge}</div>`
    +`<div class="account-grid">`
    +modelControls
    +`<div class="account-row limits"><span>${t('account.limits')}</span><div>${limits}</div></div>`
    +`<div class="account-row delegation"><span>${t('account.delegation')}</span><div>${delegation}</div></div>`
    +`<div class="account-row usage"><span>${t('account.usage')}</span><div>${usage}</div></div>`
    +`</div></div>`}
// Load balancing picks between accounts, so it needs two with a model switched on.
function balanceAccounts(accounts,callable){return Object.values(accounts||{}).filter(info=>(info.tiers||[]).some(m=>callable[m]!==false)).length}
function balanceControl(balance){if(!balance)return '';
  return `<div class="balance-row"><label class="switch"><input type="checkbox" data-balance-toggle${balance.enabled?' checked':''}><span class="slider"></span></label><b>${t('settings.balance.label')}</b>`
    +`<label class="balance-field">${t('settings.balance.busy')} <input type="number" min="0" max="100" data-balance-field="busy_percent" value="${balance.busy_percent}">%</label>`
    +`<label class="balance-field">${t('settings.balance.margin')} <input type="number" min="1" max="100" data-balance-field="margin_percent" value="${balance.margin_percent}"></label>`
    +`<span class="pref-note">${t('settings.balance.desc').replace('{window}',t('settings.balance.window.'+(balance.window||'5-hour'))).replace('{busy}',balance.busy_percent).replace('{margin}',balance.margin_percent)}</span></div>`}
function renderBalance(){const section=$('balance-section');if(!section)return;
  const shown=!!currentConfig.balance&&balanceAccounts(currentConfig.accounts,currentConfig.callable||{})>=2;
  section.hidden=!shown;$('balance-settings').innerHTML=shown?balanceControl(currentConfig.balance):''}
function renderAccounts(){const box=$('account-cards');if(!box)return;const accounts=currentConfig.accounts||{};box.innerHTML=Object.entries(accounts).map(([a,info])=>accountCard(a,info)).join('')}
async function refreshUsage(account){const errors=currentConfig.usage_errors=currentConfig.usage_errors||{};try{const r=await fetch(`/api/usage/refresh?account=${encodeURIComponent(account)}`,{method:'POST'});const body=await r.json().catch(()=>({}));if(r.ok){delete errors[account];currentConfig.accounts[account]=body.account}else errors[account]=body.error||`HTTP ${r.status}`}catch(e){errors[account]=String(e.message||e)}renderAccounts()}
const defaultWidths=[55,110,90,350,90,230,480],widthStore='model-router-column-widths-v3';
function saveWidths(table,cols){localStorage.setItem(widthStore,JSON.stringify({columns:cols.map(c=>parseFloat(c.style.width)),table:parseFloat(table.style.width)}))}
function initColumnResize(){const table=$('log-table'),cols=[...table.querySelectorAll('col')],heads=[...table.querySelectorAll('th')];let saved=null;try{saved=JSON.parse(localStorage.getItem(widthStore))}catch(e){}
 if(saved?.columns?.length===cols.length){saved.columns.forEach((w,i)=>cols[i].style.width=`${Math.max(55,w)}px`);table.style.width=`${Math.max(900,saved.table||saved.columns.reduce((a,b)=>a+b,0))}px`}
 heads.forEach((head,i)=>{const grip=document.createElement('div');grip.className='resizer';grip.title=t('resizer.title');head.append(grip);
  grip.addEventListener('pointerdown',e=>{e.preventDefault();grip.setPointerCapture(e.pointerId);document.body.classList.add('resizing');grip.classList.add('dragging');const startX=e.clientX,startWidth=cols[i].getBoundingClientRect().width,startTable=table.getBoundingClientRect().width;
   const move=ev=>{const width=Math.max(55,startWidth+ev.clientX-startX),delta=width-startWidth;cols[i].style.width=`${width}px`;table.style.width=`${Math.max(900,startTable+delta)}px`};
   const up=()=>{document.body.classList.remove('resizing');grip.classList.remove('dragging');grip.removeEventListener('pointermove',move);grip.removeEventListener('pointerup',up);grip.removeEventListener('pointercancel',up);saveWidths(table,cols)};
   grip.addEventListener('pointermove',move);grip.addEventListener('pointerup',up);grip.addEventListener('pointercancel',up)});
  grip.addEventListener('dblclick',()=>{cols[i].style.width=`${defaultWidths[i]}px`;table.style.width=`${defaultWidths.reduce((a,b)=>a+b,0)}px`;saveWidths(table,cols)})})}
function escText(el,text){el.textContent=text??''}
const cetFormatter=new Intl.DateTimeFormat('hu-HU',{timeZone:'Europe/Budapest',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23'});
function dateAndTime(value){const date=new Date(value);if(Number.isNaN(date.getTime()))return ['-','-'];const parts=Object.fromEntries(cetFormatter.formatToParts(date).filter(p=>p.type!=='literal').map(p=>[p.type,p.value]));return [`${parts.year}-${parts.month}-${parts.day}`,`${parts.hour}:${parts.minute}:${parts.second}`]}
function compact(vals){let out=[];for(const v of vals){let last=out[out.length-1];if(last&&last.v===v)last.n++;else out.push({v,n:1})}return out}
function filtered(){const t=$('tier').value,q=$('search').value.toLowerCase();return entries.filter(e=>(!t||e.tier===t)&&(!q||`${e.prompt_preview} ${e.turn_id} ${e.reason} ${e.model} ${(e.vetoed_by||[]).map(t=>'veto:'+t).join(' ')}`.toLowerCase().includes(q)))}
function searchFiltered(){const q=$('search').value.toLowerCase();return entries.filter(e=>!q||`${e.prompt_preview} ${e.turn_id} ${e.reason} ${e.model} ${(e.vetoed_by||[]).map(t=>'veto:'+t).join(' ')}`.toLowerCase().includes(q))}
function promptKey(e){return `${sessionIdFromTurn(e)}::${e.prompt_preview||`__missing__:${e.turn_id}`}`}
function promptGroups(list){const grouped=new Map();for(const e of list){const key=promptKey(e);if(!grouped.has(key))grouped.set(key,[]);grouped.get(key).push(e)}return [...grouped.values()]}function groups(list){return $('grouped').checked?promptGroups(list):list.map(e=>[e])}
function togglePrompt(key){setPromptExpanded(key,!isPromptExpanded(key));render()}
function routeKey(e){return e.effort?`${e.tier} · ${e.effort}`:(e.tier||'?')}function routeSummary(items){const counts=new Map();for(const item of items){const key=routeKey(item);counts.set(key,(counts.get(key)||0)+1)}return [...counts].map(([key,count])=>`${key} ×${count}`).join(' · ')}
function pill(t,label=t){const s=document.createElement('span');s.className=`pill ${t}`;s.textContent=label;return s}
function appendCell(row,text,className=''){const td=document.createElement('td');if(className)td.className=className;escText(td,text);row.append(td);return td}
function detailsTable(group){const table=document.createElement('table');table.className='details-table';const colgroup=document.createElement('colgroup');for(const width of ['110px','90px','34%','80px','150px','auto']){const col=document.createElement('col');col.style.width=width;colgroup.append(col)}table.append(colgroup);const head=document.createElement('thead'),headRow=document.createElement('tr');for(const key of ['th.date','th.time.short','th.prompt','th.calls','th.route','th.reason']){const th=document.createElement('th');if(key==='th.calls')th.className='calls';escText(th,t(key));headRow.append(th)}head.append(headRow);table.append(head);const tbody=document.createElement('tbody');for(const [index,item] of group.entries()){const row=document.createElement('tr'),[date,time]=dateAndTime(item.timestamp);appendCell(row,date);appendCell(row,time);appendCell(row,item.prompt_preview||t('not.recoverable'),'prompt-text');appendCell(row,item.api_call_count??index+1,'calls');const route=appendCell(row,'','route');route.append(pill(item.tier||'?',routeKey(item)));appendCell(row,item.reason||'?','reason');tbody.append(row)}table.append(tbody);return table}
function setWordWrap(){const enabled=$('word-wrap').checked;$('log-table').classList.toggle('word-wrap',enabled);localStorage.setItem('model-router-word-wrap',enabled?'1':'0')}


// Settings event listeners
for(const id of ['main-chain','worker-settings']){
  document.getElementById(id).addEventListener('click',(e)=>{
    const btn=e.target.closest('button[data-act]');
    if(!btn||!currentConfig)return;
    mutateFallback(btn.dataset.fb,Number(btn.dataset.index),btn.dataset.act);
  });
  document.getElementById(id).addEventListener('change',async(e)=>{
    if(!currentConfig)return;const el=e.target;
    if(el.matches('select.pref-add')){
      if(!el.value)return;
      const [provider,model]=el.value.split('|');
      const chains=currentConfig.hermes_fallback=currentConfig.hermes_fallback||{};
      chains[el.dataset.fb]=(chains[el.dataset.fb]||[]).concat([{provider,model}]);
      renderSettings();await saveSettings();return;
    }
    // A pick here moves Hermes's own parent, whichever account it is on; posted
    // once, with this save only, so no unrelated save can move the parent.
    if(el.id==='default-model-select'){if(!el.value)return;currentConfig.main_parent=el.value;if((currentConfig.routable||[]).includes(el.value))currentConfig.default_model=el.value}
    else if(el.matches('[data-worker-model]'))currentConfig.worker_model=Object.assign({},currentConfig.worker_model,{tier:el.value});
    else if(el.dataset.defaultTier)currentConfig.accounts[el.dataset.defaultTier].delegation.default_tier=el.value;
    else return;
    // Reload: a new primary moves Hermes's parent and leaves its own fallbacks.
    await saveSettings();await loadSettings();
  });
}

document.getElementById('pref-kinds').addEventListener('click',(e)=>{
  const btn=e.target.closest('button[data-act]');
  if(!btn||!currentConfig)return;
  mutatePreference(btn.dataset.kind,Number(btn.dataset.index),btn.dataset.act);
});

document.getElementById('pref-kinds').addEventListener('change',(e)=>{
  const select=e.target.closest('select.pref-add');
  if(!select||!select.value||!currentConfig)return;
  const kind=select.dataset.kind;
  const prefs=currentConfig.preferences=currentConfig.preferences||{};
  prefs[kind]=(prefs[kind]||[]).concat([select.value]);
  renderSettings();
  saveSettings();
});

document.getElementById('account-cards').addEventListener('change',async(e)=>{if(!currentConfig)return;const el=e.target;
  if(el.dataset.model){currentConfig.callable[el.dataset.model]=el.checked}
  else if(el.dataset.limit){const info=currentConfig.accounts[el.dataset.account];info[el.dataset.limit==='soft'?'soft_percent':'hard_percent']=Number(el.value)}
  else if(el.dataset.effort){currentConfig.effort=Object.assign({},currentConfig.effort,{[el.dataset.effort]:el.value})}
  else if(el.dataset.claudeEffort){const state=currentConfig.claude_reasoning_effort||{levels:{}};currentConfig.claude_reasoning_effort=Object.assign({},state,{levels:Object.assign({},state.levels,{[el.dataset.claudeEffort]:el.value})})}
  else return;await saveSettings();await loadSettings()});
document.getElementById('balance-settings').addEventListener('change',async(e)=>{if(!currentConfig)return;const el=e.target;
  if(el.matches('[data-balance-toggle]'))currentConfig.balance=Object.assign({},currentConfig.balance,{enabled:el.checked});
  else if(el.dataset.balanceField)currentConfig.balance=Object.assign({},currentConfig.balance,{[el.dataset.balanceField]:Number(el.value)});
  else return;await saveSettings();await loadSettings()});
document.getElementById('account-cards').addEventListener('click',e=>{const b=e.target.closest('[data-refresh-usage]');if(b)refreshUsage(b.dataset.refreshUsage)});
document.getElementById('language-select').addEventListener('change',async(e)=>{
  currentLang=e.target.value;
  localStorage.setItem('model-router-lang',currentLang);
  applyLanguage();
  renderTierFilter();groupModelCards();
  // applyLanguage only walks [data-i18n] nodes; without this the tables and
  // agent cards the renderers already built stay in the previous language.
  if(typeof render==='function')render();
  if(typeof renderAgents==='function'&&typeof agentActivity!=='undefined')renderAgents(agentActivity);
});
// Initialize language select value on load
document.getElementById('language-select').value=currentLang;
function chainChipAccount(model){
  const account=(currentConfig.tier_accounts||{})[model];
  if(!account)return null;
  const info=(currentConfig.accounts||{})[account]||{};
  const state=info.state;
  return {account,label:info.label||account,mark:(state==='soft'||state==='closed')?t('account.state.'+state):''};
}
function renderPreferences(modelLabels){
  const box=$('pref-kinds');
  if(!box)return;
  const kinds=currentConfig.work_kinds||[];
  const prefs=currentConfig.preferences||{};
  const routable=currentConfig.routable||[];
  const callable=currentConfig.callable||{};
  // Only offer what is switched on: a chain entry that can never be called is a
  // setting that silently does nothing.
  const available=Object.keys(modelLabels).filter(m=>callable[m]!==false);
  box.innerHTML='';
  for(const kind of kinds){
    const chain=prefs[kind]||[];
    const row=document.createElement('div');
    row.className='pref-kind';
    const chips=chain.map((m,i)=>{
      const external=!routable.includes(m);
      const ca=chainChipAccount(m);
      const accountClass=ca?` ${ca.account}`:'';
      const accountBits=ca
        ?`<span class="chain-account">${ca.label}</span>${ca.mark?`<span class="chain-account-mark">${ca.mark}</span>`:''}`
        :'';
      return `<span class="pref-chip${external?' external':''}${accountClass}">`
        +`<span class="rank">${i+1}.</span>${modelLabels[m]||m}${accountBits}`
        +`<button data-kind="${kind}" data-index="${i}" data-act="up" title="${t('settings.prefs.up')}">&#9650;</button>`
        +`<button data-kind="${kind}" data-index="${i}" data-act="down" title="${t('settings.prefs.down')}">&#9660;</button>`
        +`<button data-kind="${kind}" data-index="${i}" data-act="del" title="${t('settings.prefs.remove')}">&times;</button>`
        +`</span>`;
    }).join('');
    const options=available.filter(m=>!chain.includes(m))
      .map(m=>`<option value="${m}">${modelLabels[m]||m}</option>`).join('');
    row.innerHTML=`<div class="pref-kind-head"><div>`
      +`<div class="pref-kind-name">${t('kind.'+kind)}</div>`
      +`<div class="pref-kind-desc">${kind}</div></div></div>`
      +`<div class="pref-chain">${chips||`<span class="pref-empty">${t('settings.prefs.empty')}</span>`}`
      +(options?`<select class="pref-add" data-kind="${kind}"><option value="">${t('settings.prefs.add')}</option>${options}</select>`:'')
      +`</div>`;
    box.appendChild(row);
  }
  const note=document.createElement('div');
  note.className='pref-note';
  note.textContent=t('settings.prefs.note');
  box.appendChild(note);
}

function fallbackOptionLabel(o,modelLabels){return o.key&&modelLabels[o.key]?modelLabels[o.key]:`${o.key||o.provider} · ${o.model}`}
// One fallback chip per entry; `rankFrom` lets the main chain start at 2, after its primary.
// Hermes's fallback chains do not consult the router's switches, so a route
// switched off here is dropped from them (the server does the same on save)
// rather than left in place where Hermes would still fail over onto it.
function fallbackRouteOff(e){const o=(currentConfig.fallback_options||[]).find(x=>x.provider===e.provider&&x.model===e.model);return !!(o&&o.key&&(currentConfig.callable||{})[o.key]===false)}
function fallbackChips(which,chain,modelLabels,rankFrom){
  const options=currentConfig.fallback_options||[];
  return chain.map((e,i)=>{
    const known=options.find(o=>o.provider===e.provider&&o.model===e.model);
    return `<span class="pref-chip external">`
      +`<span class="rank">${i+rankFrom}.</span>${known?fallbackOptionLabel(known,modelLabels):e.provider+'/'+e.model}`
      +`<button data-fb="${which}" data-index="${i}" data-act="up" title="${t('settings.prefs.up')}">&#9650;</button>`
      +`<button data-fb="${which}" data-index="${i}" data-act="down" title="${t('settings.prefs.down')}">&#9660;</button>`
      +`<button data-fb="${which}" data-index="${i}" data-act="del" title="${t('settings.prefs.remove')}">&times;</button>`
      +`</span>`}).join('')}
function fallbackPicker(which,chain,modelLabels,exclude){
  const free=(currentConfig.fallback_options||[]).filter(o=>!fallbackRouteOff(o)&&!chain.some(e=>e.provider===o.provider&&e.model===o.model)&&!(exclude&&exclude.provider===o.provider&&exclude.model===o.model));
  return free.length?`<select class="pref-add" data-fb="${which}"><option value="">${t('settings.prefs.add')}</option>`
    +free.map(o=>`<option value="${o.provider}|${o.model}">${fallbackOptionLabel(o,modelLabels)}</option>`).join('')+`</select>`:''}
function chainRow(name,desc,body){return `<div class="pref-kind">${name?`<div class="pref-kind-head"><div><div class="pref-kind-name">${name}</div>${desc?`<div class="pref-kind-desc">${desc}</div>`:''}</div></div>`:''}<div class="pref-chain">${body}</div></div>`}
// The pair the main agent starts on: Hermes's own parent when this router serves
// it, else the router tier picked as first entry.
function mainPrimary(){const parent=currentConfig.hermes_parent||{};if(parent.model&&!parent.router_model)return parent;const o=(currentConfig.fallback_options||[]).find(x=>x.key===currentConfig.default_model);return o?{provider:o.provider,model:o.model}:parent}
// Main agent: ONE chain. Entry 1 is the model Hermes starts on; the rest is
// Hermes's fallback_providers. It used to be two settings, and the primary kept
// showing up as its own fallback. Entry 1 offers the router's own tiers (which
// also become default_model: the default route and the conductor) and the
// switched-on Claude models (Hermes's anthropic provider; the router's default
// route and conductor stay where they are).
function renderMainChain(modelLabels,models){
  const box=$('main-chain');if(!box)return;
  const chains=currentConfig.hermes_fallback=currentConfig.hermes_fallback||{};
  const primary=mainPrimary(),parent=currentConfig.hermes_parent||{};
  chains.orchestrator=(chains.orchestrator||[]).filter(e=>!(e.provider===primary.provider&&e.model===primary.model)&&!fallbackRouteOff(e));
  const chain=chains.orchestrator,defaultModel=currentConfig.default_model||'terra',callable=currentConfig.callable||{};
  // Only a routable tier can start Hermes as a router tier; a delegation-only target has
  // no model entry in the router and raises on the first decision. A switched-off tier
  // is not offered, except the current default, which must stay visible.
  const orchestrators=(currentConfig.routable||[]).filter(m=>models.includes(m)&&(callable[m]!==false||m===defaultModel));
  const claude=currentConfig.parent_options||[];
  const external=!!(parent.model&&!parent.router_model);
  const claudeParent=external?claude.find(o=>o.provider===parent.provider&&o.model===parent.model):null;
  const selected=external?(claudeParent?claudeParent.key:''):defaultModel;
  const first=`<span class="pref-chip primary"><span class="rank">1.</span><select id="default-model-select">`
    +(external&&!claudeParent?`<option value="" selected disabled>${escapeHtml(parent.model)}</option>`:'')
    +(orchestrators.length?orchestrators:models).map(m=>`<option value="${m}"${m===selected?' selected':''}>${modelLabels[m]}</option>`).join('')
    +claude.map(o=>`<option value="${o.key}"${o.key===selected?' selected':''}>${modelLabels[o.key]||o.key}</option>`).join('')
    +`</select><span class="chain-account">${t('settings.main.primary')}</span></span>`;
  box.innerHTML=chainRow('','',first+fallbackChips('orchestrator',chain,modelLabels,2)+fallbackPicker('orchestrator',chain,modelLabels,primary))
    +(external&&!claudeParent?`<div class="pref-note">${t('settings.main.external').replace('{model}',escapeHtml(parent.model))}</div>`
      :claudeParent?`<div class="pref-note">${t('settings.main.claude').replace('{tier}',modelLabels[defaultModel]||defaultModel)}</div>`:'')}
function renderWorkers(modelLabels){
  const box=$('worker-settings');if(!box)return;
  const worker=currentConfig.worker_model||{},options=worker.options||[];
  const codex=`<select data-worker-model>${worker.tier?'':`<option value="" selected>${worker.model||'—'}</option>`}`
    +options.map(m=>`<option value="${m}"${m===worker.tier?' selected':''}>${modelLabels[m]||m}</option>`).join('')+`</select>`;
  const d=((currentConfig.accounts||{}).anthropic||{}).delegation||{};
  const claude=d.tool!=='delegate_claude'?`<span class="pref-empty">—</span>`:d.enabled
    ?`<select data-default-tier="anthropic">${(d.tiers||[]).map(x=>`<option value="${x}" ${x===d.default_tier?'selected':''}>${x}</option>`).join('')}</select>`
    :`<span class="pref-empty">${t('account.delegation.off')}</span>`;
  const fallbacks=currentConfig.hermes_fallback=currentConfig.hermes_fallback||{};
  fallbacks.children=(fallbacks.children||[]).filter(e=>!fallbackRouteOff(e));
  const chain=fallbacks.children;
  box.innerHTML=chainRow(t('settings.workers.codex'),t('settings.workers.codex.desc'),codex)
    +chainRow(t('settings.workers.claude'),t('settings.workers.claude.desc'),claude)
    +chainRow(t('settings.workers.fallback'),t('settings.workers.fallback.desc'),(fallbackChips('children',chain,modelLabels,1)||`<span class="pref-empty">${t('settings.fb.empty')}</span>`)+fallbackPicker('children',chain,modelLabels))
    +`<div class="pref-note">${t('settings.fb.inherit')}</div>`}

function mutateFallback(which,index,act){
  const chains=currentConfig.hermes_fallback=currentConfig.hermes_fallback||{};
  const chain=(chains[which]||[]).slice();
  if(act==='del')chain.splice(index,1);
  else if(act==='up'&&index>0){[chain[index-1],chain[index]]=[chain[index],chain[index-1]];}
  else if(act==='down'&&index<chain.length-1){[chain[index],chain[index+1]]=[chain[index+1],chain[index]];}
  else return;
  // Kept even when empty: for a delegated worker [] means "no fallback", which is a
  // different instruction from the key being absent.
  chains[which]=chain;
  renderSettings();
  saveSettings();
}

function mutatePreference(kind,index,act){
  const prefs=currentConfig.preferences=currentConfig.preferences||{};
  const chain=(prefs[kind]||[]).slice();
  if(act==='del')chain.splice(index,1);
  else if(act==='up'&&index>0){[chain[index-1],chain[index]]=[chain[index],chain[index-1]];}
  else if(act==='down'&&index<chain.length-1){[chain[index],chain[index+1]]=[chain[index+1],chain[index]];}
  else return;
  // An emptied chain means "no preference": drop the key so the router keeps its
  // built-in route rather than seeing an empty list it would have to interpret.
  if(chain.length)prefs[kind]=chain; else delete prefs[kind];
  renderSettings();
  saveSettings();
}

function saveSettings(){
  if(!currentConfig)return Promise.resolve();
  const claudeReasoning=currentConfig.claude_reasoning_effort||{};
  const payload=JSON.parse(JSON.stringify({callable:currentConfig.callable,balance:currentConfig.balance?{enabled:!!currentConfig.balance.enabled,busy_percent:currentConfig.balance.busy_percent,margin_percent:currentConfig.balance.margin_percent}:undefined,default_model:currentConfig.default_model,main_parent:currentConfig.main_parent||undefined,effort:currentConfig.effort||{},claude_reasoning_effort:claudeReasoning.available?claudeReasoning.levels||{}:undefined,worker_model:(currentConfig.worker_model||{}).tier||undefined,preferences:currentConfig.preferences||{},hermes_fallback:currentConfig.hermes_fallback||{},usage_limits:Object.fromEntries(Object.entries(currentConfig.accounts||{}).filter(([,i])=>i.guard).map(([a,i])=>[a,{soft_percent:i.soft_percent,hard_percent:i.hard_percent}])),claude_delegation:(currentConfig.accounts||{}).anthropic?{default_tier:currentConfig.accounts.anthropic.delegation.default_tier}:undefined}));
  delete currentConfig.main_parent;
  settingsPending++;settingsLoadGeneration++;
  settingsSaveQueue=settingsSaveQueue.then(async()=>{
    const statusEl=$('settings-status');
    try{
      // A prior write failed. Keep its useful error visible until a fresh GET
      // supplies a new revision; queued snapshots must not send stale writes.
      if(settingsSaveFailed)return;
      statusEl.textContent=t('settings.saving');statusEl.style.color='#c4b5fd';
      payload.revision=currentConfig.revision;
      const response=await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      const result=await response.json();
      if(!response.ok||!result.success)throw new Error(result.error||`HTTP ${response.status}`);
      currentConfig.revision=result.revision;
      statusEl.textContent=t('settings.saved');statusEl.style.color='#88e36f';
    }catch(error){
      settingsSaveFailed=true;
      statusEl.textContent=t('settings.error.prefix')+error.message+' '+t('settings.reload');
      statusEl.style.color='#ff6b7a';
      console.error(t('settings.error.save'),error);
    }finally{settingsPending--;}
  });
  return settingsSaveQueue;
}

function formatDuration(seconds){seconds=Math.max(0,Number(seconds)||0);if(seconds<60)return `${seconds} mp`;const minutes=Math.floor(seconds/60),rest=seconds%60;return minutes<60?`${minutes} ${t('duration.m')} ${rest} ${t('duration.s')}`:`${Math.floor(minutes/60)} ${t('duration.h')} ${minutes%60} ${t('duration.m')}`}
function agentRunParts(timestamp){if(!timestamp)return ['—','—'];return dateAndTime(new Date(Number(timestamp)*1000).toISOString())}
const agentExpansionKey='ai-home-lab-agent-main-expansion-v1';function loadAgentExpansion(){try{return JSON.parse(localStorage.getItem(agentExpansionKey)||'{}')}catch(e){return {}}}function saveAgentExpansion(state){localStorage.setItem(agentExpansionKey,JSON.stringify(state))}
function renderAgents(activity){const s=activity.summary||{},tree=$('agent-tree'),allParents=activity.parents||[],parents=allParents.slice(0,12),expansion=loadAgentExpansion();$('agent-summary').textContent=`${s.running||0} ${t('agents.sum.running')} · ${s.completed||0} ${t('agents.sum.completed')} · ${s.failed||0} ${t('agents.sum.failed')} · ${parents.length}/${allParents.length} ${t('agents.recent')}`;tree.replaceChildren();if(!parents.length){const empty=document.createElement('div');empty.className='agent-empty';empty.textContent=t('agents.none');tree.append(empty);return}for(const parent of parents){const running=(parent.children||[]).some(child=>child.state==='running'),storedOpen=Object.hasOwn(expansion,parent.session_id)&&!!expansion[parent.session_id],open=running||storedOpen;const card=document.createElement('article');card.className=`agent-main-card ${open?'is-open':'is-closed'}`;const header=document.createElement('button');header.type='button';header.className='agent-main-toggle';header.setAttribute('aria-expanded',String(open));const [runDate,runTime]=agentRunParts(parent.started_at);const date=document.createElement('span');date.className='agent-main-date';date.textContent=runDate;const time=document.createElement('span');time.className='agent-main-time';time.textContent=runTime;const heading=document.createElement('div');heading.className='agent-main-heading';const preview=document.createElement('span');preview.className='agent-main-preview';preview.textContent=parent.prompt||t('agents.prev.task');heading.append(preview);const stats=document.createElement('div');stats.className='agent-main-stats';const count=document.createElement('span');count.className='agent-stat';count.textContent=`${(parent.children||[]).length} ${t('agents.subtasks')}`;const status=document.createElement('span');status.className=`agent-stat ${running?'running':''}`;status.textContent=running?t('agents.running'):t('agents.done');const chevron=document.createElement('span');chevron.className='agent-chevron';chevron.textContent=open?'⌃':'⌄';stats.append(count,status,chevron);header.append(date,time,heading,stats);header.addEventListener('click',()=>{const next=loadAgentExpansion();next[parent.session_id]=!open;saveAgentExpansion(next);renderAgents(activity)});card.append(header);const detail=document.createElement('div');detail.className='agent-main-detail';detail.hidden=!open;const prompt=document.createElement('div');prompt.className='agent-full-prompt';prompt.textContent=parent.prompt||t('agents.prev.task');detail.append(prompt);const workers=document.createElement('div');workers.className='agent-worker-grid';for(const child of parent.children||[]){const worker=document.createElement('div');worker.className=`agent-worker-card ${child.state==='running'?'running':''}`;const top=document.createElement('div');top.className='agent-worker-top';const purpose=document.createElement('span');purpose.className='agent-worker-purpose';purpose.textContent=child.reason||t('agents.reason.default');const access=child.access_mode==='requested_read_only'?document.createElement('span'):null;if(access){access.className='agent-read-only';access.textContent=t('agents.requested.ro');access.title=t('agents.requested.ro.title')}const state=document.createElement('span');state.className=`agent-worker-state ${child.state==='running'?'running':''}`;state.textContent=child.state==='running'?t('state.running.short'):t('state.done.short');if(access)top.append(purpose,access,state);else top.append(purpose,state);const details=document.createElement('div');details.className='agent-worker-details';details.textContent=`${formatDuration(child.age_seconds)} · ${child.api_calls||0} ${t('agents.calls')}`;const model=document.createElement('div');model.className='agent-worker-model';model.textContent=child.model||t('agents.unknown.model');worker.append(top,details,model);workers.append(worker)}detail.append(workers);card.append(detail);tree.append(card)}};
let agentActivity={parents:[]};function childSessionIds(){return new Set((agentActivity.parents||[]).flatMap(parent=>(parent.children||[]).map(child=>child.agent_session_id).filter(Boolean)))}function turnBelongsToChild(entry){const turn=String(entry.turn_id||'');return [...childSessionIds()].some(sessionId=>turn.startsWith(`${sessionId}:`))}const syntheticLifecyclePrefixes=['[ASYNC DELEGATION BATCH COMPLETE','[ASYNC DELEGATION COMPLETE','[Your active task list was preserved across context compression]','[IMPORTANT: Background process ','Review the conversation above and consider saving to memory if appropriate.','[CONTEXT COMPACTION'];function isSyntheticLifecyclePrompt(entry){const prompt=String(entry.prompt_preview||''),turn=String(entry.turn_id||'');if(entry&&entry.is_internal_prompt===true)return true;return turn.includes(':sa-')||syntheticLifecyclePrefixes.some(prefix=>prompt.startsWith(prefix))}function childEntries(child,list,seen=new Set()){const sessionId=child?.agent_session_id;if(!sessionId||seen.has(sessionId))return [];seen.add(sessionId);const own=(child.routed_calls&&child.routed_calls.length)?child.routed_calls:list.filter(entry=>String(entry.turn_id||'').startsWith(`${sessionId}:`));const nested=(agentActivity.parents||[]).find(parent=>parent.session_id===sessionId);const descendants=(nested?.children||[]).flatMap(next=>childEntries(next,list,seen));return [...own,...descendants]}function visibleEntries(list){const bridgeRuns=new Set(agentActivity.external_bridge_run_ids||[]);return list.filter(entry=>!bridgeRuns.has(sessionIdFromTurn(entry))&&!turnBelongsToChild(entry)&&!isSyntheticLifecyclePrompt(entry))}function sessionIdFromTurn(entry){return String(entry?.turn_id||'').split(':')[0]}function systemEntriesForRoot(root,list){const rootTurnId=String(root?.turn_id||''),sessionId=sessionIdFromTurn(root),allRoots=visibleEntries(list).sort((a,b)=>Date.parse(a.timestamp)-Date.parse(b.timestamp)),sessionRoots=allRoots.filter(entry=>sessionIdFromTurn(entry)===sessionId);return list.filter(entry=>{if(!isSyntheticLifecyclePrompt(entry))return false;const directParent=String(entry.parent_turn_id||'');if(directParent){return directParent===rootTurnId}if(sessionIdFromTurn(entry)!==sessionId)return false;const when=Date.parse(entry.timestamp),preceding=sessionRoots.filter(candidate=>Date.parse(candidate.timestamp)<=when).at(-1);return preceding&&String(preceding.turn_id||'')===rootTurnId})}function promptsMatch(left,right){const a=String(left||'').trim(),b=String(right||'').trim();return a&&b&&(a===b||a.startsWith(b)||b.startsWith(a))}function nearestParent(parents,timestamp){const target=Date.parse(timestamp)/1000;if(!Number.isFinite(target))return parents[0]||null;return parents.reduce((best,parent)=>!best||Math.abs(Number(parent.started_at)-target)<Math.abs(Number(best.started_at)-target)?parent:best,null)}function parentForGroup(group){for(const entry of group){const sessionId=sessionIdFromTurn(entry);const sessionParents=(agentActivity.parents||[]).filter(parent=>parent.session_id===sessionId);if(!sessionParents.length)continue;const prompt=String(entry.prompt_preview||'').trim();const promptMatch=sessionParents.find(parent=>promptsMatch(prompt,parent.prompt));if(promptMatch)return promptMatch;return null}return null}function groupRunState(group){const parent=parentForGroup(group);if(!parent)return 0;return (parent.children||[]).some(child=>child.state==='running')?1:0}function agentDetails(parent){const grid=document.createElement('div');grid.className='agent-worker-grid router-agent-workers';for(const child of parent.children||[]){const worker=document.createElement('div');worker.className=`agent-worker-card ${child.state==='running'?'running':''}`;const top=document.createElement('div');top.className='agent-worker-top';const purpose=document.createElement('span');purpose.className='agent-worker-purpose';purpose.textContent=child.reason||t('agents.reason.default');const state=document.createElement('span');state.className=`agent-worker-state ${child.state==='running'?'running':''}`;state.textContent=child.state==='running'?t('state.running.short'):t('state.done.short');top.append(purpose,state);if(child.access_mode==='requested_read_only'){const access=document.createElement('span');access.className='agent-read-only';access.textContent=t('agents.requested.ro');access.title=t('agents.requested.ro.title');top.append(access)}const details=document.createElement('div');details.className='agent-worker-details';details.textContent=`${formatDuration(child.age_seconds)} · ${child.api_calls||0} ${t('agents.calls')}`;const model=document.createElement('div');model.className='agent-worker-model';model.textContent=child.model||t('agents.unknown.model');worker.append(top,details,model);grid.append(worker)}return grid}function nestedParentForChild(child){const descendants=child.children||[],hasDescendants=child.children?.length;return hasDescendants?{children:descendants}:null}function nodeRoute(calls){return calls.length?routeSummary(calls):t('exec.no.route')}const taskTreeExpansionKey='model-router-task-tree-expansion-v2';function taskTreeExpansions(){try{return JSON.parse(localStorage.getItem(taskTreeExpansionKey)||'{}')}catch(e){return {}}}function taskTreeOpen(key,defaultOpen=true){const state=taskTreeExpansions();return Object.hasOwn(state,key)?!!state[key]:defaultOpen}function setTaskTreeOpen(key,open){const state=taskTreeExpansions();state[key]=open;localStorage.setItem(taskTreeExpansionKey,JSON.stringify(state))}function appendTaskTreeNode(tree,node,depth){const children=node.children||[],hasChildren=children.length>0,open=hasChildren&&taskTreeOpen(node.id,true),row=document.createElement('div');row.className=`task-tree-row ${node.state==='running'?'running':''}`;row.style.setProperty('--tree-depth',depth);row.setAttribute('role','treeitem');row.setAttribute('aria-level',String(depth+1));if(hasChildren)row.setAttribute('aria-expanded',String(open));const branch=document.createElement(hasChildren?'button':'span');branch.className=hasChildren?'task-tree-toggle':'task-tree-branch';branch.textContent=hasChildren?(open?'▾':'▸'):'•';if(hasChildren){branch.type='button';branch.title=open?t('agents.close.tasks'):t('agents.open.tasks');branch.addEventListener('click',()=>{setTaskTreeOpen(node.id,!open);render()})}const kind=document.createElement('span');kind.className='task-tree-kind';kind.textContent=node.kind;const description=document.createElement('span');description.className='task-tree-description';description.textContent=node.description||t('agents.no.desc');description.title=description.textContent;const state=document.createElement('span');state.className=`task-tree-state ${node.state==='running'?'running':''}`;state.textContent=node.state==='running'?t('state.running.short'):t('state.done.short');row.append(branch,kind,description,state);tree.append(row);if(open)for(const child of children)appendTaskTreeNode(tree,child,depth+1)}function childTaskNode(child){return {id:child.id||child.agent_session_id||child.goal,kind:child.model?.toUpperCase()||'WORKER',description:child.task_description||child.goal,state:child.state,children:(child.children||[]).map(childTaskNode)}}function lifecycleTaskNodes(systemCalls,parentId){return systemCalls.map((entry,index)=>({id:`${parentId}:lifecycle:${index}`,kind:'LIFECYCLE',description:t('internal.router.step'),state:'completed',children:[]}))}function promptExecutionTree(group,systemCalls,parent){const tree=document.createElement('div');tree.className='task-tree';tree.setAttribute('role','tree');const rootId=`${parent?.session_id||sessionIdFromTurn(group?.[0])}:root`;for(const node of [...lifecycleTaskNodes(systemCalls,rootId),...(parent?.children||[]).map(childTaskNode)])appendTaskTreeNode(tree,node,0);return tree}function legacySystem(body,systemCalls,sessionId){const key=`__system__:${sessionId}`,open=isPromptExpanded(key),first=systemCalls[0],[date,time]=dateAndTime(first.timestamp),tr=document.createElement('tr'),expand=document.createElement('td');expand.className='expand';const toggle=document.createElement('button');toggle.type='button';toggle.className='prompt-toggle';toggle.textContent=open?'▾':'▸';toggle.title=open?t('close.router.steps'):t('open.router.steps');toggle.setAttribute('aria-expanded',String(open));toggle.addEventListener('click',()=>{setPromptExpanded(key,!open);render()});expand.append(toggle);tr.append(expand);appendCell(tr,date);appendCell(tr,time);const prompt=appendCell(tr,t('root.continuation'));prompt.title=t('root.continuation.title');appendCell(tr,systemCalls.length,'calls');const route=document.createElement('td');route.className='route';for(const [index,item] of compact(systemCalls.map(routeKey)).entries()){if(index)route.append(' → ');route.append(pill(item.v.split(' · ')[0],item.v));if(item.n>1)route.append(`×${item.n}`)}tr.append(route);appendCell(tr,t('root.continuation.reason'),'reason');body.append(tr);if(open){const detail=document.createElement('tr'),cell=document.createElement('td');detail.className='detail';cell.colSpan=7;cell.append(detailsTable(systemCalls));detail.append(cell);body.append(detail)}}

/* Reference execution-tree renderer: one compact router header and one tree panel. */
function executionRouteKey(call){const tier=String(call?.tier||call?.model||'?').toLowerCase(),effort=String(call?.effort||'').toLowerCase();return effort?`${tier} · ${effort}`:tier}
function executionOwnCalls(node){const raw=Array.isArray(node?.routed_calls)?node.routed_calls:[];return raw.length?raw:[]}
function executionOwnCount(node){const raw=executionOwnCalls(node);return raw.length||Number(node?.api_calls)||0}
function executionCalls(node){return [...executionOwnCalls(node),...(node.children||[]).flatMap(executionCalls)]}
function executionTotalCalls(node){const raw=executionCalls(node);return raw.length||executionOwnCount(node)+(node.children||[]).reduce((sum,child)=>sum+executionTotalCalls(child),0)}
function executionRawCall(node){const calls=executionOwnCalls(node);return calls.length?calls[calls.length-1]:null}
function executionKind(node){const raw=executionRawCall(node),source=`${node.kind||''} ${raw?.tier||raw?.model||node.model||''}`.toLowerCase();if(node.kind==='LIFECYCLE')return 'lifecycle';if(source.includes('qwen'))return 'qwen';if(source.includes('grok'))return 'grok';if(source.includes('opus5')||source.includes('claude-opus-5'))return 'opus5';if(source.includes('sonnet5')||source.includes('claude-sonnet-5'))return 'sonnet5';if(source.includes('haiku')||source.includes('claude-haiku'))return 'haiku';if(source.includes('terra'))return 'terra';if(source.includes('spark'))return 'spark';if(source.includes('luna'))return 'luna';if(source.includes('sol'))return 'sol';return 'worker'}
function executionModelLabel(node){if(node.kind==='LIFECYCLE')return 'LIFECYCLE';const raw=executionRawCall(node),kind=executionKind(node);return String(raw?.tier||raw?.model||node.model||kind).toUpperCase()}
function executionNode(child){return {id:child.bridge_run_id||child.id||child.agent_session_id||child.goal,bridge_run_id:child.bridge_run_id,model:child.model,goal:child.goal,task_description:child.task_description||child.goal,state:child.state,external:!!child.external,access_mode:child.access_mode,metrics:child.metrics||{},routed_calls:child.routed_calls||[],api_calls:child.api_calls,children:(child.children||[]).map(executionNode)}}
function lifecycleDescription(entry){const completion=/^\[(?:ASYNC DELEGATION|CONTEXT COMPACTION|Your active task list)/i;for(const value of [entry?.lifecycle_prompt,entry?.task_description,entry?.task,entry?.first_user_message,entry?.user_prompt,entry?.prompt_preview]){const text=String(value||'').trim();if(text&&!completion.test(text))return text}return t('internal.router.step')}
function collapseCompletionReplays(systemCalls){const seen=new Map(),out=[];for(const entry of systemCalls||[]){const id=entry&&entry.delegation_id?String(entry.delegation_id):'';if(!id||entry.event_kind!=='async_delegation_completion'){out.push(entry);continue}const key=`${entry.turn_id||''}|${id}`;if(seen.has(key)){const at=seen.get(key);if(Number(entry.api_call_count||0)>=Number(out[at].api_call_count||0))out[at]=entry}else{seen.set(key,out.length);out.push(entry)}}return out}function executionLifecycleNodes(systemCalls,parentId){return collapseCompletionReplays(systemCalls).map((entry,index)=>({id:`${parentId}:lifecycle:${index}`,kind:'LIFECYCLE',task_description:lifecycleDescription(entry),state:'completed',routed_calls:[entry],api_calls:1,children:[]}))}
function executionCallTier(call){return String(call?.tier||call?.model||'?').toLowerCase()}
function executionFilteredNode(node,tier=''){const children=(node.children||[]).map(child=>executionFilteredNode(child,tier)).filter(Boolean),routed_calls=executionOwnCalls(node).filter(call=>!tier||executionCallTier(call)===tier);if(tier&&!routed_calls.length&&!children.length)return null;return {...node,routed_calls,api_calls:tier?routed_calls.length:node.api_calls,children}}
function executionSummary(runCalls){const summary={total:0,luna:0,spark:0,terra:0,sol:0,opus5:0,sonnet5:0,haiku:0,qwen:0,grok:0};for(const call of (runCalls||[]).flat()){const tier=executionCallTier(call);summary.total++;if(Object.hasOwn(summary,tier))summary[tier]++}return summary}
function appendRoutePills(row,calls){const counts=new Map();for(const call of calls||[]){const key=executionRouteKey(call);counts.set(key,(counts.get(key)||0)+1)}for(const [key,count] of counts){const [tier]=key.split(' · ');row.append(pill(tier,`${key} ×${count}`))}}
function appendExecutionNode(tree,node,depth,siblings=[]){const children=node.children||[],hasChildren=children.length>0,open=hasChildren&&taskTreeOpen(node.id,true),kind=executionKind(node),siblingIndex=siblings.indexOf(node),hasPreviousSibling=siblingIndex>0,hasNextSibling=siblingIndex>=0&&siblingIndex<siblings.length-1,row=document.createElement('div');row.className=`execution-tree-row task-tree-${kind} ${node.state==='running'?'running':''} ${hasPreviousSibling?'task-tree-has-prev-sibling':''} ${hasNextSibling?'task-tree-has-next-sibling':''}`;row.style.setProperty('--tree-depth',depth);row.setAttribute('role','treeitem');row.setAttribute('aria-level',String(depth+1));if(hasChildren)row.setAttribute('aria-expanded',String(open));const disclosure=document.createElement(hasChildren?'button':'span');disclosure.className=hasChildren?'execution-tree-toggle':'execution-tree-spacer';disclosure.textContent=hasChildren?(open?'▾':'▸'):'';if(hasChildren){disclosure.type='button';disclosure.title=open?t('agents.close.inner'):t('agents.open.inner');disclosure.addEventListener('click',event=>{event.stopPropagation();setTaskTreeOpen(node.id,!open);render()})}const marker=document.createElement('span');marker.className=`task-tree-marker ${kind}`;marker.setAttribute('aria-hidden','true');const model=document.createElement('span');model.className='execution-tree-model';model.textContent=executionModelLabel(node);const description=document.createElement('span');description.className='execution-tree-description';description.textContent=node.task_description||node.goal||t('agents.no.desc');description.title=description.textContent;if(node.external){const badges=[];badges.push(t('exec.external'));if(node.access_mode==='read_only')badges.push(t('exec.read.only'));else if(node.access_mode==='requested_read_only')badges.push(t('agents.requested.ro'));description.textContent=`${node.goal||t('exec.opus.review')} · ${badges.join(' · ')} · ${description.textContent}`}const accounting=document.createElement('span');accounting.className='execution-tree-accounting';const ownCalls=executionOwnCount(node),totalCalls=executionTotalCalls(node),metrics=node.metrics||{};accounting.textContent=node.external?`turns ${ownCalls} · input ${metrics.input_tokens??'—'} · output ${metrics.output_tokens??'—'} · cache ${metrics.cache_read_input_tokens??'—'} · cost ${metrics.total_cost_usd??'—'} · duration ${metrics.duration_seconds??'—'}s`:`${t('exec.own')} ${ownCalls} · ${t('exec.total')} ${totalCalls}`;const routes=document.createElement('span');routes.className='execution-tree-routes';appendRoutePills(routes,executionCalls(node));const state=document.createElement('span'),stateLabels={running:t('state.running.short'),success:t('state.success.short'),error:t('state.error.short'),timeout:'TIMEOUT','max-turn':'MAX-TURN',budget:'BUDGET',completed:t('state.done.short')};state.className=`execution-tree-state ${node.state==='running'?'running':''}`;state.textContent=stateLabels[node.state]||String(node.state||t('state.done.short')).toUpperCase();row.append(disclosure,marker,model,description,accounting,routes,state);tree.append(row);if(open&&hasChildren){const childrenEl=document.createElement('div');childrenEl.className='execution-tree-children';childrenEl.style.setProperty('--tree-depth',depth);for(const child of children)appendExecutionNode(childrenEl,child,depth+1,children);tree.append(childrenEl)}}
function executionScope(group,systemCalls,parent,tier=''){const rootId=`${parent?.session_id||sessionIdFromTurn(group?.[0])}:root`,rawNodes=[...executionLifecycleNodes(systemCalls,rootId),...(parent?.children||[]).map(executionNode)],nodes=rawNodes.map(node=>executionFilteredNode(node,tier)).filter(Boolean),routed_calls=(group||[]).filter(call=>!tier||executionCallTier(call)===tier),root={routed_calls,children:nodes};return {calls:executionCalls(root),nodes}}
function buildEntryIndex(allEntries){const childIds=childSessionIds(),bySession=new Map(),groupsMap=new Map(),lifecycleByTurn=new Map();for(const entry of allEntries){const session=sessionIdFromTurn(entry);if(!bySession.has(session))bySession.set(session,[]);bySession.get(session).push(entry);if(isSyntheticLifecyclePrompt(entry)){const parent=String(entry.parent_turn_id||'');if(parent){if(!lifecycleByTurn.has(parent))lifecycleByTurn.set(parent,[]);lifecycleByTurn.get(parent).push(entry)}continue}if(childIds.has(session))continue;const key=promptKey(entry);if(!groupsMap.has(key))groupsMap.set(key,[]);groupsMap.get(key).push(entry)}return {bySession,groups:[...groupsMap.values()],lifecycleByTurn}}
function childLogEntries(child,index,seen=new Set()){const sessionId=child?.agent_session_id;if(!sessionId||seen.has(sessionId))return [];seen.add(sessionId);const own=index.bySession.get(sessionId)||[],nested=(agentActivity.parents||[]).find(parent=>parent.session_id===sessionId),descendants=(nested?.children||[]).flatMap(next=>childLogEntries(next,index,seen));return [...own,...descendants]}
function indexedSystemEntries(group,index){const direct=[...new Set(group.flatMap(entry=>index.lifecycleByTurn.get(String(entry.turn_id||''))||[]))];if(direct.length)return direct;return systemEntriesForRoot(group[0],index.bySession.get(sessionIdFromTurn(group[0]))||[])}
function rootRunRecords(allEntries,tier=''){const index=buildEntryIndex(allEntries);return index.groups.map(group=>{const first=group[0],parent=parentForGroup(group),systemCalls=indexedSystemEntries(group,index),scope=executionScope(group,systemCalls,parent,tier),childLogs=(parent?.children||[]).flatMap(child=>childLogEntries(child,index));return {group,first,parent,systemCalls,scope,rawEntries:[...new Set([...group,...systemCalls,...childLogs])]}}).filter(run=>!tier||run.scope.calls.length)}
function limitRootRuns(runs,limit){const count=Number(limit);return runs.slice(-Math.max(0,Number.isFinite(count)?count:20))}
function rawEntryKey(entry){return entry?.id||`${entry?.timestamp||''}::${entry?.turn_id||''}::${entry?.api_call_count||''}::${entry?.tier||''}::${entry?.model||''}`}
function rawEntriesForRootRuns(runs,allEntries){const selected=new Set(runs.flatMap(run=>run.rawEntries||[]).map(rawEntryKey));return allEntries.filter(entry=>selected.has(rawEntryKey(entry)))}
function executionTree(nodes){const tree=document.createElement('div');tree.className='execution-tree';tree.setAttribute('role','tree');for(const node of nodes||[])appendExecutionNode(tree,node,0,nodes);return tree}
function executionState(group,parent){const active=(agentActivity.active_turns||[]).some(turn=>group.some(entry=>String(entry.turn_id||'')===String(turn.turn_id||'')))||(parent?.children||[]).some(child=>child.state==='running');return active?'running':'done'}
setWordWrap=function(){const enabled=$('word-wrap').checked;$('log-table').classList.toggle('word-wrap',enabled);$('runs').classList.toggle('word-wrap',enabled);localStorage.setItem('model-router-word-wrap',enabled?'1':'0')};$('word-wrap').addEventListener('input',setWordWrap);setWordWrap();
let delegations=[];
function tierAccount(tier){return (accountsState.tier_accounts||{})[tier]||''}
function accountLabel(account){return ((accountsState.accounts||{})[account]||{}).label||({'openai-codex':'Codex',anthropic:'Claude','qwen-token':'Qwen','xai-oauth':'Grok'})[account]||account}
function runForTurnId(runs,turnId){
  // The router log's own turn_id starts with the session id, then the turn
  // (e.g. "s1:1", with a sub-call sometimes appending ":more" beyond that) --
  // so a run "has" this turn_id when one of its raw entries is exactly it, or
  // is a sub-call nested under it.
  return runs.findIndex(run => (run.rawEntries || []).some(entry => {
    const t = String(entry.turn_id || '');
    return t === turnId || t.startsWith(`${turnId}:`);
  }));
}
function assignDelegations(runs,audits){const out=new Map(),bySession=new Map();runs.forEach((run,i)=>{const s=sessionIdFromTurn(run.first);if(!bySession.has(s))bySession.set(s,[]);bySession.get(s).push({i,at:Date.parse(run.first.timestamp)})});for(const list of bySession.values())list.sort((a,b)=>a.at-b.at);for(const d of audits){let owner=null;const turnId=String(d.turn_id||'');if(turnId){const found=runForTurnId(runs,turnId);if(found!==-1)owner=found}if(owner===null){const list=bySession.get(String(d.session_id||''));if(list){const at=Date.parse(d.timestamp);for(const r of list){if(r.at<=at)owner=r.i}}}if(owner===null)continue;if(!out.has(owner))out.set(owner,[]);out.get(owner).push(d)}return out}
function chip(account,text,marker,title){const s=document.createElement('span');s.className=`delegation-chip ${account}${marker?' marked':''}`;s.textContent=`${accountLabel(account)}: ${text}${marker?' '+marker:''}`;if(title)s.title=title;return s}
// M5: one hover format for both accounts, built from whichever clause the
// account already writes -- the router's own "X→Y (weekly N%)" / "(session
// N%)", or Claude's audit "adjusted" string "X→Y (weekly usage N%)".
function stepDownHoverText(text){const m=/(\S+)→(\S+)[^()]*\((weekly|session)(?:\s+usage)?\s+(\d+)%\)/.exec(String(text||''));return m?`${m[1]}→${m[2]}, ${m[3]} ${m[4]}%`:(text||'')}
function delegationChips(run,audits){const box=document.createElement('span');box.className='delegation-chips';const haveAudits=audits&&audits.length;
  // I3: the router appends this clause only for a real step-down; when the
  // target itself is unavailable it instead writes "...skipped (<target>
  // unavailable)", which this pattern's "(weekly|session" tail never matches.
  const stepDownRe=/usage (?:soft|hard) limit: \S+→\S+ \((?:weekly|session)/;
  // Every Claude router target name is its tier name with a trailing account
  // generation digit (sonnet5→sonnet, opus5→opus); haiku carries none and
  // passes through unchanged. Computed rather than a sonnet5/haiku pair
  // literal so this never drifts out of the codebase's own haiku-mirrors-
  // sonnet5 parity check.
  const claudeTierForTarget=target=>target.replace(/5$/,'');
  for(const node of (run.scope.nodes||[]).filter(n=>n.kind!=='LIFECYCLE')){const tier=executionKind(node),account=tierAccount(tier);if(!account||(account==='anthropic'&&haveAudits))continue;const label=account==='anthropic'?claudeTierForTarget(tier):tier;const stepped=executionCalls(node).find(c=>stepDownRe.test(String(c.reason||'')));box.append(chip(account,label,stepped?'↓':'',stepped?stepDownHoverText(stepped.reason):''))}
  for(const d of (audits||[])){const marker=d.outcome==='lowered'?'↓':d.outcome==='refused'?'✕':d.outcome==='error'?'!':'';const title=marker==='↓'?stepDownHoverText(d.adjusted):(d.message||(d.usage?`weekly ${d.usage}`:''));box.append(chip('anthropic',d.tier_used||d.tier_requested,marker,title))}
  return box}
function render(){const allEntries=searchFiltered(),groupedMode=$('grouped').checked,runs=$('runs'),tier=$('tier').value,rootRuns=rootRunRecords(allEntries,tier),limitedRoots=limitRootRuns(rootRuns,$('last').value),displayRoots=[...limitedRoots].sort((left,right)=>groupRunState(left.group)-groupRunState(right.group)),rawList=rawEntriesForRootRuns(limitedRoots,allEntries);runs.replaceChildren();const runData=groupedMode?displayRoots:rawList.map(group=>{const first=group,parent=null,systemCalls=[],scope=executionScope([group],systemCalls,parent,tier);return {group:[group],first,parent,systemCalls,scope}}).filter(run=>!tier||run.scope.calls.length);const delegationMap=assignDelegations(runData,delegations);$('status').textContent=`${limitedRoots.length} ${t('status.rootprompts')} · ${new Date().toLocaleTimeString(t('status.locale'))}`;const summary=executionSummary(runData.map(run=>run.scope.calls));for(const id of ['luna','spark','terra','sol','opus5','sonnet5','haiku','qwen','grok'])$(id).textContent=summary[id];$('total').textContent=summary.total;if(!runData.length){const empty=document.createElement('div');empty.className='empty';empty.textContent=t('no.entries');runs.append(empty);return}for(const [runIndex,record] of runData.entries()){const {group,first,parent,scope}=record,key=groupedMode?promptKey(first):`${promptKey(first)}::${first.timestamp}:${first.api_call_count}`,open=isPromptExpanded(key),hasDetails=scope.nodes.length>0,accountingCalls=scope.calls,workerCalls=scope.nodes.filter(node=>node.kind!=='LIFECYCLE').flatMap(executionCalls).length,state=executionState(group,parent),[date,time]=dateAndTime(first.timestamp),run=document.createElement('article');run.className=`router-run ${open?'is-open':'is-closed'} ${state==='running'?'running':''}`;const header=document.createElement(hasDetails?'button':'div');if(hasDetails)header.type='button';header.className=`router-run-header ${hasDetails?'':'no-details'}`;if(hasDetails)header.setAttribute('aria-expanded',String(open));if(hasDetails){const toggle=document.createElement('span');toggle.className='router-run-toggle';toggle.textContent=open?'▾':'▸';header.append(toggle)}const dateEl=document.createElement('span');dateEl.className='router-run-date';dateEl.textContent=date;const timeEl=document.createElement('span');timeEl.className='router-run-time';timeEl.textContent=time;const prompt=document.createElement('span');prompt.className='router-run-prompt';prompt.textContent=first.prompt_preview||t('not.recoverable');prompt.title=prompt.textContent;const stateEl=document.createElement('span');stateEl.className=`router-run-state ${state==='running'?'running':''}`;stateEl.textContent=state==='running'?`▶ ${t('agents.pill.running')} · ${(parent?.children||[]).filter(child=>child.state==='running').length||1} ${t('agents.pill.agent')}`:t('state.done.short');const total=document.createElement('span');total.className='router-run-total';total.innerHTML=`<b>${t('total.routing.decisions')}</b>`;total.append(` ${accountingCalls.length}`);const routes=document.createElement('span');routes.className='router-run-routes';appendRoutePills(routes,accountingCalls);routes.append(delegationChips(record,delegationMap.get(runIndex)||[]));const workers=document.createElement('span');workers.className='router-run-workers';workers.textContent=`${workerCalls} ${t('run.worker.routing')}`;header.append(dateEl,timeEl,prompt,stateEl,total,routes,workers);if(hasDetails)header.addEventListener('click',()=>{setPromptExpanded(key,!open);render()});run.append(header);if(hasDetails&&open){const panel=document.createElement('section');panel.className='execution-tree-panel';panel.setAttribute('aria-label',t('exec.tree.label'));panel.append(executionTree(scope.nodes));run.append(panel)}runs.append(run)}};
let refreshInFlight=null;
// /api/config serves guard.cached(), so a redraw alone shows the same numbers.
// Only /api/usage/refresh fetches; the guard rate-limits it to cache_seconds,
// so a click inside that window costs nothing upstream.
async function refreshAllUsage(){const accounts=Object.entries(accountsState.accounts||{}).filter(([,info])=>info&&info.has_usage_source).map(([account])=>account);await Promise.all(accounts.map(account=>fetch(`/api/usage/refresh?account=${encodeURIComponent(account)}`,{method:'POST'}).catch(()=>null)))}
// The click path: read live, then refresh. The 3s timer never comes through here.
async function manualRefresh(){await refreshAllUsage();await refreshDashboard()}
function refreshDashboard(){if(refreshInFlight)return refreshInFlight;const button=$('refresh'),previousLabel=button.textContent;button.disabled=true;button.textContent=t('status.refresh.short');$('status').className='status refreshing';$('status').textContent=entries.length?t('status.refreshing')+t('status.kept'):t('status.loading');refreshInFlight=(async()=>{try{const [entriesResponse,agentsResponse]=await Promise.all([fetch(`/api/entries?roots=${$('last').value}`,{cache:'no-store'}),fetch('/api/agents',{cache:'no-store'})]);if(!entriesResponse.ok)throw new Error(`Entries HTTP ${entriesResponse.status}`);if(!agentsResponse.ok)throw new Error(`Agents HTTP ${agentsResponse.status}`);const [payload,activity]=await Promise.all([entriesResponse.json(),agentsResponse.json()]);entries=payload.entries||[];delegations=payload.delegations||[];agentActivity=activity;render();await loadAccounts()}catch(error){$('status').className='status error';$('status').textContent=`${t('status.error.prefix')}${error.message}${t('status.error.kept')}`}finally{button.disabled=false;button.textContent=previousLabel;refreshInFlight=null}})();return refreshInFlight}
document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>setTab(tab.dataset.tab)));for(const id of ['tier','search','grouped'])$(id).addEventListener('input',()=>render());$('word-wrap').checked=localStorage.getItem('model-router-word-wrap')==='1';$('word-wrap').addEventListener('input',setWordWrap);setWordWrap();initColumnResize();setTab(selectedTab||'router',false); applyLanguage();
$('last').addEventListener('change',()=>refreshDashboard());$('refresh').addEventListener('click',()=>manualRefresh());setInterval(()=>{if($('auto').checked)refreshDashboard()},3000);refreshDashboard();
loadAccounts();setInterval(loadAccounts,60000);
</script><style>.agent-branch{margin-top:14px;padding:14px;border:1px solid #2c3951;border-radius:12px;background:#0b111c}.agent-parent-head{display:flex;justify-content:space-between;gap:12px;align-items:center}.agent-parent-prompt{margin-top:7px;color:#e9eef8;font-weight:650;white-space:normal;overflow-wrap:anywhere}.agent-children{position:relative;margin:13px 0 0 10px;padding-left:20px;border-left:1px solid #334158}.agent-child-row{display:grid;grid-template-columns:10px minmax(0,1fr) auto;gap:10px;align-items:center;position:relative;padding:10px 0}.agent-child-row:before{content:'';position:absolute;left:-20px;top:50%;width:18px;border-top:1px solid #334158}.agent-child-body{min-width:0}.agent-reason{font-weight:750;color:#e9eef8}.agent-child-goal{margin-top:2px;color:var(--muted);font-size:12px;white-space:normal;overflow-wrap:anywhere}.agent-meta{white-space:pre-line;min-width:120px}@media(max-width:700px){.agent-child-row{grid-template-columns:10px minmax(0,1fr)}.agent-child-row .agent-meta{grid-column:2;text-align:left}.agent-parent-head{align-items:flex-start;flex-direction:column;gap:2px}}.agent-main-card{margin-top:14px;border:1px solid #33435d;border-radius:16px;background:linear-gradient(145deg,rgba(20,29,45,.96),rgba(9,14,24,.98));overflow:hidden;box-shadow:0 12px 28px rgba(0,0,0,.16)}.agent-main-toggle{width:100%;display:grid;grid-template-columns:92px 64px minmax(0,1fr) auto;align-items:center;gap:10px;padding:15px 16px;border:0;border-radius:0;background:transparent;text-align:left}.agent-main-toggle:hover{background:rgba(167,139,250,.08);border-color:transparent}.agent-main-toggle:focus-visible{outline:2px solid var(--accent);outline-offset:-3px}.agent-main-heading{min-width:0;display:grid;gap:4px}.agent-main-date,.agent-main-time{color:#9dabbe;font-size:11px;font-variant-numeric:tabular-nums;white-space:nowrap}.agent-main-time{color:#c3cede}.agent-main-preview{color:#eef3ff;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.agent-main-started{color:#9dabbe;font-size:11px;font-variant-numeric:tabular-nums}.agent-main-stats{display:flex;align-items:center;justify-content:flex-end;gap:7px;flex-shrink:0}.agent-stat,.agent-worker-state,.agent-read-only{border:1px solid #394a66;border-radius:99px;padding:3px 8px;color:#aebbd0;font-size:11px;font-weight:750;white-space:nowrap}.agent-read-only{color:#ffd18a;border-color:rgba(255,180,84,.6);background:rgba(255,180,84,.1)}.agent-stat.running,.agent-worker-state.running{color:#b8f6a6;border-color:rgba(136,227,111,.55);background:rgba(136,227,111,.1)}.agent-chevron{color:#c4b5fd;font-size:18px;line-height:1}.agent-main-detail{padding:0 16px 16px;border-top:1px solid rgba(51,67,93,.72)}.agent-full-prompt{padding:13px 0;color:#c8d3e6;font-size:13px;line-height:1.55;white-space:normal;overflow-wrap:anywhere}.agent-worker-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:9px}.agent-worker-card{min-width:0;padding:11px 12px;border:1px solid #2a3850;border-radius:12px;background:linear-gradient(145deg,#101927,#0b111c)}.agent-worker-card.running{border-color:rgba(136,227,111,.46);box-shadow:inset 3px 0 0 #88e36f}.agent-worker-top{display:flex;align-items:center;justify-content:space-between;gap:8px}.agent-worker-purpose{color:#eef3ff;font-size:12px;font-weight:800}.agent-worker-details{margin-top:9px;color:#aebbd0;font-size:12px}.agent-worker-model{margin-top:3px;color:#8f9fb7;font:11px ui-monospace,SFMono-Regular,Consolas,monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}@media(max-width:700px){.agent-main-toggle{align-items:flex-start}.agent-main-preview{white-space:normal}.agent-main-stats{flex-wrap:wrap;max-width:105px}.agent-worker-grid{grid-template-columns:1fr}}.router-agent-workers{display:flex;overflow-x:auto;gap:8px;padding:2px 0 7px}.router-agent-workers .agent-worker-card{flex:0 0 300px;padding:8px 10px}.router-agent-workers .agent-worker-details{margin-top:5px}.router-agent-workers .agent-worker-model{margin-top:2px}.task-tree{padding:5px 0;display:grid;gap:2px}.task-tree-row{--tree-depth:0;display:grid;grid-template-columns:18px 92px minmax(180px,1fr) max-content;gap:9px;align-items:center;min-height:38px;padding:7px 10px 7px calc(10px + var(--tree-depth) * 28px);border-left:1px solid #334158;background:rgba(8,13,22,.45);font-size:12px}.task-tree-row:hover{background:#151d2b}.task-tree-row.running{border-left-color:#88e36f;box-shadow:inset 3px 0 0 rgba(136,227,111,.6)}.task-tree-branch,.task-tree-toggle{color:#8f9fb7;font:14px ui-monospace,SFMono-Regular,Consolas,monospace}.task-tree-toggle{width:18px;padding:0;border:0;background:transparent;text-align:center}.task-tree-toggle:hover{color:#c4b5fd;border-color:transparent}.task-tree-description{min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#e9eef8}.word-wrap .task-tree-description{white-space:normal;overflow:visible;text-overflow:clip;overflow-wrap:anywhere}.task-tree-state{border:1px solid #394a66;border-radius:99px;padding:3px 7px;color:#aebbd0;font-size:10px;font-weight:800}.task-tree-state.running{color:#b8f6a6;border-color:rgba(136,227,111,.55);background:rgba(136,227,111,.1)}@media(max-width:850px){.task-tree-row{grid-template-columns:18px 78px minmax(150px,1fr) max-content;gap:7px}.task-tree-state{grid-column:3;grid-row:2;justify-self:end}}@media(max-width:560px){.task-tree-row{grid-template-columns:18px 1fr;gap:4px;padding-left:calc(8px + var(--tree-depth) * 18px)}.task-tree-description,.task-tree-state{grid-column:2}.task-tree-state{grid-row:auto;justify-self:start}}.agent-worker-goal-details{min-width:0;max-width:100%;margin-top:8px}.agent-worker-goal-details summary{cursor:pointer;color:#c4b5fd;font-weight:700}.agent-worker-goal{margin-top:7px;max-width:100%;white-space:normal;overflow-wrap:anywhere;word-break:break-word}.router-runs{display:grid;gap:14px}.router-run{overflow:hidden;border:1px solid #304159;border-radius:15px;background:linear-gradient(145deg,rgba(13,22,34,.98),rgba(7,13,22,.98));box-shadow:0 12px 30px rgba(0,0,0,.16)}.router-run-header{width:100%;display:grid;grid-template-columns:24px 106px 78px minmax(180px,1fr) max-content max-content max-content max-content;align-items:center;gap:12px;padding:18px 22px;border:0;border-bottom:1px solid transparent;border-radius:0;background:transparent;text-align:left}.router-run-header.no-details{grid-template-columns:106px 78px minmax(180px,1fr) max-content max-content max-content max-content}.router-run.is-open .router-run-header{border-bottom-color:#304159}.router-run-header:hover{border-color:transparent;background:rgba(167,139,250,.055)}.router-run-header:focus-visible,.execution-tree-toggle:focus-visible{outline:2px solid var(--accent);outline-offset:-3px}.router-run-toggle,.execution-tree-toggle{color:#c4b5fd;font-size:27px;line-height:1}.router-run-date,.router-run-time{font:600 14px ui-monospace,SFMono-Regular,Consolas,monospace;color:#e9eef8;white-space:nowrap}.router-run-prompt{min-width:0;font-size:15px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.router-run-state,.execution-tree-state{border:1px solid #89a1be;border-radius:9px;padding:4px 9px;color:#c9d9ef;font:800 11px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap}.router-run-state.running,.execution-tree-state.running{border-color:var(--terra);color:#a8f18d;background:rgba(136,227,111,.08)}.router-run-total{display:flex;gap:9px;color:#e6edf8;font:600 14px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap}.router-run-total b{color:#91aaca;font-size:11px;letter-spacing:.04em}.router-run-routes,.execution-tree-routes{display:flex;align-items:center;gap:5px;min-width:0}.router-run-routes .pill,.execution-tree-routes .pill{margin:0}.router-run-workers{white-space:nowrap;color:#e9eef8;font-size:12px}.execution-tree-panel{margin:0;overflow:auto;background:transparent;border:0;border-radius:0}.execution-tree{position:relative;padding:0 22px 12px}.execution-tree-row{--tree-depth:0;display:grid;grid-template-columns:22px 16px 150px minmax(180px,1fr) max-content max-content max-content;align-items:center;gap:11px;min-height:52px;padding:7px 10px 7px calc(44px + var(--tree-depth) * 38px);position:relative;border-bottom:1px solid rgba(48,65,89,.48)}.execution-tree-row:last-child{border-bottom:0}.execution-tree-row:before{content:'';position:absolute;left:calc(34px + var(--tree-depth) * 38px);top:0;bottom:50%;border-left:1px solid #50637d}.execution-tree-row.task-tree-has-next-sibling:before{bottom:0}.execution-tree-row:after{content:'';position:absolute;left:calc(34px + var(--tree-depth) * 38px);top:50%;width:20px;border-top:1px solid #50637d}.execution-tree-children{position:relative}.execution-tree-children:before{content:'';position:absolute;left:calc(34px + var(--tree-depth,0) * 38px + 38px);top:0;bottom:26px;border-left:1px solid #50637d}.execution-tree-toggle,.execution-tree-spacer{position:relative;z-index:1;width:24px;min-height:28px;padding:0;border:0;background:transparent;text-align:center}.execution-tree-toggle{cursor:pointer}.execution-tree-spacer{color:transparent}.task-tree-marker{position:relative;z-index:1;width:12px;height:12px;background:#eff4fb;border-radius:50%;box-shadow:0 0 0 2px #0b1420}.task-tree-marker.lifecycle{background:#90ec70;box-shadow:0 0 12px rgba(144,236,112,.65)}.task-tree-marker.terra{background:#c49bff;box-shadow:0 0 10px rgba(196,155,255,.45)}.task-tree-marker.opus5{background:var(--opus5);box-shadow:0 0 10px rgba(214,149,255,.5)}.task-tree-marker.sonnet5{background:var(--sonnet5);box-shadow:0 0 10px rgba(159,180,255,.5)}.task-tree-marker.haiku{background:var(--haiku);box-shadow:0 0 10px rgba(255,158,207,.5)}.task-tree-opus5 .execution-tree-model{color:var(--opus5)}.execution-tree-model{font:800 14px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap}.task-tree-lifecycle .execution-tree-model{color:#d9a9ff}.task-tree-terra .execution-tree-model{color:#a8f18d}.execution-tree-description{min-width:0;color:#eff3fa;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.execution-tree-accounting{white-space:nowrap;color:#d3ddea;font-size:12px}.execution-tree-state{justify-self:end}.word-wrap .router-run-prompt,.word-wrap .execution-tree-description{white-space:normal;overflow:visible;text-overflow:clip;overflow-wrap:anywhere}.word-wrap .execution-tree-row{align-items:start;padding-top:13px;padding-bottom:13px}.word-wrap .execution-tree-row:after{top:24px}@media(max-width:1120px){.router-run-header{grid-template-columns:24px 100px 76px minmax(180px,1fr) max-content;gap:9px}.router-run-total{grid-column:5}.router-run-routes{grid-column:4 / -1;grid-row:2}.router-run-workers{grid-column:5;grid-row:2}.execution-tree-row{grid-template-columns:18px 16px 135px minmax(150px,1fr) max-content;gap:8px}.execution-tree-routes{grid-column:4 / -1}.execution-tree-state{grid-column:5;grid-row:2}}@media(max-width:700px){.router-run-header{grid-template-columns:22px 1fr max-content;padding:14px}.router-run-date{grid-column:2}.router-run-time{grid-column:3}.router-run-prompt{grid-column:2 / -1;grid-row:2}.router-run-state{grid-column:2;grid-row:3;justify-self:start}.router-run-total{grid-column:3;grid-row:3}.router-run-routes{grid-column:2 / -1;grid-row:4}.router-run-workers{grid-column:2;grid-row:5}.execution-tree-panel{margin:10px}.execution-tree{padding:8px}.execution-tree-row{grid-template-columns:18px 16px minmax(0,1fr) max-content;padding-left:calc(8px + var(--tree-depth) * 24px);gap:7px}.execution-tree-row:before{left:calc(14px + var(--tree-depth) * 24px)}.execution-tree-row:after{left:calc(14px + var(--tree-depth) * 24px)}.execution-tree-model{grid-column:3}.execution-tree-description{grid-column:3 / -1;grid-row:2}.execution-tree-accounting{grid-column:3;grid-row:3}.execution-tree-routes{grid-column:3 / -1;grid-row:4}.execution-tree-state{grid-column:4;grid-row:3;justify-self:end}}</style></body></html>'''


SYNTHETIC_LIFECYCLE_PREFIXES = (
    "[ASYNC DELEGATION BATCH COMPLETE",
    "[ASYNC DELEGATION COMPLETE",
    "[Your active task list was preserved across context compression]",
    "[IMPORTANT: Background process ",
    "Review the conversation above and consider saving to memory if appropriate.",
    "[CONTEXT COMPACTION",
)


def _session_id(entry: dict) -> str:
    return str(entry.get("turn_id") or "").split(":", 1)[0]


def _is_lifecycle(entry: dict) -> bool:
    prompt = str(entry.get("prompt_preview") or "")
    turn_id = str(entry.get("turn_id") or "")
    return bool(entry.get("is_internal_prompt")) or ":sa-" in turn_id or any(
        prompt.startswith(prefix) for prefix in SYNTHETIC_LIFECYCLE_PREFIXES
    )


def _activity_children(parent: dict) -> list[dict]:
    return list(parent.get("children") or [])


def _descendant_sessions(parent: dict, parents_by_session: dict[str, dict]) -> set[str]:
    found: set[str] = set()
    pending = _activity_children(parent)
    while pending:
        child = pending.pop()
        session_id = str(child.get("agent_session_id") or "")
        if not session_id or session_id in found:
            continue
        found.add(session_id)
        nested = parents_by_session.get(session_id)
        if nested:
            pending.extend(_activity_children(nested))
        pending.extend(_activity_children(child))
    return found


def select_recent_root_closure(entries: list[dict], activity: dict, root_limit: int) -> tuple[list[dict], int]:
    """Return the latest visible roots and only their root/child/lifecycle records."""
    parents = list(activity.get("parents") or [])
    parents_by_session = {str(parent.get("session_id") or ""): parent for parent in parents}
    all_child_sessions = {
        session_id
        for parent in parents
        for session_id in _descendant_sessions(parent, parents_by_session)
    }

    groups: dict[tuple[str, str], list[dict]] = {}
    for entry in entries:
        if _is_lifecycle(entry) or _session_id(entry) in all_child_sessions:
            continue
        key = (_session_id(entry), str(entry.get("prompt_preview") or f"__missing__:{entry.get('turn_id', '')}"))
        groups.setdefault(key, []).append(entry)
    selected_groups = list(groups.values())[-root_limit:]
    if not selected_groups:
        return [], 0

    selected_ids = {id(entry) for group in selected_groups for entry in group}
    selected_turns = {str(entry.get("turn_id") or "") for group in selected_groups for entry in group}
    selected_sessions = {_session_id(group[0]) for group in selected_groups}
    selected_child_sessions: set[str] = set()
    for group in selected_groups:
        session_id = _session_id(group[0])
        prompt = str(group[0].get("prompt_preview") or "").strip()
        candidates = [parent for parent in parents if str(parent.get("session_id") or "") == session_id]
        parent = next((item for item in candidates if str(item.get("prompt") or "").strip() == prompt), None)
        if parent is None and len(candidates) == 1:
            parent = candidates[0]
        if parent:
            selected_child_sessions.update(_descendant_sessions(parent, parents_by_session))

    for entry in entries:
        if _session_id(entry) in selected_child_sessions:
            selected_ids.add(id(entry))
            continue
        if not _is_lifecycle(entry):
            continue
        parent_turn = str(entry.get("parent_turn_id") or "")
        if parent_turn in selected_turns or (not parent_turn and _session_id(entry) in selected_sessions):
            selected_ids.add(id(entry))
    return [entry for entry in entries if id(entry) in selected_ids], len(selected_groups)


def _save_config_payload(data):
    """Validate both documents before committing either; serialize dashboard saves."""
    with _CONFIG_LOCK:
        if not isinstance(data, dict):
            return 400, {"success": False, "error": "Settings must be an object"}
        revision = _config_revision()
        if data.get("revision", revision) != revision:
            return 409, {"success": False, "error": "Settings changed. Reload before saving again."}
        shipped = _load_yaml_mapping(CONFIG_PATH)
        local_path = _local_config_path()
        local, dump = _read_router_config_for_update(local_path)
        local_plain = _load_yaml_mapping(local_path)
        legacy = _has_legacy_claude_keys(local_plain)
        config = _effective_router_config(shipped, local_plain)
        hermes_before = HERMES_CONFIG_PATH.read_bytes() if HERMES_CONFIG_PATH.exists() else None
        stamp, hermes = _read_hermes_snapshot()
        original_hermes = json.dumps(hermes, sort_keys=True)
        if "callable" in data:
            switches = data["callable"]
            if not isinstance(switches, dict) or any(type(v) is not bool for v in switches.values()):
                return 400, {"success": False, "error": "callable must map tiers to booleans"}
            _assign_in_place(config, "callable", switches)
        if "preferences" in data:
            cleaned, error = _clean_preferences(data["preferences"], config)
            if error:
                return 400, {"success": False, "error": error}
            _assign_in_place(config, "preferences", cleaned)
        for key, save in (("usage_limits", _save_usage_limits), ("effort", _save_effort),
                          ("claude_delegation", _save_claude_delegation), ("balance", _save_balance),
                          ("claude_reasoning_effort", _save_claude_reasoning_effort)):
            # ``workflow`` is retired: an old open tab may still post it, and it is ignored.
            if key in data:
                error = save(data[key], config)
                if error:
                    return 400, {"success": False, "error": error}
        if "default_model" in data:
            error = _save_default_model(str(data["default_model"]), config, hermes=hermes, persist=False)
            if error:
                return 400, {"success": False, "error": error}
        # Before the fallback chain: that save drops the (new) parent from its own chain.
        if "main_parent" in data:
            error = _save_main_parent(data["main_parent"], config, hermes=hermes, persist=False)
            if error:
                return 400, {"success": False, "error": error}
        if "hermes_fallback" in data:
            error = _save_hermes_fallback(data["hermes_fallback"], config, hermes=hermes, persist=False)
            if error:
                return 400, {"success": False, "error": error}
        if "worker_model" in data:
            error = _save_worker_model(data["worker_model"], config, hermes=hermes, persist=False)
            if error:
                return 400, {"success": False, "error": error}
        delta = _overlay(shipped, config)
        if legacy:
            # The first save after the upgrade writes the Claude switches the legacy
            # keys stood for, explicitly, so the file reads the same without them
            # (the keys themselves go: the effective config no longer has them).
            switches = delta.setdefault("callable", {})
            for name in CLAUDE_SWITCHES:
                if name in (config.get("callable") or {}):
                    switches[name] = config["callable"][name]
        holder = {"local": local}
        _assign_in_place(holder, "local", delta)
        output = io.StringIO()
        if not local_path.exists():
            output.write(LOCAL_CONFIG_HEADER)
        dump(holder["local"], output)
        if _config_revision() != revision:
            return 409, {"success": False, "error": "Settings changed while saving. Reload and try again."}
        hermes_changed = json.dumps(hermes, sort_keys=True) != original_hermes
        written_hermes = None
        try:
            if hermes_changed:
                _write_hermes_config(hermes, stamp)
                written_hermes = HERMES_CONFIG_PATH.read_bytes()
            if delta or local_path.exists():
                _atomic_write(local_path, output.getvalue().encode("utf-8"))
        except Exception:
            if written_hermes is not None:
                if HERMES_CONFIG_PATH.read_bytes() != written_hermes:
                    raise RuntimeError("Router save failed and Hermes changed concurrently; automatic rollback refused.")
                if hermes_before is None:
                    HERMES_CONFIG_PATH.unlink()
                else:
                    _atomic_write(HERMES_CONFIG_PATH, hermes_before)
            raise
        return 200, {"success": True, "revision": _config_revision()}


class Handler(BaseHTTPRequestHandler):
    log_path = DEFAULT_LOG
    state_db_path = DEFAULT_STATE_DB
    agent_log_path = DEFAULT_AGENT_LOG
    bridge_lifecycle_path = DEFAULT_BRIDGE_LIFECYCLE

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            if yaml is None:
                self._send(500, json.dumps({"error": "yaml not available"}).encode("utf-8"), "application/json")
                return
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
                status, result = _save_config_payload(data)
                self._send(status, json.dumps(result).encode("utf-8"), "application/json")
            except Exception as e:
                self._send(500, json.dumps({"error": str(e), "success": False}).encode("utf-8"), "application/json")
            return
        if parsed.path == "/api/usage/refresh":
            account = (parse_qs(parsed.query).get("account") or [""])[0]
            try:
                config = _read_router_config()
                router = _router_module()
                status = _accounts_status(config)
                if account not in status or router is None:
                    self._send(400, json.dumps({"error": f"Unknown account '{account}'"}).encode("utf-8"),
                               "application/json")
                    return
                # The button is a person asking now; the cache_seconds gate exists
                # to throttle background reads, not this.
                if router.usage_guard.read(account, config, force=True) is None:
                    reason = router.usage_guard.last_failure(account) or "the usage endpoint returned no reading"
                    self._send(502, json.dumps({"error": f"Could not refresh usage: {reason}"}).encode("utf-8"),
                               "application/json")
                    return
                self._send(200, json.dumps({"account": _accounts_status(config)[account]}).encode("utf-8"),
                           "application/json")
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}).encode("utf-8"), "application/json")
            return
        self._send(404, b'{"error":"not found"}', "application/json")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/agents":
            body = json.dumps(
                load_agent_activity(
                    self.state_db_path,
                    log_path=self.agent_log_path,
                    router_log_path=self.log_path,
                    bridge_lifecycle_path=self.bridge_lifecycle_path,
                ),
                ensure_ascii=False,
            ).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if parsed.path == "/api/entries":
            try:
                requested_root_limit = int(
                    parse_qs(parsed.query).get("roots", [str(DEFAULT_ROOT_LIMIT)])[0]
                )
            except ValueError:
                requested_root_limit = DEFAULT_ROOT_LIMIT
            if requested_root_limit not in {1, 5, 10}:
                requested_root_limit = DEFAULT_ROOT_LIMIT
            source_entries = load_entries(self.log_path)[-RAW_HISTORY_LIMIT:]
            activity = load_agent_activity(
                self.state_db_path,
                log_path=self.agent_log_path,
                router_log_path=self.log_path,
                bridge_lifecycle_path=self.bridge_lifecycle_path,
            )
            entries, selected_root_count = select_recent_root_closure(
                source_entries, activity, requested_root_limit
            )
            try:
                config_for_log = _read_router_config() if yaml is not None else {}
            except Exception:
                config_for_log = {}
            body = json.dumps(
                {
                    "entries": entries,
                    "requested_root_limit": requested_root_limit,
                    "selected_root_count": selected_root_count,
                    "source_entry_count": len(source_entries),
                    "raw_history_limit": RAW_HISTORY_LIMIT,
                    "delegations": _read_delegation_log(config_for_log)[0],
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if parsed.path == "/api/config":
            if yaml is None:
                self._send(500, json.dumps({"error": "yaml not available"}).encode("utf-8"), "application/json")
                return
            try:
                with _CONFIG_LOCK:
                    config = _read_router_config()
                    router = _router_module()
                    work_kinds = list(getattr(router, "WORK_KINDS", ())) if router else []
                    response = {
                        "revision": _config_revision(),
                        "callable": config.get("callable", {}),
                        "delegation_limits": router._host_delegation_limits() if router else {},
                        "balance": _balance_status(config),
                        "default_model": config.get("default_model", "terra"),
                        "effort": _effort_status(config),
                        "claude_reasoning_effort": _claude_reasoning_status(config),
                        "preferences": config.get("preferences") or {},
                        "work_kinds": work_kinds,
                        # Hermes's own chains, not the router's. Kept separate in the payload
                        # so the UI can say plainly which file a change lands in.
                        "hermes_fallback": {
                            "orchestrator": _hermes_chain("fallback_providers"),
                            "children": _hermes_chain("delegation", "fallback_providers"),
                        },
                        "fallback_options": _fallback_chain_options(config),
                        "hermes_parent": dict(_hermes_parent(_read_hermes_config()),
                                              router_model=_parent_is_router_model(config)),
                        "parent_options": _claude_parent_options(config),
                        "worker_model": _worker_model_status(config),
                        "accounts": _accounts_status(config),
                        "tier_accounts": _tier_accounts(config),
                        # ``routable`` (which names the router can serve itself, as opposed to
                        # delegation-only targets) already comes from _router_status().
                        **_router_status(),
                    }
                    self._send(200, json.dumps(response).encode("utf-8"), "application/json")
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}).encode("utf-8"), "application/json")
            return
        self._send(404, b'{"error":"not found"}', "application/json")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # The dashboard refreshes in parallel; closing or reloading the tab can
            # cancel either response after the server has already started writing.
            return

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="listen address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="listen port (default: 8765)")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="JSONL log path")
    args = parser.parse_args()
    Handler.log_path = args.log.expanduser()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Model-router log viewer: http://{args.host}:{args.port}", flush=True)
    print(f"Log: {Handler.log_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
