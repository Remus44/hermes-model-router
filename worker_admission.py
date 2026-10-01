"""Account admission at execution boundaries, shared by worker transports."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from . import usage_guard

import logging

_logger = logging.getLogger("model_router.worker_admission")

CLAUDE_OFF = "Claude workers are switched off in Settings. The parent model is unchanged."


def claude_switched_off(account, model, cfg):
    """Whether a Claude worker may not run: the Claude target its model maps to is off.

    A model the delegation tiers do not list (a dated snapshot, say) maps to its
    target through Hermes's own ``delegation.targets``; one neither names needs
    any Claude switch on. The switch alone decides, not the cooldown: a cooling
    tier is a wait, not a reason to stop a child mid-task.
    """
    model = str(model or "")
    if account != "anthropic" and not model.startswith("claude-"):
        return False
    from . import claude_delegation, _delegation_targets_detail

    claude_targets = tuple(claude_delegation.TARGET_FOR_TIER.values())
    target = claude_delegation.target_for_model(model, cfg) or next(
        (name for name, spec in _delegation_targets_detail().items()
         if name in claude_targets and spec.get("model") == model), None)
    targets = (target,) if target else claude_targets
    switches = cfg.get("callable") or {}
    return not any(switches.get(name) is True for name in targets)


def refusal(account, tier, cfg, *, blocking=False):
    if claude_switched_off(account, tier, cfg):
        return CLAUDE_OFF
    if not usage_guard.guarded(account, cfg):
        return ""
    reading = (usage_guard.read if blocking else usage_guard.peek)(account, cfg)
    return usage_guard.apply(account, tier, cfg, reading).refused


def delegate_task_route():
    from tools.delegate_tool_config import _load_config
    from agent.subagent_lifecycle import get_active_subagent_parent

    route = _load_config()
    parent = get_active_subagent_parent()
    return (route.get("provider") or getattr(parent, "provider", ""),
            route.get("model") or getattr(parent, "model", ""))


def guard_tool_execution(**kwargs):
    from . import _delegated_claude_review_status, _load_config, _delegation_targets_detail
    from .claude_delegation import TIER_FOR_TARGET
    from .claude_opus_bridge import CLAUDE_REVIEW_MODELS

    args, next_call = kwargs.get("args") or {}, kwargs["next_call"]
    name = str(kwargs.get("tool_name") or "").removeprefix("mcp__")
    if name != "delegate_task" or args.get("action", "spawn") in ("list", "steer", "stop"):
        return next_call(args)
    cfg = _load_config()
    if not cfg.get("enabled", True):
        return next_call(args)

    # Resolved only for an actual delegate_task spawn: every other tool call
    # (terminal, read_file, ...) must never pay for this lookup.
    try:
        from agent.subagent_lifecycle import get_active_subagent_parent
        from tools.delegate_tool_progress import _resolve_workspace_hint
        workspace_hint = _resolve_workspace_hint(get_active_subagent_parent())
        dispatch_cwd = Path(workspace_hint) if workspace_hint else None
    except Exception as exc:
        dispatch_cwd = None
        _logger.debug("worker_admission: could not resolve the parent's workspace hint: %s", exc)

    account, model = delegate_task_route()
    targets = _delegation_targets_detail()
    tasks = args.get("tasks") or [args]
    switches = cfg.get("callable") or {}
    for task in tasks if isinstance(tasks, list) else [args]:
        task = task if isinstance(task, dict) else {}
        goal = str(task.get("goal") or args.get("goal") or "")
        requested_model = str(task.get("model") or args.get("model") or "").strip()
        # The review lookup reads paths out of the goal text. Raising from here
        # would make Hermes skip this whole guard (execution middleware fails
        # open), so a failed lookup falls through to the ordinary target check.
        try:
            review, _reason = _delegated_claude_review_status(
                goal, cfg, dispatch_cwd=dispatch_cwd, requested_model=requested_model,
                at_dispatch=True,
            )
        except Exception as exc:
            review = None
            _logger.warning("worker_admission: review lookup failed, checking the target instead: %s", exc)
        if review is not None:
            _repo, alias = review
            message = refusal("anthropic", CLAUDE_REVIEW_MODELS[alias], cfg, blocking=True)
        else:
            name = requested_model.casefold()
            target = targets.get(name, {})
            # A Claude target named by its switch obeys that switch, even when its
            # model string (a dated snapshot, say) is not one the delegation tiers list.
            if name in TIER_FOR_TARGET and switches.get(name) is not True:
                message = CLAUDE_OFF
            else:
                message = refusal(target.get("provider") or account, target.get("model") or model, cfg, blocking=True)
        if message:
            return json.dumps({"error": message + " Re-dispatch on an available account; do not retry this account."})
    return next_call(args)


def stopped_response(message, model):
    """A zero-token router stop, consumable by Hermes's supported response shapes.

    Raising from execution middleware before next_call fails open in Hermes.
    Returning a final message instead stops the worker without contacting the
    provider or inventing task results.
    """
    text = "[ROUTER WORKER STOPPED] No work was performed by this call. " + message + " Re-dispatch this task on an available account."
    usage = SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0,
                            prompt_tokens=0, completion_tokens=0)
    return SimpleNamespace(
        model=model, id="router-worker-stopped", status="completed", usage=usage,
        output_text=text, output=[SimpleNamespace(type="message", status="completed",
            content=[SimpleNamespace(type="output_text", text=text)])],
        content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
        choices=[SimpleNamespace(index=0, finish_reason="stop", message=SimpleNamespace(
            role="assistant", content=text, tool_calls=None, reasoning_content=None))],
    )
