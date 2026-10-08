"""Root-turn triage: the model that receives the prompt decides who does the work.

The forced conductor (``orchestration``) planned every actionable turn in a
separate child with little context, which turned a pricing-copy fix into a
planner plus two workers (2026-09-30). Switching it off left the parent alone
with only advisory routing, and it then delegated almost nothing: a fix, a
dev release and a test plan ran as 39 parent calls (2026-10-05).

Triage sits between the two. On the first call of a user turn the parent must
call ``triage_task`` -- the same model, with the whole conversation in view --
and state whether it works alone or splits the work. The decision is its own;
the router only makes it explicit, records it, and enforces the worker
budget on every spawn that follows:

* at most ``max_workers_per_turn`` workers per user turn, sub-workers included;
* each worker may start ``max_children_per_worker`` sub-workers (one);
* a sub-worker never delegates.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

_logger = logging.getLogger("model_router.triage")

TOOL_NAME = "triage_task"
_TOOL_NAMES = frozenset({TOOL_NAME, f"mcp__{TOOL_NAME}"})
_BRIDGE_NAMES = frozenset({"tool_call", "mcp__tool_call"})
_SPAWN_TOOLS = frozenset({"delegate_task", "delegate_claude"})

_LOCK = threading.Lock()
# Worker accounting lives in memory: a budget belongs to one user turn of one
# running process, and a restart ending it early is harmless.
_TURN_WORKERS: Dict[str, int] = {}
_WORKER_CHILDREN: Dict[str, int] = {}
_ROOT_TURN_BY_SESSION: Dict[str, str] = {}

SCHEMA_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["solo", "delegate"],
            "description": "solo: you do the whole task yourself. delegate: you split it to workers.",
        },
        "kind": {
            "type": "string",
            "description": "Kind of work, e.g. question, fix, feature, deploy, investigation, review, design.",
        },
        "scope": {
            "type": "string",
            "description": "Estimated size: steps, files or systems touched.",
        },
        "rationale": {"type": "string", "description": "One sentence: why this decision."},
        "subtasks": {
            "type": "array",
            "maxItems": 4,
            "description": "For delegate only: the parts you will hand to workers.",
            "items": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "Objective and acceptance criteria."},
                    "route": {"type": "string", "description": "Worker route from the routing note, e.g. grok, sonnet5, sol."},
                },
                "required": ["goal", "route"],
            },
        },
    },
    "required": ["decision", "rationale"],
}

_DESCRIPTION = (
    "Record your triage of the current user request: whether you do it yourself or split it "
    "to workers. Call it once, first, when the router asks for it."
)


def policy(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("triage") or {}


def enabled(cfg: Dict[str, Any]) -> bool:
    return bool(policy(cfg).get("enabled"))


def _limit(cfg: Dict[str, Any], key: str, default: int) -> int:
    try:
        return max(0, int(policy(cfg).get(key, default)))
    except (TypeError, ValueError):
        return default


def max_workers(cfg: Dict[str, Any]) -> int:
    return _limit(cfg, "max_workers_per_turn", 4)


def max_children(cfg: Dict[str, Any]) -> int:
    return _limit(cfg, "max_children_per_worker", 1)


def _log(cfg: Dict[str, Any], event: Dict[str, Any]) -> None:
    from .hermes_paths import hermes_path

    try:
        path = hermes_path(str(policy(cfg).get("path") or "~/.hermes/logs/model-router-triage.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), **event}
        with _LOCK, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        _logger.debug("triage log unavailable: %s", exc)


# --- the forced first call ---------------------------------------------------

# Imperative stems, accent-free (the text is normalised first). A prompt with
# none of them that is a question or a few words is conversation: triaging it
# would cost a round trip to decide what is already obvious.
_ACTION = re.compile(
    r"\b(?:csinal|csinald|javit|rakd|rakj|add\s+ki|adj|told|tolt|modosit|irj|ird|torol|toroljuk|"
    r"keszit|epit|implemental|deploy|fix|build|refactor|vizsgal|nezd|nezz|ellenoriz|teszt|mehet|"
    r"indit|allits|cserel|frissit|telepit|migral|kuld|create|write|update|change|add|remove|"
    r"check|investigate|review|run|implement|make|do\s+it)"
)
_INTERNAL_PREFIXES = (
    "review the conversation above and consider saving to memory",
    "what do you see in this image?",
    "[important: background process",
    "delegalt feladat befejezesi esemenye",
)


def is_conversational(text: str) -> bool:
    """A question or a short remark with no imperative in it."""
    if _ACTION.search(text):
        return False
    return "?" in text or len(text) <= 60


def skip_reason(kwargs: Dict[str, Any], cfg: Dict[str, Any], kind: str = "") -> Optional[str]:
    """Why this call gets no triage, or None when it must triage."""
    from . import (
        _current_turn_has_tool_activity, _find_delegate_tool, _is_delegation_outcome_text,
        _last_user_text_and_index, _lifecycle_event_kind, _normalise, _request_items,
        _without_host_injected_context, _without_router_contract,
    )

    if not enabled(cfg):
        return "triage_disabled"
    request = kwargs.get("request")
    if not isinstance(request, dict):
        return "request_not_a_dict"
    if int(kwargs.get("api_call_count", 1) or 1) != 1:
        return "mid_loop_call"
    if str(kwargs.get("platform", "")).casefold() == "subagent" or ":sa-" in str(kwargs.get("turn_id", "")):
        return "subagent_turn"
    if _find_delegate_tool(request) is None:
        return "no_delegate_task_tool"
    if _vehicle(request) is None:
        return "no_triage_tool"
    items = _request_items(request)
    user_text, user_index = _last_user_text_and_index(items)
    if not user_text:
        return "no_user_text"
    if _is_delegation_outcome_text(user_text) or _lifecycle_event_kind(request):
        return "lifecycle_event"
    if _current_turn_has_tool_activity(items, user_index):
        return "tool_activity_before_first_call"
    operator_text = _without_host_injected_context(_without_router_contract(user_text))
    text = _normalise(operator_text)
    if not text:
        return "no_user_text"
    if text.startswith(_INTERNAL_PREFIXES):
        return "internal_prompt"
    # The operator named the delegation tool: the decision is already made.
    if re.search(r"\bdelegate_(?:claude|task)\b", operator_text):
        return "explicit_delegation_tool"
    # The classifier's chat label is not trusted over an imperative: it read
    # "rendben inditsd el ennek a javitasat majd rakjad ki developmentre" as
    # brief non-actionable conversation (2026-10-05), and that fix ran untriaged.
    if kind == "chat" and not _ACTION.search(text):
        return "chat"
    if is_conversational(text):
        return "conversational"
    return None


def _tool_name(tool: Dict[str, Any]) -> str:
    return str(tool.get("name") or (tool.get("function") or {}).get("name") or "")


def _vehicle(request: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], bool]]:
    """The tool that carries the triage call: (tool, via_bridge).

    Hermes defers plugin tools behind tool_search/tool_describe/tool_call, so
    triage_task is usually reachable only through the ``tool_call`` bridge.
    """
    tools = [tool for tool in request.get("tools") or [] if isinstance(tool, dict)]
    direct = next((tool for tool in tools if _tool_name(tool) in _TOOL_NAMES), None)
    if direct is not None:
        return direct, False
    bridge = next((tool for tool in tools if _tool_name(tool) in _BRIDGE_NAMES), None)
    return (bridge, True) if bridge is not None else None


def instruction(cfg: Dict[str, Any], *, via_bridge: bool) -> str:
    how = (f'Call it through tool_call with name "{TOOL_NAME}" and the triage as `arguments`. '
           if via_bridge else "")
    return (
        "\n\n[ROUTER TRIAGE] Before any other work, assess this request and call "
        f"{TOOL_NAME} once. {how}You decide whether you do it yourself or split it to workers.\n"
        "Delegate when: the work splits into independent parts that can run in parallel; a part "
        "fits a specialised route better (implementation, read-only exploration, design, an "
        "independent review -- see the routing note); or a part needs so much reading that it "
        "would flood your own context.\n"
        "Work solo when: it is conversation or a question; a deploy or release step; a small "
        "change in one or two files; or a refinement of what you just did in this conversation, "
        "where your context is the asset.\n"
        "Account usage counts: move work off an account that is close to its limit.\n"
        f"Limits the router enforces: at most {max_workers(cfg)} workers this turn, sub-workers "
        f"included; each worker may start {max_children(cfg)} sub-worker; sub-workers never delegate.\n"
    )


def force(request: Dict[str, Any], cfg: Dict[str, Any], *, anthropic: bool) -> Optional[Dict[str, Any]]:
    """A copy of ``request`` whose only possible move is the triage call."""
    vehicle = _vehicle(request)
    if vehicle is None:
        return None
    tool, via_bridge = vehicle
    from . import _append_user_instruction, _rejects_forced_tool_choice, _tool_schema_slot

    pinned = deepcopy(tool)
    owner, key = _tool_schema_slot(pinned)
    if via_bridge:
        schema = owner.get(key)
        if isinstance(schema, dict):
            properties = schema.setdefault("properties", {})
            properties["name"] = {"type": "string", "enum": [TOOL_NAME],
                                  "description": "The triage tool."}
            properties["arguments"] = deepcopy(SCHEMA_PARAMETERS)
            schema["required"] = ["name", "arguments"]
    routed = deepcopy(request)
    if anthropic and _rejects_forced_tool_choice(routed):
        # Anthropic refuses a forced tool_choice while thinking is on, and on Claude 5
        # models at all. One optional tool would let the reply end the turn, so the
        # toolset stays whole and the triage is asked for, not forced.
        routed["tools"] = [pinned if isinstance(item, dict) and _tool_name(item) == _tool_name(tool) else item
                           for item in routed.get("tools") or []]
        _append_user_instruction(routed, instruction(cfg, via_bridge=via_bridge))
        return routed
    routed["tools"] = [pinned]
    if anthropic:
        routed["tool_choice"] = {"type": "tool", "name": _tool_name(pinned)}
    else:
        routed["tool_choice"] = "required"
        routed["parallel_tool_calls"] = False
    _append_user_instruction(routed, instruction(cfg, via_bridge=via_bridge))
    return routed


# --- the tool ------------------------------------------------------------------

def _caller() -> Any:
    try:
        from agent.subagent_lifecycle import get_active_subagent_parent
        return get_active_subagent_parent()
    except Exception:
        return None


def handle_triage(args: Dict[str, Any], **_kwargs: Any) -> str:
    """Record the decision and tell the parent what follows from it. Never raises."""
    try:
        from . import _load_config

        cfg = _load_config()
        args = args if isinstance(args, dict) else {}
        decision = str(args.get("decision") or "").strip().casefold()
        subtasks = args.get("subtasks") if isinstance(args.get("subtasks"), list) else []
        agent = _caller()
        _log(cfg, {
            "event": "triage_decision",
            "session_id": getattr(agent, "session_id", None),
            "turn_id": getattr(agent, "_relay_pending_turn_id", None),
            "decision": decision,
            "kind": str(args.get("kind") or "")[:80],
            "scope": str(args.get("scope") or "")[:200],
            "rationale": str(args.get("rationale") or "")[:400],
            "subtasks": [{"route": str((t or {}).get("route") or "")[:40],
                          "goal_chars": len(str((t or {}).get("goal") or ""))}
                         for t in subtasks if isinstance(t, dict)],
        })
        limits = (f"at most {max_workers(cfg)} workers this turn, sub-workers included; each worker may "
                  f"start {max_children(cfg)} sub-worker for a clearly separable part; sub-workers never delegate")
        if decision == "delegate":
            return json.dumps({"recorded": "delegate", "next": (
                f"Dispatch the {len(subtasks) or 'planned'} subtask(s) now with delegate_task or "
                "delegate_claude, on the routes the routing note advises. Write each goal as objective "
                "and acceptance criteria. You keep integration and final verification; do not redo a "
                f"worker's job yourself. Router limits: {limits}.")}, ensure_ascii=False)
        return json.dumps({"recorded": "solo", "next": (
            "Do the work yourself now with your normal tools. If it proves larger than assessed you "
            f"may still delegate. Router limits: {limits}.")}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"recorded": "unknown", "next": f"Proceed with the task ({type(exc).__name__})."})


def register(ctx: Any, cfg: Dict[str, Any]) -> bool:
    try:
        handle = ctx.register_tool(
            name=TOOL_NAME, toolset="delegation",
            schema={"name": TOOL_NAME, "description": _DESCRIPTION, "parameters": SCHEMA_PARAMETERS},
            handler=handle_triage, check_fn=_available, description=_DESCRIPTION, emoji="🧭",
        )
    except Exception as exc:
        _logger.warning("triage: %s not registered: %s", TOOL_NAME, exc)
        return False
    return handle is not None


def _available() -> bool:
    try:
        from . import _load_config
        return enabled(_load_config())
    except Exception:
        return False


# --- the worker budget ---------------------------------------------------------

def _requested_count(tool_name: str, args: Dict[str, Any]) -> int:
    tasks = args.get("tasks")
    if isinstance(tasks, list):
        return max(1, len(tasks))
    return 1


def spawn_block(tool_name: str, args: Dict[str, Any], cfg: Dict[str, Any], hook: Dict[str, Any]) -> str:
    """Refusal text when this spawn breaks the turn's worker budget; "" admits it.

    Counts are charged on admission, so a refused call never consumes budget.
    """
    name = str(tool_name or "").removeprefix("mcp__")
    if not enabled(cfg) or name not in _SPAWN_TOOLS or not isinstance(args, dict):
        return ""
    if args.get("action", "spawn") in ("list", "steer", "stop"):
        return ""
    turn_id = str(hook.get("turn_id") or "")
    agent = _caller()
    session = str(getattr(agent, "session_id", "") or hook.get("session_id") or "")
    if agent is not None:
        depth = int(getattr(agent, "_delegate_depth", 0) or 0)
    else:
        depth = 1 if ":sa-" in turn_id else 0
    requested = _requested_count(name, args)
    cap = max_workers(cfg)
    with _LOCK:
        if depth >= 2:
            verdict, used = "sub_worker_cannot_delegate", None
        elif depth == 1:
            parent_session = str(getattr(agent, "_parent_session_id", "") or "")
            root = _ROOT_TURN_BY_SESSION.get(parent_session) or f"session:{parent_session or session}"
            used = _TURN_WORKERS.get(root, 0)
            own = _WORKER_CHILDREN.get(session, 0)
            if own + requested > max_children(cfg):
                verdict = "worker_child_limit"
            elif used + requested > cap:
                verdict = "turn_worker_limit"
            else:
                verdict = ""
                _WORKER_CHILDREN[session] = own + requested
                _TURN_WORKERS[root] = used + requested
        else:
            root = turn_id or f"session:{session}"
            if session:
                _ROOT_TURN_BY_SESSION[session] = root
            used = _TURN_WORKERS.get(root, 0)
            if used + requested > cap:
                verdict = "turn_worker_limit"
            else:
                verdict = ""
                _TURN_WORKERS[root] = used + requested
    _log(cfg, {"event": "spawn_blocked" if verdict else "spawn_admitted", "reason": verdict or None,
               "tool": name, "depth": depth, "requested": requested, "used_before": used,
               "turn_id": turn_id, "session_id": session})
    if verdict == "sub_worker_cannot_delegate":
        return "Sub-workers cannot delegate further. Finish this task yourself. Nothing was spawned."
    if verdict == "worker_child_limit":
        return (f"A worker may start at most {max_children(cfg)} sub-worker, and this call would exceed it. "
                "Finish the remaining work yourself. Nothing was spawned.")
    if verdict == "turn_worker_limit":
        return (f"Worker budget for this user turn: {used} of {cap} workers already started (sub-workers "
                f"included); this call asks for {requested} more. Do the remaining work yourself or wait for "
                "the running workers. Nothing was spawned.")
    return ""


def reset_state() -> None:
    """Tests only."""
    with _LOCK:
        _TURN_WORKERS.clear()
        _WORKER_CHILDREN.clear()
        _ROOT_TURN_BY_SESSION.clear()
