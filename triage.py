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
import time
import uuid
from collections import OrderedDict
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
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
# child session -> (originating root turn, recorded-at). A depth-1 caller is charged to the
# root turn it came from, not to whichever root turn the parent session admitted last.
# Bounded: entries expire after the TTL and the map keeps at most _ORIGIN_MAX of them.
_ORIGIN_BY_CHILD: Dict[str, Tuple[str, float]] = {}
_ORIGIN_MAX = 512
_ORIGIN_TTL_SECONDS = 6 * 60 * 60
_ORIGIN_AMBIGUOUS = "\x00ambiguous"
_monotonic = time.monotonic


@dataclass
class _RetainedAdmission:
    fingerprint: str
    transport: str
    scope: str
    root_turn: str
    requested: int
    recorded_at: float
    generation: str = ""
    nonce: str = ""
    state: str = "pending"


@dataclass(frozen=True)
class _AdmissionToken:
    key: Any
    fingerprint: str
    transport: str
    scope: str
    nonce: str = ""
    generation: str = ""
    response: Any = None

    @property
    def replay(self) -> bool:
        return bool(self.generation)


# This mirrors only F05's retained stable-ID window. It is not a second capacity
# owner: a pending entry prevents concurrent re-admission until F05 seals/unwinds.
_RETAINED_ADMISSIONS: "OrderedDict[Any, _RetainedAdmission]" = OrderedDict()
_ADMISSION_TOKEN: ContextVar[Optional[_AdmissionToken]] = ContextVar("model_router_retained_admission", default=None)


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
    operator_text = _without_host_injected_context(_without_router_contract(user_text))
    if not operator_text:
        return "no_user_text"
    if _is_delegation_outcome_text(operator_text) or _lifecycle_event_kind(request):
        return "lifecycle_event"
    if _current_turn_has_tool_activity(items, user_index):
        return "tool_activity_before_first_call"
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
    how = (f'Call it through tool_call in `calls` with name "{TOOL_NAME}" and the triage `arguments`. '
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

    routed = deepcopy(request)
    advisory = anthropic and _rejects_forced_tool_choice(routed)
    pinned = deepcopy(tool)
    owner, key = _tool_schema_slot(pinned)
    if via_bridge:
        schema = owner.get(key)
        if isinstance(schema, dict):
            properties = schema.setdefault("properties", {})
            properties["name"] = {"type": "string", "enum": [TOOL_NAME],
                                  "description": "The triage tool."}
            properties["arguments"] = deepcopy(SCHEMA_PARAMETERS)
            if not advisory:
                raw_calls = properties.get("calls")
                calls: Dict[str, Any] = raw_calls if isinstance(raw_calls, dict) else {"type": "array"}
                properties["calls"] = calls
                calls["minItems"] = 1
                calls["maxItems"] = 1
                raw_items = calls.get("items")
                items: Dict[str, Any] = raw_items if isinstance(raw_items, dict) else {"type": "object"}
                calls["items"] = items
                entry_properties = items.setdefault("properties", {})
                entry_properties["name"] = {"type": "string", "enum": [TOOL_NAME],
                                            "description": "The triage tool."}
                entry_properties["arguments"] = deepcopy(SCHEMA_PARAMETERS)
                items["required"] = ["name", "arguments"]
                schema["required"] = ["calls"]
            else:
                schema["required"] = ["name", "arguments"]
    if advisory:
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


def record_child_origin(child_session: Optional[str], parent_turn_id: Optional[str]) -> bool:
    """Bind a child session to the root turn that started it (the ``subagent_start`` payload).

    Recorded only when the origin is unambiguous: a non-empty parent turn that is not itself a
    sub-agent turn. A second, different origin for the same child marks it ambiguous; the gate
    then refuses its spawns rather than guess. Returns True when a link (or ambiguity) was stored.
    """
    child_session = str(child_session or "")
    parent_turn_id = str(parent_turn_id or "")
    if not child_session or not parent_turn_id or ":sa-" in parent_turn_id:
        return False
    now = _monotonic()
    with _LOCK:
        for key in [k for k, (_, at) in _ORIGIN_BY_CHILD.items() if now - at > _ORIGIN_TTL_SECONDS]:
            del _ORIGIN_BY_CHILD[key]
        known = _ORIGIN_BY_CHILD.pop(child_session, None)
        root = parent_turn_id
        if known is not None and known[0] != parent_turn_id:
            root = _ORIGIN_AMBIGUOUS
        _ORIGIN_BY_CHILD[child_session] = (root, now)
        while len(_ORIGIN_BY_CHILD) > max(1, _ORIGIN_MAX):
            del _ORIGIN_BY_CHILD[next(iter(_ORIGIN_BY_CHILD))]
    return True


def _child_origin_locked(child_session: str) -> Optional[str]:
    """Recorded root turn for a child (``_ORIGIN_AMBIGUOUS`` when conflicting); None if unknown or expired."""
    entry = _ORIGIN_BY_CHILD.get(child_session)
    if entry is None:
        return None
    if _monotonic() - entry[1] > _ORIGIN_TTL_SECONDS:
        del _ORIGIN_BY_CHILD[child_session]
        return None
    return entry[0]


def _stable_invocation(tool_name: str, args: Dict[str, Any], hook: Dict[str, Any]) -> Any:
    from .execution_adapters import stable_legacy_invocation
    return stable_legacy_invocation(tool_name, args, session_id=hook.get("session_id"),
                                    turn_id=hook.get("turn_id"), tool_call_id=hook.get("tool_call_id"))


def _purge_retained_admissions_locked() -> Tuple[int, float]:
    from .execution_adapters import (StableLegacyInvocation, retained_admission_limits,
                                     retained_invocation_generation)
    maximum, ttl = retained_admission_limits()
    now = _monotonic()
    for key, entry in list(_RETAINED_ADMISSIONS.items()):
        if now - entry.recorded_at <= ttl:
            continue
        if entry.generation:
            invocation = StableLegacyInvocation(key, entry.fingerprint, entry.transport, entry.scope)
            if retained_invocation_generation(invocation) == entry.generation:
                entry.recorded_at = now  # protected/in-flight F05 record outlives its finished-record TTL.
                continue
        del _RETAINED_ADMISSIONS[key]
    return maximum, ttl


def _bound_retained_admissions_locked(maximum: int) -> None:
    """Discard only mirrors whose exact F05 record has already gone away.

    Matching finished, protected, and pending records are never evicted here:
    F05 owns their LRU/TTL lifecycle, and in-flight records may temporarily
    exceed its finished-record target as well.
    """
    if maximum < 1:
        return
    from .execution_adapters import StableLegacyInvocation, retained_invocation_generation
    for key, entry in list(_RETAINED_ADMISSIONS.items()):
        if len(_RETAINED_ADMISSIONS) <= maximum:
            return
        if not entry.generation and entry.state == "pending":
            # A hook-only admission never reached F05. Retiring its bookkeeping
            # does not refund the already charged host attempt; a later delivery
            # must re-admit rather than borrow this abandoned permission.
            del _RETAINED_ADMISSIONS[key]
            continue
        if entry.generation:
            invocation = StableLegacyInvocation(key, entry.fingerprint, entry.transport, entry.scope)
            if retained_invocation_generation(invocation) != entry.generation:
                del _RETAINED_ADMISSIONS[key]


def _retained_replay_locked(invocation: Any) -> Optional[_RetainedAdmission]:
    entry = _RETAINED_ADMISSIONS.get(invocation.key)
    if entry is None:
        return None
    if (entry.fingerprint != invocation.fingerprint or entry.transport != invocation.transport
            or entry.scope != invocation.scope):
        return _RetainedAdmission("", "", "", "", 0, 0.0, "changed")
    if entry.generation:
        from .execution_adapters import retained_invocation_generation
        if retained_invocation_generation(invocation) != entry.generation:
            del _RETAINED_ADMISSIONS[invocation.key]
            return None
    _RETAINED_ADMISSIONS.move_to_end(invocation.key)
    return entry


def retained_replay_preflight(tool_name: str, args: Dict[str, Any], hook: Dict[str, Any]) -> Optional[str]:
    """Capture an exact sealed replay before fresh router policy checks.

    The captured bytes are carried by a ContextVar only across this host call's
    hook-to-middleware handoff.  A later F05 eviction therefore cannot turn a
    G1 replay decision into a G2 execution authorization.
    """
    invocation = _stable_invocation(tool_name, args, hook)
    if invocation is None:
        return None
    with _LOCK:
        _purge_retained_admissions_locked()
        entry = _retained_replay_locked(invocation)
        if entry is None or not entry.generation:
            return None
        if entry.generation == "changed":
            return "Retained invocation identity changed (payload, transport, or authority) for this stable tool call. Nothing was spawned."
        from .execution_adapters import retained_invocation_response
        retained, response = retained_invocation_response(invocation, entry.generation)
        if not retained:
            # An unsealed/oversized record is not a policy-exempt delivery, but
            # it remains a live F05 generation; only drop the mirror if F05 no
            # longer names this exact owner.
            from .execution_adapters import retained_invocation_generation
            if retained_invocation_generation(invocation) != entry.generation:
                _RETAINED_ADMISSIONS.pop(invocation.key, None)
            return None
        _ADMISSION_TOKEN.set(_AdmissionToken(invocation.key, invocation.fingerprint, invocation.transport,
                                             invocation.scope, generation=entry.generation, response=response))
        return ""


def consume_retained_admission(invocation: Any) -> Optional[_AdmissionToken]:
    """Return this hook's exact admission once, never a key-only exemption."""
    token = _ADMISSION_TOKEN.get()
    if token is None or invocation is None:
        return None
    if (token.key != invocation.key or token.fingerprint != invocation.fingerprint
            or token.transport != invocation.transport or token.scope != invocation.scope):
        return None
    _ADMISSION_TOKEN.set(None)
    if token.replay:
        return token
    with _LOCK:
        entry = _RETAINED_ADMISSIONS.get(invocation.key)
        if (entry is None or entry.nonce != token.nonce or entry.generation
                or entry.state != "pending"):
            return None
        entry.state = "started"
        entry.recorded_at = _monotonic()
        return token


def has_unbound_retained_admission(invocation: Any) -> bool:
    """Whether an admitted stable call is waiting for its one F05 owner."""
    if invocation is None:
        return False
    with _LOCK:
        entry = _RETAINED_ADMISSIONS.get(invocation.key)
        return bool(entry is not None and not entry.generation
                    and entry.fingerprint == invocation.fingerprint
                    and entry.transport == invocation.transport and entry.scope == invocation.scope)


def bind_retained_admission(invocation: Any, generation: str, *, create: bool = True,
                            admission: Optional[_AdmissionToken] = None) -> bool:
    """Bind triage's already-admitted stable invocation to its exact F05 generation."""
    if invocation is None or not generation:
        return False
    with _LOCK:
        maximum, _ttl = _purge_retained_admissions_locked()
        entry = _RETAINED_ADMISSIONS.get(invocation.key)
        if entry is None:
            if not create:
                return False
            entry = _RetainedAdmission(invocation.fingerprint, invocation.transport, invocation.scope,
                                       "", 1, _monotonic(), nonce=uuid.uuid4().hex)
            _RETAINED_ADMISSIONS[invocation.key] = entry
        elif (entry.fingerprint != invocation.fingerprint or entry.transport != invocation.transport
              or entry.scope != invocation.scope):
            return False
        elif admission is not None and (entry.nonce != admission.nonce or entry.state not in {"pending", "started"}
                                        or entry.generation):
            return False
        elif not create and entry.generation and entry.generation != generation:
            return False
        entry.generation = generation
        entry.state = "bound"
        entry.recorded_at = _monotonic()
        _RETAINED_ADMISSIONS.move_to_end(invocation.key)
        _bound_retained_admissions_locked(maximum)
    return True


def spawn_block(tool_name: str, args: Dict[str, Any], cfg: Dict[str, Any], hook: Dict[str, Any]) -> str:
    """Refusal text when this spawn exceeds the turn's admitted-attempt budget; an empty string admits it.

    Counts are charged on admission, so a refusal made here never consumes budget.
    Later host or handler refusals retain their admitted attempt: there is no refund seam.
    A depth-1 caller is charged to the root turn recorded for its own session at
    ``subagent_start``; a child with no (or an expired) record falls back to its parent
    session's latest root turn and the event says so (``origin_source``).
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
    stable = _stable_invocation(name, args, hook)
    root: Optional[str] = None
    origin_source: Optional[str] = None
    replayed = False
    with _LOCK:
        maximum, _ttl = _purge_retained_admissions_locked()
        retained = _retained_replay_locked(stable) if stable is not None else None
        if retained is not None and retained.generation == "changed":
            verdict, used = "retained_invocation_identity_changed", None
        elif retained is not None:
            verdict, used, replayed = "", _TURN_WORKERS.get(retained.root_turn, 0), True
            root, origin_source = retained.root_turn, "retained_replay"
        elif depth >= 2:
            verdict, used = "sub_worker_cannot_delegate", None
        elif depth == 1:
            linked = _child_origin_locked(session)
            if linked == _ORIGIN_AMBIGUOUS:
                verdict, used, origin_source = "turn_origin_ambiguous", None, "ambiguous"
            else:
                if linked is not None:
                    root, origin_source = linked, "child_link"
                else:
                    parent_session = str(getattr(agent, "_parent_session_id", "") or "")
                    root = _ROOT_TURN_BY_SESSION.get(parent_session) or f"session:{parent_session or session}"
                    origin_source = "latest_session_pointer"
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
        if not verdict and not replayed and stable is not None:
            nonce = uuid.uuid4().hex
            _RETAINED_ADMISSIONS[stable.key] = _RetainedAdmission(
                stable.fingerprint, stable.transport, stable.scope, root or "", requested, _monotonic(), nonce=nonce)
            _ADMISSION_TOKEN.set(_AdmissionToken(stable.key, stable.fingerprint, stable.transport, stable.scope,
                                                 nonce=nonce))
        elif not verdict and replayed and stable is not None and retained is not None and not retained.generation:
            _ADMISSION_TOKEN.set(_AdmissionToken(stable.key, stable.fingerprint, stable.transport, stable.scope,
                                                 nonce=retained.nonce))
        _bound_retained_admissions_locked(maximum)
    _log(cfg, {"event": "spawn_replayed" if replayed else "spawn_blocked" if verdict else "spawn_admitted",
               "reason": verdict or None, "tool": name, "depth": depth, "requested": requested,
               "used_before": used, "root_turn": root, "origin_source": origin_source,
               "turn_id": turn_id, "session_id": session})
    if verdict == "retained_invocation_identity_changed":
        return ("Retained invocation identity changed (payload, transport, or authority) for this stable tool call. "
                "Nothing was spawned.")
    if verdict == "turn_origin_ambiguous":
        return ("The user turn this worker started from is ambiguous (conflicting origin records), so its "
                "budget cannot be charged to a turn. Finish the remaining work yourself. Nothing was spawned.")
    if verdict == "sub_worker_cannot_delegate":
        return "Sub-workers cannot delegate further. Finish this task yourself. Nothing was spawned."
    if verdict == "worker_child_limit":
        return (f"A worker may start at most {max_children(cfg)} sub-worker, and this call would exceed it. "
                "Finish the remaining work yourself. Nothing was spawned.")
    if verdict == "turn_worker_limit":
        return (f"Worker budget for this user turn: {used} of {cap} admitted spawn attempts (sub-workers "
                f"included); this call asks for {requested} more. Do the remaining work yourself or wait for "
                "the running workers. Nothing was spawned.")
    return ""


def reset_state() -> None:
    """Tests only."""
    with _LOCK:
        _TURN_WORKERS.clear()
        _WORKER_CHILDREN.clear()
        _ROOT_TURN_BY_SESSION.clear()
        _ORIGIN_BY_CHILD.clear()
        _RETAINED_ADMISSIONS.clear()
