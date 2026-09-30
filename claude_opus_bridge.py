#!/usr/bin/env python3
"""Explicit Claude Code Opus 5.5 bridge for conservative coding tasks.

This is intentionally separate from ``route_llm_request``: that middleware
serves OpenAI-compatible requests and must never pretend that an Anthropic
model can be reached by replacing its model string.  The bridge invokes the
user's authenticated Claude Code CLI, validates its reported canonical model,
and records the *effective* route in the shared ModelRouter JSONL log.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# A plugin script can be run directly; make its parent plugin namespace importable
# without depending on the caller's PYTHONPATH. Only needed in script mode: a
# package load already has its parent importable through the normal package
# machinery, and inserting the resolved plugin-parent directory at sys.path[0]
# there would leak into every other import in the host Hermes process.
_PLUGIN_PARENT = Path(__file__).resolve().parent.parent

if __package__:
    # Loaded as part of the plugin package. Hermes names it
    # ``hermes_plugins.model_router``, so a top-level ``model_router`` import
    # would fail there (or load a second copy of the router elsewhere).
    from . import RouteDecision, _load_config, _log_decision, _normalise
    from .hermes_paths import hermes_path
else:
    if str(_PLUGIN_PARENT) not in sys.path:
        sys.path.insert(0, str(_PLUGIN_PARENT))
    from model_router import RouteDecision, _load_config, _log_decision, _normalise
    from model_router.hermes_paths import hermes_path

CANONICAL_OPUS_MODEL = "claude-opus-5-5"
# The review label picks the Claude tier. Two are offered so a conductor can
# spend the cheaper one on routine checks and reserve Opus for hard review --
# the point of reaching Claude at all is that it draws on a separate quota, and
# one tier would exhaust that quota on work Sonnet handles fine.
CLAUDE_REVIEW_MODELS = {"opus": CANONICAL_OPUS_MODEL, "sonnet": "claude-sonnet-5-5"}
DEFAULT_LIFECYCLE_PATH = hermes_path("~/.hermes/logs/claude-code-bridge.jsonl")
# Eight turns repeatedly truncates read-only reviews before their verdict. A
# bounded 16-turn review is cheaper than discarding and re-running an almost
# complete Opus audit; callers may lower it for narrow checks.
DEFAULT_REVIEW_MAX_TURNS = 16
DEFAULT_CODING_MAX_TURNS = 8
# Historical successful 12-turn review took 182 seconds; a 300-second outer
# process kill can therefore discard a legitimate 16-turn audit. Keep review
# wall time bounded but allow the configured 16-turn/$5 contract to finish.
DEFAULT_REVIEW_TIMEOUT_SECONDS = 600
DEFAULT_CODING_TIMEOUT_SECONDS = 300


class ClaudeBridgeFailure(RuntimeError):
    """A terminal CLI outcome with typed failure evidence for the caller."""

    def __init__(self, message: str, failure_kind: str, *, identity: Any = None,
                 mismatch: Any = None, refused: bool = False) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind
        self.failure_class = (
            "exact-route-mismatch" if failure_kind == "model-mismatch" else
            "capability" if failure_kind == "capability" else
            "timeout" if failure_kind == "timeout" else "execution-error"
        )
        self.identity = identity
        self.mismatch = mismatch
        self.refused = refused
        self.route_reason = message


CODING_SIGNAL = re.compile(
    r"\b(implement|debug|fix|refactor|patch|test|code|repository|repo|"
    r"backend|api|parser|router|middleware|python|javascript|typescript|"
    r"write a (?:script|function|module)|javitsd|hibakeres|kod|teszt|"
    r"modositsd a (?:kodot|parsert|routert))\b",
    re.IGNORECASE,
)
DESIGN_SIGNAL = re.compile(
    r"\b(ui|ux|css|layout|visual design|product design|wireframe|mockup|"
    r"design system|styling|tipograf|szinpaletta|felulet)\b",
    re.IGNORECASE,
)
OVERRIDE = re.compile(r"^\s*\[(opus5|opus)\](?:\s|$)", re.IGNORECASE)
REVIEW_OVERRIDE = re.compile(r"^\s*\[(opus|sonnet)5?-review\](?:\s|$)", re.IGNORECASE)


def review_model_alias(task: str) -> str | None:
    """Which Claude tier an explicit review label asks for, if any."""
    match = REVIEW_OVERRIDE.match(_normalise(task or ""))
    return match.group(1).casefold() if match else None


def classify_coding_dispatch(task: str) -> tuple[bool, str]:
    """Select only explicit or clearly non-design coding work for the bridge."""
    # Match against accent-stripped text, exactly as the router's own gates do.
    # Both signal sets spell their Hungarian terms unaccented ("tipograf",
    # "szinpaletta"), so matching raw input silently failed on real Hungarian —
    # which meant the Sol-only design veto did not fire for it either.
    normalised = _normalise(task or "")
    # A host-side policy has already bounded explicit [opus]/[opus5] requests.
    # Keep ordinary unlabelled design work excluded, while allowing that narrow
    # manual contract to reach the bridge without a Sol/Terra/Spark pre-loop.
    if OVERRIDE.match(normalised):
        return True, "explicit Claude Opus 5.5 coding override"
    if DESIGN_SIGNAL.search(normalised):
        return False, "Sol-only design/CSS/UI work is not eligible for Claude coding dispatch"
    if CODING_SIGNAL.search(normalised):
        return True, "conservative non-design coding classifier"
    return False, "not an explicit or conservative coding task; keep the normal ModelRouter route"


def classify_review_dispatch(task: str) -> tuple[bool, str]:
    """Allow only an explicit Claude tier request for a read-only review."""
    if REVIEW_OVERRIDE.match(_normalise(task or "")):
        return True, "explicit Claude read-only review override"
    return False, "review dispatch requires an explicit [opus-review] or [sonnet-review] prefix"


def _effective_model(payload: dict[str, Any], expected: str = CANONICAL_OPUS_MODEL) -> str:
    """Return Claude Code's served model from per-model usage evidence only."""
    usage = payload.get("modelUsage") or {}
    if isinstance(usage, dict):
        # Claude Code can report small internal/helper usage alongside the main
        # agent model. Prefer the required canonical route when it is present.
        if expected in usage:
            return expected
        for model, details in usage.items():
            if isinstance(details, dict) or details is not None:
                return str(model)
    return ""


def _append_lifecycle(path: Path | None, event: dict[str, Any]) -> None:
    """Append one self-contained event without ever persisting task text."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def _process_start_identity(pid: int) -> int | None:
    try:
        from gateway.status import get_process_start_time
        return get_process_start_time(pid)
    except Exception:
        try:
            return int(Path(f"/proc/{pid}/stat").read_text().split()[21])
        except (OSError, ValueError, IndexError):
            return None


def _terminal_state(payload: dict[str, Any] | None, *, timeout: bool = False, malformed: bool = False,
                    returncode: int = 0, expected_model: str = CANONICAL_OPUS_MODEL) -> str:
    """Apply the fixed terminal precedence required by the bridge contract."""
    if timeout:
        return "timeout"
    if malformed:
        return "error"
    payload = payload or {}
    subtype = str(payload.get("subtype") or "").casefold()
    if "max_turn" in subtype:
        return "max-turn"
    if "budget" in subtype:
        return "budget"
    if returncode:
        return "error"
    if _effective_model(payload, expected_model) != expected_model:
        return "error"
    return "success" if subtype in {"success", ""} and not payload.get("is_error") else "error"


def dispatch(task: str, repo: Path, *, write: bool = False, review: bool = False, timeout: int | None = None,
             max_turns: int | None = None, model: str | None = None, max_budget_usd: float = 5.0,
             requested_alias: str | None = None, adjustment: str = "",
             identity: Any = None, parent_session_id: str | None = None, parent_turn_id: str | None = None,
             lifecycle_path: Path | None = None) -> dict[str, Any]:
    if review and write:
        raise ValueError("a Claude review is always read-only")
    # Not defaulted at the signature: a caller that names no lifecycle log does
    # not get the production one. Defaulting there is how the test suite wrote
    # 220 mocked runs into it, which then read back as real Claude activity.
    if lifecycle_path is None:
        lifecycle_path = hermes_path(
            (_load_config().get("coding_agent") or {}).get("lifecycle_path") or ""
        ) if (_load_config().get("coding_agent") or {}).get("lifecycle_path") else None
    eligible, reason = classify_review_dispatch(task) if review else classify_coding_dispatch(task)
    if not eligible:
        raise ValueError(reason)
    # A review names its tier in the label; coding dispatch stays on Opus.
    alias = (model or (review_model_alias(task) if review else None) or "opus").casefold()
    if alias not in CLAUDE_REVIEW_MODELS:
        raise ValueError(f"unknown Claude tier {alias!r}; expected one of {sorted(CLAUDE_REVIEW_MODELS)}")
    requested_alias = (requested_alias or alias).casefold()
    if requested_alias not in CLAUDE_REVIEW_MODELS:
        raise ValueError(f"unknown requested Claude tier {requested_alias!r}")
    expected_model = CLAUDE_REVIEW_MODELS[alias]
    # A preferred admission may step Opus down to Sonnet. Validate the model the
    # admitted tier was asked to serve, while ``identity`` still records the
    # original requested tier for substitution provenance. Exact mode is the one
    # that must enforce its canonical requested identity on the wire.
    if (getattr(identity, "selection_mode", "") == "exact"
            and getattr(getattr(identity, "requested", None), "known", False)):
        expected_model = str(identity.requested.value)
    model_argument = expected_model if getattr(identity, "selection_mode", "") == "exact" else alias
    if not repo.is_dir():
        raise ValueError(f"repository directory does not exist: {repo}")
    resolved_max_turns = max_turns if max_turns is not None else (DEFAULT_REVIEW_MAX_TURNS if review else DEFAULT_CODING_MAX_TURNS)
    if isinstance(resolved_max_turns, bool) or not isinstance(resolved_max_turns, int) or resolved_max_turns < 1:
        raise ValueError("max_turns must be a positive integer")
    budget = float(max_budget_usd)
    if not math.isfinite(budget) or budget < 0.01:
        raise ValueError("max_budget_usd must be at least 0.01 and finite")
    default_timeout = DEFAULT_REVIEW_TIMEOUT_SECONDS if review else DEFAULT_CODING_TIMEOUT_SECONDS
    resolved_timeout = default_timeout if timeout is None else timeout
    if resolved_timeout < 1:
        raise ValueError("timeout must be positive")
    # A review must not be killed by an accidental legacy --timeout 300 while
    # its documented 16-turn budget remains valid.
    if review:
        resolved_timeout = max(resolved_timeout, DEFAULT_REVIEW_TIMEOUT_SECONDS)
    prompt = task if not review else (
        f"{task}\n\nYou are a reviewer. Inspect and report evidence, risks, and recommendations only. "
        "Do not modify files or execute commands. Return your final PASS/FAIL/UNAVAILABLE verdict and P0-P3 findings "
        "as soon as you have enough evidence; do not continue exploratory reading after the verdict is supported."
    )
    # Keep the prompt off argv: a review request may contain a large dirty diff,
    # and Linux otherwise rejects the process before Claude can start with
    # E2BIG/"Argument list too long". Claude Code's print mode accepts text on
    # stdin when no positional prompt is supplied.
    command = [
        "claude", "-p", "--model", model_argument,
        "--max-turns", str(resolved_max_turns), "--max-budget-usd", f"{budget:.2f}", "--output-format", "json",
    ]
    # Results are accepted only when Claude reports the requested canonical tier.
    # Do not request a lower-tier fallback that this contract must reject.
    # Read-only runs override the available tool set and additionally deny every
    # mutating or shell-capable built-in. `--allowedTools Read` alone can be
    # widened by an existing Claude Code permission profile, so it is not a
    # sufficient reviewer boundary.
    if write:
        command += ["--allowedTools", "Read,Edit,Write,Bash"]
    else:
        command += ["--tools", "Read", "--disallowedTools", "Edit,Write,Bash,Agent,WebSearch,WebFetch"]
    bridge_run_id = str(uuid.uuid4())
    started_at = time.time()
    base_event = {
        "schema_version": 1, "bridge_run_id": bridge_run_id,
        "parent_session_id": parent_session_id or "", "parent_turn_id": parent_turn_id or "",
        "review": review, "requested_read_only": not write, "pid": os.getpid(),
        "requested_model": expected_model,
        "requested_tier": requested_alias, "effective_tier": alias,
        **({"adjusted": adjustment} if adjustment else {}),
        "process_started_at": _process_start_identity(os.getpid()),
    }
    _append_lifecycle(lifecycle_path, {**base_event, "event": "started", "state": "running", "timestamp": started_at})
    try:
        completed = subprocess.run(command, cwd=str(repo), text=True, input=prompt, capture_output=True, timeout=resolved_timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        _append_lifecycle(lifecycle_path, {**base_event, "event": "terminal", "state": "timeout",
                                          "timestamp": time.time(), "duration_seconds": time.time() - started_at})
        raise ClaudeBridgeFailure("Claude Code timed out", "timeout", identity=identity) from exc
    except OSError:
        _append_lifecycle(lifecycle_path, {**base_event, "event": "terminal", "state": "error",
                                          "timestamp": time.time(), "duration_seconds": time.time() - started_at})
        raise
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        _append_lifecycle(lifecycle_path, {**base_event, "event": "terminal", "state": "error",
                                          "timestamp": time.time(), "duration_seconds": time.time() - started_at,
                                          "malformed": True, "returncode": completed.returncode})
        raise ClaudeBridgeFailure("Claude Code did not return JSON output", "malformed-json", identity=identity) from exc
    if not isinstance(payload, dict):
        _append_lifecycle(lifecycle_path, {**base_event, "event": "terminal", "state": "error",
                                          "timestamp": time.time(), "duration_seconds": time.time() - started_at,
                                          "malformed": True, "returncode": completed.returncode})
        raise ClaudeBridgeFailure("Claude Code did not return a JSON result object", "malformed-json", identity=identity)
    effective_model = _effective_model(payload, expected_model)
    state = _terminal_state(payload, returncode=completed.returncode, expected_model=expected_model)
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    _append_lifecycle(lifecycle_path, {
        **base_event, "event": "terminal", "state": state, "timestamp": time.time(),
        "duration_seconds": time.time() - started_at, "subtype": str(payload.get("subtype") or ""),
        "returncode": completed.returncode, "canonical_model": effective_model, "num_turns": payload.get("num_turns"),
        "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        "total_cost_usd": payload.get("total_cost_usd"),
    })
    observed_identity, mismatch, substitution = identity, None, None
    if identity is not None:
        if __package__:
            from .target_identity import verify_observed
        else:
            from model_router.target_identity import verify_observed
        observed_identity, mismatch, substitution = verify_observed(
            identity, effective_model, "claude_cli.result.modelUsage",
        )
    if state == "max-turn":
        raise ClaudeBridgeFailure("Claude Code reached max turns", "max-turn", identity=observed_identity)
    if state == "budget":
        raise ClaudeBridgeFailure("Claude Code exhausted its budget", "budget", identity=observed_identity)
    if completed.returncode:
        raise ClaudeBridgeFailure(
            f"Claude Code failed with exit {completed.returncode}: {completed.stderr.strip()[:500]}",
            "nonzero-exit", identity=observed_identity,
        )
    if effective_model != expected_model or mismatch:
        raise ClaudeBridgeFailure(
            f"Claude Code did not serve {expected_model}; effective model was {effective_model or 'missing'}",
            "model-mismatch", identity=observed_identity, mismatch=mismatch,
        )
    if state != "success":
        raise ClaudeBridgeFailure(
            f"Claude Code failed with subtype {payload.get('subtype') or 'unknown'}",
            "execution-error", identity=observed_identity,
        )
    cfg = _load_config()
    _log_decision(
        RouteDecision("sonnet5" if alias == "sonnet" else "opus5", effective_model,
                      reason + (f"; usage soft limit: {adjustment}" if adjustment else ""), "external"),
        {"turn_id": f"{bridge_run_id}:{payload.get('num_turns', 1)}", "api_call_count": payload.get("num_turns", 1), "request": {}},
        cfg,
    )
    return {"bridge_run_id": bridge_run_id, "effective_model": effective_model, "reason": reason,
            "result": payload.get("result", ""), "num_turns": payload.get("num_turns"),
            "total_cost_usd": payload.get("total_cost_usd"),
            "identity": observed_identity.as_dict() if observed_identity is not None else None,
            "substitution": substitution}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--write", action="store_true", help="permit normal Claude Code edit/write/bash tools; no bypass flags are used")
    parser.add_argument("--review", action="store_true", help="require explicit [opus-review] and enforce Read-only tools")
    parser.add_argument("--timeout", type=int, default=None,
                        help="process wall-time override in seconds (reviews use at least 600 seconds)")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="override the bounded Claude Code turn limit (reviews default to 16; coding to 8)")
    parser.add_argument("--parent-session-id", default=os.getenv("HERMES_PARENT_SESSION_ID"))
    parser.add_argument("--parent-turn-id", default=os.getenv("HERMES_PARENT_TURN_ID"))
    parser.add_argument("--lifecycle-path", type=Path, default=DEFAULT_LIFECYCLE_PATH)
    args = parser.parse_args()
    print(json.dumps(dispatch(args.task, args.repo.expanduser().resolve(), write=args.write, review=args.review,
                              timeout=args.timeout, max_turns=args.max_turns,
                              parent_session_id=args.parent_session_id,
                              parent_turn_id=args.parent_turn_id, lifecycle_path=args.lifecycle_path.expanduser()), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
