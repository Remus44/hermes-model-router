#!/usr/bin/env python3
"""Readable, dependency-free viewer for the model-router JSONL log."""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

try:
    from .hermes_paths import hermes_path
except ImportError:
    from hermes_paths import hermes_path


DEFAULT_LOG = hermes_path("~/.hermes/logs/model-router.jsonl")
DEFAULT_STATE_DB = hermes_path("~/.hermes/state.db")


def _second_timestamp(value: str) -> str:
    """Normalize both legacy fractional and current whole-second timestamps."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(
            microsecond=0
        ).isoformat()
    except (AttributeError, TypeError, ValueError):
        return str(value)


def _prompt_preview(value: object) -> str:
    return " ".join(str(value or "").split())


def _redacted_preview(value: object, limit: int = 280) -> str:
    """Mirror the dashboard's existing credential-redaction policy."""
    text = _prompt_preview(value)
    if not text:
        return ""
    text = re.sub(
        r"(?i)\b(password|passwd|api[ _-]?key|secret|token)\s*([=:])\s*([^\s,;]+)",
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)\bauthorization\s*:\s*bearer\s+[^\s,;]+", "Authorization: Bearer ***", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "sk-[REDACTED]", text)
    text = re.sub(r"-----BEGIN [^-]+-----.*?-----END [^-]+-----", "[REDACTED KEY MATERIAL]", text, flags=re.DOTALL)
    return text if len(text) <= limit else f"{text[:limit - 1]}…"


INTERNAL_PROMPT_PREFIXES = (
    "[context compaction",
    "[async delegation batch complete",
    "[async delegation complete",
    "[your active task list was preserved across context summary",
    "[important: background process",
    "review the conversation above and consider saving to memory",
    "[sol] ",
)


def _session_id(turn_id: object) -> str:
    turn = str(turn_id or "")
    return turn.split(":", 1)[0] if ":" in turn else ""


def _strip_compaction_prompt(text: str) -> str:
    normalized = str(text or "").strip()
    marker = "[END OF CONTEXT SUMMARY"
    lowered = normalized.lower()
    compaction_start = lowered.find("[context compaction")
    if compaction_start == 0 and marker.lower() in lowered:
        marker_index = lowered.find(marker.lower())
        end = normalized.find("]", marker_index)
        if end >= 0:
            end += 1
            return normalized[end:].strip()
    return normalized


def _looks_like_internal_prompt(turn_id: object, prompt: object) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return True
    if not str(turn_id or "").strip():
        return True
    lower = text.lower()
    if any(lower.startswith(prefix) for prefix in INTERNAL_PROMPT_PREFIXES):
        return True
    if "[contex" in lower and "compaction" in lower:
        return True
    return False


def _completion_delegation_id(prompt: object) -> str:
    match = re.match(r"^\[ASYNC DELEGATION(?: BATCH)? COMPLETE\s+[—-]\s*([^\]\s]+)", str(prompt or ""), re.I)
    return match.group(1) if match else ""


def _entry_completion_delegation_id(entry: Dict) -> str:
    return str(entry.get("delegation_id") or "") or _completion_delegation_id(entry.get("prompt_preview"))


def _attach_lifecycle_provenance(entries: List[Dict], state_db: Path = DEFAULT_STATE_DB) -> None:
    """Join synthetic completion rows to durable delegation/task provenance.

    Old logs remain valid: unavailable tables, unrecognised envelopes, or a
    missing predecessor simply leave the optional fields unset.
    """
    synthetic = [entry for entry in entries if _entry_completion_delegation_id(entry)]
    if not synthetic:
        return
    try:
        connection = sqlite3.connect(state_db.resolve().as_uri() + "?mode=ro", uri=True)
    except (OSError, sqlite3.Error):
        return
    try:
        for entry in synthetic:
            delegation_id = _entry_completion_delegation_id(entry)
            entry["event_kind"] = "async_delegation_completion"
            try:
                row = connection.execute(
                    "SELECT parent_session_id, dispatched_at, task_json FROM async_delegations WHERE delegation_id=?",
                    (delegation_id,),
                ).fetchone()
            except sqlite3.Error:
                row = None
            if not row:
                entry["lifecycle_prompt"] = "Delegált feladat befejezési eseménye"
                continue
            parent_session, dispatched_at, task_json = row
            try:
                task = json.loads(task_json or "{}")
            except (TypeError, ValueError):
                task = {}
            goal = task.get("goal") or (task.get("goals") or [""])[0]
            origin = _redacted_preview(goal)
            try:
                origin_row = connection.execute(
                    "SELECT id, content FROM messages WHERE session_id=? AND role='user' "
                    "AND timestamp<=? AND content NOT LIKE '[ASYNC DELEGATION%' "
                    "ORDER BY timestamp DESC, id DESC LIMIT 1",
                    (parent_session, float(dispatched_at or 0)),
                ).fetchone()
            except sqlite3.Error:
                origin_row = None
            if origin_row:
                entry["origin_message_id"] = origin_row[0]
                origin = origin or _redacted_preview(origin_row[1])
            if origin:
                entry["origin_preview"] = origin
                entry["lifecycle_prompt"] = f"Delegált feladat befejezési eseménye · {origin}"
            else:
                entry["lifecycle_prompt"] = "Delegált feladat befejezési eseménye"
    finally:
        connection.close()


def _annotate_internal_prompts(entries: List[Dict]) -> None:
    """Mark continuation/system prompts so dashboard can hide and re-parent them."""
    last_root_by_session: Dict[str, str] = {}
    last_root_turn = None
    for entry in entries:
        turn_id = str(entry.get("turn_id", ""))
        prompt = str(entry.get("prompt_preview", ""))
        session = _session_id(turn_id)
        text = _strip_compaction_prompt(prompt)
        if text != prompt:
            entry["prompt_preview"] = text
            prompt = text

        is_internal = (
            entry.get("event_kind") == "async_delegation_completion"
            or _looks_like_internal_prompt(turn_id, prompt)
        )
        entry["is_internal_prompt"] = is_internal

        if not is_internal and turn_id:
            last_root_by_session[session] = turn_id
            last_root_turn = turn_id
            continue

        parent = last_root_by_session.get(session) or last_root_turn
        if parent:
            entry["parent_turn_id"] = parent
            entry.setdefault("origin_turn_id", parent)


def _fill_prompt_previews(entries: List[Dict], state_db: Path = DEFAULT_STATE_DB) -> None:
    """Backfill old router records from Hermes's canonical session store."""
    # Resolve every mappable entry, not only empty ones: older JSONL rows may
    # contain a legacy 50-character preview which must be replaced in the UI.
    candidates = entries
    session_ids = {
        str(entry.get("turn_id", "")).split(":", 1)[0]
        for entry in candidates
        if ":" in str(entry.get("turn_id", ""))
    }
    if not session_ids or not state_db.exists():
        return
    placeholders = ",".join("?" for _ in session_ids)
    try:
        connection = sqlite3.connect(state_db.resolve().as_uri() + "?mode=ro", uri=True)
        rows = connection.execute(
            f"SELECT session_id, timestamp, content FROM messages "
            f"WHERE role='user' AND session_id IN ({placeholders}) "
            f"ORDER BY session_id, timestamp, id",
            tuple(session_ids),
        ).fetchall()
        connection.close()
    except (OSError, sqlite3.Error):
        return

    messages: Dict[str, tuple[list, list]] = {}
    for session_id, timestamp, content in rows:
        times, previews = messages.setdefault(str(session_id), ([], []))
        times.append(float(timestamp))
        previews.append(_prompt_preview(content))

    for entry in candidates:
        session_id = str(entry.get("turn_id", "")).split(":", 1)[0]
        timeline = messages.get(session_id)
        if not timeline:
            continue
        try:
            route_time = datetime.fromisoformat(
                str(entry.get("timestamp", "")).replace("Z", "+00:00")
            ).timestamp()
        except (TypeError, ValueError):
            continue
        index = bisect.bisect_right(timeline[0], route_time + 0.5) - 1
        if index >= 0:
            entry["prompt_preview"] = timeline[1][index]


def _collapse_completion_replays(entries: List[Dict]) -> List[Dict]:
    """One async-delegation completion is one routing decision.

    The gateway may replay a completion turn once per parent call, so the same
    delegation_id appears many times.  Keep the last row (highest call count)
    at the position of the first, so accounting counts the delivery once.
    """
    merged: Dict[tuple, int] = {}
    result: List[Dict] = []
    for entry in entries:
        is_completion = entry.get("event_kind") == "async_delegation_completion" or bool(_completion_delegation_id(entry.get("prompt_preview")))
        delegation_id = _entry_completion_delegation_id(entry) if is_completion else ""
        if not delegation_id:
            result.append(entry)
            continue
        key = (entry.get("turn_id"), delegation_id)
        if key in merged:
            position = merged[key]
            if int(entry.get("api_call_count") or 0) >= int(result[position].get("api_call_count") or 0):
                result[position] = {**entry, "timestamp": result[position]["timestamp"]}
        else:
            merged[key] = len(result)
            result.append(entry)
    return result


def load_entries(path: Path) -> List[Dict]:
    entries: List[Dict] = []
    if not path.exists():
        return entries
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"Warning: invalid JSON on line {line_number}: {exc}")
            continue
        entry["timestamp"] = _second_timestamp(entry.get("timestamp", ""))
        entries.append(entry)
    entries = _collapse_completion_replays(entries)
    _attach_lifecycle_provenance(entries)
    _fill_prompt_previews(entries)
    _annotate_internal_prompts(entries)
    return entries


def _compact_runs(values: Iterable[str]) -> str:
    runs = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    return " → ".join(value if count == 1 else f"{value} ×{count}" for value, count in runs)


def _prompt(entry: Dict) -> str:
    return str(entry.get("prompt_preview", "")) or "-"


def _route(entry: Dict) -> str:
    tier = str(entry.get("tier", "?"))
    effort = str(entry.get("effort", ""))
    return f"{tier}/{effort}" if effort else tier


def _print_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    template = "  ".join(f"{{:<{width}}}" for width in widths)
    print(template.format(*headers))
    print(template.format(*("─" * width for width in widths)))
    for row in rows:
        print(template.format(*row))


def grouped_rows(entries: List[Dict]) -> List[List[str]]:
    groups: List[List[Dict]] = []
    for entry in entries:
        if groups and groups[-1][0].get("turn_id") == entry.get("turn_id"):
            groups[-1].append(entry)
        else:
            groups.append([entry])
    rows = []
    for group in groups:
        first = group[0]
        rows.append([
            str(first.get("timestamp", "")),
            _prompt(first),
            str(len(group)),
            _compact_runs(_route(item) for item in group),
            _compact_runs(str(item.get("reason", "?")) for item in group),
        ])
    return rows


def raw_rows(entries: List[Dict]) -> List[List[str]]:
    return [[
        str(entry.get("timestamp", "")),
        _prompt(entry),
        str(entry.get("api_call_count", "")),
        _route(entry),
        str(entry.get("reason", "")),
    ] for entry in entries]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="JSONL log path")
    parser.add_argument("--last", type=int, default=50, help="show the last N records (default: 50)")
    parser.add_argument("--raw", action="store_true", help="do not group calls by turn")
    args = parser.parse_args()

    entries = load_entries(args.log)
    if args.last > 0:
        entries = entries[-args.last:]
    if not entries:
        print(f"No routing entries found in {args.log}")
        return 0

    counts = Counter(str(entry.get("tier", "unknown")) for entry in entries)
    print(
        f"Log: {args.log} | Records: {len(entries)} | "
        + " | ".join(f"{tier}: {counts.get(tier, 0)}" for tier in ("luna", "spark", "terra", "sol", "opus5"))
    )
    print()
    headers = ["TIME (UTC)", "PROMPT", "CALLS" if not args.raw else "CALL", "ROUTE", "REASON"]
    _print_table(headers, raw_rows(entries) if args.raw else grouped_rows(entries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
