"""Append-only audit trail for money-affecting actions.

A bank has to be able to say what it did, when, and on whose say-so. Every write
passes through the confirmation gate in `graph.py`, and each stage is recorded
here: what the agent asked to do, what the customer answered, what actually ran.

It exists because of a real failure. An earlier build let the agent ask for
confirmation in prose instead of calling the write tool, so it told the customer
"I'm blocking that right now" while nothing ran — and nothing recorded the gap.
A `write_requested` with no matching `write_executed` now shows it.

Format is JSON Lines: one record per line, appended, never rewritten. Greppable,
tailable, and a crash mid-write costs one line rather than the file.

One quirk when reading it: `interrupt()` re-runs the node from the top when the
turn resumes, so one write logs two `write_requested` lines — one from the pass
that halted, one from the pass that carried on. Dedupe on `tool_call_id`. What
matters is that the requested IDs match the executed ones, not the line counts.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Overridable so tests write to a temporary path instead of the repo, and so a
# deployment can point it at a mounted volume.
_ENV_VAR = "THINKMONEY_AUDIT_LOG"
_DEFAULT_PATH = "audit.log"


def audit_log_path() -> Path:
    """Where the trail is written. Read per call so tests can repoint it."""
    return Path(os.environ.get(_ENV_VAR, _DEFAULT_PATH))


def record(event: str, **fields) -> dict:
    """Append one event and return it.

    Returning the record lets callers assert on what was written without
    re-reading the file, and keeps the logging call a single expression at the
    call site.

    Auditing must never be the reason a customer's action fails, so a file that
    cannot be written is swallowed rather than raised. The trade-off is
    deliberate and one-directional: a lost line is bad, a refused cancellation
    because the disk was full is worse.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }

    try:
        path = audit_log_path()
        if path.parent != Path(""):
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    except OSError:
        pass

    return entry


def read_entries(path: Path | None = None) -> list[dict]:
    """Every recorded event, oldest first. Malformed lines are skipped.

    A truncated final line — the signature of a crash mid-append — should not
    make the rest of the trail unreadable.
    """
    target = path or audit_log_path()
    if not target.exists():
        return []

    entries = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _summarise(tool_call: dict) -> dict:
    """The identifying fields of a write, without dragging the whole call in."""
    return {
        "tool": tool_call.get("name", ""),
        "tool_call_id": tool_call.get("id", ""),
        "arguments": tool_call.get("args") or {},
    }


def write_requested(tool_call: dict) -> dict:
    """The agent has asked to move money. Nothing has happened yet."""
    return record("write_requested", **_summarise(tool_call))


def write_approved(tool_call: dict, answer) -> dict:
    """The customer said yes. Records the raw answer that was read as approval."""
    return record("write_approved", answer=str(answer), **_summarise(tool_call))


def write_refused(tool_call: dict, answer) -> dict:
    """The customer did not approve — including by saying nothing intelligible."""
    return record("write_refused", answer=str(answer), **_summarise(tool_call))


def write_executed(tool_call: dict, succeeded: bool, detail: str = "") -> dict:
    """The tool ran. `succeeded` reflects the tool's own success flag."""
    return record(
        "write_executed",
        succeeded=succeeded,
        detail=detail[:500],
        **_summarise(tool_call),
    )
