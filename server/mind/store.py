"""Per-mind persistent store.

v0 schema invariant (docs/architecture.md): every derived row is append-only
and stamped with the event seq that produced it; corrections are supersede
links, never in-place edits or deletes. Fork/regenerate/interrupt all reduce
to superseding a tail and appending a new one, and state-at-any-seq stays a
query.
"""

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_ts REAL DEFAULT (unixepoch('subsec')),
    client_system TEXT
);
CREATE TABLE IF NOT EXISTS events (
    session TEXT NOT NULL,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,          -- normalized JSON of the full message
    source TEXT NOT NULL,           -- 'client' | 'mind'
    complete INTEGER NOT NULL DEFAULT 1,
    confirmed INTEGER NOT NULL DEFAULT 0,
    superseded_by INTEGER,          -- seq of the event that replaced this one
    ts REAL DEFAULT (unixepoch('subsec')),
    PRIMARY KEY (session, seq)
);
CREATE TABLE IF NOT EXISTS summaries (
    session TEXT NOT NULL,
    seq INTEGER NOT NULL,           -- creation order among summaries
    upto_seq INTEGER NOT NULL,      -- covers live events with seq <= upto_seq
    content TEXT NOT NULL,
    superseded INTEGER NOT NULL DEFAULT 0,
    ts REAL DEFAULT (unixepoch('subsec')),
    PRIMARY KEY (session, seq)
);
"""


def normalize_message(message: dict[str, Any]) -> str:
    """Canonical JSON for comparison and storage (role + semantic fields)."""
    keep = {
        key: value
        for key, value in message.items()
        if key in ("role", "content", "tool_calls", "tool_call_id", "name") and value is not None
    }
    return json.dumps(keep, sort_keys=True, ensure_ascii=False)


@dataclass
class Event:
    seq: int
    role: str
    message: dict[str, Any]
    source: str
    complete: bool
    confirmed: bool


class MindStore:
    def __init__(self, db_path: str | Path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    # -- sessions -----------------------------------------------------------

    def create_session(self, client_system: str | None) -> str:
        session_id = uuid.uuid4().hex[:12]
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO sessions (id, client_system) VALUES (?, ?)",
                (session_id, client_system),
            )
        return session_id

    def list_session_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT id FROM sessions ORDER BY created_ts").fetchall()
        return [row[0] for row in rows]

    def set_client_system(self, session_id: str, client_system: str | None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE sessions SET client_system = ? WHERE id = ?",
                (client_system, session_id),
            )

    def get_client_system(self, session_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT client_system FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row[0] if row else None

    # -- events -------------------------------------------------------------

    def live_events(self, session_id: str) -> list[Event]:
        rows = self._conn.execute(
            "SELECT seq, role, content, source, complete, confirmed FROM events"
            " WHERE session = ? AND superseded_by IS NULL ORDER BY seq",
            (session_id,),
        ).fetchall()
        return [
            Event(
                seq=row[0],
                role=row[1],
                message=json.loads(row[2]),
                source=row[3],
                complete=bool(row[4]),
                confirmed=bool(row[5]),
            )
            for row in rows
        ]

    def append_event(
        self,
        session_id: str,
        message: dict[str, Any],
        source: str,
        complete: bool = True,
        confirmed: bool = False,
    ) -> int:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM events WHERE session = ?", (session_id,)
            ).fetchone()
            seq = row[0] + 1
            self._conn.execute(
                "INSERT INTO events (session, seq, role, content, source, complete, confirmed)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    seq,
                    message.get("role", ""),
                    normalize_message(message),
                    source,
                    int(complete),
                    int(confirmed),
                ),
            )
        return seq

    def confirm_event(self, session_id: str, seq: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE events SET confirmed = 1 WHERE session = ? AND seq = ?",
                (session_id, seq),
            )

    def supersede_from(self, session_id: str, from_seq: int, by_seq: int) -> None:
        """Supersede live events in [from_seq, by_seq) (fork/regenerate/truncation).

        The replacing event (by_seq) is always newer than the range it
        replaces, and must not supersede itself.
        """
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE events SET superseded_by = ? WHERE session = ? AND seq >= ?"
                " AND seq < ? AND superseded_by IS NULL",
                (by_seq, session_id, from_seq, by_seq),
            )
            # Summaries covering superseded ground truth are invalid.
            self._conn.execute(
                "UPDATE summaries SET superseded = 1 WHERE session = ? AND upto_seq >= ?",
                (session_id, from_seq),
            )

    # -- summaries ----------------------------------------------------------

    def latest_summary(self, session_id: str) -> tuple[int, str] | None:
        """Returns (upto_seq, content) of the newest valid summary, if any."""
        row = self._conn.execute(
            "SELECT upto_seq, content FROM summaries WHERE session = ? AND superseded = 0"
            " ORDER BY seq DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return (row[0], row[1]) if row else None

    def append_summary(self, session_id: str, upto_seq: int, content: str) -> None:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM summaries WHERE session = ?", (session_id,)
            ).fetchone()
            self._conn.execute(
                "INSERT INTO summaries (session, seq, upto_seq, content) VALUES (?, ?, ?, ?)",
                (session_id, row[0] + 1, upto_seq, content),
            )
