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
CREATE TABLE IF NOT EXISTS records (
    session TEXT NOT NULL,
    seq INTEGER NOT NULL,           -- creation order among records
    generation INTEGER NOT NULL,    -- steward pass that produced it
    kind TEXT NOT NULL,             -- 'fact' | 'decision' | 'commitment'
    payload TEXT NOT NULL,          -- JSON per kind
    provenance_seq INTEGER NOT NULL,
    superseded INTEGER NOT NULL DEFAULT 0,
    ts REAL DEFAULT (unixepoch('subsec')),
    PRIMARY KEY (session, seq)
);
CREATE TABLE IF NOT EXISTS episodes (
    session TEXT NOT NULL,
    seq INTEGER NOT NULL,
    span_from INTEGER NOT NULL,
    span_to INTEGER NOT NULL,
    summary TEXT NOT NULL,
    superseded INTEGER NOT NULL DEFAULT 0,
    ts REAL DEFAULT (unixepoch('subsec')),
    PRIMARY KEY (session, seq)
);
CREATE TABLE IF NOT EXISTS threads (
    session TEXT NOT NULL,
    seq INTEGER NOT NULL,           -- creation order among thread rows
    generation INTEGER NOT NULL,    -- steward pass that produced it
    key TEXT NOT NULL,              -- stable slug; carries dynamics across generations
    payload TEXT NOT NULL,          -- JSON: kind, summary, anchors[], open_questions[]
    provenance_seq INTEGER NOT NULL,
    superseded INTEGER NOT NULL DEFAULT 0,
    ts REAL DEFAULT (unixepoch('subsec')),
    PRIMARY KEY (session, seq)
);
-- Deliberate exception to the append-only invariant: activation/importance
-- are runtime-computed dynamics, reconstructible from events — a cache of
-- the mind's attention, not conversation truth. Updated in place.
CREATE TABLE IF NOT EXISTS thread_dynamics (
    session TEXT NOT NULL,
    key TEXT NOT NULL,
    activation REAL NOT NULL,
    importance REAL NOT NULL,
    updated_seq INTEGER NOT NULL,   -- last event seq applied (tick idempotence)
    PRIMARY KEY (session, key)
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
    ts: float = 0.0  # wall-clock unix seconds (v4 chronos)


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
            "SELECT seq, role, content, source, complete, confirmed, ts FROM events"
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
                ts=row[6] or 0.0,
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
        ts: float | None = None,
    ) -> int:
        """ts overrides the wall clock (fake-clock eval runs); None = real now."""
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM events WHERE session = ?", (session_id,)
            ).fetchone()
            seq = row[0] + 1
            if ts is None:
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
            else:
                self._conn.execute(
                    "INSERT INTO events (session, seq, role, content, source, complete, confirmed, ts)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        seq,
                        message.get("role", ""),
                        normalize_message(message),
                        source,
                        int(complete),
                        int(confirmed),
                        ts,
                    ),
                )
        return seq

    def next_seq(self, session_id: str) -> int:
        """The seq the next appended event will receive."""
        row = self._conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM events WHERE session = ?", (session_id,)
        ).fetchone()
        return row[0] + 1

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
            # Derived state built from superseded ground truth is invalid.
            # (Ledger generations from before the fork are not auto-restored;
            # the next steward pass rebuilds from live events. v1 tradeoff.)
            self._conn.execute(
                "UPDATE summaries SET superseded = 1 WHERE session = ? AND upto_seq >= ?",
                (session_id, from_seq),
            )
            self._conn.execute(
                "UPDATE records SET superseded = 1 WHERE session = ? AND provenance_seq >= ?",
                (session_id, from_seq),
            )
            self._conn.execute(
                "UPDATE episodes SET superseded = 1 WHERE session = ? AND span_to >= ?",
                (session_id, from_seq),
            )
            self._conn.execute(
                "UPDATE threads SET superseded = 1 WHERE session = ? AND provenance_seq >= ?",
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

    # -- structured records (v1 steward) --------------------------------------

    def live_records(self, session_id: str) -> dict[str, list[dict[str, Any]]]:
        """Current ledger, grouped by kind (facts / decisions / commitments)."""
        rows = self._conn.execute(
            "SELECT kind, payload FROM records WHERE session = ? AND superseded = 0 ORDER BY seq",
            (session_id,),
        ).fetchall()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for kind, payload in rows:
            grouped.setdefault(kind, []).append(json.loads(payload))
        return grouped

    def replace_records(
        self,
        session_id: str,
        records: dict[str, list[dict[str, Any]]],
        provenance_seq: int,
    ) -> None:
        """Version the whole ledger: supersede the old generation, append the new.

        The steward proposes the complete updated state each pass; generation
        versioning keeps application deterministic (no record matching) while
        preserving the append-only history.
        """
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(generation), 0), COALESCE(MAX(seq), 0) FROM records"
                " WHERE session = ?",
                (session_id,),
            ).fetchone()
            generation, seq = row[0] + 1, row[1]
            self._conn.execute(
                "UPDATE records SET superseded = 1 WHERE session = ? AND superseded = 0",
                (session_id,),
            )
            for kind, items in records.items():
                for item in items:
                    seq += 1
                    self._conn.execute(
                        "INSERT INTO records (session, seq, generation, kind, payload, provenance_seq)"
                        " VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            session_id,
                            seq,
                            generation,
                            kind,
                            json.dumps(item, ensure_ascii=False, sort_keys=True),
                            provenance_seq,
                        ),
                    )

    # -- threads (v2 CRS) ------------------------------------------------------

    def live_threads(self, session_id: str) -> list[dict[str, Any]]:
        """Current thread structure: payload dicts with 'key' merged in."""
        rows = self._conn.execute(
            "SELECT key, payload FROM threads WHERE session = ? AND superseded = 0 ORDER BY seq",
            (session_id,),
        ).fetchall()
        return [{"key": row[0], **json.loads(row[1])} for row in rows]

    def replace_threads(
        self, session_id: str, threads: list[dict[str, Any]], provenance_seq: int
    ) -> None:
        """Version the whole thread set, same pattern as replace_records."""
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(generation), 0), COALESCE(MAX(seq), 0) FROM threads"
                " WHERE session = ?",
                (session_id,),
            ).fetchone()
            generation, seq = row[0] + 1, row[1]
            self._conn.execute(
                "UPDATE threads SET superseded = 1 WHERE session = ? AND superseded = 0",
                (session_id,),
            )
            for thread in threads:
                key = str(thread.get("key") or "").strip()
                if not key:
                    continue
                payload = {k: v for k, v in thread.items() if k != "key"}
                seq += 1
                self._conn.execute(
                    "INSERT INTO threads (session, seq, generation, key, payload, provenance_seq)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        seq,
                        generation,
                        key,
                        json.dumps(payload, ensure_ascii=False, sort_keys=True),
                        provenance_seq,
                    ),
                )

    def get_dynamics(self, session_id: str) -> dict[str, tuple[float, float, int]]:
        """key -> (activation, importance, updated_seq)."""
        rows = self._conn.execute(
            "SELECT key, activation, importance, updated_seq FROM thread_dynamics"
            " WHERE session = ?",
            (session_id,),
        ).fetchall()
        return {row[0]: (row[1], row[2], row[3]) for row in rows}

    def set_dynamics(
        self, session_id: str, key: str, activation: float, importance: float, updated_seq: int
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO thread_dynamics (session, key, activation, importance, updated_seq)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(session, key) DO UPDATE SET"
                " activation = excluded.activation, importance = excluded.importance,"
                " updated_seq = excluded.updated_seq",
                (session_id, key, activation, importance, updated_seq),
            )

    # -- episodes -------------------------------------------------------------

    def live_episodes(self, session_id: str) -> list[tuple[int, int, str]]:
        rows = self._conn.execute(
            "SELECT span_from, span_to, summary FROM episodes WHERE session = ? AND superseded = 0"
            " ORDER BY seq",
            (session_id,),
        ).fetchall()
        return [(row[0], row[1], row[2]) for row in rows]

    def append_episode(self, session_id: str, span_from: int, span_to: int, summary: str) -> None:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM episodes WHERE session = ?", (session_id,)
            ).fetchone()
            self._conn.execute(
                "INSERT INTO episodes (session, seq, span_from, span_to, summary) VALUES (?, ?, ?, ?, ?)",
                (session_id, row[0] + 1, span_from, span_to, summary),
            )
