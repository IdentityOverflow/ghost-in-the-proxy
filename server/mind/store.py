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
-- Memory v6: the fold log. One immutable row per fold carries everything
-- the fold derived — steward ops, the leaf episode, and (via span_to) the
-- coverage watermark — so a fold commits or fails atomically, and ledger
-- state is a pure replay of the live rows (server/mind/ledger.py).
CREATE TABLE IF NOT EXISTS folds (
    session TEXT NOT NULL,
    seq INTEGER NOT NULL,           -- creation order among folds
    span_from INTEGER NOT NULL,
    span_to INTEGER NOT NULL,       -- covers live events with seq <= span_to
    kind TEXT NOT NULL,             -- 'steward' | 'prose' (episode-only fallback)
    ops TEXT NOT NULL,              -- JSON list of steward ops (may be [])
    episode TEXT NOT NULL,
    superseded INTEGER NOT NULL DEFAULT 0,
    ts REAL DEFAULT (unixepoch('subsec')),
    PRIMARY KEY (session, seq)
);
-- Consolidations are a disposable index over leaf episodes (level 1 = era
-- over folds, level 2 = epoch over eras). Never truth: regenerated after
-- a fork, and the leaves they summarize are never deleted.
CREATE TABLE IF NOT EXISTS consolidations (
    session TEXT NOT NULL,
    seq INTEGER NOT NULL,
    level INTEGER NOT NULL,
    child_from INTEGER NOT NULL,    -- first fold seq (level 1) / consolidation seq (level 2+)
    child_to INTEGER NOT NULL,
    span_from INTEGER NOT NULL,     -- event span, for fork invalidation
    span_to INTEGER NOT NULL,
    content TEXT NOT NULL,
    superseded INTEGER NOT NULL DEFAULT 0,
    ts REAL DEFAULT (unixepoch('subsec')),
    PRIMARY KEY (session, seq)
);
-- Text-embedding cache for derived text (ledger records, episodes). Keyed
-- by content hash, so it needs no invalidation and survives forks.
CREATE TABLE IF NOT EXISTS embed_cache (
    model TEXT NOT NULL,
    hash TEXT NOT NULL,
    vec BLOB NOT NULL,
    PRIMARY KEY (model, hash)
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
CREATE TABLE IF NOT EXISTS embeddings (
    session TEXT NOT NULL,
    seq INTEGER NOT NULL,
    model TEXT NOT NULL,            -- embeddings from different models never mix
    vec BLOB NOT NULL,              -- float32 array bytes; dim implied by model
    PRIMARY KEY (session, seq, model)
);
"""


def content_text(message: dict[str, Any]) -> str | None:
    """The message's text, whatever the wire shape: a plain string, or the
    joined text parts of an OpenAI content-parts array (PI et al. send
    [{"type": "text", "text": ...}]). None when there is no text at all.
    Every mind organ that READS text goes through here; assuming str cost a
    live session its whole router/steward/cue pipeline (belt stripped on an
    empty user text, model play-acted the tool call)."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        joined = "\n".join(text for text in texts if text)
        return joined if joined else None
    return None


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
            # A fold whose span reaches the fork point goes entirely; its
            # surviving prefix is re-folded (coverage falls back with it).
            # Earlier folds are untouched, so the replayed ledger is exactly
            # the state as of the fork — restore is a query, not a feature.
            self._conn.execute(
                "UPDATE folds SET superseded = 1 WHERE session = ? AND span_to >= ?",
                (session_id, from_seq),
            )
            self._conn.execute(
                "UPDATE consolidations SET superseded = 1 WHERE session = ? AND span_to >= ?",
                (session_id, from_seq),
            )
            # Attention is a cache reconstructible from events: drop it and
            # let the next tick replay the live user turns.
            self._conn.execute("DELETE FROM thread_dynamics WHERE session = ?", (session_id,))

    # -- embeddings ---------------------------------------------------------

    def put_embedding(self, session_id: str, seq: int, model: str, vec: bytes) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (session, seq, model, vec) VALUES (?, ?, ?, ?)",
                (session_id, seq, model, vec),
            )

    def get_embeddings(self, session_id: str, model: str) -> list[tuple[int, bytes]]:
        with self._lock:
            return [
                (row[0], row[1])
                for row in self._conn.execute(
                    "SELECT seq, vec FROM embeddings WHERE session = ? AND model = ? ORDER BY seq",
                    (session_id, model),
                )
            ]

    def get_cached_vectors(self, model: str, hashes: list[str]) -> dict[str, bytes]:
        if not hashes:
            return {}
        found: dict[str, bytes] = {}
        with self._lock:
            for start in range(0, len(hashes), 500):
                chunk = hashes[start : start + 500]
                marks = ",".join("?" for _ in chunk)
                for row in self._conn.execute(
                    f"SELECT hash, vec FROM embed_cache WHERE model = ? AND hash IN ({marks})",
                    (model, *chunk),
                ):
                    found[row[0]] = row[1]
        return found

    def put_cached_vectors(self, model: str, vectors: dict[str, bytes]) -> None:
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embed_cache (model, hash, vec) VALUES (?, ?, ?)",
                [(model, key, vec) for key, vec in vectors.items()],
            )

    # -- fold log (memory v6) ---------------------------------------------------

    def live_folds(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT seq, span_from, span_to, kind, ops, episode FROM folds"
            " WHERE session = ? AND superseded = 0 ORDER BY seq",
            (session_id,),
        ).fetchall()
        return [
            {
                "seq": row[0],
                "span_from": row[1],
                "span_to": row[2],
                "kind": row[3],
                "ops": json.loads(row[4]),
                "episode": row[5],
            }
            for row in rows
        ]

    def append_fold(
        self,
        session_id: str,
        span_from: int,
        span_to: int,
        ops: list[dict[str, Any]],
        episode: str,
        kind: str = "steward",
    ) -> int | None:
        """Commit one fold atomically. Returns its seq, or None when the span
        is no longer live (a fork landed inside it while the steward was
        thinking) — a stale fold must never enter the log."""
        with self._lock, self._conn:
            live = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE session = ? AND seq IN (?, ?)"
                " AND superseded_by IS NULL",
                (session_id, span_from, span_to),
            ).fetchone()[0]
            if live < len({span_from, span_to}):
                return None
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM folds WHERE session = ?", (session_id,)
            ).fetchone()
            seq = row[0] + 1
            self._conn.execute(
                "INSERT INTO folds (session, seq, span_from, span_to, kind, ops, episode)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id, seq, span_from, span_to, kind,
                    json.dumps(ops, ensure_ascii=False), episode,
                ),
            )
        return seq

    def live_consolidations(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT seq, level, child_from, child_to, span_from, span_to, content"
            " FROM consolidations WHERE session = ? AND superseded = 0 ORDER BY seq",
            (session_id,),
        ).fetchall()
        keys = ("seq", "level", "child_from", "child_to", "span_from", "span_to", "content")
        return [dict(zip(keys, row)) for row in rows]

    def append_consolidation(
        self,
        session_id: str,
        level: int,
        child_from: int,
        child_to: int,
        span_from: int,
        span_to: int,
        content: str,
    ) -> None:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM consolidations WHERE session = ?",
                (session_id,),
            ).fetchone()
            self._conn.execute(
                "INSERT INTO consolidations (session, seq, level, child_from, child_to,"
                " span_from, span_to, content) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, row[0] + 1, level, child_from, child_to, span_from, span_to, content),
            )

    # -- thread dynamics (v2 CRS) ---------------------------------------------

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
