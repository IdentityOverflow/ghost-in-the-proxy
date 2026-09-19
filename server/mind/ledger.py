"""Ledger state as a pure replay of fold operations (memory v6).

The steward proposes small OPS against runtime-issued ids; this module
validates and applies them. Nothing here touches storage: state is a
function of the live fold log, which is what makes fork restore a query
(docs/memory-v6.md §1) and the whole thing unit-testable without sqlite.

Validation is per-op. A bad op is dropped and reported; the rest of the
proposal still commits. Rejecting whole folds was tried (the abandoned
hardening pass) and turns one confused op into a fold that fails forever.
"""

import re
from dataclasses import dataclass, field
from typing import Any

RECORD_KINDS = ("fact", "decision", "commitment")
# The field that names a record within its kind — used for de-duplication
# and for human-readable telemetry, never as identity (ids are identity).
KEY_FIELD = {"fact": "subject", "decision": "topic", "commitment": "statement"}
ALLOWED_FIELDS = {
    "fact": ("subject", "claim", "thread", "core"),
    "decision": ("topic", "status", "choice", "reason", "thread", "core"),
    "commitment": ("actor", "statement", "trigger", "due", "status", "thread"),
}
REQUIRED_FIELDS = {
    "fact": ("subject", "claim"),
    "decision": ("topic",),
    "commitment": ("statement",),
}
DECISION_STATUSES = ("decided", "leaning", "open")
COMMITMENT_STATUSES = ("open", "done", "dropped")
THREAD_FIELDS = ("name", "kind", "summary", "anchors", "open_questions")
FIELD_CHAR_CAP = 400
# Fields an update may CLEAR with an explicit null ("only when I ask"
# replacing "tomorrow" must be able to drop the deadline).
NULLABLE_FIELDS = ("due", "trigger", "reason")


@dataclass
class Record:
    id: str
    kind: str
    data: dict[str, Any]
    src: int | None = None  # event seq the steward cited as evidence
    created_fold: int = 0
    updated_fold: int = 0
    updated_seq: int = 0  # span_to of the fold that last touched it

    @property
    def key(self) -> str:
        return normalize_key(str(self.data.get(KEY_FIELD[self.kind], "")))

    @property
    def is_open(self) -> bool:
        if self.kind == "commitment":
            return self.data.get("status", "open") == "open"
        return True

    def text(self) -> str:
        """Flat text for relevance scoring and embedding."""
        parts = [
            str(self.data.get(name, ""))
            for name in ("subject", "topic", "statement", "claim", "choice", "reason", "trigger")
        ]
        return " ".join(part for part in parts if part)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, **self.data}


@dataclass
class Thread:
    id: str
    data: dict[str, Any]
    updated_seq: int = 0

    @property
    def name(self) -> str:
        return str(self.data.get("name") or self.id)

    def as_dict(self) -> dict[str, Any]:
        # 'key' is what the dynamics/assembler layer calls thread identity.
        return {"key": self.id, **self.data}


@dataclass
class Episode:
    fold: int
    span_from: int
    span_to: int
    text: str


@dataclass
class LedgerState:
    records: dict[str, Record] = field(default_factory=dict)
    threads: dict[str, Thread] = field(default_factory=dict)
    episodes: list[Episode] = field(default_factory=list)
    covered_upto: int = 0
    next_record: int = 1
    next_thread: int = 1

    def by_kind(self, kind: str) -> list[Record]:
        return [record for record in self.records.values() if record.kind == kind]

    def grouped(self) -> dict[str, list[dict[str, Any]]]:
        """{kind: [record dicts]} — the shape the assembler and telemetry use."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in self.records.values():
            grouped.setdefault(record.kind, []).append(record.as_dict())
        return grouped


@dataclass
class ApplyReport:
    applied: int = 0
    dropped: list[str] = field(default_factory=list)  # human-readable reasons
    deduped: int = 0


def normalize_key(text: str) -> str:
    # Unicode-aware: an ASCII-only filter maps every non-English key to ""
    # and de-duplication then merges unrelated records into one.
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def _clean(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()[:FIELD_CHAR_CAP]
    if isinstance(value, list):
        return [str(item).strip()[:120] for item in value[:12] if str(item).strip()]
    return value


def _record_fields(kind: str, op: dict[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for name in ALLOWED_FIELDS[kind]:
        if name in op and op[name] is not None:
            data[name] = _clean(op[name])
    if "core" in data:
        data["core"] = bool(data["core"])
    if kind == "decision" and "status" in data:
        status = str(data["status"]).lower()
        data["status"] = status if status in DECISION_STATUSES else "open"
    if kind == "commitment" and "status" in data:
        status = str(data["status"]).lower()
        data["status"] = status if status in COMMITMENT_STATUSES else "open"
    return data


def _src(op: dict[str, Any]) -> int | None:
    try:
        return int(op["src"]) if op.get("src") is not None else None
    except (TypeError, ValueError):
        return None


def apply_ops(
    state: LedgerState,
    ops: list[Any],
    fold: int,
    span_to: int,
) -> ApplyReport:
    """Apply one fold's ops to `state` in place. Never raises on bad ops."""
    report = ApplyReport()
    handles: dict[str, str] = {}  # proposal-local thread handle -> real id

    def resolve_thread(value: Any) -> str | None:
        if value is None:
            return None
        ref = str(value).strip()
        ref = handles.get(ref, ref)
        if ref in state.threads:
            return ref
        # Tolerate the model citing a thread by name instead of id.
        wanted = normalize_key(ref)
        for thread in state.threads.values():
            if normalize_key(thread.name) == wanted and wanted:
                return thread.id
        return None

    # Threads first, whatever order the model emitted: records in the same
    # proposal may point at a thread declared after them.
    ordered = [op for op in ops if isinstance(op, dict) and op.get("op") == "thread"] + [
        op for op in ops if not (isinstance(op, dict) and op.get("op") == "thread")
    ]
    for op in ordered:
        if not isinstance(op, dict):
            report.dropped.append(f"not an object: {str(op)[:60]}")
            continue
        verb = str(op.get("op", "")).lower()

        if verb == "thread":
            fields = {name: _clean(op[name]) for name in THREAD_FIELDS if op.get(name) is not None}
            if "<" in str(fields.get("name", "")) or not str(fields.get("name", "")).strip():
                # A template placeholder copied verbatim ("<kebab-slug>" —
                # observed on gemma-3-12b): name it from its own anchors.
                words = [normalize_key(str(a)).replace(" ", "-") for a in fields.get("anchors") or []]
                fields["name"] = "-".join(word for word in words[:3] if word) or ""
                if not fields["name"]:
                    fields.pop("name")
            ref = str(op.get("id") or "").strip()
            existing = resolve_thread(ref) or resolve_thread(fields.get("name"))
            if existing:
                state.threads[existing].data.update(fields)
                state.threads[existing].updated_seq = span_to
                if ref:
                    # The model re-declared a thread that already exists:
                    # records citing its local handle still belong to it.
                    handles[ref] = existing
            else:
                if not fields.get("name") and not fields.get("summary"):
                    report.dropped.append(f"thread without name/summary: {ref}")
                    continue
                thread_id = f"t{state.next_thread}"
                state.next_thread += 1
                fields.setdefault("name", thread_id)
                fields.setdefault("kind", "topic")
                state.threads[thread_id] = Thread(thread_id, fields, updated_seq=span_to)
                if ref:
                    handles[ref] = thread_id
            report.applied += 1
            continue

        if verb == "add":
            kind = str(op.get("kind", "")).lower()
            if kind not in RECORD_KINDS:
                report.dropped.append(f"add with unknown kind {kind!r}")
                continue
            data = _record_fields(kind, op)
            missing = [name for name in REQUIRED_FIELDS[kind] if not data.get(name)]
            if missing:
                report.dropped.append(f"add {kind} missing {missing}")
                continue
            thread = resolve_thread(data.get("thread"))
            if thread:
                data["thread"] = thread
            else:
                data.pop("thread", None)
            if kind == "decision":
                data.setdefault("status", "open")
            if kind == "commitment":
                data.setdefault("status", "open")
                data.setdefault("actor", "assistant")
            key = normalize_key(str(data[KEY_FIELD[kind]]))
            duplicate = next(
                (
                    record
                    for record in state.records.values()
                    if key and record.kind == kind and record.key == key and record.is_open
                ),
                None,
            )
            if duplicate is not None:
                # The slice did not show the model this record, or it forgot
                # the id: same key means same thing — update, don't fork it.
                duplicate.data.update(data)
                duplicate.updated_fold, duplicate.updated_seq = fold, span_to
                duplicate.src = _src(op) or duplicate.src
                report.deduped += 1
                report.applied += 1
                continue
            record_id = f"r{state.next_record}"
            state.next_record += 1
            state.records[record_id] = Record(
                record_id, kind, data, src=_src(op),
                created_fold=fold, updated_fold=fold, updated_seq=span_to,
            )
            report.applied += 1
            continue

        if verb in ("update", "close", "supersede"):
            record = state.records.get(str(op.get("id") or "").strip())
            if record is None:
                report.dropped.append(f"{verb} of unknown id {op.get('id')!r}")
                continue
            data = _record_fields(record.kind, op)
            if verb == "close":
                if record.kind == "commitment":
                    status = str(op.get("status") or "done").lower()
                    data = {"status": status if status in ("done", "dropped") else "done"}
                elif record.kind == "decision":
                    data = {"status": "decided", **{k: v for k, v in data.items() if k != "status"}}
                else:
                    report.dropped.append(f"close on a fact {record.id}")
                    continue
            if "thread" in data:
                thread = resolve_thread(data["thread"])
                if thread:
                    data["thread"] = thread
                else:
                    data.pop("thread")
            cleared = [
                name for name in NULLABLE_FIELDS
                if name in op and op[name] is None and name in record.data
                and name in ALLOWED_FIELDS[record.kind]
            ]
            if not data and not cleared:
                report.dropped.append(f"{verb} {record.id} changed nothing")
                continue
            record.data.update(data)
            for name in cleared:
                record.data.pop(name)
            record.updated_fold, record.updated_seq = fold, span_to
            record.src = _src(op) or record.src
            report.applied += 1
            continue

        report.dropped.append(f"unknown op {verb!r}")
    return report


def replay(folds: list[dict[str, Any]]) -> LedgerState:
    """Rebuild state from live fold rows, oldest first.

    Each row: {"seq", "span_from", "span_to", "ops": [...], "episode": str}.
    """
    state = LedgerState()
    for row in folds:
        apply_ops(state, row.get("ops") or [], fold=row["seq"], span_to=row["span_to"])
        episode = str(row.get("episode") or "").strip()
        if episode:
            state.episodes.append(Episode(row["seq"], row["span_from"], row["span_to"], episode))
        state.covered_upto = max(state.covered_upto, row["span_to"])
    return state
