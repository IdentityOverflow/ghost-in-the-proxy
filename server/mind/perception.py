"""Session resolution and transcript reconciliation.

The client's resent transcript is not memory — it is evidence of what the
client believes happened, and its prefix is the session id. Reconciliation
compares it against each session's live events and classifies the request:

- continue:   live events are a prefix of the incoming messages
- truncation: ...except the client kept a prefix of our last reply (stop button)
- regenerate/fork: incoming diverges at seq k — supersede the tail, adopt theirs
- new:        nothing matches

One mechanism, per docs/architecture.md decisions 3 and 5.
"""

import json
from dataclasses import dataclass
from typing import Any

from .store import Event, MindStore, normalize_message


@dataclass
class Reconciliation:
    session_id: str
    outcome: str  # "new" | "continue" | "truncation" | "fork"
    new_events: list[dict[str, Any]]  # incoming messages not yet in the store


def _content_text(message: dict[str, Any]) -> str | None:
    content = message.get("content")
    return content if isinstance(content, str) else None


def _matches(event: Event, message: dict[str, Any]) -> str | None:
    """Classify how an incoming message relates to a stored event.

    Returns "exact", "truncation" (incoming is a non-empty prefix of our
    recorded assistant reply — the stop-button case), or None.
    """
    if normalize_message(event.message) == normalize_message(message):
        return "exact"
    if event.role == "assistant" and message.get("role") == "assistant":
        recorded = _content_text(event.message)
        incoming = _content_text(message)
        if recorded and incoming and recorded.rstrip().startswith(incoming.rstrip()) and incoming.strip():
            return "truncation"
    return None


def _split_system(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    if messages and messages[0].get("role") == "system":
        return _content_text(messages[0]), messages[1:]
    return None, messages


def _score_match(events: list[Event], incoming: list[dict[str, Any]]) -> tuple[int, str]:
    """Longest pairwise match; returns (matched_count, kind_of_last_match)."""
    matched = 0
    last_kind = "exact"
    for event, message in zip(events, incoming):
        kind = _matches(event, message)
        if kind is None:
            break
        matched += 1
        last_kind = kind
        if kind == "truncation":
            break  # a truncated reply can only be the last retained message
    return matched, last_kind


def reconcile(store: MindStore, messages: list[dict[str, Any]]) -> Reconciliation:
    client_system, incoming = _split_system(messages)

    best: tuple[int, str, str, list[Event]] | None = None  # matched, kind, session, events
    for session_id in store.list_session_ids():
        events = store.live_events(session_id)
        if not events:
            continue
        matched, kind = _score_match(events, incoming)
        if matched == 0:
            continue
        if best is None or matched > best[0]:
            best = (matched, kind, session_id, events)

    # A session continues only if every incoming message up to the stored
    # history matched (no divergence inside the shared span). Divergence with
    # a solid shared prefix (>=2 messages) is a fork; anything less is new.
    if best is not None:
        matched, kind, session_id, events = best
        diverged = matched < min(len(events), len(incoming))
        if kind == "truncation":
            # Stop button: client kept a prefix of our reply. Supersede our
            # record with the truncation the client actually saw.
            truncated_at = events[matched - 1].seq
            replacement = incoming[matched - 1]
            new_seq = store.append_event(session_id, replacement, source="client", confirmed=True)
            store.supersede_from(session_id, truncated_at, new_seq)
            # Everything the store had after the truncation point is gone
            # from the client's world; re-ingest their tail as truth.
            _ingest(store, session_id, incoming[matched:])
            store.set_client_system(session_id, client_system)
            return Reconciliation(session_id, "truncation", incoming[matched:])
        if not diverged and matched == len(events):
            # Clean continuation; confirm our now-acknowledged events.
            for event in events:
                if not event.confirmed:
                    store.confirm_event(session_id, event.seq)
            new = incoming[matched:]
            _ingest(store, session_id, new)
            store.set_client_system(session_id, client_system)
            return Reconciliation(session_id, "continue", new)
        # Edit/regenerate: shared prefix, then divergence. Unambiguous when
        # both timelines diverge at an assistant message (regenerate), or
        # when the shared prefix is substantial (edit deeper in).
        divergence_is_regenerate = (
            diverged
            and matched < len(incoming)
            and events[matched].role == "assistant"
            and incoming[matched].get("role") == "assistant"
        )
        if diverged and (matched >= 2 or divergence_is_regenerate):
            fork_seq = events[matched].seq
            tail = incoming[matched:]
            if tail:
                first_new = store.append_event(session_id, tail[0], source="client", confirmed=True)
                store.supersede_from(session_id, fork_seq, first_new)
                _ingest(store, session_id, tail[1:])
            store.set_client_system(session_id, client_system)
            return Reconciliation(session_id, "fork", tail)

    session_id = store.create_session(client_system)
    _ingest(store, session_id, incoming)
    return Reconciliation(session_id, "new", incoming)


def _ingest(store: MindStore, session_id: str, messages: list[dict[str, Any]]) -> None:
    for message in messages:
        store.append_event(session_id, message, source="client", confirmed=True)
