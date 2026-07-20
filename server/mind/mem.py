"""The Mem socket (reel phase v0): cue→episode behind a backend interface.

docs/architecture.md reserved "Mem behind cue→episode so holographic reel can
slot in". s13 confirmed the two retrieval classes the distilled surfaces
cannot carry — dropped detail (semantic reach into raw events the steward
discarded) and incidental order over folded material — so the socket now
exists as code: a MemBackend protocol with the current lexical search as its
default, behavior-preserving occupant. Future backends: embeddings (content
queries past the lexical wall) and the holographic reel (order/"next"
queries, time-tagged natively).

Design constraints carried in from the experiments:
- Queries carry a TRAJECTORY, not just text (ROFramework-PyLib #5: single
  moments are ambiguous; episodic addressing needs trajectory-stub cues —
  and the mind's live texture is exactly that stub).
- "next" is a first-class query kind: successor retrieval is what no
  distilled surface preserves (s13 probe B). Backends without order return
  nothing rather than pretending.
- Spans keep seq provenance: s13's baseline failed WITH the answer in
  context — retrieval is not enough, the pointed presentation (provenance-
  marked spans) is load-bearing.
- observe()/boundary() are write-path hooks. Lexical ignores them; a reel
  writes films on observe and closes slides on boundary (fold and thread-
  dormancy transitions both signal boundaries).
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from .dynamics import coverage, tokenize
from .store import Event, content_text


@dataclass
class MemSpan:
    seq: int
    role: str
    text: str
    score: float
    backend: str


@dataclass
class MemQuery:
    text: str
    # Recent live events — the trajectory stub. Backends may ignore it
    # (lexical) or use it for episodic addressing (reel).
    trajectory: list[Event] = field(default_factory=list)
    kind: str = "content"  # "content" | "next"
    # For kind="next": return what followed this seq.
    anchor_seq: int | None = None
    k: int = 3


class MemBackend(Protocol):
    name: str

    def observe(self, session_id: str, event: Event) -> None:
        """A new confirmed event entered the session's live history."""

    def boundary(self, session_id: str, reason: str, upto_seq: int) -> None:
        """An episode boundary: a fold happened or a thread went dormant."""

    def query(self, session_id: str, query: MemQuery, events: list[Event]) -> list[MemSpan]:
        """Resolve a cue against this session's raw memory.

        `events` is the session's live history, supplied so stateless
        backends need no storage of their own; stateful backends may ignore
        it in favor of their own index.
        """


class LexicalMem:
    """The v3 recall search, unchanged in behavior, as the default backend.

    Stateless: searches the live events it is handed. Coverage ranks; an
    exact-phrase hit dominates (verbatim requests usually carry a
    distinctive fragment of the sought material). Cannot answer "next" —
    order queries return nothing rather than a guess.
    """

    name = "lexical"

    def observe(self, session_id: str, event: Event) -> None:
        pass

    def boundary(self, session_id: str, reason: str, upto_seq: int) -> None:
        pass

    def query(self, session_id: str, query: MemQuery, events: list[Event]) -> list[MemSpan]:
        if query.kind != "content":
            return []
        query_tokens = tokenize(query.text)
        if not query_tokens:
            return []
        scored: list[MemSpan] = []
        needle = query.text.strip().lower()
        for event in events:
            text = content_text(event.message) or ""
            if not text:
                continue
            cov, hits = coverage(query_tokens, tokenize(text))
            if hits == 0:
                continue
            score = cov + (1.0 if needle and needle in text.lower() else 0.0)
            scored.append(MemSpan(event.seq, event.role, text, score, self.name))
        scored.sort(key=lambda span: span.score, reverse=True)
        return scored[: query.k]


_BACKENDS: dict[str, type] = {"lexical": LexicalMem}


def create_mem_backend(name: str) -> MemBackend:
    """Fail-open to lexical: a misconfigured backend name must not take the
    recall path down with it."""
    backend_cls = _BACKENDS.get(name)
    if backend_cls is None:
        print(f"[mind] unknown mem backend {name!r}; using lexical", flush=True)
        backend_cls = LexicalMem
    return backend_cls()
