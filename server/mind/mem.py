"""The Mem socket (reel phase): cue→episode behind a backend interface.

docs/architecture.md reserved "Mem behind cue→episode so holographic reel can
slot in". s13 confirmed the two retrieval classes the distilled surfaces
cannot carry — dropped detail (semantic reach into raw events the steward
discarded) and incidental order over folded material. The socket:

- LexicalMem — v3 recall search, behavior-preserving default.
- EmbeddingMem — semantic vectors (bge-m3 class) hybrid-scored with the
  lexical signals; order answered honestly from the event log (seq order is
  total and already true — no memory substrate needs to re-store it).

Design constraints carried in from the experiments:
- Queries carry a TRAJECTORY, not just text (ROFramework-PyLib #5: single
  moments are ambiguous; the mind's live texture is the stub cue).
- "next" is a first-class query kind (s13 probe B): backends answer from
  the log or return nothing — never a guess.
- Spans keep seq provenance: s13's baseline failed WITH the answer in
  context — retrieval is not enough, pointed presentation is load-bearing.
- observe()/boundary() are write-path hooks (fold = episode boundary).
  All methods are async: embedding is network I/O inside an async server.

The vector store is a BLOB table in minds.sqlite3 + numpy brute force: at
conversation scale (thousands of events × 1024 dims) a matmul is
sub-millisecond; a vector-db dependency would be decoration.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import numpy as np

from .config import MindConfig
from .dynamics import coverage, tokenize
from .store import Event, MindStore, content_text

EMBED_INPUT_CHAR_CAP = 6000  # bge-m3 takes 8k tokens; cap keeps calls bounded


@dataclass
class MemSpan:
    seq: int
    role: str
    text: str
    score: float
    backend: str
    # Semantic component alone (cosine); 0.0 for purely lexical hits. The
    # auto-cue channel filters on this: it exists to serve the class lexical
    # CANNOT reach, and must not double-serve lexical hits into the prompt.
    sim: float = 0.0


@dataclass
class MemQuery:
    text: str
    # Recent live events — the trajectory stub. Backends may ignore it
    # (lexical, pointwise embedding) or use it for episodic addressing.
    trajectory: list[Event] = field(default_factory=list)
    kind: str = "content"  # "content" | "next"
    # For kind="next": return what followed this seq.
    anchor_seq: int | None = None
    k: int = 3


class MemBackend(Protocol):
    name: str
    autocue: bool  # whether prepare() should inject this backend's spans per turn

    async def observe(self, session_id: str, event: Event) -> None:
        """A new confirmed event entered the session's live history."""

    async def boundary(self, session_id: str, reason: str, upto_seq: int) -> None:
        """An episode boundary: a fold happened or a thread went dormant."""

    async def query(self, session_id: str, query: MemQuery, events: list[Event]) -> list[MemSpan]:
        """Resolve a cue against this session's raw memory.

        `events` is the session's live history, supplied so stateless
        backends need no storage of their own; stateful backends may ignore
        it in favor of their own index (but must respect liveness: only
        seqs present in `events` may be returned — forks supersede).
        """


class LexicalMem:
    """The v3 recall search, unchanged in behavior, as the default backend.

    Stateless: searches the live events it is handed. Coverage ranks; an
    exact-phrase hit dominates (verbatim requests usually carry a
    distinctive fragment of the sought material). Cannot answer "next" —
    order queries return nothing rather than a guess.
    """

    name = "lexical"
    autocue = False  # per-turn injection of lexical hits would change v3/v4 behavior

    async def observe(self, session_id: str, event: Event) -> None:
        pass

    async def boundary(self, session_id: str, reason: str, upto_seq: int) -> None:
        pass

    async def query(self, session_id: str, query: MemQuery, events: list[Event]) -> list[MemSpan]:
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


class EmbeddingMem:
    """Semantic raw memory: an embedding per event, hybrid-scored with the
    lexical signals so verbatim reach (s9) survives while paraphrase reach
    (s13 probe A) opens up. Vectors live in the mind's own sqlite; failures
    fail open to the lexical component — the recall path must never die of
    an embedding outage.
    """

    name = "embedding"
    autocue = True

    def __init__(self, config: MindConfig, store: MindStore):
        self.config = config
        self.store = store
        self._lexical = LexicalMem()
        self._client: httpx.AsyncClient | None = None

    async def _embed(self, texts: list[str]) -> list[np.ndarray] | None:
        try:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=30)
            response = await self._client.post(
                f"{self.config.embed_base_url.rstrip('/')}/embeddings",
                json={"model": self.config.embed_model, "input": texts},
            )
            response.raise_for_status()
            return [
                np.asarray(item["embedding"], dtype=np.float32)
                for item in response.json()["data"]
            ]
        except Exception as error:
            print(f"[mind] embed call failed ({error!r}); continuing without", flush=True)
            return None

    async def observe(self, session_id: str, event: Event) -> None:
        text = (content_text(event.message) or "").strip()
        if not text:
            return
        vecs = await self._embed([text[:EMBED_INPUT_CHAR_CAP]])
        if vecs:
            self.store.put_embedding(session_id, event.seq, self.config.embed_model, vecs[0].tobytes())

    async def boundary(self, session_id: str, reason: str, upto_seq: int) -> None:
        pass

    async def query(self, session_id: str, query: MemQuery, events: list[Event]) -> list[MemSpan]:
        if query.kind == "next":
            # Order is the log's to answer: seq order is total and true.
            if query.anchor_seq is None:
                return []
            following = [event for event in events if event.seq > query.anchor_seq]
            return [
                MemSpan(e.seq, e.role, content_text(e.message) or "", 1.0, self.name)
                for e in following[: query.k]
            ]

        lex_spans = await self._lexical.query(
            session_id, MemQuery(text=query.text, k=max(len(events), 1)), events
        )
        lex_by_seq = {span.seq: span.score for span in lex_spans}

        sims: dict[int, float] = {}
        query_vecs = await self._embed([query.text[:EMBED_INPUT_CHAR_CAP]])
        if query_vecs is not None:
            live = {event.seq for event in events}
            stored = [
                (seq, blob)
                for seq, blob in self.store.get_embeddings(session_id, self.config.embed_model)
                if seq in live
            ]
            if stored:
                matrix = np.stack([np.frombuffer(blob, dtype=np.float32) for _, blob in stored])
                qvec = query_vecs[0]
                norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(qvec) + 1e-9)
                scores = (matrix @ qvec) / np.maximum(norms, 1e-9)
                sims = {seq: float(s) for (seq, _), s in zip(stored, scores)}

        by_seq = {event.seq: event for event in events}
        candidates = set(lex_by_seq) | {
            seq for seq, sim in sims.items() if sim >= self.config.embed_min_sim
        }
        scored: list[MemSpan] = []
        for seq in candidates:
            event = by_seq.get(seq)
            if event is None:
                continue
            sim = max(0.0, sims.get(seq, 0.0))
            scored.append(
                MemSpan(
                    seq,
                    event.role,
                    content_text(event.message) or "",
                    lex_by_seq.get(seq, 0.0) + sim,
                    self.name,
                    sim=sim,
                )
            )
        scored.sort(key=lambda span: span.score, reverse=True)
        return scored[: query.k]


def create_mem_backend(
    name: str, config: MindConfig | None = None, store: MindStore | None = None
) -> MemBackend:
    """Fail-open to lexical: a misconfigured backend must not take the
    recall path down with it."""
    if name == "embedding":
        if config is not None and store is not None:
            return EmbeddingMem(config, store)
        print("[mind] embedding backend needs config+store; using lexical", flush=True)
    elif name != "lexical":
        print(f"[mind] unknown mem backend {name!r}; using lexical", flush=True)
    return LexicalMem()
