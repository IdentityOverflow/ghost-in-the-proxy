# Memory v6 — the endless conversation

Status: experimental branch `experiment/endless-memory`. Design by Claude,
reviewed by gpt-6-astra (2026-09-19). Supersedes the v1 "complete ledger
rewrite" steward and the v0 assembler budget.

## Why

v0–v5 were gated on scenarios of at most 20 turns. Past that horizon three
structural mechanisms degrade the mind, all confirmed in code:

1. **The memory section is unbounded.** One episode line per fold forever,
   facts and decisions only accumulate, nothing caps the system message.
   Texture shrinks to its two-turn floor and the prompt becomes a long flat
   list — the dilution the project exists to prevent, recreated inside the
   memory section. No hard guard: the request can exceed the window.
2. **The steward plays telephone.** Every fold a small model re-emits the
   COMPLETE ledger; nothing checks carry-forward, so drops and mutations
   compound. The call grows with the ledger (in AND out) until it truncates,
   the parse fails, and the ledger freezes behind a prose fallback. A valid
   `{}` reply erases everything.
3. **Deep forks wipe the ledger.** Generation-replace marks old generations
   superseded, so invalidating the newest leaves nothing live, while older
   episodes keep the coverage watermark high: the next fold starts from
   empty memory and never re-reads the early events.

## Shape

```
raw events (append-only, ground truth)          <- never summarized, always reachable
   │ fold (background, after the reply)
   ▼
fold log: one immutable row per fold = {span, ops[], episode}
   │ replay of LIVE folds (fork = supersede folds whose span touches the fork)
   ▼
ledger state: id'd records r1..rN, threads t1..tN, leaf episodes
   │ consolidation (background): 6 leaf episodes -> 1 era line, once
   ▼
bounded rendering: tiers under a token budget; the rest is retrieval-only
```

The brain analogy, for orientation rather than decoration: raw events are
the hippocampal trace; the fold is encoding; the ledger is semantic memory;
leaf episodes are episodic memory; eras are systems consolidation (gist
survives, detail stays reachable by cue); rendering is working memory —
small on purpose, filled by salience and cue, never by abundance.

## 1. The fold log (store)

`folds(session, seq, span_from, span_to, kind, ops JSON, episode, superseded)`

- A fold commits atomically: ops + episode + coverage in one row. Coverage
  watermark = max `span_to` over live folds. A malformed proposal commits
  nothing and advances nothing.
- Ledger state is a pure function: replay live folds in order. No mutable
  record rows, so there is nothing to half-invalidate.
- Fork at seq k: folds with `span_to >= k` are superseded. Earlier state is
  intact by construction; the surviving prefix of a fold that straddled the
  fork is simply re-folded (coverage fell back with it).
- A fold finishing after a fork landed inside its span is discarded at
  commit (span liveness check).

## 2. The delta steward

The model proposes small operations against runtime-issued ids; the runtime
applies them deterministically. Omission means unchanged.

```json
{"ops": [
  {"op": "thread", "id": "n1", "name": "heating", "kind": "topic", "summary": "...", "anchors": ["diesel"], "open_questions": []},
  {"op": "add", "kind": "fact", "subject": "battery", "claim": "280Ah LiFePO4", "thread": "t1", "core": false, "src": 43},
  {"op": "add", "kind": "decision", "topic": "heater", "status": "leaning", "choice": "diesel", "reason": "...", "thread": "n1", "src": 49},
  {"op": "add", "kind": "commitment", "actor": "assistant", "statement": "...", "trigger": "...", "due": null, "src": 23},
  {"op": "update", "id": "r7", "claim": "280Ah (was 200Ah)", "src": 43},
  {"op": "close", "id": "r4", "status": "done"}
 ],
 "episode": "2-3 sentences"}
```

- Ids are runtime-allocated and monotonic (`r12`, `t3`); the model only
  copies them. New threads use proposal-local handles (`n1`) mapped at commit.
- Validation is per-op, never per-fold: an op with an unknown id or missing
  field is dropped and counted; the rest commits. (The abandoned hardening
  pass rejected whole folds and then failed the same fold forever.)
- Adds are de-duplicated in the runtime: same kind + same normalized key as
  a live record becomes an update.
- `core: true` marks identity-level knowledge (names, hard constraints, key
  dates, allergies) — the profile tier that always renders.
- The steward sees a SLICE of the ledger rendered as compact id'd lines:
  threads, open commitments, undecided decisions, then records ranked by
  relevance to the fold span (lexical + embedding when available), under a
  fixed token budget. Input and output no longer grow with the conversation.
- JSON-schema constrained decoding is requested when the backend supports
  it; fallback is a plain-text episode-only fold (no ops), never a rewrite.

## 3. Bounded rendering

`memory_budget = clamp(35% of workspace budget, 900, 4000)`; the workspace
budget itself is capped (`MIND_WORKSPACE_CAP`, default 16k) so a 128k
window does not reintroduce transcript stuffing — what scales with the
window is verbatim texture and retrieval depth, not the default scene.

Tiers fill in priority order; each has a share cap, unused share flows down:

| tier | content | ordering |
|---|---|---|
| now | clock line | — |
| commitments | open commitments | overdue/due-soon, cue relevance, recency |
| profile | `core` facts | cue relevance, recency |
| decisions | undecided first, then relevant decided | cue relevance, recency |
| threads | admitted threads + their facts | activation |
| recalled | any other record matching the latest message | lexical hits / cosine |
| episodes | era lines, last 3 leaf episodes, 2 cue-matched older leaves | time |
| verbatim | auto-cued raw spans (v5) | seq |

A partial list says so ("+4 more tracked — ask to list them") and is never
labelled complete. Everything not rendered is still reachable: by cue next
turn, by the recall tool, by the auto-cue channel.

**Hard guard.** After assembly the full request (system + texture + tool
schemas + reserve for the reply) must fit the window. If not: uncovered old
texture is evicted anyway (it stays in raw memory and gets folded), then an
oversized single message is middle-truncated with a recall pointer. The
token estimate self-calibrates per session from the backend's reported
`usage.prompt_tokens`.

## 4. Background maintenance (the idle loop, finally)

After the reply is recorded, a per-session task runs under a session lock:
fold if pressure warrants, then consolidate. The next request for that
session awaits the lock — a human's typing time usually covers it; a fast
client simply waits for the fold it would have waited for anyway. If no
background pass happened (restart, first contact with a long transcript),
the request path folds synchronously as before. Assistant replies are only
folded once confirmed by the next request (unchanged).

**Consolidation.** Leaf episodes older than the newest 6 are grouped in
sixes and summarized ONCE into an era line; six eras fold into an epoch.
Each piece of text is therefore rewritten O(log n) times over the life of
the conversation, not once per fold, and the leaves are never deleted —
eras are a disposable index over them, regenerated after a fork.

## 5. Not in this phase

- Cross-conversation identity (one persistent agent across chats, keyed by
  model alias) — the next step toward "persistent agent with personality".
- A mind-owned self-model (the agent's own stances, running jokes, voice).
- Quick thoughts (fast Q/A micro-reasoning) — after memory is proven.
