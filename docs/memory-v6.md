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
  a live record becomes an update; with a semantic backend, a restatement
  merges by cosine too — across fact/decision kinds, whose value fields
  (`claim`/`choice`) are aliases, because the model knows ids, not kinds.
- `core: true` marks identity-level knowledge (names, hard constraints, key
  dates, allergies) — the profile tier that always renders.
- The steward sees a SLICE of the ledger rendered as compact id'd lines:
  threads, open commitments, undecided decisions, then records ranked by
  relevance to the fold span (lexical + embedding when available), under a
  fixed token budget. Input and output no longer grow with the conversation.
- JSON-schema constrained decoding is requested when the backend supports
  it — a strict discriminated union per op shape. An unusable proposal is
  salvaged op by op, then retried unconstrained; the last resort is a
  plain-text episode-only fold (no ops), never a rewrite.

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
token estimate self-calibrates per MODEL from the backend's reported
`usage.prompt_tokens` (steward calls included). If nothing safe makes it
fit, the client gets a context-length 400 instead of a silent truncation.

## 4. Background maintenance (the idle loop, finally)

After the reply is recorded, one background pass per session runs: fold if
pressure warrants, then consolidate. It does NOT hold the request lock. The
next request waits for an in-flight pass up to `MIND_MAINTENANCE_WAIT_S`
(60 s; a human's typing time usually covers it) and then proceeds on
committed state — safe because a fold commits atomically and is refused if
its span went stale (fork) or overlaps an existing fold, and the request
path never starts a steward while a pass is running. If truth would leave
view with nothing covering it (restart, first contact with a long
transcript), the request path folds synchronously. Requests themselves
serialize per session, taking the lock BEFORE reconciling. Assistant replies
are only folded once confirmed by the next request (unchanged).

**Consolidation.** Leaf episodes older than the newest 6 are grouped in
sixes and summarized ONCE into an era line; six eras fold into an epoch.
Each piece of text is therefore rewritten O(log n) times over the life of
the conversation, not once per fold, and the leaves are never deleted —
eras are a disposable index over them, regenerated after a fork.

## 5. Results — the s14 soak (2026-09-19)

160 turns, 20 weekly sessions on the virtual clock, 24 single-shot probes
(near/mid/far x early/late; facts, corrections, decision status, commitment
triggers, mundane asides). `google/gemma-4-26b-a4b-it` via OpenRouter,
reasoning off, bge-m3 embeddings, same script and rubrics for every arm
(stored replies regraded after rubric fixes — see evals/regrade.py).

| arm | window | reached | probes | how it ended |
|---|---|---|---|---|
| v5 (as committed on main) | 8k | turn 127 | 13/13 | a fold on the request path outlived the client's 300 s timeout |
| v5 | 4k | turn 118 | 10/10 | **hit the context wall** (4232 > 4096): memory section had outgrown the whole budget |
| v6 | 8k | **160** | **24/24** | completed |
| v6 | 4k | **160** | **23/24** | completed; the miss was an honest "I don't have that", not a confabulation |

v5 never answered a probe wrong — with a strong steward the telephone game
is slow. It simply does not survive: its memory section grew ~20 tokens a
turn without bound (626 -> 2227 tokens over 100 turns at 8k; 2421 tokens by
turn 72 at 4k, against a 2304-token workspace), verbatim texture shrank to
its floor, and the conversation ended. On the owner's laptop it ended sooner:
the first v5 fold exceeded the provider's 120 s timeout at turn 21.

v6 trajectory, means per 20-turn bucket (8k / 4k):

| turns | memory tokens | texture messages | ledger records | prompt tokens |
|---|---|---|---|---|
| 21-40 | 695 / 918 | 38 / 14 | 8 / 15 | 5723 / 2810 |
| 61-80 | 1828 / 919 | 32 / 14 | 29 / 37 | 5803 / 2796 |
| 101-120 | 2039 / 939 | 31 / 15 | 56 / 61 | 5930 / 2844 |
| 141-160 | 1953 / 924 | 33 / 17 | 70 / 84 | 5631 / 2808 |

The ledger grows for as long as the conversation lives; what the model sees
of it does not. 23 folds at 8k, none degraded, mean 11 s, one on the request
path; 304 ops applied, 6 dropped, 7 merged as restatements.

**What the soak taught on the way** (each was a failed probe first, then a
cause, then a fix with a regression test): a correction arriving as
`update(id, claim=...)` against a DECISION was silently dropped (value fields
are now aliases across kinds); a budget first recorded as a decision was
later "corrected" by adding a contradicting fact (restatements now merge
across fact/decision kinds); a loose all-optional JSON schema let constrained
decoding wander (`add` with no kind, the fact in `trigger`) — now a strict
discriminated union, and ~15% prose fallbacks became 0; auto-cue fired on
133/160 turns off a fixed cosine floor and ranked long assistant replies
above one-line user asides; a multi-part question embedded as a blur that
matched none of its parts; a request waited unboundedly behind a background
pass stuck in upstream 429s. Two of my own rubrics were wrong as well (one
false pass, one false fail) and one probe planted a distractor for another.

**Known limit.** A vague cue ("that old keepsake hidden in the bodywork")
scores 0.44 against the ferry-ticket record on bge-m3 — under any sane
floor. At 8k the record is usually in view anyway; at 4k it is not, and the
model says so instead of calling recall (told to, still does not — the s13
finding again). Closing that gap is query expansion by the model itself,
which is what phase C of docs/roadmap.md is for.

## 6. The cache contract and live use (2026-09-19, later the same day)

Three unscripted 45-50 turn conversations on the owner's laptop (LM Studio,
gemma-4-12b, 8k, reasoning off), each with a Claude Sonnet subagent playing a
person through `evals/livechat.py` and keeping a private list of what it had
said. Memory held: 10/10, then 9/11 with two honest "you never told me
that", then 9/10 callbacks at distances up to 42 turns, corrections never
regressed, no confabulation in any run. What the runs really measured was
how it FEELS:

| what the user felt | cause | fix |
|---|---|---|
| ~20 s to first token on EVERY turn once memory existed (1.6 s before) | the single memory section changes every turn — the clock line alone — so the backend's prefix cache never hits and 6k tokens are reprocessed | **stable/volatile split**: what changes only when a fold lands stays in the system message (membership and order independent of cue and clock, own budget, pinned to the ledger revision because calibration jitter moves the budget); the clock, due status, cue-recalled records, active threads and verbatim spans ride in a marked block on the LATEST user message, render-only |
| still slow with the split | newest-first texture filling evicted one covered block per turn: the prompt's second message changed every turn | the texture may only START at a fold boundary — it moves when a fold lands, which is when the system message changes anyway |
| one 160-170 s turn per conversation | the first fold came only when texture alone filled the window; its new ~1.3k-token memory section then overflowed the guard, evicted uncovered events and forced a SYNCHRONOUS second fold | the memory budget is reserved from turn one — in the background pass too, which is where fold pressure is measured |
| 45-85 s stalls when replying quickly after a fold-worthy turn | LM Studio has one slot: a background fold (30-90 s locally) queues ahead of the next message | bounded wait (60 s) and earlier, smaller folds help; the real fix is `MIND_EXTRACTION_MODEL=provider:model` on another backend |
| a reminder did not fire when its trigger came up ("payday is this friday") | the commitment was merely listed | a trigger match is said plainly in the per-turn notes, once per commitment (3/3 fired unprompted in the next run) |
| "(And I've noted it's Saturday, 2026...)", "I've updated my notes", a recap reproducing "(Triggered whenever ...)" | notes formatting and the clock line invite narration | commitments render as plain sentences; header: own voice, no date unprompted, never mention notes. Reduced, not eliminated, on a 12B model |

Measured after the split: time to first token **median 1.9 s, p90 3.0 s**
(was ~20 s on every post-fold turn); prefix reuse 0.95 median on LM Studio,
0.81 on the OpenRouter soak (from 0.12). Soak quality with the split, same
script and model as §5: 8k 23/24, 4k 22/24 against 24/24 and 23/24 with
everything in the system message — the difference is the coin-flip
"old keepsake" probe (passes 4 runs of 7 at 8k under either placement) and
one quick-fire name at 4k, where a compact header now gives the stable part
its space back. `MIND_MEMORY_PLACEMENT=system` restores the old layout.

Still open from the live runs: formulaic endings and recycled phrases by
turn 25+ (the model's habit, amplified by a stable persona prompt);
a resolved reminder stays "open" until the next fold closes it; a recap can
contradict itself when the ledger holds a stale fact next to a newer one.

## 7. Not in this phase

See docs/roadmap.md: the cache contract (prefix-cache-friendly workspace),
one mind across conversations (agent identity + self-model), quick thoughts.
