# The Mind Middleware — conceptual architecture

A persistent, structured, living cognitive layer between any OpenAI-compatible
client and any OpenAI-compatible model. The client believes it is talking to a
model with a transcript; the model believes it is being handed a small, fresh,
perfectly relevant context every turn. Neither is wrong. Both are being lied
to constructively.

**Thesis: context is not memory.** The context window is a conscious
workspace — a narrow field where current goals, entities, commitments, and
retrieved facts are bound for the next action. Durable memory lives outside
it as structured state. The transcript the client resends every request is
not the conversation's substrate; it is a *sensory event stream* we diff for
what just happened.

This document synthesizes three prior designs into one buildable system:

| source | contributes | timescale |
|---|---|---|
| [Cognitive Workspace Architecture](https://github.com/IdentityOverflow/idea_drawer/blob/main/cognitive_workspace_architecture.md) | anatomy — stores, components, contracts | per-turn |
| [Cognitive Runtime Architecture](https://github.com/IdentityOverflow/idea_drawer/blob/main/cognitive_runtime_with_SLMs.md) (CRS) | physiology — activation dynamics, reflection/idle modes | seconds–minutes |
| [Organic Cognitive Architecture](https://github.com/IdentityOverflow/idea_drawer/blob/main/organic_cognitive_architecture_oca.md) (OCA) | development — RPE-gated learning, sleep consolidation | days–weeks (future) |

Formally, the middleware is an RO Observer tuple **O = (B, M, R, Mem)**:
B = semantic world model, M = workspace, R = router/executive policy,
Mem = episodic index. Mem is an interface (`cue → episode`), so the vector
store can later be swapped for the holographic reel from ROFramework-PyLib
without touching the rest.

## Position in the stack

```
client (openclaw / hermes / any app)
   │  full transcript resent per request (OpenAI chat API)
   ▼
┌─ proxy (server/) ───────────────────────────────────────────────┐
│  endpoint → agent graph (LangGraph) → provider                  │
│                 │                                               │
│     ┌───────────┴─ mind (MIND_ENABLED, else passthrough) ─────┐ │
│     │                                                         │ │
│     │  1 Session Resolver   prefix-hash → persistent mind     │ │
│     │  2 Perception         diff transcript → new events      │ │
│     │  3 Dynamics (CRS)     thread activations update/decay   │ │
│     │  4 Router             direct | retrieve | worker        │ │
│     │  5 Workspace Assembler  build small context, budgeted   │ │
│     │       ── forward to provider, stream reply back ──      │ │
│     │  6 Steward (async)    extract → stores after response   │ │
│     │  7 Idle loop          reflection / consolidation        │ │
│     │                                                         │ │
│     │  stores (SQLite): semantic | episodic | commitments |   │ │
│     │                   threads  | artifacts                  │ │
│     └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
   ▼
model backend (LM Studio / Ollama / frontier API)
```

The passthrough path stays intact and default-on. The mind is config-gated
and **fails open**: any internal error forwards the original request
untouched. A broken mind must degrade to today's proxy, never block traffic.

## Eval-driven design

Every organ exists because the baseline failed a measured probe
(`evals/`, gemma-3-12b + gemma4-8B, 2026-07-05). This table is the
architecture's justification and its acceptance test:

| measured failure | evidence | organ that fixes it | probe |
|---|---|---|---|
| conversations die at the window (43% @4k vs 76% @8k) | s1/s2/s5 ABORTED | Workspace Assembler (bounded context) | whole suite @4k |
| commitment lost *while in context* (salience dilution) | s1 t20, both models, 4.1k & 16.8k | Commitment Ledger, promoted above episodic recall | s1 t20, s3 |
| decision status confabulated ("leaning" → "locked") | s6 t12, judge, both models | Semantic store: decisions carry `status: open/taken` | s6 t12 |
| silent truncation → confabulated memories | s5 apple-pie reply | never truncate: assembler owns the budget | s5 t15 |
| heavy tasks tax every later turn | s6 token curve; s2 dies @t3 through proxy | Worker containment: artifacts, not transcript bulk | s6 curve, s2 completion |
| synthesis miss (capability, not memory) | s2 t5, both models | Router → worker route (later); memory layers must NOT fix this | s2 t5 as control |

## The organs

**1. Session Resolver.** The chat API is stateless; minds are not. Recognize
a continuing conversation by hashing the message prefix: the client's
history-so-far matches what we saw last turn (plus our own previous reply).
Match → same mind; no match → new session (or a fork — take the longest
matching prefix). Optional `X-Mind-Session` header for cooperating clients.
This is the one place the resent transcript helps: it is its own session id.

**2. Perception.** Diff the incoming transcript against the last seen state:
the suffix is the new event(s) — user turn, tool results the client executed,
etc. The transcript is never trusted as memory, only as evidence of what
happened. Store raw events append-only (provenance ground truth).

**3. Cognitive state + dynamics (CRS).** The part that makes it a mind
rather than a database. Threads — topic / inquiry / narrative / commitment —
carry `activation` (fast, decays per event: `A ← λA + input + reinforcement`,
λ≈0.85) and `importance` (slow: `I ← 0.98I + 0.02A`). Activation controls
workspace admission; importance controls archival. Open questions act as
attractors (relevant input boosts their threads). Inertia margin prevents
thrashing. Nothing is deleted by decay — archived threads remain retrievable
by cue (s5's contract: unprompted resurfacing = fail, cued recall = pass).

**4. Router.** Deterministic first: does this turn need retrieval, tools, a
worker, or nothing? Selects the scoped tool pack (prune the client's tool
list to what the route needs — schema bulk is context tax). LLM-oracle
routing only if metrics later justify it.

**5. Workspace Assembler.** Deterministic builder of the model's entire
context, in fixed order: identity/policy → active thread state (narrative +
3-4 threads, 80-120 token summaries) → open commitments (always, highest
priority after identity) → retrieved semantic facts → retrieved episodic
shards → recent turn texture (last 2-3 turns verbatim) → scoped tools.

Budget model — *fixed core, elastic evidence, capped total*:

```
window W (from config/probe)
reserve      = max(25% of W, 1k)          -- response + tool round-trips
core         ≈ 1.2-2k absolute            -- identity, threads, commitments, facts
evidence     = percentage of remainder    -- episodic shards, texture, tool digests
workspace cap ≈ 12-16k regardless of W    -- a 200k window must not reintroduce stuffing
```

The core is a *sufficiency* size, not a proportion: the current scene does
not get 8x richer because the window is 8x bigger — s1's commitment loss
happened at 4.1k inside an 8k window, so **abundance causes dilution**; a
bigger default scene is the disease, not a feature. (Ceiling if v1 shows
genuine need: ~3k. Not more.) What legitimately scales with W is elastic
evidence depth (more retrieved shards, longer verbatim texture, fatter tool
results) and reserve. Beyond the cap, extra window is spent on tool
round-trips and bursts, never on default scene bulk. A mostly-stable
workspace size also keeps prefix caching effective. Every item must justify
why it is active now; salience decides, the assembler only enforces.

Window tiers:

| tier | window | promise |
|---|---|---|
| survival (eval stress condition) | 4k | the mind never dies mid-conversation; conversational scenarios pass; tool-heavy work (s2-class) explicitly exempt — single tool payloads alone crowd a 4k window |
| **performance (design floor)** | **8k** | full-suite target: score ≥ transcript-stuffing at any window, flat tokens, tool-heavy included (~2k core + ~2k reserve + ~4k evidence) |
| comfort | 32k+ | realistic operating case: same core, richer elastic evidence, workspace cap still holds |

4k stays as a *measurement*, not a promise: it is where the baseline
collapse (43%) makes the middleware's survival property most legible.

**6. Steward (after the reply, asynchronously).** Extraction model reads the
finalized turn pair and proposes structured updates: semantic records
(fact/preference/decision+status/constraint/entity/relationship), episodic
event summaries, commitment deltas (opened/closed/blocked), contradiction
records (supersede, never overwrite — s4's contract: new value wins current-
state queries, transition stays recoverable). The runtime applies updates
deterministically; the LLM only proposes (CRS §19). Extraction lag is
acceptable; the workspace uses last-known state plus raw recent texture.

**7. Idle loop.** Between requests: reflection (short idle — refine thread
summaries, generate open questions, update predictions) and consolidation
(long idle — merge similar threads, decay salience, compact episodes).
This is what "living" means operationally: state evolves when nobody is
talking. Implemented as a background task in the server process; the OCA
sleep/consolidation ideas land here eventually.

**Provenance escape hatch.** Because the assembler *replaces* the
transcript, a workspace miss must be recoverable: the mind exposes a
`recall` tool to the model (query → episodic shards / raw transcript spans
with provenance). The model can ask for what the assembler didn't give it.
This also turns retrieval misses into observable events for tuning.

## Data model (sketch)

```
events        (session, seq, role, content, ts)          -- append-only ground truth
threads       (id, kind, summary, activation, importance, state, updated_seq)
semantic      (id, kind, subject, claim, status, confidence, provenance_seq,
               superseded_by)                             -- contradiction = link, not overwrite
episodes      (id, summary, entities[], decisions[], open_questions[],
               provenance_span)                           -- retrieval handles, not content
commitments   (id, actor, statement, trigger, status, due, provenance_seq)
artifacts     (id, kind, content, ttl, promoted)          -- worker outputs, tool digests
```

SQLite per mind (same pattern the openclaw cognitive-workspace proved),
embeddings optional and additive (FTS first; vectors when needed; the
holographic reel as a future Mem backend behind the same cue→episode call).

**v0 schema invariant:** every derived row is append-only and stamped with
the event seq that produced it; corrections are supersede links, never
in-place edits. This is what makes fork/regenerate (decision 3) a query
instead of a feature, and it gives time-travel debugging of the mind for
free. Reply events additionally carry `complete` and `confirmed_seq`
(decision 5): a reply is provisional until the next request's prefix shows
what the client retained.

## Phasing — each phase graduates on eval numbers

| phase | build | graduates when (vs 43% @4k / 76% @8k baselines) |
|---|---|---|
| **v0 skeleton** | session resolver, perception + reply confirmation, event store, naive assembler (recent texture + running summary), fail-open plumbing | conversational scenarios *complete* at 4k (no aborts); score ≥ baseline-8k on s3/s4 |
| **v1 stores** | steward extraction, semantic + commitments + episodes, assembler uses them | **≥76% @8k, flat tokens** (s1 3/3, s6 t12, s4 stays 4/4); conversational scenarios ≥ baseline-8k *at 4k* |
| **v2 dynamics** | CRS thread activations, decay, question attractors, idle reflection | s5 pair holds with *smaller* mean workspace; no regressions; retrieval hit-rate up |
| **v3 routing/workers** | router, scoped tool packs, contained workers, `recall` tool | s2 completes @8k with synthesis probe attempted; s6 curve flat after dump |
| **v4 chronos** | wall-clock time as a first-class input (`MIND_TIME`): event timestamps read back, current time + elapsed-gap markers rendered, steward converts relative triggers to absolute `due` datetimes, OVERDUE surfaces on return | s11 duration + time-of-day probes pass; v3 and baseline fail them identically (no clock anywhere); no 8k regression |

The s2 synthesis probe doubles as a leak detector throughout: if it starts
passing before v3's worker route exists, the harness is leaking hints.

v4 is the first phase where the mind diverges from what any transcript
could carry: the client protocol transmits no clock, so a transcript-stuffing
baseline *cannot* know how long the user was away — only a mind with its own
timeline can. Eval support: the harness advances a virtual clock per turn
(`Turn.advance_clock_s`) sent as an `X-Mind-Clock` header, honored only when
`MIND_FAKE_CLOCK=1`. Deliberately out of v4 scope (tier 2/3, planned as the
repurposed-MDCS custom client): the idle reflection loop acting *between*
requests, agenda records, and true proactive push — an OpenAI-compatible
proxy cannot initiate messages, so proactivity through standard clients
surfaces at next contact.

## Design decisions (settled 2026-07-05)

1. **Extraction model: same backend, configurable.** The conversation model
   also runs steward extraction (`MIND_EXTRACTION_MODEL` overrides when a
   faster/second model is wanted — gemma-3/4 12B class is expected to be
   fast enough; MoE models like gemma4-26b-a4b are candidates). Measure
   steward quality on the eval transcripts before optimizing.
2. **Workspace framing: hybrid, by nature of the content.** The workings of
   the mind — identity, narrative, thread state, commitments, semantic
   beliefs — live in the **system prompt**: they are the model's self and
   current awareness, not quotable data (and this composes with the
   DynamicSystemPrompt line of work). Evidence — episodic shards, tool
   digests, recalled spans — lives in **structured blocks in the message
   stream** with provenance markers: it is quotable material the model
   should treat as data, not identity. Recent texture stays as real
   user/assistant messages (the shape models are trained on). Within the
   system prompt, most-stable sections come first for prefix-cache locality.
3. **Fork semantics: first-class, via replayable state.** Edit/regenerate is
   a routine workflow with small models, so "new session" (continuity loss)
   is unacceptable. Mechanism: *state-at-any-seq is a query, not a
   snapshot*. All store writes are append-only and stamped with the event
   seq that produced them (`provenance_seq`), and nothing is destructively
   updated (the supersede-link discipline from contradictions applies to
   every store). Then: **regenerate** = same prefix, our last reply event
   superseded; **edit** = longest-common-prefix fork — branch shares events
   up to seq N and derived state is rebuilt by filtering applied updates to
   `provenance_seq ≤ N`. Cheap, lazy, and it falls out of the schema rather
   than being a feature. This makes append-only + seq-stamping a **v0
   schema requirement**, not a later concern. (Future: an s7 eval scenario
   probing edit/regenerate continuity.)
4. **One mind per session.** Cross-session identity (what graduates from
   session memory to durable user memory) is Steward policy, deferred.
5. **Interruption is a first-class event, unified with fork semantics.**
   (Flagged early from MDCS experience: stop/interrupt handled as an
   exception path causes state desync.) When the client hits stop, our
   record of the reply and the client's record diverge — the client keeps a
   truncation, or nothing. Same divergence as edit/regenerate, one
   mechanism:

   - **We know what we sent.** The proxy is the stream, so on disconnect
     the reply event is recorded with exactly the streamed-so-far content
     and `complete: false`. No guessing, unlike client-side attempts.
   - **Every reply event is unconfirmed until the next request.** The next
     transcript's prefix shows what the client actually retained: our full
     reply (confirm), a prefix of it (interrupt — supersede our record with
     their truncation), a different reply (regenerate), or a divergent
     earlier prefix (edit/fork). One reconciliation step in Perception
     covers all four; message-level prefix matching must therefore treat
     "their last assistant message is a prefix of our recorded one" as a
     match-with-truncation, not a mismatch.
   - **The Steward extracts from user turns immediately, from our replies
     only once confirmed.** Never extract commitments or facts from words
     the user cut off and may never have read; the async design makes this
     free (extraction already lags by design).
   - **Upstream cancellation propagates.** Client disconnect cancels the
     backend request (stream loop already checks `is_disconnected`); a
     mind must not keep paying for a reply nobody wants.
   - **The interrupt itself is recorded** as an event. Even before anything
     consumes it, it is a salience signal (the user cut us off — wrong
     direction, or too long) that CRS dynamics and future RPE-style
     learning (OCA) can use. (Future: an s8 eval scenario — interrupt
     mid-reply, continue the conversation, probe that the mind's state
     reflects the truncation the client saw, not the full reply it
     generated.)
