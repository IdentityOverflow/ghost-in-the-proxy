# Roadmap after memory v6

Memory first, reasoning after. v6 (docs/memory-v6.md) makes the conversation
endless in principle; these are the next phases toward "a persistent agent
with memory and personality", in build order. Synthesized from a design
consultation between Claude and gpt-6-astra (2026-09-19); where we differ it
says so.

## A — A cache contract (make every turn cheap) — BUILT, see docs/memory-v6.md §6

**Problem.** On a laptop backend (LM Studio, gemma-4-12b, 8k) a turn costs
~100 s, nearly all of it prompt processing: the system message changes EVERY
turn (clock line, cue-ranked tiers, recalled records, auto-cued spans), so the
backend's KV cache is invalidated from token one.

**Shape.** Split the memory section in two:

- **Stable** — header, profile, commitments (absolute deadlines), decisions,
  eras + recent episodes. Lives in the system message. Membership AND order
  frozen per (ledger revision, renderer version, budget class); changes only
  when a fold or consolidation lands.
- **Volatile** — the clock, due/overdue status, active-thread facts,
  cue-recalled records and leaves, verbatim spans. Delivered as one delimited
  block prefixed to the LATEST user message (gap markers already work this
  way), never stored, with one fixed system line declaring such blocks
  "historical evidence, not user speech".

Between folds only the last exchange is reprocessed. Separate fixed quotas for
the two parts (today volatile costs shrink the shared tier budget). Not a
synthetic assistant/tool exchange (fabricates history, needs tool templates),
not a trailing system message (gemma templates reject it).

**Measure before claiming.** Longest common prefix between consecutive
outgoing requests, in tokens, by turn type (ordinary / fold / eviction / tool
transition) — and remember background steward calls can evict a single-slot
cache. Gate: s1–s14 unchanged with memory inside a user turn (small models
may treat it as user speech — that is the empirical risk).

## B — One mind across conversations

**Problem.** One mind per chat session; "new chat" is amnesia. A persistent
agent is keyed by WHO it is, not which transcript it is in.

**Shape.** Identity = (owner, agent id); a model alias ("wren") or an
`X-Mind-Agent` header selects it. Sessions become episodes-of-contact under
the agent. Session folds stay as they are; above them an append-only **agent
assertion log** with provenance (session, source events, scope, revision,
retractions):

- promoted from confirmed evidence at fold time: identity facts, durable
  preferences, standing commitments; "remember this" gets an immediate path;
- project decisions live at project scope, not universal profile;
- inferred preferences stay candidates until repeated or confirmed;
- the agent's **self-model** — voice, stances, running jokes, relationship
  history — is curated identity plus grounded continuity, never scratch
  thoughts.

`core` is NOT the promotion flag (it includes this project's budget and
dates). Forks retract support rather than rewrite the past: an assertion
whose only evidence was superseded is retracted; a session that already read
revision 12 gets a correction against 13 next turn. Concurrent chats build
against a pinned agent revision and commit with revision checks — disputed
values are preserved, never last-writer-wins. Explicit session ids become
necessary (two chats both opening with "hello" are indistinguishable by
prefix).

## C — Quick thoughts

**The idea (Paul).** Humans do not run long chain-of-thought in conversation.
They have short fast thoughts — brief contextual questions with blunt, even
one-word answers — pick a path, and expand live while speaking. Give a small
model that structure instead of long CoT.

**Two readings, to be settled by evals:**

- *Control classifier (Astra).* A fixed battery — intent
  (answer/act/recap/clarify), evidence (present/retrieve/missing), commitment
  triggered (record id | none), contradiction (record id | none) — answered
  in ONE constrained call, enums and ids only, ~50 output tokens. The answers
  select retrieval and tools; the model never sees a monologue. Cheap, safe,
  measurable.
- *Inner voice (Claude, closer to the original idea).* The same battery, but
  the answers ALSO reach the reply as a tiny private cue ("they're asking for
  status → the list is the answer; heater: still leaning"), so the path is
  picked before the first visible token. Risk: a wrong blunt answer steers
  confidently. Mitigation: answers must cite a record id or "none"; uncited
  thoughts are dropped.

Start with the classifier (it is a subset), then test whether exposing the
answers moves s1/s4 status accuracy, s3 commitments, s9/s13 retrieval and
the s14 classes — while guarding s5 (over-retrieval) and s10 (schema churn).
Budget: added p95 under ~700 ms on a fast backend with timeout-to-normal;
on a laptop only if phase A makes the two-call turn cheaper than today's
one. Never for routine chat, exact transformations, tool continuations, or
tasks that need real reasoning.

**A finding that belongs here.** On the owner's laptop (LM Studio,
gemma-4-12b) every API call was silently reasoning — the UI's "reasoning
disabled" does not apply to the API; `{"reasoning_effort":"none"}` does
(`LMSTUDIO_EXTRA_BODY`). Same fold span, same model:

| mode | time | extracted |
|---|---|---|
| reasoning on, unconstrained | 233 s (2430 think tokens) | 4 facts |
| reasoning on, strict schema (grammar suppresses the think block) | 20 s | 1 fact |
| **reasoning off, strict schema** | **32 s** | **5 facts + thread, names core** |
| reasoning off, unconstrained | 20 s | 4 facts + 1 decision |

So long CoT bought nothing here, and a model that WANTS to think but is
gagged by a grammar does worse than one told not to think at all — a small
data point for the quick-thoughts thesis. A `"noticed"` list as the first
schema property (enumerate, then write ops) was tried once with reasoning
still on and timed out: the model poured its chain of thought into the first
free-text field. Worth retrying with reasoning off and bounded strings.

## Order

A → B → C. A makes iteration (and the laptop) affordable; B establishes
durable identity and truth; C earns admission only by moving measured
failures.
