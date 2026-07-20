# ghost-in-the-proxy

**A persistent, structured, living mind for any OpenAI-compatible model — instead of a context window full of dead transcript.**

This is an OpenAI-compatible proxy with a cognitive middleware inside. Your client talks to it exactly like it would talk to the model; the model never sees the client's raw transcript. Instead, the mind treats the incoming transcript as a *sensory event stream*, maintains its own persistent state per conversation, and assembles the model's entire context fresh on every request: distilled memory plus verbatim recent texture, inside a fixed budget that stays flat no matter how long the conversation runs.

It measurably improves small models at small windows — the founding result: gemma-4-12B at an 8k window went from **57% (transcript-stuffing baseline) to 86%** on a memory-coherence eval suite, and to **31/32** with the full architecture, where the single remaining failure is a model-capability probe that fails the baseline identically.

## What the mind does

| organ | job |
|---|---|
| **Perception** | Diffs each incoming transcript against the event store: continue, fork (edited message), regenerate, or stop-button truncation — real client behaviors, all first-class. The transcript is evidence, never memory. |
| **Steward** | An LLM extraction pass at fold time proposes the complete updated ledger — facts, decisions with status, commitments with triggers — plus thread structure and an episode line. The runtime applies it deterministically; the LLM only proposes. |
| **CRS dynamics** | Threads carry activation (fast decay, λ≈0.85) and importance (slow). Activation gates workspace admission: dormant material leaves the context entirely, and a lexical cue reinjects it. Forgetting is not deletion. |
| **Workspace assembler** | Builds the model's context in fixed order — client system prompt, memory sections with true status labels, recent turns verbatim — under a hard budget (~25% reserve, flat curve). Abundance causes dilution; the scene stays small on purpose. |
| **Recall** | The provenance escape hatch: a proxy-side tool over the raw event store. When the model needs exact wording that folded away, it reaches back and gets the verbatim span. Invisible to the client, on streamed and non-streamed requests alike (pure recall replies are held back, resolved, and re-queried mid-stream). |
| **Raw memory (Mem)** | A pluggable cue→episode backend over the raw events. Default is the lexical search; `MIND_MEM_BACKEND=embedding` adds semantic vectors (bge-m3 class) hybrid-scored with the lexical signals — and an auto-cue channel that pushes folded, semantically-matched verbatim spans into the workspace each turn, seq-tagged. Built because evals showed distillation drops exactly the mundane one-shot details real questions come back for. |
| **Router + containment** | The client's tool belt is scoped per turn (schema bulk is context tax), and stale tool payloads compress to digests under budget pressure — verbatim first, recall always available. |
| **Chronos** | Wall-clock time as a first-class input: the workspace carries the current time, real elapsed-time markers between turns ("[4 hours pass — it is now …]"), and due/OVERDUE status on time-triggered commitments — the steward converts "in two hours" to an absolute datetime at extraction. A transcript cannot carry any of this: the client protocol transmits no clock. |

Everything derived is append-only and provenance-stamped; corrections are supersede links. State at any point in the conversation is a query, which is what makes fork/regenerate/interrupt a reconciliation step instead of a feature.

See [`docs/architecture.md`](docs/architecture.md) for the full design and the phased eval gates (v0 skeleton → v1 ledger → v2 dynamics → v3 recall/routing/containment → v4 chronos → v5 Mem socket). Every phase ships only after a scenario that *fails without it* starts passing — and the suite has twice returned a verdict of "don't build it": the semantic-callback scenario showed distillation already covers paraphrased recall of memorable asides, and the holographic-reel backend was retired when embeddings plus log order passed every retrieval gate it would have claimed.

## The eval suite

`evals/` contains a 13-scenario harness that plays scripted multi-turn conversations as a real client (full transcript resent every turn) against any OpenAI-compatible endpoint, and grades probes deterministically plus with an optional LLM judge (majority-of-N voting). Scenarios cover decision coherence, tool-heavy synthesis, delayed commitment triggers, contradiction handling, salience decay with cued recall, containment, fork/regenerate continuity, mid-reply interrupts, verbatim recall, tool-schema tax, time awareness across real gaps (a virtual clock advances per turn via the `X-Mind-Clock` header, honored only with `MIND_FAKE_CLOCK=1`), zero-word-overlap semantic callbacks, and dropped-detail + incidental-order recall. `--stream` runs the whole suite over SSE (usage measured via `stream_options`); `evals/regrade.py` re-scores stored replies after rubric fixes without re-sampling.

```bash
# baseline (direct to your backend)
python -m evals.run --base-url http://localhost:1234/v1 --model <model> --wall 8192 --label baseline

# through the mind
MIND_ENABLED=1 MIND_FAIL_MODE=strict MIND_WINDOW=8192 DEFAULT_PROVIDER=lmstudio \
  uvicorn server.main:app --port 8000 &
python -m evals.run --base-url http://127.0.0.1:8000/v1 --model <model> --label mind
```

## Running

```bash
pip install -r server/requirements.txt
MIND_ENABLED=1 MIND_WINDOW=8192 DEFAULT_PROVIDER=lmstudio LMSTUDIO_BASE_URL=http://localhost:1234 \
  uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Point any OpenAI-compatible client at it. With `MIND_ENABLED` unset the server is a faithful passthrough (that bare proxy lives on as its own project: [LLM-passthrough-endpoint](https://github.com/IdentityOverflow/LLM-passthrough-endpoint)).

Key environment variables (see `server/mind/config.py` for all):

| variable | default | meaning |
|---|---|---|
| `MIND_ENABLED` | `0` | turn the mind on |
| `MIND_WINDOW` | `8192` | the model window the budget is built for (8k is the design floor; 4k is the survival tier) |
| `MIND_FAIL_MODE` | `open` | `open` = mind errors fall back to passthrough; `strict` = fail loudly (required for eval runs) |
| `MIND_EXTRACTION_MODEL` | request model | separate/faster model for steward extraction |
| `MIND_RECALL` / `MIND_TOOL_ROUTER` | `1` | v3 organs |
| `MIND_TIME` | `1` | v4 chronos: current time, gap markers, due/overdue commitments |
| `MIND_MEM_BACKEND` | `lexical` | raw-memory backend: `lexical` or `embedding` (unknown names fail open to lexical) |
| `MIND_EMBED_BASE_URL` / `MIND_EMBED_MODEL` | `http://localhost:1234/v1` / `text-embedding-bge-m3` | OpenAI-compatible embeddings endpoint for the embedding backend |
| `MIND_EMBED_MIN_SIM` | `0.45` | cosine floor for semantic-only hits and auto-cue injection |
| `MIND_DB_DIR` | `var/minds` | SQLite event stores (and embedding vectors), one mind per session |

```bash
pytest tests/   # unit tests: perception, store invariants, assembler, dynamics, recall, router
```

## Provenance

Grown from three design documents — a [conscious-workspace anatomy](https://github.com/IdentityOverflow/idea_drawer/blob/main/cognitive_workspace_architecture.md), a [cognitive-runtime physiology](https://github.com/IdentityOverflow/idea_drawer/blob/main/cognitive_runtime_with_SLMs.md) (CRS), and an [organic-development sketch](https://github.com/IdentityOverflow/idea_drawer/blob/main/organic_cognitive_architecture_oca.md) (OCA) — synthesized into one middleware and gated phase by phase on eval numbers. The name is the project: a ghost living inside the shell of a passthrough proxy.

## License
MIT
