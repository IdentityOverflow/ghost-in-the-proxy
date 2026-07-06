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
| **Recall** | The provenance escape hatch: a proxy-side tool over the raw event store. When the model needs exact wording that folded away, it reaches back and gets the verbatim span. Invisible to the client. |
| **Router + containment** | The client's tool belt is scoped per turn (schema bulk is context tax), and stale tool payloads compress to digests under budget pressure — verbatim first, recall always available. |

Everything derived is append-only and provenance-stamped; corrections are supersede links. State at any point in the conversation is a query, which is what makes fork/regenerate/interrupt a reconciliation step instead of a feature.

See [`docs/architecture.md`](docs/architecture.md) for the full design and the phased eval gates (v0 skeleton → v1 ledger → v2 dynamics → v3 recall/routing/containment).

## The eval suite

`evals/` contains a 10-scenario harness that plays scripted multi-turn conversations as a real client (full transcript resent every turn) against any OpenAI-compatible endpoint, and grades probes deterministically plus with an optional LLM judge (majority-of-N voting). Scenarios cover decision coherence, tool-heavy synthesis, delayed commitment triggers, contradiction handling, salience decay with cued recall, containment, fork/regenerate continuity, mid-reply interrupts, verbatim recall, and tool-schema tax.

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
| `MIND_DB_DIR` | `var/minds` | SQLite event stores, one mind per session |

```bash
pytest tests/   # unit tests: perception, store invariants, assembler, dynamics, recall, router
```

## Provenance

Grown from three design documents — a conscious-workspace anatomy, a cognitive-runtime physiology (CRS), and an organic-development sketch (OCA) — synthesized into one middleware and gated phase by phase on eval numbers. The name is the project: a ghost living inside the shell of a passthrough proxy.

## License
MIT
