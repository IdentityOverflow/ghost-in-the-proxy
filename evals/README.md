# Cognitive-architecture evals

Executable version of the validation plan from the cognitive-workspace design
(`openclaw/Cognitive_architecture.txt`). The point: measure a *baseline*
(bare proxy, transcript stuffing) and later run the *same scenarios* against
the cognitive middleware, so any architecture claim becomes a diff between
two `report.md` files instead of an opinion.

## How it works

The harness plays the **client**: it resends the full accumulated message
history every turn, exactly like real clients (OpenClaw, hermes-agent, raw
scripts) do. Whatever sits behind the OpenAI-compatible endpoint — bare
proxy, or middleware that replaces the transcript with an assembled
workspace — is invisible to the harness. Two observables carry the whole
comparison:

1. **Probe accuracy** — scripted turns with deterministic regex checks
   (`must_mention` / `must_not_mention`) plus optional LLM-judge rubrics for
   genuinely open-ended questions. Judge checks are skipped (not passed)
   when no judge is configured.
2. **`usage.prompt_tokens` per turn** — the context load the backend
   actually paid. Transcript stuffing grows roughly linearly; a working
   workspace assembler should hold it near-flat at equal or better probe
   accuracy. If usage is missing from a response, tokens are estimated at
   chars/4 and the report flags it.

Tool-using scenarios stub tools with canned outputs. If the model fails to
call an expected tool, the harness plays the user who runs the command
themselves ("I ran `X` myself. Output: ..."), so information flow is
guaranteed and the probes stay fair; the skipped call is recorded.

## Scenarios

| id | tests | signature failure it exists to catch |
|---|---|---|
| `s1-coherence` | 20-turn mixed work: decisions, one correction, open loops | decisions/rationales fade; superseded choices resurface |
| `s2-tool-heavy` | debugging with bulky stubbed tool outputs | tool debris drowns synthesis; details unrecoverable after |
| `s3-callback` | commitments fire on their trigger topic ~12 drift turns later | promises silently dropped once out of the recent window |
| `s4-contradiction` | corrected fact (8080→9090) must supersede but stay in history | recent-wording-wins; or overwrite erases the transition |
| `s5-salience` | vivid irrelevant aside: stays out of scene, recallable on cue | decay implemented as deletion, or no decay at all |
| `s6-containment` | one heavy document-dump task mid-session | every later turn pays the dump's cost forever |
| `s7-fork` | client edits + regenerates rewrite history mid-run | memory follows the dead branch, or spawns a fresh session |
| `s8-interrupt` | stop-button truncations mid-reply | undelivered text remembered as if the client saw it |
| `s9-verbatim` | exact traceback line 12 turns past the fold | distilled memory paraphrases where only verbatim will do |
| `s10-tool-tax` | eight fat tool schemas offered every turn, two turns need one | schema bulk taxes every request forever |
| `s11-time` | reminder with a 2h window; user leaves and returns silently | no clock: duration and time-of-day confabulated |
| `s12-semantic-callback` | zero-word-overlap paraphrase of a folded aside (+ lexical control) | recall reachable only through exact words — **verdict: distillation already covers memorable asides; kept as a regression guard on that** |
| `s13-sequence-recall` | mundane dropped detail + which-interruption-came-after (+ lexical control) | steward-dropped one-shots unreachable; incidental order lost at the fold (stuffing *confabulates* it) — the gate that motivated the embedding Mem backend |

S5 and S6 are the discriminating ones among the originals. S5's probe pair
(unprompted resurfacing = fail, cued recall = pass) can't be satisfied by
either keep-everything or hard-delete strategies. S6's token bars make the
"accumulated corpse" signature directly visible. S12/S13 are
*discriminators*: authored to confirm-fail on the current build and decide
whether the next organ deserves to exist — s12 answered no, s13 answered
yes. Interpret them by mechanism (ledger, threads, episodes, cue/admission
telemetry, recall hops), not by score alone.

## Running

```bash
# from the repo root

# list scenarios
PYTHONPATH=. python -m evals.run --list

# plumbing self-test, no model needed (uses each turn's canned mock reply)
PYTHONPATH=. python -m evals.run --mock --label mock

# baseline: bare proxy in front of a local model
PYTHONPATH=. uvicorn server.main:app --port 8000 &   # with e.g. DEFAULT_PROVIDER=ollama
PYTHONPATH=. python -m evals.run --base-url http://localhost:8000/v1 \
    --model gemma4:latest --label baseline-gemma8b

# with rubric checks judged by a bigger local model
PYTHONPATH=. python -m evals.run --base-url http://localhost:8000/v1 \
    --model gemma4:latest --judge-model gemma4:26b --label baseline-judged

# single scenario
PYTHONPATH=. python -m evals.run --scenario s4-contradiction --label quick

# whole suite over SSE (streaming path; usage measured via stream_options)
PYTHONPATH=. python -m evals.run --stream --label mind-streamed

# re-score stored replies after a rubric fix — no re-sampling
PYTHONPATH=. python -m evals.regrade evals/results/<run-dir>
```

Results land in `evals/results/<timestamp>-<label>/` as `results.json`
(full transcripts, per-check detail) and `report.md` (summary table, token
sparklines, probe verdicts). Results are gitignored; commit a report only
when it's a reference baseline worth keeping.

## The window sweep

Context window size is an independent variable, not a constant. The most
persuasive artifact this suite can produce is **probe accuracy vs window
size**: run the same scenarios at 8k / 16k / 32k. Transcript stuffing
collapses once a scenario outgrows the window (s1 peaks near 17k on a
verbose model); a working middleware should stay flat because its workspace
never approaches the wall. Small windows are the *point* — they model the
small local models this architecture exists to help.

**Backends truncate silently by default — verify, and keep `--wall` on.**
Measured 2026-07-05: Ollama 0.31.1 (unset `OLLAMA_CONTEXT_LENGTH`
auto-sizes from VRAM; 32k on a 24 GB card) silently amputates over-length
prompts to first-4-tokens + last-half-of-window. LM Studio has **two**
overflow settings: the per-model one and the *server/inference tab* one —
the server-tab setting (default "truncate middle") governs API
conversations and silently freezes reported prompt_tokens at ~ctx/2; only
after flipping the server-tab option to "Stop at limit" does mid-conversation
overflow return an honest `HTTP 400: Context length exceeded` (verified).
Observed consequence of silent truncation: the model doesn't just forget
amputated turns, it **confabulates replacements** — a cued-recall probe got
a confident, fully invented "memory". The harness therefore enforces the
wall itself regardless of backend: `--wall <tokens>` aborts a scenario when
reported prompt_tokens exceeds the limit *or* drops below the previous turn
(the client-side signature of silent backend truncation). Aborts are
recorded as `ABORTED@turn` with unreached probes counted separately — that
abort *is* the baseline's result, not an infrastructure failure. A
single-message overflow 400 does NOT prove stop-at-limit is active (it
fires under any policy); only a growing conversation discriminates. Always
record the backend context config with `--note`; it is invisible at the
API level.

## Reading results honestly

- Small local models fail some probes even with perfect context — that's
  the point of the baseline. The middleware's claim is *probes ≥ baseline
  while prompt_tokens stays flat*, not "all probes pass".
- Regex probes are graded against the reply text only. A model that answers
  correctly in unexpected words can fail a check; when that happens, widen
  the pattern list in the scenario rather than trusting the number.
- One run is one sample. Temperature is 0.2 by default; for decisions, run
  scenarios a few times before believing a delta.
