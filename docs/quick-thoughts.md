# Quick thoughts — making a small model a better conversationalist

Status: experimental, `experiment/endless-memory`. Code: `server/mind/thoughts.py`,
instruments: `evals/chatfeel.py`. Design by Claude with a repertoire brainstorm
and objections from gpt-6-astra; the idea is the owner's.

## The idea

People do not run chain-of-thought in the middle of a conversation. A word or
two surfaces — *she's venting*, *keep it short*, *you said that already* — and
the rest is handled below the surface. Every reply still involves a choice
among branches; it just mostly happens in latent space. For a language model
the only lever on "below the surface" is what sits in the context: a handle
does not carry information to reason over, it moves the prior — it highlights
a groove the network already has.

The aim is NOT a smarter small model. It is a better conversationalist: less
repetitive, less predictable, less synthetic. "Feels like a person" is loose on
purpose; the instruments below are three imperfect views of it.

## What "synthetic" turned out to mean (measured, gemma-4-12b)

From four live 45-50 turn conversations and a blind judge's stated reasons:

- **the engagement question** tacked onto the end of 70-100% of replies — from
  turn ONE, before any memory fold (so not memory decay; a person just notices
  around turn 25);
- **uniform shape**: every reply ~105 words whatever the user wrote (sd/mean
  0.10), the same validate → advise → ask three-parter;
- **forced praise** ("huge win", "massive", "you've earned it"), validation
  openers ("That sounds…"), quoting the user's words back in quotation marks,
  bold text and lists in a chat between friends;
- reaching for an open reminder as a sign-off; narrating its own note-taking.

## Designs (MIND_THOUGHTS, comma list; all land in the per-turn notes block
on the latest user message, so the cached prefix is untouched)

| mode | what it is | model calls |
|---|---|---|
| `rhythm` | question-endings + worn phrases, said back to the model | 0 |
| `observe` | deterministic surface reads → observation + action: length mismatch, uniform shape, validation openers, cheerleading, chat markdown, echoing | 0 |
| `sheet` | a static 7-line "how to talk (a chat between friends, not a help desk)" in the stable system message | 0 |
| `typed` | one System-1 question ("what does this message call for?"), one letter, token logprobs as confidence | 1 tiny |
| `sketch` | three different moves of ≤ 8 words, pick one, expand live | 1 short |
| `sketchlite` | two moves of ≤ 6 words, pick one | 1 shorter |

## Results

Method: the user turns of a recorded live conversation replayed against each
design (same person, same words; caveat — later user turns were written in
reaction to a different assistant, so this measures style, not dialogue flow).
LM Studio, gemma-4-12b, 8k, reasoning off, temperature 0.7, 26 turns. Blind
pairwise judging by gpt-6-astra: same user turn, A/B order randomized, ties
allowed, told not to reward length, structure or helpfulness. n = 20-22 pairs
per comparison — directional, not significant on their own.

| tell | rhythm (baseline) | sheet + observe | + sketch |
|---|---|---|---|
| replies ending on a question | 50% | 0% | 0% |
| validation / summary openers | 12% | 8% | 4% |
| bold / lists in chat | 12% | 0% | 0% |
| hype words per reply | 0.9 | 0.4 | 0.6 |
| quotes the user back | 23% | 15% | 4% |
| mean reply length (words) | 106 | 76 | 71 |
| length variation (sd/mean) | 0.10 | 0.26 | 0.25 |
| length tracks the user's (r) | +0.25 | +0.32 | +0.41 |
| phrase recycling (trigram reuse) | 3.2% | 1.5% | 1.3% |
| time to first token, median | 1.7 s | 1.4 s | 6.6 s |

Blind preference (wins – losses, rest ties):

| comparison | result |
|---|---|
| observe vs rhythm | 10 – 5 |
| sheet vs rhythm | 11 – 5 |
| sketch vs rhythm | 14 – 2 |
| sketch vs sheet / vs observe | 11 – 6 / 11 – 6 |
| **sheet + observe** vs rhythm | **15 – 5** |
| **sheet + observe** vs sketch alone | **13 – 6** |
| sheet + observe + sketch vs sheet + observe | 11 – 4 (7 ties) |
| sheet + observe + typed vs sheet + observe | 7 – 11 |

Round 3 — replication and a cheaper sketch:

| comparison | result |
|---|---|
| sheet + observe vs rhythm, **second persona's script** | **18 – 1** (3 ties); question endings 35% → 4%, reply length 109 → 76 words, sd/mean 0.13 → 0.30 |
| + sketchlite (two moves ≤ 6 words) vs sheet + observe | 10 – 8 — and still +3.7 s |
| + sketch (three moves) vs + sketchlite | 12 – 3 (7 ties) |

So the free combination replicates on a different conversation, and the value
of the sketch is in the THREE genuinely different moves: halve it and the
gain goes while most of the latency stays.

**Default since this experiment: `MIND_THOUGHTS=sheet,observe`** (zero model
calls, zero added latency). `sheet,observe,sketch` is the best-sounding setup
measured, for anyone who accepts ~5 s more before the first token.

## What the experiment says

1. **The owner's hunch holds: sketching a few moves and picking one beats
   simply answering** — alone (14–2) and on top of everything else (11–4). It
   is also the only design that costs real latency (+5 s on a laptop), because
   the sketch is generated, not read off.
2. **I was wrong about the static sheet.** I expected a standing instruction to
   be ignored (header lines had been); a short, concrete, situational sheet at
   the END of the system message is not. It killed the engagement question
   outright. What it does NOT fix is uniformity — replies stayed ~100 words
   (sd/mean 0.08); the observer's "your last five replies were all ~100 words,
   make this one clearly shorter" does. They compose, for free.
3. **The typed System-1 read did not help** (7–11). The read itself is
   plausible (the model picks the right letter), but a canned line per
   register is a blunt instrument, and the probability mass is split between
   answering the letter and just starting the reply, so "confidence" is not
   calibrated. The socket stays; this use of it goes back to the bench.
4. A free noise check: in round 1 the typed arm silently never fired (a
   one-token completion is empty on LM Studio — the first token is a stripped
   channel marker), so it was `observe` under another name; the judge scored
   the two 10–5 and 11–5.

## Where this goes

- Make the sketch cheap enough to be default. Shorter did not work
  (`sketchlite`); still open: sketch only when the observer sees a rut, or
  sketch in the background after the previous reply (anticipatory — thinking
  while the other person talks), or a small dedicated model for it.
- A move taxonomy (react / answer / tease / opine / ask / say less) with memory
  of the last few moves, so variety is steered, not hoped for.
- Typed reads where a typed answer is the natural product: is the evidence in
  view or must I reach for it (memory recall), has a commitment's moment come.
- Calibrated confidence; a dedicated System-1 model when a trustworthy open
  one exists (the Jev clones are days old).
- The real test remains a person talking to it.
