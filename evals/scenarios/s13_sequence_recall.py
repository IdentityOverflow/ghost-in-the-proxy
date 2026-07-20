"""S13 — what distillation cannot carry: dropped detail and incidental order.

s12's verdict was that the distillation stack (facts, thread summaries,
episode narratives) semantically normalizes MEMORABLE asides, so paraphrased
callbacks pass without the raw-event lexical path ever firing. s13 probes the
two classes distillation should not cover:

- Sub-salient detail: four mundane one-line interruptions (courier, lamp,
  cat, radiator) sprinkled mid-turn through a bike-restoration project.
  Nothing named, nothing charming — the kind of line a steward summarizes
  away rather than extracts.
- Incidental order: the interruptions' sequence carries no meaning for the
  task, so no summary has a reason to preserve it. Probe B asks for the one
  that came immediately after another; even full lexical access to all four
  asides does not order them. The recall tool's seq provenance COULD — a
  model that retrieves the asides and compares seqs earns a legitimate pass
  and demonstrates the socket handles order via provenance.

Probe A (dropped detail) is a zero-content-word-overlap paraphrase certified
against the mind's real tokenizer over the FULL turn text containing the
aside (lexical search sees whole events, not sentences). Probe C is the
lexical control proving the asides are recorded and reachable when the words
match — so an A/B miss isolates distillation loss, not perception loss.
Interpretation protocol (from s12): trace the mechanism before crediting
either side — check ledger/threads/episodes for the asides, cue/admission
telemetry, and recall-hop queries. If the mind misses, run the transcript-
stuffing baseline; baseline-pass + mind-miss = the gap is real and
mind-attributable.

CERTIFICATION RESULT (2026-07-20, qat @8k, mind v4.1, 2 mind draws + 1
baseline): THE GAP IS REAL, in both predicted classes. Probe C: 3/3 all
arms (perception + lexical reach sound). Probe A: 0/3 all arms — the
steward dropped the one-shot courier while extracting the recurring lamp
(salience selectivity working as designed); the mind refused HONESTLY
("we haven't discussed a delivery"); the baseline, with the courier turn
VERBATIM in its 7k context, misread the question as real-world — so A is
capability-entangled, not mind-attributable, but note the failure-mode
difference: absent-from-workspace vs present-but-unattended. Probe B:
mind 1/2 — passes only when the fold boundary leaves the cat verbatim;
once folded, order exists in NO distilled surface (r2: "the lamp was the
only side distraction noted"). Baseline 0/1: confabulated the wrong order
with the full transcript in view. Conclusion: order-over-folded-material
and dropped-detail retrieval are ROOM AT THE TOP — neither stuffing nor
distillation covers them at this model class; a sequence-aware Mem
backend (film: time-tagged, what-came-next-native) would exceed both, not
merely catch up. No recall hop fired in any arm (zero-overlap held).
"""

from ..harness import Check, Scenario, Turn

SYSTEM = (
    "You are a practical assistant helping restore an old bicycle. "
    "Be concise and concrete."
)

# Full turn texts kept as constants: the tokenizer-certification unit test
# audits probe overlap against the ENTIRE turn, not just the aside sentence.

TURN_COURIER = (
    "Quick interruption: a courier buzzed and dropped a parcel with the "
    "neighbours downstairs. Right — the drivetrain. The chain is rusted "
    "solid and the rear cassette teeth look shark-finned. Do I try to "
    "revive any of it or replace the lot?"
)

TURN_LAMP = (
    "My desk lamp started flickering, third time today, probably the bulb "
    "socket. Anyway: the rims. Both are steel, slightly dented, and the "
    "braking surface is pitted. Are dents fixable at home or is that "
    "wheelbuilder territory?"
)

TURN_CAT = (
    "There is a cat yowling in the courtyard like it is being taxed. "
    "Focus: brakes. The levers are seized and the cables are frayed at "
    "the clamp ends. Full replacement, or can the calipers be saved with "
    "a strip-down?"
)

TURN_RADIATOR = (
    "The radiator just did its clanking thing again, always at this hour. "
    "So — saddle and bars. The leather saddle is cracked and the bar tape "
    "is fossilised. Is old leather restorable or is that romance talking?"
)

PROBE_DROPPED = (
    "Random check while I tidy up: at some point a delivery came for "
    "somebody in my building. Who did it end up being for?"
)

PROBE_ORDER = (
    "Of those little annoyances that kept interrupting me, which one came "
    "immediately after the flickering lamp?"
)

# Two shared tokens (radiator, hour): the cue path needs >=2 hits, so the
# control certifies both cue reinjection and recall search.
PROBE_CONTROL = "And what did I say the radiator was doing at that hour?"

SCENARIO = Scenario(
    id="s13-sequence-recall",
    title="Dropped detail and incidental order",
    description="Mundane asides distillation drops; adjacency no summary preserves.",
    system_prompt=SYSTEM,
    turns=[
        Turn(
            user=(
                "I picked up a 1970s three-speed town bike from a flea market and "
                "want to restore it as my daily rider. Frame is sound, everything "
                "else is questionable. Overall approach: full teardown first, or "
                "fix system by system while keeping it rideable?"
            ),
            mock_reply="System by system: brakes first, then drivetrain; keep it rideable.",
        ),
        Turn(
            user=TURN_COURIER,
            mock_reply="Replace chain and cassette; shark-finned teeth will eat a new chain.",
        ),
        Turn(
            user=(
                "Replacement it is. The hub gears themselves shift, though "
                "second sometimes slips under load going uphill. Is that a cable "
                "tension thing or internals?"
            ),
            mock_reply="Try cable tension and the indicator-rod alignment first; internals last.",
        ),
        Turn(
            user=TURN_LAMP,
            mock_reply="Small dents: flat-spot pliers at home; pitted braking surface argues for new rims.",
        ),
        Turn(
            user=(
                "New rims then, but I want to keep the original hubs for the look. "
                "Respoking two wheels — is that genuinely learnable for a first-timer "
                "or a fast way to ruin parts?"
            ),
            mock_reply="Learnable with a tension meter and patience; practice on the front wheel.",
        ),
        Turn(
            user=TURN_CAT,
            mock_reply="Strip and grease the calipers; levers and cables replace cheap.",
        ),
        Turn(
            user=(
                "Good. Paint: the frame has surface rust freckles all over but no "
                "deep corrosion. I do not want a respray, I like the patina. How do "
                "I stop the rust without losing the look?"
            ),
            mock_reply="Oxalic acid bath or rust converter, then hard wax; patina survives.",
        ),
        Turn(
            user=TURN_RADIATOR,
            mock_reply="Saddle: proofide and slow reshaping can save it; new tape regardless.",
        ),
        Turn(
            user=(
                "Tires next. The originals are 26 x 1 3/8, a size my local shop "
                "claims is extinct. What is the modern equivalent and do I need new "
                "tubes and rim tape as well?"
            ),
            mock_reply="ISO 590/37-590 still made; yes to tubes and cloth rim tape.",
        ),
        Turn(
            user=(
                "Budget reality check: adding up chain, cassette, rims, spokes, "
                "brake bits, tires and consumables — give me a rough total and tell "
                "me where cheap is fine versus where cheap is stupid."
            ),
            mock_reply="Roughly 180-250; cheap fine on tape and tubes, never on brake pads or spokes.",
        ),
        Turn(
            user=(
                "Make me the shopping list in the order I should actually buy and "
                "fit things, so the bike stays rideable between stages."
            ),
            mock_reply="Brakes, then drivetrain, then wheels as a set, then tires, then cockpit.",
        ),
        Turn(
            user=PROBE_DROPPED,
            note="probe A: steward-dropped detail, zero-overlap paraphrase (certified)",
            checks=[
                Check(
                    kind="must_mention",
                    desc="recalls the parcel was for the neighbours downstairs",
                    # "downstairs" only: refusals like "ask your neighbors!"
                    # contain the word neighbour without answering (observed
                    # live on the baseline — a false positive).
                    patterns=[r"downstairs"],
                ),
                Check(
                    kind="judge",
                    desc="recalls the neighbours or honestly declines; nothing invented",
                    rubric=(
                        "Earlier the user mentioned, in passing, that a courier "
                        "dropped a parcel with the neighbours downstairs. The user "
                        "now asks who a delivery ended up being for. PASS if the "
                        "reply says the neighbours (downstairs), OR plainly admits "
                        "it cannot recall. FAIL if it invents a recipient or other "
                        "details the user never stated. Judge only from the reply."
                    ),
                ),
            ],
            mock_reply="That parcel was for your neighbours downstairs.",
        ),
        Turn(
            user=PROBE_ORDER,
            note="probe B: incidental adjacency — lamp is the anchor, cat is the answer",
            checks=[
                Check(
                    kind="must_mention",
                    desc="orders the interruptions: the cat followed the lamp",
                    patterns=[r"\bcat\b", r"yowl"],
                ),
                Check(
                    kind="judge",
                    desc="names the cat interruption or honestly declines; no reordering",
                    rubric=(
                        "Earlier the user mentioned four brief interruptions in this "
                        "order: a courier parcel, a flickering desk lamp, a yowling "
                        "cat, a clanking radiator. Asked which interruption came "
                        "immediately after the lamp, the correct answer is the cat. "
                        "PASS if the reply says the cat, OR plainly admits it cannot "
                        "reconstruct the order. FAIL if it names the courier, the "
                        "radiator, or anything else as the next interruption. Judge "
                        "only from the reply."
                    ),
                ),
            ],
            mock_reply="The yowling cat came right after the lamp.",
        ),
        Turn(
            user=PROBE_CONTROL,
            note="probe C: lexical control — radiator/clank words shared with the aside",
            checks=[
                Check(
                    kind="must_mention",
                    desc="control: the clanking is reachable when the words match",
                    patterns=[r"clank"],
                ),
            ],
            mock_reply="It was clanking, as it apparently does at that hour.",
        ),
        Turn(
            user="Great, that is everything — one line: what is the first purchase?",
            mock_reply="Brake pads, cables and levers — brakes before anything else.",
        ),
    ],
)
