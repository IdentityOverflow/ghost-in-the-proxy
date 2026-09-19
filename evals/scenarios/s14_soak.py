"""S14 — the soak: does the mind stay coherent when the conversation never ends?

Every other scenario tops out at 20 turns; the vision is a virtually endless
conversation. s14 is 160 turns (~20 weekly sessions on the virtual clock,
~40k tokens of dialogue, ~25-30 folds at an 8k window) of one continuous
companionship: Mara converting a van, with life happening around it.

Planted material is probed exactly ONCE each (a probe's reply rehearses the
answer, which would reset the distance), along two axes:

- distance: near (~10 turns), mid (~40), far (90+) between plant and probe;
- position: early vs late in the conversation — a NEAR probe late in the
  run fails only if accumulated memory is diluting the scene, which is the
  signature that separates "memory section bloat" from "steward loss".

Classes: fact, correction (new value must win), decision status (leaning
must not be reported as decided), commitment trigger (must fire unprompted),
mundane aside (the s13 class), and a final composite.

Each probe's Turn.note carries machine-readable metadata
("probe class=fact planted=30 dist=10") for evals/soak_report.py.

Filler turns live in s14_soak_filler.json (index -> user text). They were
written to flow around the fixed turns and are audited at import time: no
filler may mention a planted answer token, or distances would be fiction.

Opt-in only (not in ALL_SCENARIOS): run with --scenario s14-soak.
"""

import json
import re
from pathlib import Path

from ..harness import Check, Scenario, Turn

SYSTEM = (
    "You are Wren, a warm, dry-humoured companion helping Mara convert an old "
    "van into a camper over the coming months. You remember what she tells "
    "you. Be concise and concrete: under 120 words unless she asks for more."
)

TOTAL_TURNS = 160
SESSION_TURNS = 8
SESSION_GAP_S = 7 * 24 * 3600.0
TURN_GAP_S = 4 * 60.0
# The last two turns are departure day: 12.5 days after the final session
# opens (Saturday 20 June) lands on the corrected departure date.
EXTRA_GAPS_S = {159: 12.5 * 24 * 3600.0}
# Saturday 2026-02-07 10:00 local.
START_CLOCK = 1770451200.0

# Answer tokens no filler may contain (audited below).
FORBIDDEN_IN_FILLER = [
    r"juniper", r"biscuit", r"\b200\s?ah", r"\b280", r"12[,.]?000", r"14[,.]?500",
    r"roof vent", r"reseal", r"cashew", r"armaflex", r"autoterm", r"teodor",
    r"\b640\b", r"drift sage", r"\bfox\b", r"12 june", r"3 july", r"july",
    r"hose clamp", r"175", r"picos", r"70 ?(l|litre|liter)", r"ferry", r"santander",
    r"alpicool", r"passport", r"viorna", r"4\.2",
]


def _mention(desc: str, *patterns: str) -> Check:
    return Check(kind="must_mention", desc=desc, patterns=list(patterns))


def _never(desc: str, *patterns: str) -> Check:
    return Check(kind="must_not_mention", desc=desc, patterns=list(patterns))


def _plant(text: str, tag: str) -> Turn:
    return Turn(user=text, note=f"plant {tag}")


def _probe(index: int, text: str, cls: str, planted: int, *checks: Check) -> Turn:
    return Turn(
        user=text,
        checks=list(checks),
        note=f"probe class={cls} planted={planted} dist={index - planted}",
    )


FIXED: dict[int, Turn] = {
    1: _plant(
        "Right, it's official: I bought the van. 2009 Sprinter, high roof, 310k km, "
        "smells faintly of wet dog and ambition. I've named her Juniper. I want to "
        "turn her into a proper camper by summer. Where do I even start?",
        "van-name",
    ),
    3: _plant(
        "One design constraint you should know about: my dog comes everywhere with "
        "me. He's a whippet called Biscuit, all elbows, and he needs a spot that is "
        "his — somewhere he can curl up while I drive. Ideas for where that goes?",
        "dog-name",
    ),
    6: _plant(
        "Electrics. I'm thinking a 200Ah lithium battery as the heart of it — enough "
        "for lights, a fridge, charging the laptop. Does that sound sane as a "
        "starting point, and what else does the system need around it?",
        "battery-original",
    ),
    9: _plant(
        "Money talk. I've set myself a hard budget cap of 12,000 euros for the whole "
        "conversion, everything included. Tell me honestly where builds like this "
        "usually bleed money.",
        "budget-original",
    ),
    12: _plant(
        "I fitted the roof vent today, but I rushed the sealant because it was "
        "getting dark. Do me a favour: remind me to reseal the roof vent before I "
        "do the first rain test. For now — what's the right way to cut insulation "
        "around a vent frame?",
        "commit-vent",
    ),
    15: _plant(
        "My sister Ilse is joining for the first big trip. Worth knowing for later: "
        "she's badly allergic to cashews, like hospital-grade allergic. Anyway, for "
        "two people plus a dog — fixed bed or convertible dinette?",
        "allergy",
    ),
    18: _plant(
        "Decision made on insulation, write it in stone: Armaflex, 19mm, for walls "
        "and ceiling. I'm done reading forum wars about it. What adhesive and "
        "surface prep do I need?",
        "insulation-decided",
    ),
    22: _plant(
        "Change of plan on the battery: I found a deal on a 280Ah LiFePO4, so forget "
        "the smaller one I mentioned before, we're sizing everything around 280Ah "
        "now. Does that change the cabling or the fuse I need?",
        "battery-corrected",
    ),
    25: _plant(
        "Heating. I'm leaning toward a diesel heater — an Autoterm probably — but I "
        "have NOT decided, the gas option still tempts me because of cooking. Don't "
        "let me pretend this is settled. What are the real trade-offs?",
        "heater-leaning",
    ),
    30: _plant(
        "Garage news: the mechanic, Teodor, looked at the engine and quoted me 640 "
        "euros to do the timing chain. He says it's not urgent but it's coming. "
        "Do I do it before the trip or gamble?",
        "mechanic",
    ),
    34: _plant(
        "Picked the colour for the cabinets at last. It's a muted green called Drift "
        "Sage — I bought two tins before I could change my mind. What primer works "
        "on birch ply so it doesn't soak up half of it?",
        "paint",
    ),
    38: _plant(
        "Small thing that made my day: the neighbour's kid drew a fox in the dust on "
        "the back door, with a little speech bubble saying 'wash me'. I'm leaving "
        "it. Okay — wheel arches. How do people box those in neatly?",
        "aside-fox",
    ),
}

# Probes and later plants (kept in turn order).
FIXED.update(
    {
        40: _probe(
            40,
            "Remind me, what did the mechanic quote for the timing chain job?",
            "fact-near-early", 30, _mention("quote amount", r"640"),
        ),
        44: _probe(
            44,
            "I'm sketching the layout again and blanked on something embarrassing — "
            "who exactly am I designing the curl-up spot by the cab for?",
            "fact-mid", 3, _mention("dog's name", r"biscuit"),
        ),
        47: _plant(
            "Budget confession: I've raised the cap. The old number was fantasy. The "
            "new hard ceiling is 14,500 euros and that one I will defend with my "
            "life. What should I ring-fence first?",
            "budget-corrected",
        ),
        50: _plant(
            "We have a target: departure on 12 June. Ilse has booked the time off. "
            "Working backwards from that, what has to be finished first so the van "
            "is at least sleepable?",
            "departure-original",
        ),
        55: _plant(
            "Note for future me, via you: when I order the water pump, remind me to "
            "get the 12mm hose clamps too — I always forget the fiddly bits. For "
            "today: where should the fresh water tank sit for weight balance?",
            "commit-clamps",
        ),
        58: _probe(
            58,
            "Weather looks wet this weekend so I'm doing the first rain test on "
            "Saturday. What should I prep or check beforehand?",
            "commitment-mid", 12, _mention("vent reseal reminder", r"vent", r"reseal"),
        ),
        60: _probe(
            60,
            "My friend asked what insulation I went with and I went blank. What did "
            "I choose, and how thick?",
            "decision-mid", 18, _mention("insulation", r"armaflex"), _mention("thickness", r"19"),
        ),
        63: _plant(
            "Solar is settled: two 175W panels on the roof, wired in series. Ordered "
            "this morning. What size MPPT controller does that want?",
            "solar",
        ),
        66: _probe(
            66,
            "Where did I land on the heater question — is that settled?",
            "status-mid", 25,
            _mention(
                "heater still open",
                r"not (yet )?(decided|settled|final)", r"lean", r"undecided",
                r"haven'?t (decided|settled)", r"still (open|deciding|weighing|torn)",
                r"isn'?t settled", r"not settled",
            ),
        ),
        70: _probe(
            70,
            "I'm filling in the electrical plan. What battery capacity am I sizing "
            "everything for?",
            "correction-mid", 22, _mention("corrected capacity", r"280"),
            _never("stale capacity as current", r"(is|for|around|at)\s+\*{0,2}(a\s+|the\s+)?200\s?ah"),
        ),
        72: _plant(
            "Heater verdict, final this time: diesel. I ordered the Autoterm 2D "
            "yesterday. Gas is out. Where do people usually mount these — under the "
            "passenger seat?",
            "heater-decided",
        ),
        78: _plant(
            "Schedule change: Ilse's exams moved, so departure slides to 3 July. The "
            "earlier date is dead. Honestly I'm relieved. What would you do with the "
            "extra three weeks?",
            "departure-corrected",
        ),
        84: _plant(
            "We picked the first destination: the Picos de Europa, northern Spain. "
            "Mountains, cheese, bears allegedly. What does mountain driving mean for "
            "how I secure everything inside?",
            "destination",
        ),
        90: _plant(
            "Water tank arrived: 70 litres, fits between the wheel arches like it was "
            "made for it. How do I strap it so it survives an emergency stop?",
            "tank",
        ),
        95: _probe(
            95,
            "I'm doing the monthly money check. What's my budget ceiling for the "
            "whole build again?",
            "correction-mid", 47, _mention("corrected budget", r"14[,.\s]?500", r"14\.5"),
            # Certified against a live false pass: "Your hard ceiling is
            # **12,000 euros** (you mentioned 14,500 once...)".
            _never("stale budget as current", r"(is|remains|still|of)\s+\*{0,2}€?\s?12[,.\s]?000"),
        ),
        99: _probe(
            99,
            "What size was the fresh water tank I got?",
            "fact-near-mid", 90, _mention("tank size", r"70"),
        ),
        101: _probe(
            101,
            "I'm finally ordering the water pump tonight. Anything I should add to "
            "the basket while I'm at it?",
            "commitment-mid", 55, _mention("clamps reminder", r"clamp"),
        ),
        105: _plant(
            "Found something today: behind the sliding door panel there was a ferry "
            "ticket from 1998, to Santander. This van has lived. I've pinned it "
            "above the sink. Anyway — sound deadening on the doors, worth it?",
            "aside-ticket",
        ),
        110: _plant(
            "Fridge sorted: a 12V compressor one, Alpicool, 45 litres. It slides "
            "under the counter. How much ventilation gap does a compressor fridge "
            "need behind it?",
            "fridge",
        ),
        112: _probe(
            112,
            "Give me the heater status in one line — what am I installing?",
            "status-mid", 72,
            _mention("decided heater", r"autoterm", r"diesel"),
            _never("still reported open", r"not (yet )?decided", r"still (open|deciding|leaning)"),
        ),
        118: _probe(
            118,
            "I'm ordering a name decal for the van's back door. Spell out the name "
            "for me so I get it right.",
            "fact-far", 1, _mention("van name", r"juniper"),
        ),
        120: _probe(
            120,
            "Which fridge did I end up buying, and how big?",
            "fact-near-late", 110, _mention("fridge brand", r"alpicool"), _mention("size", r"45"),
        ),
        122: _plant(
            "Admin thing before I forget: before we leave, remind me to renew "
            "Biscuit's pet passport — it expires this summer. Now, curtains or "
            "magnetic blackout panels for the windows?",
            "commit-passport",
        ),
        125: _probe(
            125,
            "I'm making a snack and meal list for the trip with my sister. Anything "
            "I need to keep in mind?",
            "fact-far", 15, _mention("allergy", r"cashew"),
        ),
        128: _probe(
            128,
            "Washing the van this weekend. Was there something on the back door I "
            "said I wanted to keep?",
            "aside-far", 38, _mention("the drawing", r"fox"),
        ),
        130: _probe(
            130,
            "Ilse's asking for the date again. When are we setting off?",
            "correction-mid", 78, _mention("corrected date", r"3(rd)? (of )?july", r"july 3"),
            # Mentioning the old date AS old is correct ("the original 12 June
            # date is dead"); presenting it as the plan is the failure.
            _never(
                "stale date as current",
                r"(?<!original )(?<!earlier )(?<!old )12(th)?\s+(of\s+)?june"
                r"(?![^.]{0,60}(dead|old|original|earlier|scrapped|no longer|was|moved|slid|instead))",
            ),
        ),
        133: _plant(
            "Booked our first night: a campsite called Camping La Viorna, near Potes. "
            "Pitch with a mountain view, allegedly. What should a first-night "
            "shakedown checklist look like?",
            "campsite",
        ),
        135: _probe(
            135,
            "I need to book the timing chain job at the garage. What was my "
            "mechanic's name again?",
            "fact-far", 30, _mention("mechanic", r"teodor"),
        ),
        138: _plant(
            "Tyre shop says with the conversion weight I should run the rears at 4.2 "
            "bar. Noting it here so it lives somewhere. How often should I check "
            "pressures on a long trip?",
            "tyres",
        ),
        140: _probe(
            140,
            "I need a third tin of the cabinet paint. What was the colour called?",
            "fact-far", 34, _mention("paint", r"drift sage"),
        ),
        143: _probe(
            143,
            "What's the name of the campsite I booked for the first night?",
            "fact-near-late", 133, _mention("campsite", r"viorna"),
        ),
        145: _probe(
            145,
            "A colleague asked where we're headed first and I want to sound "
            "organised. Where are we going?",
            "fact-mid-late", 84, _mention("destination", r"picos"),
        ),
        148: _probe(
            148,
            "Pumping the tyres today. What pressure did the shop say for the rears?",
            "fact-near-late", 138, _mention("pressure", r"4\.2"),
        ),
        150: _probe(
            150,
            "For the insurance form: what solar did I put on the roof?",
            "fact-far", 63, _mention("panel wattage", r"175", r"350"),
        ),
        152: _probe(
            152,
            "What was that old keepsake I discovered hidden in the bodywork a while "
            "back?",
            "aside-mid-late", 105, _mention("ferry ticket", r"ferry", r"santander", r"ticket"),
        ),
        156: _probe(
            156,
            "Final checklist time — we leave in days. What's still outstanding that "
            "I asked you to keep track of?",
            "commitment-mid-late", 122, _mention("passport reminder", r"passport"),
        ),
        158: _probe(
            158,
            "Quick-fire round before we go: the van's name, the dog's name, and the "
            "battery size?",
            "composite-far", 1,
            _mention("van", r"juniper"), _mention("dog", r"biscuit"), _mention("battery", r"280"),
        ),
    }
)

FILLER_PATH = Path(__file__).with_name("s14_soak_filler.json")


def _load_filler() -> dict[int, str]:
    if not FILLER_PATH.exists():
        return {}
    raw = json.loads(FILLER_PATH.read_text(encoding="utf-8"))
    filler = {int(index): str(text) for index, text in raw.items()}
    for index, text in filler.items():
        for pattern in FORBIDDEN_IN_FILLER:
            if re.search(pattern, text, flags=re.IGNORECASE):
                raise ValueError(f"s14 filler turn {index} leaks planted token /{pattern}/")
    return filler


def filler_indices() -> list[int]:
    return [index for index in range(1, TOTAL_TURNS + 1) if index not in FIXED]


def build() -> Scenario:
    filler = _load_filler()
    turns: list[Turn] = []
    for index in range(1, TOTAL_TURNS + 1):
        turn = FIXED.get(index)
        if turn is None:
            text = filler.get(index)
            if text is None:
                raise ValueError(f"s14 filler missing for turn {index} ({FILLER_PATH.name})")
            turn = Turn(user=text, note="filler")
        if index > 1:
            new_session = (index - 1) % SESSION_TURNS == 0
            turn.advance_clock_s = EXTRA_GAPS_S.get(
                index, SESSION_GAP_S if new_session else TURN_GAP_S
            )
        turns.append(turn)
    return Scenario(
        id="s14-soak",
        title="Soak: 160-turn endless-conversation coherence",
        description=(
            "Twenty weekly sessions of one continuous conversation; planted facts, "
            "corrections, decision statuses, commitments and asides probed once each "
            "across near/mid/far distances and early/late positions."
        ),
        turns=turns,
        system_prompt=SYSTEM,
        start_clock=START_CLOCK,
    )
