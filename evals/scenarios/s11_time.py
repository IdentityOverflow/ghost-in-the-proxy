"""S11 — chronos: time-aware commitments across a real wall-clock gap.

v4 discriminator. The user starts pizza dough proofing ("needs exactly two
hours, remind me when I'm back"), chats for ~25 minutes, leaves for the
garden, and returns FOUR hours later. The virtual clock advances via
Turn.advance_clock_s (X-Mind-Clock header); the return message never says
how long the user was gone.

Four probes, three kinds of teeth:
- t3  no-nag: minutes after the request, the reminder must not fire as due.
- t9  recall parity: the reminder itself must fire on return (transcript
  stuffing passes this too — it is not the time probe).
- t10 duration: "how long was I actually out?" — only a system with a clock
  can answer ~4 hours; a model anchored to the scripted "two hours" fails.
- t11 time of day: "what's the time now?" — deterministic, ~14:28.

Confirm-fail contract: v3 (no clock anywhere) must fail t10/t11 exactly like
the baseline; v4 chronos must pass all four.
"""

from datetime import datetime

from ..harness import Check, Scenario, Turn

SYSTEM = (
    "You are a helpful assistant for a home cook. Be concise and friendly. "
    "Keep track of promises and follow-ups across the conversation."
)

# Saturday morning, local time. Fixed so rubrics can pin absolute times.
START_CLOCK = datetime(2026, 3, 14, 10, 0, 0).timestamp()

SCENARIO = Scenario(
    id="s11-time",
    title="Time-aware commitment across a 4-hour gap",
    description="Dough reminder set at 10:03 with a 2h window; user returns at 14:26 without saying so. Duration and time-of-day probes need a real clock.",
    system_prompt=SYSTEM,
    start_clock=START_CLOCK,
    turns=[
        Turn(
            user=(
                "Saturday baking day! I just mixed a batch of pizza dough and set "
                "it to proof in the big blue bowl on the counter."
            ),
            mock_reply="Sounds great — pizza dough proofing, noted.",
        ),
        Turn(
            advance_clock_s=180,  # 10:03
            user=(
                "The dough needs exactly two hours to proof, no more. I'll be in "
                "and out of the garden all day, so remind me to punch it down and "
                "divide it into four 250 g balls once the two hours are up."
            ),
            mock_reply="Will do: when the two hours are up, punch down and divide into four 250 g balls.",
        ),
        Turn(
            advance_clock_s=120,  # 10:05
            user=(
                "While it rises — I want records on while I cook later. I've got "
                "mostly jazz. Pick me two albums that fit an afternoon of baking."
            ),
            note="probe: no-nag — 2 minutes in, the reminder must not fire as due",
            checks=[
                Check(
                    kind="judge",
                    desc="answers the music question without claiming the dough is due now",
                    rubric=(
                        "The user asked for two jazz album picks a couple of "
                        "minutes after setting a two-hour dough reminder. Does "
                        "the reply answer the music question WITHOUT telling "
                        "the user to punch down the dough now and WITHOUT "
                        "claiming the proofing time is up or nearly up? "
                        "Mentioning the future reminder in passing is fine; "
                        "treating it as currently due is a FAIL."
                    ),
                ),
            ],
            mock_reply="Kind of Blue and Getz/Gilberto — perfect baking-afternoon records.",
        ),
        Turn(
            advance_clock_s=300,  # 10:10
            user=(
                "My tomato seedlings are getting leggy on the windowsill. Worth "
                "moving them to the cold frame this early in March?"
            ),
            mock_reply="Harden them off gradually; a cold frame with a night cover works in March.",
        ),
        Turn(
            advance_clock_s=240,  # 10:14
            user=(
                "Baking-adjacent question: pizza steel or pizza stone for a home "
                "oven that tops out at 275 C? I keep reading conflicting takes."
            ),
            mock_reply="Steel — better conduction at low oven temps; stones want more heat.",
        ),
        Turn(
            advance_clock_s=300,  # 10:19
            user=(
                "My sourdough starter smells like acetone since yesterday. Is it "
                "ruined or just hungry?"
            ),
            mock_reply="Just hungry — acetone smell means underfed; do two feedings 12 h apart.",
        ),
        Turn(
            advance_clock_s=240,  # 10:23
            user=(
                "One more garden thing: the rhubarb crowns I planted in autumn "
                "are pushing up. Can I pull any stalks this first year?"
            ),
            mock_reply="Don't harvest year one — let the crowns build strength; light picking next year.",
        ),
        Turn(
            advance_clock_s=180,  # 10:26
            user=(
                "Alright, heading out to the garden. Hold my questions — I'll "
                "check back in when I'm inside again."
            ),
            mock_reply="Enjoy the garden! I'll be here when you're back.",
        ),
        Turn(
            advance_clock_s=14400,  # 14:26 — four hours, unstated
            user="Ok, I'm back inside now. Anything I should be doing?",
            note="probe: reminder fires on return (recall parity — not the time probe)",
            checks=[
                Check(
                    kind="must_mention",
                    desc="raises the dough reminder unprompted",
                    patterns=[r"dough", r"punch", r"250\s*g", r"four\s+balls"],
                ),
                Check(
                    kind="judge",
                    desc="reminds unprompted to punch down and divide the dough",
                    rubric=(
                        "Earlier the user asked to be reminded, once two hours "
                        "of proofing were up, to punch down the pizza dough and "
                        "divide it into four 250 g balls. They just said only "
                        "'Ok, I'm back inside now. Anything I should be doing?'. "
                        "Does the reply raise the dough task unprompted?"
                    ),
                ),
            ],
            mock_reply="The dough! The two hours are long past — punch it down and divide into four 250 g balls now.",
        ),
        Turn(
            advance_clock_s=60,  # 14:27
            user="How long was I actually out there?",
            note="probe: duration — needs a clock; 'two hours' anchoring fails",
            checks=[
                Check(
                    kind="judge",
                    desc="states the absence was roughly four hours",
                    rubric=(
                        "The user was away for four hours (left ~10:26, "
                        "returned ~14:26) and asks how long they were out. "
                        "Does the reply give a concrete duration of roughly "
                        "four hours (anything from 3.5 to 4.5 hours counts)? "
                        "Saying 'about two hours', refusing to estimate, or "
                        "giving no duration at all is a FAIL."
                    ),
                ),
            ],
            mock_reply="About four hours — you left around 10:26 and came back at 14:26.",
        ),
        Turn(
            advance_clock_s=90,  # ~14:28
            user="What's the time now, by the way?",
            note="probe: time of day — deterministic, ~14:28",
            checks=[
                Check(
                    kind="must_mention",
                    desc="states the current time (~14:28, accepts 14:2x/14:3x or 2:2x/2:3x pm)",
                    patterns=[r"14:[23]\d", r"\b2:[23]\d"],
                ),
            ],
            mock_reply="It's 14:28.",
        ),
    ],
)
