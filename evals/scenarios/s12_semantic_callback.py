"""S12 — semantic callback: a paraphrased probe with ZERO content-word overlap.

The Mem-backend discriminator. An early aside (landlord's tortoise Muriel)
folds out of verbatim view under network-planning filler; the first probe
asks about it in entirely different words — the lexical cue (coverage>=0.18,
hits>=2) and the recall tool's lexical search have nothing to grip. The
second probe asks with the original words as the control: it certifies the
memory is present and lexically reachable, so a probe-A miss isolates the
purely semantic gap.

Zero overlap between the aside and probe A is certified by a unit test
against the mind's real tokenizer (tests/test_mind.py), so the property
cannot rot silently. Caveat for live runs: the model's own probe-A recall
QUERY may translate the paraphrase back into original words ("slow reptile"
-> "tortoise"); if it does and passes, that is a legitimate pass — evidence
that lexical search plus model translation covers semantic callbacks. The
assistant's live reply to the aside may also echo probe words; check recall
telemetry before crediting either side.

CERTIFICATION RESULT (2026-07-20, qat @8k, mind v4.1): probe A PASSED in
both runs, each via a different DISTILLATION surface, with the raw-event
lexical path never firing — run 1: the steward paraphrased "called" to
"named" into the thread fact and CRS coverage readmitted the thread; run 2
(hardened who-was probe): the episode narrative ("a tortoise named Muriel
from a previous apartment") rendered un-gated by thread dormancy. Verdict:
the distillation stack (facts, thread summaries, episodes) already acts as
a semantic normalizer for memorable asides, so paraphrased callbacks are
NOT forcing evidence for a semantic Mem backend. The scenario stays in the
suite as a regression guard on exactly that property. A future s13 should
probe what distillation cannot cover: sequence recall ("what did we discuss
right after X?") and steward-dropped detail.
"""

from ..harness import Check, Scenario, Turn

SYSTEM = (
    "You are a practical assistant helping plan a home network overhaul. "
    "Be concise and concrete."
)

# Kept as named constants so the tokenizer-certification test imports the
# exact strings the scenario sends.
ASIDE = (
    "Oh — random memory before I forget it again: back when I was renting on "
    "Vasgatan, the landlord had an ancient tortoise called Muriel living in "
    "the stairwell, and she outlasted three of my neighbours' leases. "
    "Anyway — where were we with the switch question?"
)

# "who was" rather than "what was it named/called": name/call are the two
# canonical verbs a steward paraphrase converges on (observed live — the
# first certification run passed because the steward rewrote the aside's
# "called" as "named" and the probe said "named": distillation expands the
# lexical surface). The discriminator dodges that axis.
PROBE_SEMANTIC = (
    "One more thing as we wrap up the rack stuff — who was the slow reptile "
    "at my old place? The one the building owner kept indoors."
)

PROBE_LEXICAL = "And the landlord's tortoise — what was she called again?"

SCENARIO = Scenario(
    id="s12-semantic-callback",
    title="Semantic callback without shared words",
    description="Zero-overlap paraphrased recall; lexical machinery must translate or miss.",
    system_prompt=SYSTEM,
    turns=[
        Turn(
            user=(
                "I'm redoing my home network this month. Current state: ISP router in "
                "the hallway, one dumb 8-port switch in the living room, wifi dead in "
                "the office, and a NAS that drops off the network weekly. I want to do "
                "this properly — where should I start?"
            ),
            mock_reply="Start with a wiring map and a managed switch; we'll segment from there.",
        ),
        Turn(
            user=ASIDE,
            mock_reply="Muriel sounds indestructible. Back to switches: managed, 16 ports.",
        ),
        Turn(
            user=(
                "Right, the switch. I can get a used enterprise managed switch cheap, "
                "but it sounds like a jet engine. The alternatives are a fanless "
                "16-port smart switch or sticking with dumb switches and giving up on "
                "VLANs entirely. The rack would live in the hallway closet, which "
                "shares a wall with the bedroom. What's the sane choice?"
            ),
            mock_reply="Fanless smart switch: VLANs without the noise penalty next to a bedroom.",
        ),
        Turn(
            user=(
                "Fanless it is. Now VLANs: I want the NAS and backups separated from "
                "the flat's general traffic, guests isolated, and the smart-home "
                "gadgets quarantined because I don't trust a single one of them. "
                "Sketch me a VLAN layout with sensible numbering."
            ),
            mock_reply="VLAN 10 trusted, 20 servers/NAS, 30 IoT, 40 guest; firewall between.",
        ),
        Turn(
            user=(
                "Good. For wifi: the office dead zone is two brick walls from the "
                "router. I'd rather run one ethernet drop and hang an access point "
                "than mess with mesh repeaters that halve the bandwidth. Where exactly "
                "would you put the AP, and does it join the VLAN scheme?"
            ),
            mock_reply="Ceiling-mount the AP in the hallway midpoint; trunk port, SSIDs mapped to VLANs.",
        ),
        Turn(
            user=(
                "The NAS dropping off weekly turned out to be a power-saving setting "
                "on its NIC, by the way — found it in the admin panel logs. While I'm "
                "in there: the backup job runs nightly at 3am over SMB and saturates "
                "everything for an hour. Better schedule or better protocol?"
            ),
            mock_reply="Keep 3am, switch to rsync over SSH with bwlimit; SMB overhead is the hog.",
        ),
        Turn(
            user=(
                "Cabling plan: I can fish cat6 through the old telephone conduits to "
                "the office and bedroom, but the living room run means either drilling "
                "through a load-bearing wall or a 12-meter surface-mounted duct along "
                "the skirting. Which ugliness do I accept?"
            ),
            mock_reply="Surface duct: reversible, no structural risk, paintable.",
        ),
        Turn(
            user=(
                "Fine, duct it is. Power question: the closet has a single socket on "
                "the same circuit as the washing machine, which trips things twice a "
                "year. Small UPS for the switch, router and NAS — what capacity, and "
                "does the NAS need a clean-shutdown hookup?"
            ),
            mock_reply="600-900VA line-interactive; yes, USB signalling to the NAS for clean shutdown.",
        ),
        Turn(
            user=(
                "Addressing: static IPs for the NAS, printer and AP, DHCP for the "
                "rest? And should the smart-home VLAN get internet at all, or only "
                "local broker access? Half those gadgets phone home to servers I "
                "can't audit."
            ),
            mock_reply="Reservations over statics; IoT gets broker-only egress with explicit allowlist.",
        ),
        Turn(
            user=(
                "Last infrastructure bit: monitoring. I want to know when the NAS "
                "misses a backup, when a gadget starts chattering to a new address, "
                "and when the office AP drops. One lightweight dashboard, not an "
                "enterprise suite. What do I run and where does it live?"
            ),
            mock_reply="Uptime-Kuma plus ntopng on the NAS in a container; alerts to your phone.",
        ),
        Turn(
            user=PROBE_SEMANTIC,
            note="probe A: zero content-word overlap with the aside (certified by unit test)",
            checks=[
                Check(
                    kind="must_mention",
                    desc="recalls the name Muriel through the paraphrase",
                    patterns=[r"[Mm]uriel"],
                ),
                Check(
                    kind="judge",
                    desc="either recalls Muriel or honestly declines; no invented name",
                    rubric=(
                        "Earlier in this conversation the user mentioned, as an aside, "
                        "that a former landlord kept a tortoise called Muriel in the "
                        "stairwell. The user now asks for the animal's name using only "
                        "a paraphrase. PASS if the reply names Muriel, OR plainly says "
                        "it cannot recall the name. FAIL if it invents a different "
                        "name, a different animal, or details the user never said. "
                        "Judge only from the reply text."
                    ),
                ),
            ],
            mock_reply="That was Muriel, the landlord's tortoise.",
        ),
        Turn(
            user=PROBE_LEXICAL,
            note="probe B: lexical control — same aside, original words (landlord/tortoise/called)",
            checks=[
                Check(
                    kind="must_mention",
                    desc="control: name reachable when the words match",
                    patterns=[r"[Mm]uriel"],
                ),
            ],
            mock_reply="Muriel.",
        ),
        Turn(
            user="Perfect. Summarize the network plan in five bullets so I can shop.",
            mock_reply=(
                "Fanless 16-port managed switch; VLANs 10/20/30/40; one cat6 drop + "
                "ceiling AP; UPS with NAS shutdown; Uptime-Kuma monitoring."
            ),
        ),
    ],
)
