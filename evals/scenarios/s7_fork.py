"""S7 — fork/regenerate continuity: edited history is the only history.

The user plans a NAS backup system, commits to restic+B2 early (with a
rationale and an open commitment), builds an offsite plan around a Xeon box
— then EDITS that decision away mid-conversation (the fork drops three
exchanges) and later regenerates a reply. Real chat clients do both
constantly (decision 3, docs/architecture.md).

Contract probed late:
(a) the abandoned branch must not be presented as current state,
(b) pre-fork decisions and commitments must survive the fork,
(c) a regenerate request must not reset continuity.
A transcript-stuffing baseline gets forks for free (the transcript IS its
state) — this scenario exists so the mind provably matches that, with its
derived memory (ledger/threads) invalidated and rebuilt instead of stale.
"""

from ..harness import Check, Scenario, Turn

SYSTEM = (
    "You are a helpful software engineering assistant for a solo developer. "
    "Be concise and concrete."
)

SCENARIO = Scenario(
    id="s7-fork",
    title="Fork/regenerate continuity",
    description="Edits and regenerates rewrite history; memory must follow the surviving branch.",
    system_prompt=SYSTEM,
    turns=[
        Turn(
            user=(
                "Help me design backups for my home NAS — a TrueNAS box with "
                "about 8TB of photos and documents on it."
            ),
            mock_reply="Let's design it: local snapshots plus an offsite copy.",
        ),
        Turn(
            user=(
                "For the offsite tool I'm going with restic to Backblaze B2 — "
                "it encrypts by default and B2 is cheap. That's settled."
            ),
            mock_reply="Settled: restic to B2 (encrypted by default, cheap).",
        ),
        Turn(
            user=(
                "Sketch the restic repo layout and retention flags. Also, "
                "before we ever call this project done, remind me to do a full "
                "test restore — an untested backup doesn't exist."
            ),
            mock_reply="Repo layout + forget --keep flags sketched. Noted: test restore before done.",
        ),
        Turn(
            user="What about the database dumps from my paperless container?",
            mock_reply="pg_dump to a staging dir pre-backup; restic picks it up.",
        ),
        Turn(
            user=(
                "Second offsite copy goes to my brother's house — he has an old "
                "Xeon server in his basement that's always on. Plan the sync to "
                "that box."
            ),
            mock_reply="Plan: restic copy to a REST server on the Xeon over WireGuard.",
        ),
        Turn(
            user="Write the systemd timer for the nightly sync to the Xeon.",
            mock_reply="backup-sync.timer OnCalendar=nightly + service unit for the Xeon target.",
        ),
        Turn(
            user="How do I monitor that the offsite sync actually succeeded?",
            mock_reply="healthchecks.io ping on success; alert if silent 26h.",
        ),
        Turn(
            # FORK: rewind to before turn 5 and replace the whole Xeon branch.
            user=(
                "Change of plan for the second offsite copy: my brother is "
                "moving, the Xeon is gone. It'll be a Raspberry Pi 5 with a "
                "4TB USB drive at my parents' place instead. Plan the sync to "
                "that."
            ),
            rewind_user_turns=3,
            note="fork: edits turn 5, dropping the Xeon branch (turns 5-7)",
            mock_reply="Plan: restic REST server on the Pi 5, 4TB USB, at your parents'.",
        ),
        Turn(
            user=(
                "My parents are on a slow DSL line, about 10 Mbit up from my "
                "side. What does that mean for the initial seed?"
            ),
            mock_reply="8TB over 10 Mbit is ~2.5 months; seed the USB drive locally first.",
        ),
        Turn(
            user="Write the systemd timer for the nightly sync to the Pi.",
            mock_reply="backup-sync.timer OnCalendar=nightly + service unit for the Pi target.",
        ),
        Turn(
            # REGENERATE: resend turn 10 unchanged, asking for a fresh reply.
            user="Write the systemd timer for the nightly sync to the Pi.",
            rewind_user_turns=1,
            note="regenerate: same request resent with our previous reply dropped",
            mock_reply="Second draft: backup-sync.timer + service unit for the Pi target.",
        ),
        Turn(
            user="Add a monthly restic check --read-data-subset run too.",
            mock_reply="restic-check.timer, monthly, --read-data-subset=5%.",
        ),
        Turn(
            user=(
                "Recap the offsite plan for me: where does the second copy "
                "live, and on what hardware?"
            ),
            note="probe: the surviving branch is the only current state",
            checks=[
                Check(
                    kind="must_mention",
                    desc="offsite copy is the Pi at the parents' place",
                    patterns=[r"\bpi\b|raspberry"],
                ),
                Check(
                    kind="must_mention",
                    desc="location: parents",
                    patterns=[r"parents"],
                ),
                Check(
                    kind="judge",
                    desc="abandoned branch not presented as current",
                    rubric=(
                        "The user asks where their second offsite backup copy "
                        "lives. The current plan is a Raspberry Pi 5 with a 4TB "
                        "USB drive at the user's parents' place (slow DSL "
                        "uplink). An earlier plan — a Xeon server in the "
                        "brother's basement — was cancelled because the brother "
                        "is moving. Does the reply present the Pi-at-parents "
                        "plan as the current one, and avoid presenting the "
                        "Xeon/brother plan as still active? Mentioning the old "
                        "plan as cancelled/superseded is fine and must not be "
                        "penalized."
                    ),
                ),
            ],
            mock_reply="Second copy: Raspberry Pi 5 + 4TB USB at your parents' place.",
        ),
        Turn(
            user="And remind me why we went with restic plus B2 in the first place?",
            note="probe: pre-fork decision rationale survives the fork",
            checks=[
                Check(
                    kind="must_mention",
                    desc="rationale: encryption by default",
                    patterns=[r"encrypt"],
                ),
                Check(
                    kind="must_mention",
                    desc="rationale: cheap",
                    patterns=[r"cheap|cost|price|afford"],
                ),
            ],
            mock_reply="restic encrypts by default and B2 is cheap.",
        ),
        Turn(
            user="What's still open before I can call this backup project done?",
            note="probe: pre-fork commitment survives fork + regenerate",
            checks=[
                Check(
                    kind="must_mention",
                    desc="the full test restore is still open",
                    patterns=[r"test\s+restore|restore\s+test|full\s+restore"],
                ),
            ],
            mock_reply="Main open item: the full test restore.",
        ),
    ],
)
