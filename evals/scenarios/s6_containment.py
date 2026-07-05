"""S6 — heavy-task containment: deep work must not become permanent bulk.

Validation-plan scenario 6, adapted to black-box form (the baseline has no
workers; the middleware may implement the heavy synthesis however it wants).
A light conversation establishes pre-task state, then one turn dumps three
long design documents for comparison, then the conversation returns to light
work. Graded on: (a) synthesis quality over the dump, (b) whether pre-dump
state survived the flood, and (c) the token curve — on a transcript-stuffing
baseline every turn after the dump pays its full cost forever, which is the
"accumulated corpse" signature the middleware exists to remove. The report's
per-turn token bars make that signature directly visible.
"""

from ..harness import Check, Scenario, Turn


def _proposal(name: str, author: str, position: str, body_points: list[str]) -> str:
    lines = [
        f"PROPOSAL {name} — {position}",
        f"Author: {author}",
        "Status: draft for review",
        "",
        "Summary",
        "-------",
    ]
    for index, point in enumerate(body_points, start=1):
        lines.append(f"{index}. {point}")
    lines.append("")
    return "\n".join(lines)


DOC_A = _proposal(
    "A 'Ledger'",
    "mira",
    "full event-sourced sync",
    [
        "Every mutation on every device is an immutable event appended to a local log.",
        "Devices exchange logs pairwise; the merged log is totally ordered by hybrid logical clocks.",
        "Current state is a pure fold over the merged log, so conflicts are impossible by construction: both writes are kept and the fold decides.",
        "Storage cost grows without bound; a snapshot-plus-truncate compaction runs monthly and is the riskiest component.",
        "Estimated implementation effort: 8 weeks. New dependencies: none, but a custom HLC implementation is required.",
        "Offline behavior: perfect — the log simply grows until a peer is reachable.",
        "Main risk: compaction bugs can silently lose history; the design demands extensive property-based testing.",
        "The author considers CRDT libraries overkill and prefers owning the whole semantic.",
    ],
)

DOC_B = _proposal(
    "B 'Patchwork'",
    "jonas",
    "CRDT library on Automerge",
    [
        "State lives in an Automerge document per notebook; the library guarantees convergence.",
        "Sync is delta-based over any transport; peers exchange compressed change sets.",
        "Conflicts are resolved by Automerge's built-in last-writer-wins registers except for text, which merges character-wise.",
        "Storage overhead measured at roughly 3x plaintext in the prototype; acceptable for notes, painful for attachments.",
        "Estimated implementation effort: 3 weeks, most of it integration and migration of existing data.",
        "Offline behavior: perfect, inherited from the CRDT model.",
        "Main risk: the library dictates the data model; schema migrations inside Automerge documents are awkward and under-documented.",
        "Attachments stay outside the CRDT as content-addressed blobs synced separately.",
    ],
)

DOC_C = _proposal(
    "C 'Turnstile'",
    "priya",
    "central server, no peer-to-peer",
    [
        "A small always-on server (the user already runs one) holds the single source of truth.",
        "Devices sync via a queue of pending mutations with optimistic concurrency; a version check rejects stale writes.",
        "Rejected writes surface to the user as a three-way merge dialog — conflicts are rare in single-user practice.",
        "Storage cost is 1x plus a bounded undo window of 90 days.",
        "Estimated implementation effort: 2 weeks with boring, well-understood technology.",
        "Offline behavior: degraded — mutations queue locally and reconcile on reconnect; two offline devices can still diverge until both reach the server.",
        "Main risk: the server is a single point of failure and a privacy commitment; backups become critical.",
        "The author argues single-user reality makes P2P machinery unjustifiable complexity.",
    ],
)

HEAVY_TASK = (
    "Here are three competing sync-design proposals for my notes app. Read "
    "all three and compare them properly — where they genuinely disagree, "
    "what each is best at, and which risks are unique to each.\n\n"
    "=== DOCUMENT 1 ===\n" + DOC_A + "\n=== DOCUMENT 2 ===\n" + DOC_B +
    "\n=== DOCUMENT 3 ===\n" + DOC_C
)

SYSTEM = (
    "You are a helpful software engineering assistant for a solo developer. "
    "Be concise and concrete."
)

SCENARIO = Scenario(
    id="s6-containment",
    title="Heavy-task containment",
    description="A bulky one-off synthesis task must not permanently tax every later turn.",
    system_prompt=SYSTEM,
    turns=[
        Turn(
            user=(
                "I'm working on my notes app 'inkwell' today. Two work items: "
                "the calendar feature, and picking a sync design."
            ),
            mock_reply="Calendar feature and sync design — let's start.",
        ),
        Turn(
            user=(
                "Calendar first. Decision: recurring events are stored as RRULE "
                "strings, expanded at render time, never materialized into "
                "occurrence rows. Occurrences would drift out of sync with edits."
            ),
            mock_reply="Agreed: store RRULE, expand at render, no occurrence rows.",
        ),
        Turn(
            user="What edge cases should the RRULE expansion tests cover?",
            mock_reply="DST transitions, COUNT vs UNTIL, timezone-crossing events, leap days, EXDATE.",
        ),
        Turn(
            user="Good. Park the calendar; now the sync design decision.",
            mock_reply="Ready for the proposals.",
        ),
        Turn(
            user=HEAVY_TASK,
            note="probe: synthesis across the three documents",
            checks=[
                Check(
                    kind="must_mention",
                    desc="engages all three proposals",
                    patterns=[r"(ledger|event.?sourc).{0,2000}(patchwork|automerge|crdt).{0,2000}(turnstile|central|server)"],
                ),
                Check(
                    kind="judge",
                    desc="comparison captures the real trade-off axes",
                    rubric=(
                        "Three sync proposals were compared. Trade-off axes: "
                        "A 'Ledger' event-sourcing — conflicts impossible by "
                        "construction, 8 weeks effort, unbounded log with risky "
                        "monthly compaction; B 'Patchwork' Automerge CRDT — 3 "
                        "weeks, ~3x storage, library dictates the data model, "
                        "awkward schema migrations; C 'Turnstile' central "
                        "server — 2 weeks, boring tech, single point of failure "
                        "plus privacy/backup burden, degraded offline. The full "
                        "documents contained more supporting detail than this "
                        "summary (hybrid logical clocks, LWW/character-wise "
                        "merge, content-addressed attachments, P2P machinery, "
                        "single-user context) — do NOT penalize the reply for "
                        "referencing document details missing from this "
                        "summary. Does the reply accurately capture at least "
                        "two of the trade-off axes without contradicting them?"
                    ),
                ),
            ],
            mock_reply=(
                "Ledger: strongest guarantees, costliest (8 weeks, risky compaction). "
                "Patchwork/Automerge: fastest to correctness (3 weeks) but 3x storage "
                "and the library owns your schema. Turnstile: simplest (2 weeks) but "
                "central point of failure and weaker offline."
            ),
        ),
        Turn(
            user=(
                "I'll sleep on it, leaning Turnstile. Back to light work: write a "
                "SQL query for this week's notes, table notes(id, title, "
                "created_at)."
            ),
            mock_reply="SELECT id, title FROM notes WHERE created_at >= date('now', '-7 days');",
        ),
        Turn(
            user="Back to the calendar feature — remind me what we decided about recurring events, and why?",
            note="probe: pre-dump state survived the document flood",
            checks=[
                Check(
                    kind="must_mention",
                    desc="recalls the RRULE decision",
                    patterns=[r"rrule"],
                ),
                Check(
                    kind="must_mention",
                    desc="recalls expansion at render / no materialized occurrences",
                    patterns=[r"render", r"expan", r"not?\s+materializ", r"occurrence"],
                ),
            ],
            mock_reply="Store recurring events as RRULE strings, expand at render time; no occurrence rows because they'd drift from edits.",
        ),
        Turn(
            user="Add a UI copy line for the recurring-event editor, friendly tone.",
            mock_reply="\"Repeats every week — edit the rule and every occurrence follows.\"",
        ),
        Turn(
            user="Which proposal had the shortest implementation estimate, and how long was it?",
            note="probe: a small fact from the heavy task is still addressable afterwards",
            checks=[
                Check(
                    kind="must_mention",
                    desc="Turnstile / central server at 2 weeks",
                    patterns=[r"turnstile", r"central.{0,40}server"],
                ),
                Check(
                    kind="must_mention",
                    desc="the 2-week estimate",
                    patterns=[r"2\s*weeks?", r"two\s*weeks?"],
                ),
            ],
            mock_reply="Turnstile, the central-server design — 2 weeks.",
        ),
        Turn(
            user="Draft tomorrow's todo list from today's session, max five items.",
            mock_reply="1) decide sync (leaning Turnstile) 2) RRULE tests 3) editor copy review 4) backup plan if central server 5) migration sketch.",
        ),
        Turn(
            user="What's a quick keyboard shortcut convention for the notes list? Just a suggestion.",
            mock_reply="j/k navigate, enter opens, e edits, / searches — vim-ish and mnemonic.",
        ),
        Turn(
            user="Last thing: one-sentence status for my project journal.",
            note="probe: closing state is intact after the whole session",
            checks=[
                Check(
                    kind="judge",
                    desc="status covers both work items truthfully",
                    rubric=(
                        "Today's session: decided recurring events are stored as "
                        "RRULE and expanded at render time, and compared three "
                        "sync proposals (event-sourcing, Automerge CRDT, central "
                        "server) with the user leaning toward the central-server "
                        "'Turnstile' option. Does the one-sentence status "
                        "truthfully reflect this session?"
                    ),
                ),
            ],
            mock_reply="Locked the RRULE render-time design for recurring events and compared three sync proposals, leaning Turnstile pending a night's sleep.",
        ),
    ],
)
