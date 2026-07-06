"""S8 — interrupt mid-reply: the delivered prefix is the only truth.

The user stops the assistant twice mid-explanation (the harness keeps only
a short prefix of those replies in history, exactly like a chat client's
stop button), then work continues long enough for the folds to run over the
truncated material. Decision 5, docs/architecture.md: a reply is provisional
until the next request shows what the client retained.

Contract probed late:
(a) the mind must claim only the DELIVERED portion of an interrupted reply
    — a mind that folded its own full provisional reply has phantom memory
    of content the user never saw (the baseline cannot even make this
    mistake: its transcript was cut client-side),
(b) commitments and decisions planted around the interrupts must survive,
(c) the whole scenario must simply complete — under strict fail mode a
    reconciliation desync after a truncation dies loudly.
"""

from ..harness import Check, Scenario, Turn

SYSTEM = (
    "You are a helpful software engineering assistant for a solo developer. "
    "Be concise and stay on topic."
)

SCENARIO = Scenario(
    id="s8-interrupt",
    title="Interrupt mid-reply",
    description="Stop-button truncations must supersede replies; only delivered text is remembered.",
    system_prompt=SYSTEM,
    turns=[
        Turn(
            user=(
                "I'm building a Python log parser for my router's syslog "
                "stream. Where do we start?"
            ),
            mock_reply="Start with the line format: capture a sample and classify record types.",
        ),
        Turn(
            user=(
                "Decision: we parse with regex named groups and store parsed "
                "records in SQLite. Locked in."
            ),
            mock_reply="Locked: regex named groups -> SQLite.",
        ),
        Turn(
            user=(
                "Walk me through the whole parsing pipeline in detail, step "
                "by step, from raw syslog line to database row."
            ),
            truncate_reply_chars=180,
            note="interrupt 1: stop button after ~180 chars; the tail was never delivered",
            mock_reply=(
                "Step 1: ingest — tail the syslog socket and buffer complete lines. "
                "Step 2: classify the record type by facility. Step 3: apply the named-group "
                "regex for that type. Step 4: normalize timestamps to UTC. Step 5: validate "
                "fields. Step 6: batch-insert rows into SQLite. Step 7: commit and checkpoint."
            ),
        ),
        Turn(
            user=(
                "Sorry, stopped you there — too much detail for now. Just give "
                "me the one-line version of the pipeline instead."
            ),
            mock_reply="Ingest -> classify -> regex parse -> normalize -> insert into SQLite.",
        ),
        Turn(
            user="Write the named-group regex for the dnsmasq query lines.",
            mock_reply="(?P<ts>...) dnsmasq\\[\\d+\\]: query\\[(?P<qtype>[A-Z]+)\\] (?P<domain>\\S+)",
        ),
        Turn(
            user=(
                "One thing to not forget — before we finish this project, "
                "remind me to add a unit test for malformed lines."
            ),
            mock_reply="Noted: unit test for malformed lines before we finish.",
        ),
        Turn(
            user="Sketch the SQLite schema for the parsed records.",
            mock_reply="records(id, ts, host, facility, qtype, domain, raw) + index on (ts, host).",
        ),
        Turn(
            user="How should I dedupe repeated identical lines within a second?",
            mock_reply="Hash (ts-second, host, raw); skip inserts whose hash was just seen.",
        ),
        Turn(
            user="What about log rotation on the router side — anything to handle?",
            mock_reply="Follow the socket, not files; on file mode use inode-aware reopen.",
        ),
        Turn(
            user=(
                "Explain in detail how logrotate would interact with a "
                "long-running parser process if I did read from files."
            ),
            truncate_reply_chars=150,
            note="interrupt 2: stop button again mid-explanation",
            mock_reply=(
                "In detail: logrotate renames the file, so your open fd points at the rotated "
                "inode and you keep reading stale data. With copytruncate the file is copied "
                "then truncated, so you must detect size shrink and seek to zero. The robust "
                "pattern is watching inode changes and reopening, plus handling the race window."
            ),
        ),
        Turn(
            user="Okay, I get the gist. Moving on — how do I run this as a systemd service?",
            mock_reply="A simple unit with Restart=on-failure and journald for its own logs.",
        ),
        Turn(
            user=(
                "That first walkthrough I cut off — how far did you actually "
                "get before I stopped you?"
            ),
            note="probe: memory must reflect the DELIVERED prefix, not the full undelivered reply",
            checks=[
                Check(
                    kind="judge",
                    desc="claims only the delivered portion of the interrupted reply",
                    rubric=(
                        "Earlier, the assistant was interrupted almost "
                        "immediately while walking through a parsing pipeline "
                        "step by step: only the very beginning was delivered — "
                        "roughly one or two sentences, at most the first step. "
                        "The user now asks how far the assistant got before "
                        "being stopped. Does the reply claim only a small "
                        "delivered portion (getting cut off early / around the "
                        "first step)? It FAILS if it claims to have covered "
                        "many or all steps, or recites detailed later steps as "
                        "having been said."
                    ),
                ),
            ],
            mock_reply="Barely started — I'd only gotten through the ingest step when you stopped me.",
        ),
        Turn(
            user="What's still open before we can call this parser finished?",
            note="probe: commitment planted between interrupts survives",
            checks=[
                Check(
                    kind="must_mention",
                    desc="the malformed-lines unit test is still open",
                    patterns=[r"malformed"],
                ),
            ],
            mock_reply="Open item: the unit test for malformed lines.",
        ),
        Turn(
            user="Quick one-paragraph summary of the design so far.",
            note="probe: continuity intact after two interrupts",
            checks=[
                Check(
                    kind="must_mention",
                    desc="core design: regex parsing",
                    patterns=[r"regex"],
                ),
                Check(
                    kind="must_mention",
                    desc="core design: SQLite storage",
                    patterns=[r"sqlite"],
                ),
            ],
            mock_reply="Regex named-group parser feeding SQLite, dedupe by hash, socket-based ingest, systemd service.",
        ),
    ],
)
