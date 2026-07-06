"""S9 — verbatim recall past the fold horizon: the recall-tool discriminator.

The user pastes a chunky traceback early, then does twelve turns of other
work — more than enough fold pressure to evict the paste from texture and
distill it into ledger claims. Then they need the EXACT first line back for
a bug report. Distilled memory cannot produce an exact quote; only reaching
back into the raw event store can (v3 `recall`, the provenance escape
hatch — docs/architecture.md). The check is a deterministic regex on the
exact line, so nothing about this probe is judge-dependent.

Expected: v2 mind FAILS this (paraphrase or honest refusal), which is the
point — it is built to graduate v3, and doubles as a confabulation probe:
inventing a plausible-but-wrong quote is the worst outcome.
"""

from ..harness import Check, Scenario, Turn

SYSTEM = (
    "You are a helpful software engineering assistant for a solo developer. "
    "Be concise and concrete."
)

TRACEBACK = """\
Traceback (most recent call last):
  File "/opt/trailcam/app/ingest.py", line 214, in process_upload
    photo = persist_photo(session, upload, sha)
  File "/opt/trailcam/app/ingest.py", line 158, in persist_photo
    session.execute(insert_stmt, params)
  File "/usr/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1412, in execute
    return meth(self, ...)
sqlite3.OperationalError: no such column: photos.review_flag_v2 [ref: ingest-9f4c2e71]"""

SCENARIO = Scenario(
    id="s9-verbatim",
    title="Verbatim recall past the fold horizon",
    description="An exact quote of long-evicted raw material; distilled memory cannot fake it.",
    system_prompt=SYSTEM,
    turns=[
        Turn(
            user=(
                "My trailcam app started failing uploads this morning after "
                "last night's migration. Here's the traceback:\n\n" + TRACEBACK
            ),
            mock_reply="The migration didn't add photos.review_flag_v2 — the insert references a missing column.",
        ),
        Turn(
            user=(
                "Right, the migration must not have applied. I'll rerun it "
                "tonight from the laptop — park that for now, I have other "
                "things to get through first."
            ),
            mock_reply="Parked: rerun the migration tonight. What's next?",
        ),
        Turn(
            user="First: my grafana dashboard shows gaps every night at 3am. Ideas?",
            mock_reply="Check whether the nightly vacuum/backup job blocks the datasource.",
        ),
        Turn(
            user="It's the backup job. How do I make the two not collide?",
            mock_reply="Stagger them: backup at 3am, move the report job to 4am, or use snapshots.",
        ),
        Turn(
            user="Move the report to 4am. Write the updated cron entries.",
            mock_reply="0 3 * * * backup.sh / 0 4 * * * report.sh — updated.",
        ),
        Turn(
            user="Next topic: I want rate limiting on my public photo-share links.",
            mock_reply="Token bucket per link id; nginx limit_req is the zero-code option.",
        ),
        Turn(
            user="Use nginx limit_req. Write the location block.",
            mock_reply="limit_req_zone + location /share/ { limit_req zone=share burst=10; }",
        ),
        Turn(
            user="Now help me pick a UPS for the server rack — it draws about 120W.",
            mock_reply="A 600-900VA line-interactive UPS gives 15-30 min at 120W.",
        ),
        Turn(
            user="Ordered a 900VA one. How do I get clean shutdowns when it kicks in?",
            mock_reply="apcupsd or NUT; trigger shutdown at 20% battery.",
        ),
        Turn(
            user="Set up NUT config for that.",
            mock_reply="ups.conf + upsmon.conf with SHUTDOWNCMD at battery.charge.low 20.",
        ),
        Turn(
            user="Also — the wifi camera on the north fence drops every few days.",
            mock_reply="Check RSSI first; if below -75 dBm you need a repeater or directional antenna.",
        ),
        Turn(
            user="RSSI is -79. So a repeater then — where roughly?",
            mock_reply="Midpoint with line of sight to both; powered outlet near the shed works.",
        ),
        Turn(
            user="Good enough. Summarize today's infra changes in two lines for my notes.",
            mock_reply="Cron staggered 3am/4am; nginx rate limit on /share/; UPS+NUT shutdown at 20%; repeater planned.",
        ),
        Turn(
            user=(
                "Back to this morning's upload failure — I'm filing the bug "
                "against the migration tool right now. Quote me the exact "
                "final line of that traceback, character for character; the "
                "maintainers want it verbatim including the ref id."
            ),
            note="probe: verbatim retrieval of long-evicted raw material (v3 recall discriminator)",
            checks=[
                Check(
                    kind="must_mention",
                    desc="the exact error line, verbatim",
                    patterns=[
                        r"sqlite3\.OperationalError: no such column: photos\.review_flag_v2 \[ref: ingest-9f4c2e71\]"
                    ],
                ),
                Check(
                    kind="must_not_mention",
                    desc="no confabulated ref id (worst outcome is a plausible wrong quote)",
                    patterns=[r"ingest-(?!9f4c2e71)[0-9a-f]{8}"],
                ),
            ],
            mock_reply=(
                "sqlite3.OperationalError: no such column: photos.review_flag_v2 [ref: ingest-9f4c2e71]"
            ),
        ),
        Turn(
            user="Thanks. And what was the plan for the migration itself?",
            note="probe: the parked commitment context survived alongside",
            checks=[
                Check(
                    kind="must_mention",
                    desc="rerun the migration tonight",
                    patterns=[r"rerun|re-run|tonight"],
                ),
            ],
            mock_reply="Rerun it tonight from the laptop.",
        ),
    ],
)
