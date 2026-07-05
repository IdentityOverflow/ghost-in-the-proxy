"""S5 — salience decay with cued recall: forgetting is not deletion.

Validation-plan scenario 5, sharpened into a pair of complementary probes.
A vivid but irrelevant aside is planted early (grandmother's plum dumplings,
distinctive detail: semolina instead of flour). Sixteen turns of unrelated
pipeline-design work follow. Then:
(a) an open-ended question where dragging the stale detail back in would be
    wrong — tests that low-salience material stays out of the active scene;
(b) an explicit cue — tests that it was decayed, not destroyed.
A system that passes (a) by hard-deleting will fail (b); a system that
passes (b) by keeping everything hot tends to fail (a) and shows in the
token curve. Passing both cheaply is the mark of real salience handling.
"""

from ..harness import Check, Scenario, Turn

SYSTEM = (
    "You are a helpful software engineering assistant for a solo developer. "
    "Be concise and stay on topic."
)

SCENARIO = Scenario(
    id="s5-salience",
    title="Salience decay with cued recall",
    description="A vivid irrelevant aside must fade from the active scene yet stay retrievable on cue.",
    system_prompt=SYSTEM,
    turns=[
        Turn(
            user=(
                "Let's design a logging pipeline for my fleet of five Raspberry "
                "Pis. They run assorted sensors and I want the logs in one place."
            ),
            mock_reply="Central log host + shippers on each Pi. Let's spec it.",
        ),
        Turn(
            user=(
                "Ha, completely unrelated — I'm cooking tonight and just "
                "remembered my grandmother's plum dumpling trick: she used "
                "semolina instead of flour in the dough, made them lighter. "
                "Anyway! Back to logs."
            ),
            mock_reply="Nice trick — semolina dough. Back to the pipeline: what's the log volume?",
        ),
        Turn(
            user="Volume is small, maybe 50 MB/day across all five. What shipper do you suggest?",
            mock_reply="At 50 MB/day, vector or even rsyslog forwarding is plenty.",
        ),
        Turn(
            user="Let's use vector. Central store: Loki or plain files with logrotate?",
            mock_reply="Loki if you want queries and Grafana; files+logrotate if you want zero services.",
        ),
        Turn(
            user="Loki. Sketch the vector.toml for a Pi shipping journald to the central Loki.",
            mock_reply="[sources.journald] ... [sinks.loki] endpoint=http://loghost:3100.",
        ),
        Turn(
            user="What labels should I put on the streams? I don't want cardinality explosions.",
            mock_reply="host and unit only; put everything else in the line, not in labels.",
        ),
        Turn(
            user="Now the retention policy — I want 30 days hot, then delete.",
            mock_reply="Loki limits_config: retention_period=720h with the compactor enabled.",
        ),
        Turn(
            user="One Pi is on flaky wifi. What happens to logs when the link drops?",
            mock_reply="Vector buffers on disk; set buffer.max_size and it replays on reconnect.",
        ),
        Turn(
            user="Configure that disk buffer, say 200 MB cap.",
            mock_reply="[sinks.loki.buffer] type=disk, max_size=209715200, when_full=block.",
        ),
        Turn(
            user="Add a Grafana alert: any Pi silent for 15 minutes.",
            mock_reply="absent-style query on count_over_time by host, 15m window.",
        ),
        Turn(
            user="What dashboard panels would you start with?",
            mock_reply="Log volume by host, error-rate by unit, silent-host table, disk buffer fill.",
        ),
        Turn(
            user="Should the central box also ship its own logs into Loki?",
            mock_reply="Yes — same vector config pointed at localhost; one pipeline for everything.",
        ),
        Turn(
            user=(
                "We're batching sinks now. Open question: anything else in this "
                "pipeline you think is worth batching or buffering that we "
                "haven't covered?"
            ),
            note="probe: stale aside must not intrude on an open-ended on-topic question",
            checks=[
                Check(
                    kind="must_not_mention",
                    desc="the dumpling aside stays out of the active scene",
                    patterns=[r"dumpling", r"semolina", r"plum", r"grandmother", r"recipe"],
                ),
                Check(
                    kind="judge",
                    desc="answer is on-topic for the logging pipeline",
                    rubric=(
                        "The conversation is about a Raspberry Pi logging "
                        "pipeline (vector, Loki, buffering). Is the reply an "
                        "on-topic answer about batching/buffering in that "
                        "pipeline, free of unrelated personal/cooking content?"
                    ),
                ),
            ],
            mock_reply="Batch Loki pushes (batch.max_bytes), buffer journald reads, and batch alert evaluation; nothing else needs it at 50 MB/day.",
        ),
        Turn(
            user="Write the final checklist for rolling this out to all five Pis.",
            mock_reply="1) install vector 2) drop config 3) enable service 4) verify streams in Grafana 5) silence-alert test.",
        ),
        Turn(
            user=(
                "Totally different thing before I go cook — what was that trick "
                "from my grandmother's recipe I mentioned way back?"
            ),
            note="probe: cued recall — decay must not have destroyed the memory",
            checks=[
                Check(
                    kind="must_mention",
                    desc="recalls semolina instead of flour in the dumpling dough",
                    patterns=[r"semolina"],
                ),
            ],
            mock_reply="Semolina instead of flour in the plum dumpling dough — lighter dumplings.",
        ),
        Turn(
            user="Right! Okay, one-paragraph summary of what we designed today.",
            note="probe: closing summary keeps the aside out of the design record",
            checks=[
                Check(
                    kind="must_mention",
                    desc="summary covers the actual pipeline",
                    patterns=[r"loki", r"vector", r"pipeline", r"logging"],
                ),
                Check(
                    kind="must_not_mention",
                    desc="summary excludes the cooking aside",
                    patterns=[r"dumpling", r"semolina", r"recipe"],
                ),
            ],
            mock_reply="We designed a vector->Loki pipeline for five Pis: journald sources, host/unit labels, 30-day retention, disk buffering for flaky wifi, and a silent-host alert.",
        ),
    ],
)
