"""S2 — tool-heavy 10-turn sequence: repo reads, command outputs, synthesis.

Validation-plan scenario 2. The harness stubs two tools with bulky canned
outputs (a few KB each) so the transcript accumulates realistic tool debris.
The final probes are answerable only by synthesizing across three separate
tool outputs: the bug is a unit mismatch (interval_ms consumed as seconds)
that no single output states outright.
"""

from ..harness import CannedResult, Check, Scenario, ToolDef, Turn

SYSTEM = (
    "You are a coding assistant helping debug the fictional 'pulsehub' repo. "
    "Use the provided tools to read files and run commands when you need "
    "facts; do not invent file contents."
)

SCHEDULER_PY = '''\
$ read_file app/scheduler.py
"""pulsehub job scheduler."""

import time
import threading

from app.config import settings
from app.jobs import registry


class Scheduler:
    """Runs registered jobs on a fixed heartbeat.

    The heartbeat is configured by `settings.heartbeat_interval_ms`.
    """

    def __init__(self):
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            started = time.monotonic()
            for job in registry.due_jobs(now=time.time()):
                try:
                    job.run()
                except Exception as error:
                    print(f"job {job.name} failed: {error}")
            elapsed = time.monotonic() - started
            # Sleep the remainder of the heartbeat.
            time.sleep(max(0.0, settings.heartbeat_interval_ms - elapsed))

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
'''

CONFIG_PY = '''\
$ read_file app/config.py
"""pulsehub settings, loaded from environment."""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Heartbeat between scheduler ticks, in milliseconds.
    heartbeat_interval_ms: int = int(os.getenv("PULSEHUB_HEARTBEAT_MS", "250"))
    # Maximum jobs run per tick.
    max_jobs_per_tick: int = int(os.getenv("PULSEHUB_MAX_JOBS", "16"))
    # Database path.
    db_path: str = os.getenv("PULSEHUB_DB", "pulsehub.sqlite3")


settings = Settings()
'''

PYTEST_OUTPUT = '''\
$ run_command pytest tests/test_scheduler.py -x -q
============================= test session starts ==============================
platform linux -- Python 3.10.14, pytest-8.2.0
collected 3 items

tests/test_scheduler.py .F

=================================== FAILURES ===================================
______________________ test_ticks_every_quarter_second _______________________

    def test_ticks_every_quarter_second():
        scheduler = Scheduler()
        scheduler.start()
        time.sleep(1.2)
        scheduler.stop()
>       assert 4 <= tick_counter.count <= 6, (
            f"expected ~5 ticks in 1.2s at 250ms heartbeat, got {tick_counter.count}"
        )
E       AssertionError: expected ~5 ticks in 1.2s at 250ms heartbeat, got 1
E       assert 4 <= 1

tests/test_scheduler.py:41: AssertionError
=========================== short test summary info ============================
FAILED tests/test_scheduler.py::test_ticks_every_quarter_second - AssertionEr...
1 failed, 2 passed in 3.41s
'''

GREP_OUTPUT = '''\
$ run_command grep -rn "heartbeat_interval_ms" app/ tests/
app/config.py:11:    heartbeat_interval_ms: int = int(os.getenv("PULSEHUB_HEARTBEAT_MS", "250"))
app/scheduler.py:14:    The heartbeat is configured by `settings.heartbeat_interval_ms`.
app/scheduler.py:33:            time.sleep(max(0.0, settings.heartbeat_interval_ms - elapsed))
tests/test_scheduler.py:12:HEARTBEAT_MS = 250
tests/test_scheduler.py:39:    # 250ms heartbeat -> ~5 ticks in 1.2 seconds
'''

TOOLS = [
    ToolDef(
        name="read_file",
        description="Read a file from the pulsehub repository. Args: {path}.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "repo-relative path"}},
            "required": ["path"],
        },
        results=[
            CannedResult(match="scheduler", content=SCHEDULER_PY),
            CannedResult(match="config", content=CONFIG_PY),
            CannedResult(match=None, content="error: file not found"),
        ],
    ),
    ToolDef(
        name="run_command",
        description="Run a shell command in the pulsehub repo. Args: {command}.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        results=[
            CannedResult(match="pytest", content=PYTEST_OUTPUT),
            CannedResult(match="grep", content=GREP_OUTPUT),
            CannedResult(match=None, content="(command produced no output)"),
        ],
    ),
]

SCENARIO = Scenario(
    id="s2-tool-heavy",
    title="Tool-heavy debugging sequence",
    description="Synthesis across bulky tool outputs; detail retention afterwards.",
    system_prompt=SYSTEM,
    tools=TOOLS,
    turns=[
        Turn(
            user=(
                "Users report pulsehub jobs run way less often than configured. "
                "Start by running the scheduler tests."
            ),
            expects_tool=True,
            fallback_tool="run_command",
            mock_reply="Running pytest... the quarter-second tick test fails: 1 tick instead of ~5.",
        ),
        Turn(
            user="Interesting. Read the scheduler implementation.",
            expects_tool=True,
            fallback_tool="read_file",
            mock_reply="scheduler.py sleeps settings.heartbeat_interval_ms each loop.",
        ),
        Turn(
            user="Now read the config so we know what that setting actually is.",
            expects_tool=True,
            fallback_tool="read_file",
            mock_reply="config.py: heartbeat_interval_ms defaults to 250, documented as milliseconds.",
        ),
        Turn(
            user="Check where else that setting is referenced, just to be safe.",
            expects_tool=True,
            fallback_tool="run_command",
            mock_reply="grep shows config default 250, the scheduler sleep, and tests assuming 250ms.",
        ),
        Turn(
            user="So what's the bug? Name the file and the exact mistake.",
            note="probe: cross-output synthesis — unit mismatch",
            checks=[
                Check(
                    kind="must_mention",
                    desc="locates the bug in the scheduler sleep",
                    patterns=[r"scheduler\.py", r"time\.sleep"],
                ),
                Check(
                    kind="must_mention",
                    desc="names the unit mismatch (ms consumed as seconds)",
                    patterns=[
                        r"millisecond.{0,80}second",
                        r"second.{0,80}millisecond",
                        r"ms.{0,60}(as|vs\.?|instead of|not).{0,20}s(econds)?\b",
                        r"250\s*(ms|millisecond).{0,80}250\s*s(econds)?",
                        r"/\s*1000|divide.{0,30}1000|1000\.0",
                    ],
                ),
            ],
            mock_reply=(
                "app/scheduler.py passes heartbeat_interval_ms (250, milliseconds) "
                "straight to time.sleep, which takes seconds — so it sleeps 250s. "
                "Divide by 1000."
            ),
        ),
        Turn(
            user="Write the one-line fix.",
            mock_reply="time.sleep(max(0.0, settings.heartbeat_interval_ms / 1000.0 - elapsed))",
        ),
        Turn(
            user="Draft a short commit message for it.",
            mock_reply="fix(scheduler): convert heartbeat interval from ms to seconds before sleeping",
        ),
        Turn(
            user="While we're here — what would you add to the test suite to catch unit bugs like this earlier?",
            mock_reply="A fast-clock test asserting tick cadence, and a settings unit-suffix convention test.",
        ),
        Turn(
            user="What was the default heartbeat value in the config, and in what unit?",
            note="probe: retention of a small detail buried in tool output bulk",
            checks=[
                Check(
                    kind="must_mention",
                    desc="recalls default 250",
                    patterns=[r"\b250\b"],
                ),
                Check(
                    kind="must_mention",
                    desc="recalls the unit is milliseconds",
                    patterns=[r"millisecond", r"\bms\b"],
                ),
            ],
            mock_reply="250, in milliseconds (PULSEHUB_HEARTBEAT_MS).",
        ),
        Turn(
            user="Summarize the whole investigation in three sentences for the issue tracker.",
            note="probe: end-of-session synthesis stays grounded",
            checks=[
                Check(
                    kind="must_mention",
                    desc="summary still names scheduler and the ms/seconds confusion",
                    patterns=[r"scheduler"],
                ),
                Check(
                    kind="judge",
                    desc="three-sentence summary is accurate to the investigation",
                    rubric=(
                        "The investigation found: pulsehub's scheduler passed a "
                        "milliseconds config value (heartbeat_interval_ms, default "
                        "250) directly to time.sleep which takes seconds, making "
                        "jobs run ~1000x too rarely; the fix divides by 1000. "
                        "Is the reply an accurate summary of that, without "
                        "invented findings?"
                    ),
                ),
            ],
            mock_reply=(
                "Jobs ran rarely because scheduler.py slept heartbeat_interval_ms "
                "seconds instead of milliseconds. Config default is 250ms. Fix: "
                "divide by 1000 before time.sleep."
            ),
        ),
    ],
)
