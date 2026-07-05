"""S3 — long-horizon callback: commitments must survive topic drift.

Validation-plan scenario 3. Two commitments are planted early — one the user
asks the assistant to hold ("when we get to deployment, remind me to rotate
the Hetzner API token") and one the assistant is asked to track (flag when
combined disk estimates cross 40 GB). Then twelve turns of genuinely
unrelated work. The deployment probe deliberately avoids the words "remind"
or "token": a keyword-echo strategy fails it; only durable commitment state
passes.
"""

from ..harness import Check, Scenario, Turn

SYSTEM = (
    "You are a helpful assistant for a solo developer. Be concise. Keep track "
    "of promises and follow-ups across the conversation."
)

SCENARIO = Scenario(
    id="s3-callback",
    title="Long-horizon commitment callback",
    description="Early commitments must fire when their trigger topic returns, ~12 drift turns later.",
    system_prompt=SYSTEM,
    turns=[
        Turn(
            user=(
                "Today I want to knock out a bunch of small unrelated chores, and "
                "at the very end we'll plan the deployment of my 'lighthouse' app "
                "to my Hetzner box."
            ),
            mock_reply="Sounds good — chores first, deployment planning at the end.",
        ),
        Turn(
            user=(
                "Important, before I forget: WHEN we get to the deployment topic "
                "later, remind me that I must rotate the Hetzner API token first — "
                "the old one leaked into a shell history file. Don't bring it up "
                "before then, just when deployment comes up."
            ),
            mock_reply="Understood. When deployment comes up, I'll remind you to rotate the Hetzner API token first.",
        ),
        Turn(
            user=(
                "Also track this for me: I'll mention disk sizes of a few datasets "
                "as we go. If the running total ever passes 40 GB, flag it, "
                "because that's all the free space the box has. First one: the "
                "bird-audio dataset is 18 GB."
            ),
            mock_reply="Tracking disk total: 18 GB so far; I'll flag when it passes 40 GB.",
        ),
        Turn(
            user="First chore: write a bash one-liner to find files over 100 MB modified in the last week.",
            mock_reply="find . -type f -size +100M -mtime -7",
        ),
        Turn(
            user="Now a regex that matches ISO-8601 dates but only in 2025 or 2026.",
            mock_reply=r"\b202[56]-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b",
        ),
        Turn(
            user="Convert this to a markdown table: name,age / ada,36 / grace,45.",
            mock_reply="| name | age |\n|---|---|\n| ada | 36 |\n| grace | 45 |",
        ),
        Turn(
            user="Oh, second dataset: the trail-camera archive is 14 GB. Anyway, next chore.",
            mock_reply="Noted, running total 32 GB. What's next?",
        ),
        Turn(
            user="Explain the difference between git rebase and git merge in two sentences.",
            mock_reply="Rebase rewrites your commits onto a new base for linear history; merge joins histories with a merge commit.",
        ),
        Turn(
            user="Write a haiku about a message queue.",
            mock_reply="Messages waiting / patient in the broker's hall / consumers arrive",
        ),
        Turn(
            user="What's a good name for a CLI that syncs dotfiles? Give me five options.",
            mock_reply="tether, homing, dotdock, symlinkd, nestsync.",
        ),
        Turn(
            user=(
                "Third dataset for the box: the OpenStreetMap extract is 11 GB. "
                "Also, how do I make systemd restart a service on failure?"
            ),
            note="probe: threshold commitment fires — 18+14+11 = 43 GB > 40 GB",
            checks=[
                Check(
                    kind="must_mention",
                    desc="flags that the 40 GB disk budget is now exceeded",
                    patterns=[r"40\s*GB", r"\b43\s*GB", r"over.{0,30}(budget|limit|free space)", r"exceed"],
                ),
            ],
            mock_reply="Heads up: 18+14+11 = 43 GB — past your 40 GB free space. For systemd: Restart=on-failure in the unit.",
        ),
        Turn(
            user="Give me a .gitignore for a Python project, short version.",
            mock_reply="__pycache__/, *.pyc, .venv/, dist/, *.egg-info/, .env",
        ),
        Turn(
            user="What does HTTP 418 mean?",
            mock_reply="I'm a teapot — an April Fools RFC status code.",
        ),
        Turn(
            user="Recommend a color scheme for a terminal, something low-contrast for night use.",
            mock_reply="Gruvbox dark soft or Nord — both easy on the eyes at night.",
        ),
        Turn(
            user="Last chore: one-sentence explanation of what a bloom filter trades away.",
            mock_reply="It trades exactness for space: false positives possible, false negatives never.",
        ),
        Turn(
            user="Alright, chores done. Let's plan the lighthouse deployment on the Hetzner box now.",
            note="probe: trigger-based callback with zero keyword overlap ('remind'/'token' unsaid)",
            checks=[
                Check(
                    kind="must_mention",
                    desc="unprompted reminder to rotate the Hetzner API token first",
                    patterns=[r"rotat.{0,40}token", r"token.{0,60}(rotat|leak)", r"api\s+token"],
                ),
            ],
            mock_reply="Before anything: rotate the Hetzner API token (the old one leaked into shell history). Then: build, transfer, systemd unit, reverse proxy.",
        ),
        Turn(
            user="Anything else you were tracking for me from earlier that's still relevant?",
            note="probe: commitment inventory — the disk total should resurface",
            checks=[
                Check(
                    kind="must_mention",
                    desc="recalls the disk-space tracking situation (43 GB vs 40 GB)",
                    patterns=[r"43\s*GB", r"40\s*GB", r"disk", r"free space"],
                ),
            ],
            mock_reply="Yes — the dataset total hit 43 GB, over the 40 GB free space, so plan storage before deploying.",
        ),
        Turn(
            user="Good session. Summarize the deployment plan in four bullet points.",
            mock_reply="- rotate token - free disk space - build & transfer - systemd + reverse proxy",
        ),
    ],
)
