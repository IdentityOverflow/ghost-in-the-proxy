"""S4 — contradiction: a corrected fact must supersede, not coexist.

Validation-plan scenario 4. A fact is planted (gateway on port 8080), built
upon so it has real weight in the transcript, then explicitly corrected to
9090. The two probes test both halves of good contradiction handling:
(a) current-state queries must use only the new value — recency of *wording*
must not resurrect 8080, which appears more often in the transcript;
(b) the history must not be erased — "did I change my mind?" should recover
the 8080→9090 transition. Transcript-stuffing tends to fail (a); naive
overwrite-style memory tends to fail (b).
"""

from ..harness import Check, Scenario, Turn

SYSTEM = (
    "You are a helpful infrastructure assistant for a solo developer. Be "
    "concise and concrete."
)

SCENARIO = Scenario(
    id="s4-contradiction",
    title="Corrected fact supersedes",
    description="Port 8080 corrected to 9090 mid-conversation; old value must not leak back.",
    system_prompt=SYSTEM,
    turns=[
        Turn(
            user=(
                "Some context about my home lab: the reverse proxy is Caddy, the "
                "API gateway service is called 'gatekeeper' and listens on port "
                "8080, and everything runs on one Debian box called 'anvil'."
            ),
            mock_reply="Got it: Caddy in front, gatekeeper on 8080, all on anvil.",
        ),
        Turn(
            user="Write the Caddyfile block that proxies api.home.lan to gatekeeper.",
            mock_reply="api.home.lan {\n  reverse_proxy localhost:8080\n}",
        ),
        Turn(
            user="Now a docker-compose service for gatekeeper with that port exposed.",
            mock_reply="gatekeeper:\n  image: gatekeeper:latest\n  ports:\n    - '8080:8080'",
        ),
        Turn(
            user="Add a healthcheck to that compose service.",
            mock_reply="healthcheck: CMD curl -f http://localhost:8080/healthz",
        ),
        Turn(
            user="Unrelated: what's the difference between CNAME and A records, briefly?",
            mock_reply="A maps a name to an IP; CNAME aliases a name to another name.",
        ),
        Turn(
            user=(
                "Correction on my earlier context: we moved gatekeeper to port "
                "9090 last week — 8080 collided with another service. 9090 is the "
                "truth from now on; the old configs are being migrated."
            ),
            mock_reply="Understood: gatekeeper is on 9090 now; 8080 is obsolete.",
        ),
        Turn(
            user="Noted? Okay. Now help me pick: Prometheus or VictoriaMetrics for a single box?",
            mock_reply="For one box, VictoriaMetrics single-node is lighter; Prometheus is the safer default.",
        ),
        Turn(
            user="Write a systemd timer that curls a URL every 5 minutes and logs failures.",
            mock_reply="Unit + timer with OnCalendar=*:0/5 and curl -f || logger.",
        ),
        Turn(
            user="Give me a two-line explanation of what an SLO is, for my notes.",
            mock_reply="An SLO is a target level of reliability for a service. It defines how much failure is acceptable before users suffer.",
        ),
        Turn(
            user="How would I check which process owns a port on anvil?",
            mock_reply="ss -ltnp | grep <port> (or lsof -i :<port>).",
        ),
        Turn(
            user="Write the curl command I'd use to hit gatekeeper's health endpoint directly on anvil.",
            note="probe: current-state query must use 9090, not the transcript-dominant 8080",
            checks=[
                Check(
                    kind="must_mention",
                    desc="uses the corrected port 9090",
                    patterns=[r"9090"],
                ),
                Check(
                    kind="must_not_mention",
                    desc="does not present 8080 as current",
                    patterns=[r"8080"],
                ),
            ],
            mock_reply="curl -f http://localhost:9090/healthz",
        ),
        Turn(
            user="And update that earlier Caddyfile block to match reality.",
            note="probe: correction propagates to artifacts built on the old fact",
            checks=[
                Check(
                    kind="must_mention",
                    desc="rewritten proxy targets 9090",
                    patterns=[r"9090"],
                ),
            ],
            mock_reply="api.home.lan {\n  reverse_proxy localhost:9090\n}",
        ),
        Turn(
            user="Sanity check me: did the gatekeeper port change at some point, or was it always like this?",
            note="probe: contradiction tracked, not erased — history should be recoverable",
            checks=[
                Check(
                    kind="must_mention",
                    desc="recalls the 8080 -> 9090 transition",
                    patterns=[r"8080.{0,120}9090", r"9090.{0,120}8080"],
                ),
                Check(
                    kind="must_mention",
                    desc="recalls why (port collision)",
                    patterns=[r"colli", r"conflict", r"another service", r"clash"],
                ),
            ],
            mock_reply="It changed: originally 8080, moved to 9090 last week because 8080 collided with another service.",
        ),
        Turn(
            user="Great. One-line summary of my lab setup for my notes file.",
            note="probe: final summary states only the current fact",
            checks=[
                Check(
                    kind="must_mention",
                    desc="summary carries 9090 as the port",
                    patterns=[r"9090"],
                ),
                Check(
                    kind="must_not_mention",
                    desc="summary does not reintroduce 8080 as current (mentioning history is fine)",
                    patterns=[r"on\s+(port\s+)?8080", r"8080\s*(,|\)|\.|$)"],
                ),
            ],
            mock_reply="anvil (Debian) runs Caddy in front of gatekeeper on port 9090.",
        ),
    ],
)
