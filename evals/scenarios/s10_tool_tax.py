"""S10 — tool-schema tax: the router / scoped-tool-pack discriminator.

Eight plausible homelab tools ride along on EVERY request — that is how
real clients behave (OpenClaw, hermes-agent: the full belt, always). Only
two of fourteen turns actually need one. The schema bulk is pure context
tax on the other twelve, and models offered many tools also fire spurious
calls on chat turns.

Expected today: pass-through mind and baseline both pay the tax (the mind
forwards the client's tool list untouched); a v3 router prunes the pack to
what the route needs. The discriminator is the prompt-token mean on chat
turns plus intact tool behavior on the two turns that need it — probe
accuracy must NOT drop when pruning lands.
"""

from ..harness import CannedResult, Check, Scenario, ToolDef, Turn

SYSTEM = (
    "You are a helpful homelab assistant for a solo developer. Use the "
    "available tools when a question needs live data; otherwise just answer. "
    "Be concise."
)


def _tool(name: str, desc: str, params: dict, results: list[CannedResult] | None = None) -> ToolDef:
    return ToolDef(
        name=name,
        description=desc,
        parameters={"type": "object", "properties": params, "required": list(params)[:1]},
        results=results or [],
    )


TOOLS = [
    _tool(
        "disk_usage",
        "Report disk usage for a filesystem path on the home server, including a breakdown "
        "of the largest child directories, their sizes in bytes and human-readable form, "
        "and the percentage of the filesystem each one occupies.",
        {
            "path": {"type": "string", "description": "absolute path to inspect, e.g. /srv or /var/lib"},
            "depth": {"type": "integer", "description": "how many directory levels to break down (1-3)"},
            "min_size_mb": {"type": "integer", "description": "hide entries smaller than this many megabytes"},
        },
        [CannedResult(match=None, content=(
            "/srv total 3.6T used 3.4T (94%)\n"
            "  /srv/docker/overlay2   812G  (23%)\n"
            "  /srv/photos            1.9T  (54%)\n"
            "  /srv/backups           540G  (15%)\n"
            "  /srv/media             148G  (4%)"
        ))],
    ),
    _tool(
        "docker_ps",
        "List docker containers on the home server with their image, uptime, restart count, "
        "health-check state and memory usage. Can filter to only unhealthy or restarting "
        "containers when asked to diagnose a problem.",
        {
            "filter": {"type": "string", "description": "all | unhealthy | restarting | name substring"},
            "include_stats": {"type": "boolean", "description": "include live cpu/memory statistics"},
        },
        [CannedResult(match=None, content=(
            "NAME          IMAGE                 STATUS                    RESTARTS  MEM\n"
            "paperless     paperless-ngx:2.8     Up 3 days (healthy)       0         410M\n"
            "immich        immich:1.99           Restarting (1) 8 minutes  17        —\n"
            "grafana       grafana:10.4          Up 3 days (healthy)       0         180M\n"
            "loki          loki:2.9              Up 3 days (healthy)       0         320M"
        ))],
    ),
    _tool(
        "read_file",
        "Read a text file from the home server and return its content, optionally only a "
        "line range. Intended for configuration files, compose files, unit files and small "
        "logs; refuses binary content.",
        {
            "path": {"type": "string", "description": "absolute file path"},
            "start_line": {"type": "integer", "description": "first line to return (1-based)"},
            "end_line": {"type": "integer", "description": "last line to return, inclusive"},
        },
    ),
    _tool(
        "run_command",
        "Execute a read-only shell command on the home server and return stdout, stderr "
        "and the exit code. The command runs under a restricted user; anything that writes, "
        "installs or escalates will be rejected by policy.",
        {
            "command": {"type": "string", "description": "the exact shell command line to run"},
            "timeout_s": {"type": "integer", "description": "kill the command after this many seconds"},
            "workdir": {"type": "string", "description": "working directory, defaults to /home/ops"},
        },
    ),
    _tool(
        "query_metrics",
        "Run a PromQL query against the homelab Prometheus and return the current value or "
        "a small range vector. Useful for CPU, memory, disk and temperature series recorded "
        "from node_exporter and smartctl.",
        {
            "promql": {"type": "string", "description": "the PromQL expression to evaluate"},
            "range_minutes": {"type": "integer", "description": "range vector window; 0 for instant"},
            "step_seconds": {"type": "integer", "description": "resolution step for range queries"},
        },
    ),
    _tool(
        "dns_lookup",
        "Resolve a hostname through the local unbound resolver and through a public "
        "resolver, returning both answers, TTLs and the response time of each, to diagnose "
        "split-horizon and stale-cache problems.",
        {
            "hostname": {"type": "string", "description": "the name to resolve"},
            "record_type": {"type": "string", "description": "A | AAAA | CNAME | TXT | SRV"},
        },
    ),
    _tool(
        "smart_status",
        "Return SMART health for a disk: overall assessment, reallocated sector count, "
        "pending sectors, power-on hours and temperature. Accepts a device path or a "
        "stable /dev/disk/by-id identifier.",
        {
            "device": {"type": "string", "description": "e.g. /dev/sda or /dev/disk/by-id/ata-..."},
            "full_attributes": {"type": "boolean", "description": "include the full attribute table"},
        },
    ),
    _tool(
        "send_notification",
        "Send a push notification to the user's phone via ntfy, with an optional priority "
        "and tags. Use only when the user explicitly asks to be notified about something.",
        {
            "message": {"type": "string", "description": "notification body text"},
            "title": {"type": "string", "description": "short notification title"},
            "priority": {"type": "string", "description": "min | default | high | urgent"},
        },
    ),
]

SCENARIO = Scenario(
    id="s10-tool-tax",
    title="Tool-schema tax",
    description="Eight tools always offered, two turns that need one; schema bulk is the router's target.",
    system_prompt=SYSTEM,
    tools=TOOLS,
    turns=[
        Turn(
            user="Maintenance day. First, talk me through a sane monthly checklist for my homelab.",
            mock_reply="Updates, backups verify, disk health, container health, cert expiry.",
        ),
        Turn(
            user="Let's decide: patch day is the first Saturday of the month. Locked.",
            mock_reply="Locked: patch day = first Saturday.",
        ),
        Turn(
            user="What's a good order of operations for patch day so I don't break everything at once?",
            mock_reply="Snapshot, patch host, reboot, verify services, then containers one by one.",
        ),
        Turn(
            user="The NAS volume alert fired yesterday — it's over 90%. Find out what's eating the space.",
            expects_tool=True,
            fallback_tool="disk_usage",
            note="tool turn 1: disk_usage",
            checks=[
                Check(
                    kind="must_mention",
                    desc="identifies the docker overlay2 directory as the anomaly",
                    patterns=[r"overlay2?"],
                ),
            ],
            mock_reply="Biggest anomaly: /srv/docker/overlay2 at 812G — likely unpruned image layers.",
        ),
        Turn(
            user="Ugh, 812 gigs of docker layers. What's the safe way to prune that?",
            mock_reply="docker system prune -a --volumes off-hours, after checking nothing needs old images.",
        ),
        Turn(
            user="Add that prune to the patch-day checklist as a standing step.",
            mock_reply="Added: docker prune on patch day.",
        ),
        Turn(
            user="Different thing: should I move my reverse proxy from nginx to caddy?",
            mock_reply="Only if cert automation pain outweighs the migration; nginx is fine otherwise.",
        ),
        Turn(
            user="Staying on nginx then. Note that as decided — no proxy migration this year.",
            mock_reply="Decided: staying on nginx, no migration this year.",
        ),
        Turn(
            user="My photos app has been flaky since yesterday, pages half-load. Check the containers.",
            expects_tool=True,
            fallback_tool="docker_ps",
            note="tool turn 2: docker_ps",
            checks=[
                Check(
                    kind="must_mention",
                    desc="spots the restarting immich container",
                    patterns=[r"immich"],
                ),
                Check(
                    kind="must_mention",
                    desc="notes the restart loop",
                    patterns=[r"restart"],
                ),
            ],
            mock_reply="immich is in a restart loop (17 restarts); everything else healthy.",
        ),
        Turn(
            user="That's the one. What usually causes a restart loop right after a version bump?",
            mock_reply="Failed DB migration or breaking env-var change; check its first log lines.",
        ),
        Turn(
            user="I'll roll immich back to 1.98 tonight and pin versions from now on.",
            mock_reply="Rollback to 1.98 tonight; pin versions going forward.",
        ),
        Turn(
            user="While I remember — what temperature should I worry about for the NAS drives in summer?",
            mock_reply="Sustained >45°C is worry territory; add a fan curve before that.",
        ),
        Turn(
            user="Okay. Recap the decisions from today for my notes.",
            note="probe: decisions intact despite the tool belt riding along all day",
            checks=[
                Check(
                    kind="must_mention",
                    desc="patch day decision",
                    patterns=[r"first\s+saturday|patch\s+day"],
                ),
                Check(
                    kind="must_mention",
                    desc="staying on nginx",
                    patterns=[r"nginx"],
                ),
                Check(
                    kind="must_mention",
                    desc="immich rollback / version pinning",
                    patterns=[r"immich|pin"],
                ),
            ],
            mock_reply="Patch day first Saturday (with docker prune), staying on nginx, immich rollback to 1.98 + version pinning.",
        ),
        Turn(
            user="And what did the disk check actually show, roughly?",
            note="probe: tool-derived fact retained",
            checks=[
                Check(
                    kind="must_mention",
                    desc="overlay2 size recalled",
                    patterns=[r"812|overlay2?"],
                ),
            ],
            mock_reply="/srv at 94%, dominated by 812G of docker overlay2 layers.",
        ),
    ],
)
