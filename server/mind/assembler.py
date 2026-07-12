"""Naive v0 workspace assembler: system core + running summary + recent texture.

Budget model (docs/architecture.md): reserve is a fraction of the window,
the summary section is capped, and texture (recent verbatim messages) fills
the remainder newest-first without splitting tool blocks. Older live events
must be covered by the running summary before they can be evicted from
texture — the summarizer guarantees that ordering.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .config import MindConfig
from .dynamics import ThreadState
from .store import Event


@dataclass
class ThreadsView:
    """Attention state the runtime computed for this request (v2 CRS).

    Facts attached to a thread render only while that thread is admitted or
    cued — salience decides, the assembler only enforces. Facts whose thread
    key is unknown (or absent) always render: a steward that proposes no
    threads degrades exactly to v1 behavior, never to silence.
    """

    admitted: list[ThreadState] = field(default_factory=list)
    cued: list[ThreadState] = field(default_factory=list)
    all_keys: set[str] = field(default_factory=set)

MIND_HEADER = (
    "## Conversation memory\n"
    "You have a persistent memory of this conversation. Earlier turns are "
    "condensed below; recent turns follow verbatim. Treat these records as "
    "true history you remember, and obey their status labels:\n"
    "- When asked what is open, outstanding, or left to do, the 'Open "
    "commitments' list IS the answer — lead with those items. If you add "
    "anything beyond them, you MUST label it as a new suggestion, never as "
    "something already agreed or discussed.\n"
    "- A decision marked LEANING is NOT decided. Never say 'we decided' "
    "about it; say it is still open.\n"
    "- When asked for a status, recap, or summary, name the concrete items "
    "from these records with their true status (decided vs still open) — "
    "not vague phrases like 'finalizing the architecture'.\n"
    "- Never invent decisions, agreements, or tracked items that are not in "
    "these records or the recent turns."
)


def format_clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%A %Y-%m-%d %H:%M")


def format_gap(seconds: float) -> str:
    if seconds < 3600:
        return f"{max(1, round(seconds / 60))} minutes"
    if seconds < 48 * 3600:
        hours = seconds / 3600
        return f"{hours:.1f}".rstrip("0").rstrip(".") + " hours"
    return f"{seconds / 86400:.1f}".rstrip("0").rstrip(".") + " days"


def _parse_due(value: Any) -> float | None:
    """Steward-proposed due datetimes are ISO strings; garbage parses to None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).strip()).timestamp()
    except ValueError:
        return None


DIGEST_NOTICE = (
    "NOTE: some earlier tool outputs below are shown as truncated digests "
    "marked 'folded away'. If your answer depends on the exact content of a "
    "digested output, you MUST call recall(...) to read it in full BEFORE "
    "answering. Answering from a digest guess is an error."
)


def render_memory(
    summary_text: str,
    records: dict[str, list[dict[str, Any]]],
    episodes: list[tuple[int, int, str]],
    threads: ThreadsView | None = None,
    now: float | None = None,
) -> str:
    """Render the structured ledger + episodes (+ prose fallback) as the
    memory section of the system prompt. Empty sections are omitted. With a
    clock but no memory yet (fresh session), only a bare time line renders —
    the memory framing would be a lie on turn one, and noise primes behavior
    (observed: an s2 model exploring the environment instead of acting)."""
    sections: list[str] = []
    if threads and threads.admitted:
        lines = []
        for thread in threads.admitted:
            lines.append(f"- {thread.key}: {thread.summary}")
            for question in thread.open_questions:
                lines.append(f"  - open question: {question}")
        sections.append("### Active threads (what is currently in play)\n" + "\n".join(lines))
    decisions = records.get("decision", [])
    if decisions:
        lines = []
        for item in decisions:
            status = str(item.get("status", "open")).upper()
            if status == "LEANING":
                status = "LEANING (not yet decided)"
            reason = f" (reason: {item['reason']})" if item.get("reason") else ""
            lines.append(f"- {item.get('topic')}: {status} — {item.get('choice', '')}{reason}")
        sections.append("### Decisions\n" + "\n".join(lines))
    commitments = records.get("commitment", [])
    open_commitments = [c for c in commitments if c.get("status", "open") == "open"]
    if open_commitments:
        lines = []
        for item in open_commitments:
            line = f"- ({item.get('actor', 'user')}) {item.get('statement')}"
            if item.get("trigger"):
                line += f" — trigger: {item['trigger']}"
            due = _parse_due(item.get("due")) if now is not None else None
            if due is not None:
                if now >= due:
                    line += (
                        f" — was due {format_clock(due)}, OVERDUE by "
                        f"{format_gap(now - due)}: raise this NOW"
                    )
                else:
                    line += f" — due {format_clock(due)} (in {format_gap(due - now)})"
            lines.append(line)
        sections.append("### Open commitments (complete list of tracked items)\n" + "\n".join(lines))
    facts = records.get("fact", [])
    if threads is not None:
        visible_keys = {thread.key for thread in threads.admitted}
        facts = [
            item
            for item in facts
            if item.get("thread") not in threads.all_keys or item.get("thread") in visible_keys
        ]
    if facts:
        lines = [f"- {item.get('subject')}: {item.get('claim')}" for item in facts]
        sections.append("### Facts\n" + "\n".join(lines))
    if threads and threads.cued:
        lines = []
        for thread in threads.cued:
            lines.append(f"- {thread.key}: {thread.summary}")
            for fact in thread.facts:
                lines.append(f"  - {fact.get('subject')}: {fact.get('claim')}")
        sections.append(
            "### Recalled (dormant memory cued by the latest message)\n" + "\n".join(lines)
        )
    if episodes:
        lines = [f"- {summary}" for _, _, summary in episodes]
        sections.append("### Earlier events\n" + "\n".join(lines))
    if summary_text:
        sections.append("### Earlier conversation (condensed)\n" + summary_text)
    if not sections:
        return f"Current time: {format_clock(now)}." if now is not None else ""
    if now is not None:
        sections.insert(
            0,
            "### Now\n"
            f"Current time: {format_clock(now)} — the time AS OF the user's "
            "latest message (any elapsed-time markers in the conversation are "
            "already counted into it; never add them on top). Use it for any "
            "question about time, duration, or how long the user was away, "
            "and check open commitments' due times against it.",
        )
    return MIND_HEADER + "\n\n" + "\n\n".join(sections)


def estimate_tokens(payload: Any) -> int:
    if isinstance(payload, str):
        return max(1, len(payload) // 4)
    return max(1, len(json.dumps(payload, ensure_ascii=False)) // 4)


@dataclass
class Workspace:
    messages: list[dict[str, Any]]
    texture_from_seq: int  # first live seq actually included verbatim
    desired_from_seq: int  # first seq the BUDGET wanted — eviction pressure
    estimated_tokens: int


def _texture_blocks(events: list[Event]) -> list[list[Event]]:
    """Group events so tool results never separate from their tool_calls.

    A block is one message, except tool messages attach to the preceding
    block (which ends with the assistant tool_calls they answer). Slicing
    at block boundaries can therefore never orphan a tool exchange.
    """
    blocks: list[list[Event]] = []
    for event in events:
        if event.role == "tool" and blocks:
            blocks[-1].append(event)
        else:
            blocks.append([event])
    return blocks


def assemble(
    config: MindConfig,
    client_system: str | None,
    events: list[Event],
    summary: tuple[int, str] | None,
    records: dict[str, list[dict[str, Any]]] | None = None,
    episodes: list[tuple[int, int, str]] | None = None,
    threads: ThreadsView | None = None,
    now: float | None = None,
) -> Workspace:
    reserve = max(int(config.window * config.reserve_fraction), 1024)
    # chars/4 underestimates real tokenizers by ~20% (measured against
    # gemma-3 via LM Studio usage); the safety factor absorbs that.
    workspace_budget = int((config.window - reserve) * 0.75)


    system_parts = []
    if client_system:
        system_parts.append(client_system)
    summary_upto, summary_text = (summary or (0, ""))
    memory = render_memory(summary_text, records or {}, episodes or [], threads, now=now)
    if memory:
        system_parts.append(memory)
    system_content = "\n\n".join(system_parts)
    system_cost = estimate_tokens(system_content) if system_content else 0

    texture_budget = workspace_budget - system_cost

    # Containment (v3): under budget pressure, stale tool payloads render as
    # digests — the full text stays in the event store, reachable via recall.
    # Verbatim comes FIRST: digestion is a compression stage before eviction,
    # never a default (digesting a payload the budget could have carried cost
    # s2-t5 at 4k while the baseline still had the evidence in view).
    render: dict[int, Any] = {event.seq: event.message for event in events}
    candidate_cost = sum(
        estimate_tokens(event.message) for event in events if event.seq > summary_upto
    )
    if candidate_cost > texture_budget:
        render = _digest_stale_tool_events(config, events)
    digested = any(render[event.seq] is not event.message for event in events)
    if digested and system_content:
        system_content += "\n\n" + DIGEST_NOTICE
        system_cost = estimate_tokens(system_content)
        texture_budget = workspace_budget - system_cost

    # Chronos (v4): real elapsed time between turns renders as an inline
    # marker on the later user message — no role changes, so chat-template
    # alternation is untouched.
    if now is not None:
        render = _mark_gaps(config, events, render)

    blocks = _texture_blocks(events)

    # Newest blocks first until the budget is spent; never evict events the
    # summary doesn't cover yet (the summarizer runs before assembly to make
    # that impossible in steady state), and always include the newest block.
    chosen: list[list[Event]] = []
    spent = 0
    for block in reversed(blocks):
        cost = sum(estimate_tokens(render[event.seq]) for event in block)
        if chosen and spent + cost > texture_budget:
            break
        chosen.append(block)
        spent += cost
    chosen.reverse()

    # Chat templates (gemma et al.) demand user/assistant alternation, so
    # texture must open on a user block: extend backward (over budget) until
    # it does. The eviction boundary then always ends on an assistant turn.
    start_index = len(blocks) - len(chosen)
    while start_index > 0 and chosen and chosen[0][0].role != "user":
        start_index -= 1
        chosen.insert(0, blocks[start_index])

    texture_events = [event for block in chosen for event in block]
    # desired_from_seq is what the budget selected — the runtime uses it to
    # measure eviction pressure and trigger summarization. Until coverage
    # exists, texture extends backward (over budget) rather than lose truth;
    # measuring pressure from the EXTENDED texture would never trigger the
    # summarizer (the v0 first-run bug), so the two seqs are kept separate.
    desired_from_seq = texture_events[0].seq if texture_events else 0
    if texture_events:
        uncovered = [
            event for event in events if summary_upto < event.seq < desired_from_seq
        ]
        if uncovered:
            texture_events = [event for event in events if event.seq > summary_upto]
    # Final alternation guard: whatever path built the texture, it must open
    # on a user turn (duplicating an already-summarized event is safe; a
    # template rejection is not).
    while texture_events and texture_events[0].role != "user":
        earlier = [event for event in events if event.seq < texture_events[0].seq]
        if not earlier:
            break
        texture_events.insert(0, earlier[-1])

    messages: list[dict[str, Any]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.extend(render[event.seq] for event in texture_events)

    return Workspace(
        messages=messages,
        texture_from_seq=texture_events[0].seq if texture_events else 0,
        desired_from_seq=desired_from_seq,
        estimated_tokens=system_cost
        + sum(estimate_tokens(render[event.seq]) for event in texture_events),
    )


def _mark_gaps(
    config: MindConfig, events: list[Event], render: dict[int, Any]
) -> dict[int, Any]:
    """Prefix user messages with a '[N hours pass]' marker when real wall-clock
    time elapsed since the previous event. Events without a timestamp (legacy
    rows) neither get marked nor anchor a gap."""
    threshold = config.gap_mark_minutes * 60
    if threshold <= 0:
        return render
    previous_ts: float | None = None
    for event in events:
        if not event.ts:
            continue
        if (
            previous_ts is not None
            and event.role == "user"
            and event.ts - previous_ts >= threshold
        ):
            message = render[event.seq]
            content = message.get("content")
            if isinstance(content, str):
                # The marker pins the absolute arrival time, so the model
                # never has to add the gap to anything itself (gemma-12B
                # added it to the header's current time when it had to).
                marker = (
                    f"[{format_gap(event.ts - previous_ts)} pass — "
                    f"it is now {format_clock(event.ts)}]"
                )
                render[event.seq] = {**message, "content": f"{marker}\n\n{content}"}
        previous_ts = event.ts
    return render


def _digest_stale_tool_events(config: MindConfig, events: list[Event]) -> dict[int, Any]:
    """Per-seq render map: stale bulky tool payloads become head digests.

    'Stale' = before the latest user turn; the current turn's in-flight tool
    exchange stays verbatim (the model needs it to answer NOW). The event
    store is untouched — reconciliation still sees full payloads, and recall
    retrieves them verbatim.
    """
    render: dict[int, Any] = {event.seq: event.message for event in events}
    cap = config.tool_digest_chars
    if cap <= 0:
        return render
    last_user_seq = max((event.seq for event in events if event.role == "user"), default=0)
    for event in events:
        content = event.message.get("content")
        if (
            event.role == "tool"
            and event.seq < last_user_seq
            and isinstance(content, str)
            and len(content) > cap + 200  # only digest when it actually saves
        ):
            digest = (
                content[:cap]
                + f"\n…[{len(content) - cap} chars of tool output folded away — "
                'call recall("<distinctive words>") for the full text]'
            )
            render[event.seq] = {**event.message, "content": digest}
    return render


