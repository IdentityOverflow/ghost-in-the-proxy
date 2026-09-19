"""Workspace assembler: system core + bounded memory + recent texture.

Budget model (docs/architecture.md, docs/memory-v6.md §3): the reply reserve
is a fraction of the window; the workspace budget is what remains, capped so
a huge window cannot reintroduce transcript stuffing; the memory section is
rendered elsewhere under its own budget (memory_view.py) and handed in as
text; texture (recent verbatim messages) fills the remainder newest-first
without splitting tool blocks.

Older live events should be covered by a fold before they leave texture, so
texture extends backward over budget while coverage catches up — but only
up to the HARD limit. Past it, truth-in-view loses to not-dying: uncovered
events are evicted anyway (they stay in the raw store, reachable by recall
and auto-cue, and the pending fold will cover them).
"""

import json
from dataclasses import dataclass
from typing import Any

from .config import MindConfig
from .memory_view import ThreadsView, format_clock, format_gap  # noqa: F401 (re-exported)
from .store import Event, content_text

# chars/4 undercounts real tokenizers (~20-25% measured on gemma via LM
# Studio usage). Until a session has calibrated itself from the backend's
# reported usage, assume this many real tokens per estimated token.
DEFAULT_TOKEN_SCALE = 1.33

DIGEST_NOTICE = (
    "NOTE: some earlier tool outputs below are shown as truncated digests "
    "marked 'folded away'. If your answer depends on the exact content of a "
    "digested output, you MUST call recall(...) to read it in full BEFORE "
    "answering. Answering from a digest guess is an error."
)


class WorkspaceOverflow(Exception):
    """The request cannot be made to fit the window by any safe eviction or
    truncation (a huge client system prompt, giant tool-call arguments).
    Surfaced to the client as a context-length error: silently sending it
    would either fail at the backend or be truncated there — the
    confabulation source the mind exists to remove."""


def estimate_tokens(payload: Any) -> int:
    """Cheap token estimate. ~4 chars/token holds for English prose; text
    outside ASCII (CJK, Cyrillic, emoji) runs far denser, up to a token per
    character, so it is counted separately — a first request in Chinese must
    not look four times smaller than it is. Calibration refines the rest."""
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    dense = sum(1 for char in text if ord(char) > 0x2FF)
    return max(1, (len(text) - dense) // 4 + int(dense * 0.8))


@dataclass
class Workspace:
    messages: list[dict[str, Any]]
    texture_from_seq: int  # first live seq actually included verbatim
    desired_from_seq: int  # first seq the BUDGET wanted — eviction pressure
    estimated_tokens: int
    # Telemetry split: the mind's own memory section vs the whole system
    # message (client prompt + memory) vs the workspace budget it lives in.
    memory_tokens: int = 0
    system_tokens: int = 0
    budget_tokens: int = 0
    hard_limit_tokens: int = 0
    # Events evicted by the hard guard before any fold covered them — should
    # be zero in steady state; nonzero means folding fell behind.
    evicted_uncovered: int = 0


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


def token_budgets(config: MindConfig, tools_tokens: int = 0, scale: float = DEFAULT_TOKEN_SCALE) -> tuple[int, int]:
    """(workspace budget, hard limit), both in ESTIMATED tokens.

    `scale` converts estimates to real tokens (self-calibrated per session
    from usage.prompt_tokens); tool schemas ride in the same request, so
    their cost comes off the top. The workspace cap keeps a 128k window from
    turning back into a transcript dump — what legitimately grows with the
    window is bounded too, just higher.
    """
    reserve = max(int(config.window * config.reserve_fraction), 1024)
    budget = int((config.window - reserve) / scale) - tools_tokens
    budget = min(budget, config.workspace_cap_tokens)
    # Over-budget backward extension may eat half the reply reserve, no more.
    hard = int((config.window - reserve // 2) / scale) - tools_tokens
    hard = min(hard, int(config.workspace_cap_tokens * 1.25))
    return max(budget, 256), max(hard, 512)


def assemble(
    config: MindConfig,
    client_system: str | None,
    events: list[Event],
    covered_upto: int = 0,
    memory_text: str = "",
    now: float | None = None,
    tools_tokens: int = 0,
    scale: float = DEFAULT_TOKEN_SCALE,
) -> Workspace:
    workspace_budget, hard_limit = token_budgets(config, tools_tokens, scale)

    system_parts = []
    if client_system:
        system_parts.append(client_system)
    if memory_text:
        system_parts.append(memory_text)
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
        estimate_tokens(event.message) for event in events if event.seq > covered_upto
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

    def block_cost(block: list[Event]) -> int:
        return sum(estimate_tokens(render[event.seq]) for event in block)

    # Newest blocks first until the budget is spent; always the newest block.
    chosen: list[list[Event]] = []
    spent = 0
    for block in reversed(blocks):
        cost = block_cost(block)
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

    # desired_from_seq is what the budget selected — the runtime measures
    # eviction pressure from it. Until a fold covers them, older events stay
    # in view over budget rather than lose truth (measuring pressure from the
    # EXTENDED texture would never trigger the fold: the v0 first-run bug).
    desired_from_seq = chosen[0][0].seq if chosen else 0
    evicted_uncovered = 0
    while start_index > 0 and blocks[start_index - 1][-1].seq > covered_upto:
        start_index -= 1
        chosen.insert(0, blocks[start_index])

    # HARD GUARD: the request must fit the window, whatever it costs. Evict
    # from the old end (keeping the opening on a user block), never the
    # newest block.
    def total() -> int:
        return system_cost + sum(block_cost(block) for block in chosen)

    if total() > hard_limit:
        # Smallest eviction that fits AND still opens on a user block; if no
        # user-opening suffix fits, keep the last one and shrink below.
        openers = [index for index, block in enumerate(chosen) if block[0].role == "user"]
        keep_from = next(
            (
                index
                for index in openers
                if system_cost + sum(block_cost(block) for block in chosen[index:]) <= hard_limit
            ),
            openers[-1] if openers else 0,
        )
        for block in chosen[:keep_from]:
            evicted_uncovered += sum(1 for event in block if event.seq > covered_upto)
        chosen = chosen[keep_from:]

    texture_events = [event for block in chosen for event in block]
    # Final alternation guard: whatever path built the texture, it must open
    # on a user turn (duplicating an already-covered event is safe; a
    # template rejection is not).
    while texture_events and texture_events[0].role != "user":
        earlier = [event for event in events if event.seq < texture_events[0].seq]
        if not earlier:
            break
        texture_events.insert(0, earlier[-1])

    # Still over: a single exchange alone overflows (a giant paste or tool
    # dump in the current turn). Middle-truncate the largest messages in
    # place until it fits; the store keeps the full text.
    for _ in range(4):
        excess = system_cost + sum(estimate_tokens(render[e.seq]) for e in texture_events) - hard_limit
        if excess <= 0 or not texture_events:
            break
        _shrink_block(texture_events, render, excess)
    final = system_cost + sum(estimate_tokens(render[e.seq]) for e in texture_events)
    # Small slack: the guard works in estimates, and a few tokens over the
    # (already conservative) hard limit is not worth failing a request for.
    if final > hard_limit * 1.03:
        raise WorkspaceOverflow(
            f"request needs ~{final} tokens after eviction and truncation; "
            f"the limit for a {config.window}-token window is ~{hard_limit} "
            f"(system prompt {system_cost}, tools {tools_tokens})"
        )

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
        memory_tokens=estimate_tokens(memory_text) if memory_text else 0,
        system_tokens=system_cost,
        budget_tokens=workspace_budget,
        hard_limit_tokens=hard_limit,
        evicted_uncovered=evicted_uncovered,
    )


def _shrink_block(block: list[Event], render: dict[int, Any], excess_tokens: int) -> None:
    """Middle-truncate the largest text message of a block by ~excess tokens.
    The store keeps the full text; the marker tells the model how to reach it."""
    largest = max(block, key=lambda event: len(content_text(render[event.seq]) or ""))
    message = render[largest.seq]
    text = content_text(message) or ""
    keep = max(800, len(text) - (excess_tokens + 60) * 4)
    if keep >= len(text):
        return
    head, tail = keep * 2 // 3, keep // 3
    cut = (
        f"{text[:head]}\n…[{len(text) - keep} characters omitted here to fit the context window — "
        f"ask the user to send it in parts, or call recall(...) for a specific passage]…\n{text[-tail:]}"
    )
    render[largest.seq] = {**message, "content": cut}


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
            # The marker pins the absolute arrival time, so the model never
            # has to add the gap to anything itself (gemma-12B added it to
            # the header's current time when it had to).
            marker = (
                f"[{format_gap(event.ts - previous_ts)} pass — "
                f"it is now {format_clock(event.ts)}]"
            )
            if isinstance(content, str):
                render[event.seq] = {**message, "content": f"{marker}\n\n{content}"}
            elif isinstance(content, list):
                render[event.seq] = {
                    **message,
                    "content": [{"type": "text", "text": marker}] + content,
                }
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
        content = content_text(event.message)
        if (
            event.role == "tool"
            and event.seq < last_user_seq
            and content is not None
            and len(content) > cap + 200  # only digest when it actually saves
        ):
            digest = (
                content[:cap]
                + f"\n…[{len(content) - cap} chars of tool output folded away — "
                'call recall("<distinctive words>") for the full text]'
            )
            render[event.seq] = {**event.message, "content": digest}
    return render
