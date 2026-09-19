"""Bounded memory rendering (memory v6 §3): working memory is small on purpose.

The ledger may grow for as long as the conversation lives; what the model
SEES of it each turn may not. This module selects, under a hard token
budget, what enters the memory section of the system prompt:

  commitments -> profile (core facts) -> decisions -> active threads ->
  cue-recalled records -> episodes (eras + recent + cue-matched leaves)

Each tier has a share cap so none can crowd the rest out; unused share
flows down in a second pass. Whatever is not selected is not lost: it comes
back by cue on a later turn, through the recall tool, or the auto-cue
channel. A partial list always says it is partial — the v1 header called
the commitments list "complete", which is a lie the moment anything is cut.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .config import MindConfig
from .dynamics import ThreadState
from .ledger import LedgerState, Record
from .mem import MemBackend
from .relevance import Scored, score_texts

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
    "these records or the recent turns. This memory shows what is relevant "
    "now, not everything: if something seems missing, say you would need to "
    "look it up rather than guessing."
)

RECALLED_SPAN_CHAR_CAP = 400
RECENT_LEAF_EPISODES = 3
CUED_LEAF_EPISODES = 2
MAX_RECALLED_RECORDS = 6
MAX_FACTS_PER_THREAD = 8
DUE_SOON_S = 24 * 3600

HEADINGS_RESERVE = 110  # tokens: six tier headings incl. the long "partial list" one

# Share of the tier budget each tier may claim in the first pass.
TIER_SHARES = {
    "commitments": 0.28,
    "profile": 0.18,
    "decisions": 0.14,
    "threads": 0.22,
    "recalled": 0.12,
    "episodes": 0.22,
}


@dataclass
class ThreadsView:
    """Attention state the runtime computed for this request (v2 CRS)."""

    admitted: list[ThreadState]
    cued: list[ThreadState]
    all_keys: set[str]


@dataclass
class _Item:
    text: str
    cost: int
    record_id: str | None = None


def estimate_tokens(text: str) -> int:
    dense = sum(1 for char in text if ord(char) > 0x2FF)
    return max(1, (len(text) - dense) // 4 + int(dense * 0.8))


def format_clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%A %Y-%m-%d %H:%M")


def format_gap(seconds: float) -> str:
    if seconds < 3600:
        return f"{max(1, round(seconds / 60))} minutes"
    if seconds < 48 * 3600:
        hours = seconds / 3600
        return f"{hours:.1f}".rstrip("0").rstrip(".") + " hours"
    return f"{seconds / 86400:.1f}".rstrip("0").rstrip(".") + " days"


def parse_due(value: Any) -> float | None:
    """Steward-proposed due datetimes are ISO strings; garbage parses to None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).strip()).timestamp()
    except ValueError:
        return None


def memory_budget(config: MindConfig, workspace_budget: int) -> int:
    wanted = int(workspace_budget * config.memory_fraction)
    return max(config.memory_min_tokens, min(wanted, config.memory_max_tokens))


def _commitment_line(record: Record, now: float | None) -> str:
    data = record.data
    line = f"- ({data.get('actor', 'assistant')}) {data.get('statement')}"
    if data.get("trigger"):
        line += f" — trigger: {data['trigger']}"
    due = parse_due(data.get("due")) if now is not None else None
    if due is not None:
        if now >= due:
            line += f" — was due {format_clock(due)}, OVERDUE by {format_gap(now - due)}: raise this NOW"
        else:
            line += f" — due {format_clock(due)} (in {format_gap(due - now)})"
    return line


def _decision_line(record: Record) -> str:
    data = record.data
    status = str(data.get("status", "open")).upper()
    if status == "LEANING":
        status = "LEANING (not yet decided)"
    reason = f" (reason: {data['reason']})" if data.get("reason") else ""
    return f"- {data.get('topic')}: {status} — {data.get('choice', '')}{reason}"


def _fact_line(record: Record, indent: str = "") -> str:
    return f"{indent}- {record.data.get('subject')}: {record.data.get('claim')}"


def _item(text: str, record: Record | None = None) -> _Item:
    return _Item(text, estimate_tokens(text) + 1, record.id if record else None)


async def render_memory(
    config: MindConfig,
    state: LedgerState,
    consolidations: list[dict[str, Any]],
    threads: ThreadsView | None,
    cue: str,
    mem: MemBackend | None,
    budget_tokens: int,
    now: float | None = None,
    recalled_spans: list | None = None,
    seq_ts: dict[int, float] | None = None,
) -> str:
    """The memory section of the system prompt, never above budget_tokens.

    With a clock but no memory yet (fresh session) only a bare time line
    renders — the memory framing would be a lie on turn one, and noise
    primes behavior (observed: an s2 model exploring instead of acting).
    """
    has_memory = bool(state.records or state.episodes or recalled_spans)
    if not has_memory:
        return f"Current time: {format_clock(now)}." if now is not None else ""

    records = list(state.records.values())
    scores: dict[str, Scored] = {}
    if cue.strip() and records:
        scored = await score_texts(cue, [record.text() for record in records], mem, config.embed_min_sim)
        scores = {record.id: item for record, item in zip(records, scored)}

    def matched(record: Record) -> bool:
        return record.id in scores and scores[record.id].matched

    def relevance(record: Record) -> float:
        return scores[record.id].score if record.id in scores else 0.0

    # -- fixed parts ----------------------------------------------------------
    now_section = ""
    if now is not None:
        now_section = (
            "### Now\n"
            f"Current time: {format_clock(now)} — the time AS OF the user's "
            "latest message (any elapsed-time markers in the conversation are "
            "already counted into it; never add them on top). Use it for any "
            "question about time, duration, or how long the user was away, "
            "and check open commitments' due times against it."
        )
    verbatim_section = ""
    if recalled_spans:
        # Raw-memory rescue (s13). seq provenance rendered on purpose: order
        # questions are only answerable if the model can SEE the numbers.
        lines = []
        for span in sorted(recalled_spans, key=lambda s: s.seq):
            text = span.text
            if len(text) > RECALLED_SPAN_CHAR_CAP:
                text = text[:RECALLED_SPAN_CHAR_CAP] + " …[truncated]"
            lines.append(f"- [seq {span.seq}, verbatim] {span.role}: {text}")
        verbatim_section = (
            "### Recalled verbatim (raw memory matched to the latest message)\n" + "\n".join(lines)
        )
    fixed_cost = sum(
        estimate_tokens(part) + 2 for part in (MIND_HEADER, now_section, verbatim_section) if part
    )
    # Section headings are text too: reserve them up front (the longest
    # variant of each), or a full render lands just over budget.
    tier_budget = max(0, budget_tokens - fixed_cost - HEADINGS_RESERVE)

    # -- tier candidates, best first -------------------------------------------
    open_commitments = [r for r in state.by_kind("commitment") if r.is_open]

    def commitment_rank(record: Record) -> tuple:
        due = parse_due(record.data.get("due")) if now is not None else None
        pressing = due is not None and due - now <= DUE_SOON_S
        return (0 if pressing else 1, 0 if matched(record) else 1, -record.updated_seq)

    open_commitments.sort(key=commitment_rank)

    core = [r for r in state.by_kind("fact") if r.data.get("core")]
    core.sort(key=lambda r: (0 if matched(r) else 1, -relevance(r), -r.updated_seq))

    admitted_ids = {thread.key for thread in threads.admitted} if threads else set()
    decisions = state.by_kind("decision")

    def decision_rank(record: Record) -> tuple:
        undecided = record.data.get("status") != "decided"
        near = matched(record) or record.data.get("thread") in admitted_ids
        return (0 if undecided else 1, 0 if near else 1, -record.updated_seq)

    decisions.sort(key=decision_rank)

    plain_facts = [r for r in state.by_kind("fact") if not r.data.get("core")]
    thread_items: list[_Item] = []
    known_threads = threads.all_keys if threads else set()
    if threads:
        for thread in threads.admitted:
            thread_items.append(_item(f"- {thread.name}: {thread.summary}"))
            for question in thread.open_questions:
                thread_items.append(_item(f"  - open question: {question}"))
            facts = [r for r in plain_facts if r.data.get("thread") == thread.key]
            facts.sort(key=lambda r: (0 if matched(r) else 1, -r.updated_seq))
            for record in facts[:MAX_FACTS_PER_THREAD]:
                thread_items.append(_item(_fact_line(record, "  "), record))
    # Facts the steward attached to no (known) thread have no salience gate:
    # they ride along by recency, exactly the v1 behavior, but under budget.
    loose = [r for r in plain_facts if r.data.get("thread") not in known_threads]
    loose.sort(key=lambda r: (0 if matched(r) else 1, -r.updated_seq))
    loose_items = [_item(_fact_line(record), record) for record in loose]

    # -- fill: first pass under share caps, second pass spends the remainder --
    tiers: dict[str, list[_Item]] = {
        "commitments": [_item(_commitment_line(r, now), r) for r in open_commitments],
        "profile": [_item(_fact_line(r), r) for r in core],
        "decisions": [_item(_decision_line(r), r) for r in decisions],
        "threads": thread_items + loose_items,
    }
    chosen: dict[str, list[_Item]] = {name: [] for name in TIER_SHARES}
    remaining = tier_budget

    def take(name: str, items: list[_Item], cap: int) -> list[_Item]:
        nonlocal remaining
        left: list[_Item] = []
        spent = 0
        for item in items:
            if spent + item.cost > cap or item.cost > remaining:
                # Skip, don't stop: one oversized entry must not block every
                # shorter one queued behind it.
                left.append(item)
                continue
            chosen[name].append(item)
            spent += item.cost
            remaining -= item.cost
        return left

    overflow: dict[str, list[_Item]] = {}
    for name in ("commitments", "profile", "decisions", "threads"):
        overflow[name] = take(name, tiers[name], int(tier_budget * TIER_SHARES[name]))

    shown = {item.record_id for items in chosen.values() for item in items if item.record_id}

    # Recalled: anything the cue genuinely reaches that is not already in view
    # — dormant-thread facts, old decisions, closed commitments.
    recalled_pool = [r for r in records if r.id not in shown and matched(r)]
    recalled_pool.sort(key=relevance, reverse=True)
    recalled_items: list[_Item] = []
    if threads:
        for thread in threads.cued:
            recalled_items.append(_item(f"- (thread) {thread.name}: {thread.summary}"))
    for record in recalled_pool[:MAX_RECALLED_RECORDS]:
        if record.kind == "fact":
            recalled_items.append(_item(_fact_line(record), record))
        elif record.kind == "decision":
            recalled_items.append(_item(_decision_line(record), record))
        else:
            status = record.data.get("status", "open")
            recalled_items.append(_item(f"{_commitment_line(record, now)} [{status}]", record))
    overflow["recalled"] = take("recalled", recalled_items, int(tier_budget * TIER_SHARES["recalled"]))

    episode_items = await _episode_items(config, state, consolidations, cue, mem, seq_ts)
    overflow["episodes"] = take("episodes", episode_items, int(tier_budget * TIER_SHARES["episodes"]))

    for name in ("commitments", "profile", "recalled", "threads", "decisions", "episodes"):
        if remaining <= 0:
            break
        overflow[name] = take(name, overflow[name], remaining)

    def compose() -> str:
        return MIND_HEADER + "\n\n" + "\n\n".join(_sections())

    def _sections() -> list[str]:
        return _render_sections(
            chosen, overflow, len(open_commitments), now_section, verbatim_section, episode_items
        )

    # The estimate above is per-line; the bound is on the composed text. Trim
    # from the least important tier until the promise holds exactly.
    text = compose()
    for name in ("episodes", "recalled", "threads", "decisions", "profile", "commitments"):
        while chosen[name] and estimate_tokens(text) > budget_tokens:
            overflow[name].insert(0, chosen[name].pop())
            text = compose()
    return text


def _render_sections(
    chosen: dict[str, list[_Item]],
    overflow: dict[str, list[_Item]],
    total_open_commitments: int,
    now_section: str,
    verbatim_section: str,
    episode_items: list[_Item],
) -> list[str]:
    sections: list[str] = []
    if now_section:
        sections.append(now_section)
    if not chosen["commitments"] and total_open_commitments:
        sections.append(
            f"### Open commitments\n- ({total_open_commitments} tracked items exist but are not "
            "shown here — if asked what is outstanding, say you need to look them up)"
        )
    if chosen["commitments"]:
        hidden = len(overflow["commitments"])
        title = (
            "### Open commitments (complete list of tracked items)"
            if not hidden
            else f"### Open commitments (the {len(chosen['commitments'])} most pressing of "
            f"{total_open_commitments} tracked — if asked for everything, say more exist)"
        )
        sections.append(title + "\n" + "\n".join(i.text for i in chosen["commitments"]))
    if chosen["profile"]:
        sections.append("### Always true (who and what matters)\n" + "\n".join(i.text for i in chosen["profile"]))
    if chosen["decisions"]:
        sections.append("### Decisions\n" + "\n".join(i.text for i in chosen["decisions"]))
    if chosen["threads"]:
        sections.append(
            "### Active threads (what is currently in play)\n"
            + "\n".join(i.text for i in chosen["threads"])
        )
    if chosen["recalled"]:
        sections.append(
            "### Recalled (older memory cued by the latest message)\n"
            + "\n".join(i.text for i in chosen["recalled"])
        )
    if verbatim_section:
        sections.append(verbatim_section)
    if chosen["episodes"]:
        # Selected newest-first (budget eats the oldest); read oldest-first.
        position = {id(item): index for index, item in enumerate(episode_items)}
        ordered = sorted(chosen["episodes"], key=lambda item: position[id(item)], reverse=True)
        sections.append("### Earlier events (oldest first)\n" + "\n".join(i.text for i in ordered))
    return sections


async def _episode_items(
    config: MindConfig,
    state: LedgerState,
    consolidations: list[dict[str, Any]],
    cue: str,
    mem: MemBackend | None,
    seq_ts: dict[int, float] | None,
) -> list[_Item]:
    """Episode lines in CHRONOLOGICAL order; selection priority is encoded by
    which lines exist at all (eras stand in for the leaves they cover, except
    leaves the cue reaches for)."""

    def dated(span_to: int, text: str, tag: str = "") -> str:
        stamp = ""
        if seq_ts and seq_ts.get(span_to):
            stamp = datetime.fromtimestamp(seq_ts[span_to]).strftime("%b %d") + ": "
        return f"- {tag}{stamp}{text}"

    # The highest consolidation level wins: an epoch hides its eras, an era
    # hides its leaves.
    level2 = [c for c in consolidations if c["level"] == 2]
    hidden_eras = {
        seq for c in level2 for seq in range(c["child_from"], c["child_to"] + 1)
    }
    eras = [c for c in consolidations if c["level"] == 1 and c["seq"] not in hidden_eras]
    covered_folds = {
        fold
        for c in consolidations
        if c["level"] == 1
        for fold in range(c["child_from"], c["child_to"] + 1)
    }
    leaves = state.episodes
    recent = {episode.fold for episode in leaves[-RECENT_LEAF_EPISODES:]}
    cued: set[int] = set()
    older = [episode for episode in leaves if episode.fold in covered_folds]
    if cue.strip() and older:
        scored = await score_texts(cue, [episode.text for episode in older], mem, config.embed_min_sim)
        best = sorted((s for s in scored if s.matched), key=lambda s: s.score, reverse=True)
        cued = {older[s.index].fold for s in best[:CUED_LEAF_EPISODES]}

    timeline: list[tuple[int, str]] = []
    for c in level2:
        timeline.append((c["span_to"], dated(c["span_to"], c["content"], "(long ago, condensed) ")))
    for c in eras:
        timeline.append((c["span_to"], dated(c["span_to"], c["content"], "(condensed) ")))
    for episode in leaves:
        if episode.fold in covered_folds and episode.fold not in cued and episode.fold not in recent:
            continue
        tag = "(recalled detail) " if episode.fold in cued and episode.fold in covered_folds else ""
        timeline.append((episode.span_to, dated(episode.span_to, episode.text, tag)))
    timeline.sort(key=lambda pair: pair[0])
    items = [_item(text) for _, text in timeline]
    # Budget pressure should eat the OLDEST lines first; take() walks the list
    # in order, so hand it newest-first and let the renderer re-sort.
    items.reverse()
    return items
