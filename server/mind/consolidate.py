"""Consolidation (memory v6 §4): leaf episodes -> era lines -> epoch lines.

One leaf episode per fold, forever, is a list that cannot stay in view. But
re-summarizing a running summary every fold is the telephone game again. So
consolidation is hierarchical and ONE-SHOT: once `era_size` leaf episodes
have aged past the recent tail they are condensed, together, exactly once,
into an era line; `era_size` eras condense into an epoch. Any piece of text
is rewritten O(log n) times over the life of the conversation.

Eras are a disposable INDEX, never truth: the leaves stay in the fold log
(cue-matched leaves still render beside their era), the raw events stay in
the event store, and a fork simply supersedes the consolidations whose span
it touches — the next idle pass rebuilds them.
"""

import re
from typing import Any

from .config import MindConfig
from .ledger import replay
from .store import MindStore

CONSOLIDATE_SYSTEM = (
    "You are given consecutive notes from one long ongoing conversation, oldest "
    "first. Condense them into ONE paragraph of 2-3 sentences: the gist of what "
    "happened over this stretch and where things stood at the end of it. Keep "
    "names, the FINAL value of any number or date that changed, and decisions "
    "(say whether each was final or only a leaning). Drop blow-by-blow detail. "
    "Third person, plain text, no preamble."
)


async def consolidate_once(
    config: MindConfig,
    store: MindStore,
    session_id: str,
    provider: Any,
    model: str,
) -> int | None:
    """Do at most one consolidation (bounded idle work). Returns its level."""
    size = max(2, config.era_size)
    state = replay(store.live_folds(session_id))
    consolidations = store.live_consolidations(session_id)

    eras = [c for c in consolidations if c["level"] == 1]
    covered = {fold for c in eras for fold in range(c["child_from"], c["child_to"] + 1)}
    aged = [episode for episode in state.episodes[:-size] if episode.fold not in covered]
    if len(aged) >= size:
        group = aged[:size]
        text = await _condense(config, provider, model, [episode.text for episode in group])
        if text:
            store.append_consolidation(
                session_id, 1, group[0].fold, group[-1].fold,
                group[0].span_from, group[-1].span_to, text,
            )
            return 1
        return None

    epochs = [c for c in consolidations if c["level"] == 2]
    hidden = {seq for c in epochs for seq in range(c["child_from"], c["child_to"] + 1)}
    aged_eras = [c for c in eras[:-size] if c["seq"] not in hidden]
    if len(aged_eras) >= size:
        group = aged_eras[:size]
        text = await _condense(config, provider, model, [c["content"] for c in group])
        if text:
            store.append_consolidation(
                session_id, 2, group[0]["seq"], group[-1]["seq"],
                group[0]["span_from"], group[-1]["span_to"], text,
            )
            return 2
    return None


async def _condense(config: MindConfig, provider: Any, model: str, notes: list[str]) -> str:
    payload = {
        "model": config.extraction_model or model,
        "messages": [
            {"role": "system", "content": CONSOLIDATE_SYSTEM},
            {"role": "user", "content": "\n".join(f"{i}. {note}" for i, note in enumerate(notes, 1))},
        ],
        "temperature": 0.1,
        "stream": False,
        "max_tokens": min(config.extraction_max_tokens, 1500),
    }
    response = await provider.chat_completions(payload)
    content = response["choices"][0]["message"].get("content") or ""
    # Same think-block hygiene as the steward.
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
