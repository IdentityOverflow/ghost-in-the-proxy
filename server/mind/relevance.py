"""Relevance of derived memory (records, episodes) to a cue (memory v6).

One scorer for both consumers: the steward slice (cue = the fold span, long)
and per-turn rendering (cue = the latest user message, short). Lexical
overlap is the floor that always works; the Mem backend's semantic
similarity is additive when it has one (embedding), exactly the v5 split.
"""

import math
from dataclasses import dataclass

from .dynamics import tokenize
from .mem import MemBackend


@dataclass
class Scored:
    index: int
    score: float
    hits: int
    sim: float

    @property
    def matched(self) -> bool:
        """Real evidence the cue reaches for this item — not mere topicality."""
        return self.hits >= 2 or self.sim > 0.0


async def score_texts(
    cue: str,
    texts: list[str],
    mem: MemBackend | None = None,
    min_sim: float = 0.45,
) -> list[Scored]:
    """Score every text against the cue; same order as `texts`."""
    cue_tokens = tokenize(cue)
    sims: list[float] | None = None
    if mem is not None and texts and cue.strip():
        sims = await mem.text_sims(cue, texts)
    scored = []
    for index, text in enumerate(texts):
        tokens = tokenize(text)
        hits = len(cue_tokens & tokens)
        # Set-cosine: robust to a long cue (fold span) and a short one alike.
        lexical = hits / math.sqrt(len(cue_tokens) * len(tokens)) if hits else 0.0
        sim = sims[index] if sims is not None and sims[index] >= min_sim else 0.0
        # A tiny record ("dog: Biscuit") can only ever share one word.
        if hits == 1 and len(tokens) <= 3:
            hits = 2
        scored.append(Scored(index, lexical + sim, hits, sim))
    return scored
