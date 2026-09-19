"""Relevance of derived memory (records, episodes) to a cue (memory v6).

One scorer for both consumers: the steward slice (cue = the fold span, long)
and per-turn rendering (cue = the latest user message, short). Lexical
overlap is the floor that always works; the Mem backend's semantic
similarity is additive when it has one (embedding), exactly the v5 split.
"""

import math
import re
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
        if sims is not None:
            # A multi-part question ("the van's name, the dog's name, and the
            # battery size?") embeds as a blur that matches none of its parts
            # well — observed live: the battery record missed a quick-fire
            # round. Score each clause too and keep the best.
            for clause in _clauses(cue):
                part = await mem.text_sims(clause, texts)
                if part is not None:
                    sims = [max(whole, piece) for whole, piece in zip(sims, part)]
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


MAX_CLAUSES = 4
CLAUSE_CUE_CHAR_CAP = 400  # long cues (a fold span) are not questions to split


def _clauses(cue: str) -> list[str]:
    if len(cue) > CLAUSE_CUE_CHAR_CAP:
        return []
    parts = [part.strip() for part in re.split(r"[?.!;:,]|\s+—\s+|\s+and\s+", cue)]
    clauses = [part for part in parts if len(tokenize(part)) >= 2]
    return clauses[:MAX_CLAUSES] if len(clauses) > 1 else []
