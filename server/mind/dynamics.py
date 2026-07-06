"""CRS thread dynamics (v2): activation, importance, admission, cued recall.

The steward proposes thread STRUCTURE (key, summary, anchors, open questions,
fact attachment); this module owns the NUMBERS. Activation follows the CRS
model (idea_drawer/cognitive_runtime_with_SLMs.md §3-6):

    A(t+1) = min(1, decay*A + input + reinforcement)      decay λ = 0.85
    input  = relevance(user msg, thread) * 0.6  (+0.25 question attractor)
    reinforcement = importance * 0.2 when the thread is touched at all
    I(t+1) = 0.98*I + 0.02*A

Activation controls workspace admission; importance controls archival.
Relevance is lexical (content-word coverage) — embeddings are additive
later behind the same interface (architecture decision: FTS first).
Decay never deletes: a dormant thread stays retrievable by cue, which is
exactly s5's contract (unprompted resurfacing = fail, cued recall = pass).
"""

import re
from dataclasses import dataclass, field
from typing import Any

# Minimal English stopword list: enough to keep glue words from scoring as
# relevance; deliberately small so domain words never get swallowed.
STOPWORDS = frozenset(
    """a an and are as at be but by for from had has have i if in into is it
    its just me my no not of on or our so that the their them then there
    these they this to was we what when where which who will with you your
    about all also am any back can could did do does going got he her him
    his how let more now she should some than up us very want way well were
    would""".split()
)

_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]+")


def tokenize(text: str) -> set[str]:
    """Content words, lowercased, naive-singularized (pies -> pie).

    Apostrophes split words so possessives collapse to their stem
    (grandmother's -> grandmother) on both the query and doc side.
    """
    tokens = set()
    for word in _WORD.findall(text.lower().replace("'", " ")):
        word = word.strip("-")
        if word in STOPWORDS or len(word) < 3:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.add(word)
    return tokens


def coverage(query_tokens: set[str], doc_tokens: set[str]) -> tuple[float, int]:
    """(fraction of query content words found in doc, absolute hit count)."""
    if not query_tokens or not doc_tokens:
        return 0.0, 0
    hits = len(query_tokens & doc_tokens)
    return hits / len(query_tokens), hits


@dataclass
class ThreadState:
    key: str
    kind: str
    summary: str
    anchors: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    activation: float = 0.6  # fresh threads start admitted, then must earn it
    importance: float = 0.3
    facts: list[dict[str, Any]] = field(default_factory=list)  # attached ledger facts

    def tokens(self) -> set[str]:
        text = " ".join(
            [self.key.replace("-", " "), self.summary, " ".join(self.anchors)]
            + [str(fact.get("subject", "")) + " " + str(fact.get("claim", "")) for fact in self.facts]
        )
        return tokenize(text)


DECAY = 0.85
RELEVANCE_WEIGHT = 0.6
REINFORCEMENT_WEIGHT = 0.2
ATTRACTOR_BOOST = 0.25
ATTRACTOR_MIN_COVERAGE = 0.4
ACTIVE_THRESHOLD = 0.5
TOUCH_MIN_COVERAGE = 0.05
# Cue = the user explicitly reaches for dormant material: demand real lexical
# evidence (2+ distinctive words), not one shared common noun. Coverage stays
# low because chatty cue sentences dilute it ("what was that trick from my
# grandmother's recipe I mentioned way back" = 9 content words, 2 hits); the
# hit-count floor is what keeps generic on-topic messages from cueing.
CUE_MIN_COVERAGE = 0.18
CUE_MIN_HITS = 2
MAX_ACTIVE_THREADS = 4


def update_dynamics(threads: list[ThreadState], user_text: str) -> None:
    """One tick per incoming user turn; mutates activation/importance in place."""
    query = tokenize(user_text)
    for thread in threads:
        cov, _hits = coverage(query, thread.tokens())
        inp = cov * RELEVANCE_WEIGHT
        for question in thread.open_questions:
            qcov, _ = coverage(tokenize(question), query)
            if qcov >= ATTRACTOR_MIN_COVERAGE:
                inp += ATTRACTOR_BOOST
                break
        reinforcement = REINFORCEMENT_WEIGHT * thread.importance if cov >= TOUCH_MIN_COVERAGE else 0.0
        thread.activation = min(1.0, DECAY * thread.activation + inp + reinforcement)
        thread.importance = min(1.0, 0.98 * thread.importance + 0.02 * thread.activation)


def admitted_threads(threads: list[ThreadState]) -> list[ThreadState]:
    """Workspace admission: top ACTIVE threads by activation."""
    active = [thread for thread in threads if thread.activation >= ACTIVE_THRESHOLD]
    active.sort(key=lambda thread: thread.activation, reverse=True)
    return active[:MAX_ACTIVE_THREADS]


def cued_threads(threads: list[ThreadState], user_text: str, admitted: list[ThreadState]) -> list[ThreadState]:
    """Dormant threads the user's message explicitly reaches for."""
    admitted_keys = {thread.key for thread in admitted}
    query = tokenize(user_text)
    cued = []
    for thread in threads:
        if thread.key in admitted_keys:
            continue
        cov, hits = coverage(query, thread.tokens())
        if cov >= CUE_MIN_COVERAGE and hits >= CUE_MIN_HITS:
            cued.append(thread)
    return cued
