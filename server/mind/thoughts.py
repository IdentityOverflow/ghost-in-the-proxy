"""Quick thoughts (roadmap C, experimental): what pops into mind before a reply.

The owner's thesis: people do not run chain-of-thought mid-conversation. A
word or two surfaces — "she's venting", "keep it short", "you said that
already" — and the rest is handled below the surface. For a language model
the only lever on "below the surface" is what sits in the context: a handle
does not carry information to reason over, it moves the prior. The aim here
is not a smarter model but a better CONVERSATIONALIST: less repetitive, less
predictable, less synthetic.

What we know so far (docs/memory-v6.md §6): a standing instruction in the
system header did nothing on a 12B model; a specific, current, machine-made
observation WITH an action, at the end of the context, was obeyed every time
it fired. So every design here produces lines for the per-turn notes block
on the latest user message (cache-friendly: nothing before it changes).

Designs, switchable and combinable via MIND_THOUGHTS (comma list):

  rhythm   the proven baseline: question-endings + worn phrases (style_note)
  observe  D5 — deterministic reads of the conversation's surface: length
           mismatch, uniform reply shape, validation openers, echoing, chat
           markdown. Zero model calls.
  typed    D3 — one System-1 question ("what does this message call for?")
           answered by the conversation model in ONE token; token logprobs are
           the confidence; only a confident read becomes a line.
  sketch   D2 — one short generation: three different moves, then a pick; the
           picked path is the line ("pick a path, expand live").
  sketchlite  the same with two moves of <= 6 words (about a third of the cost)
  sheet    D1 — a static "how to talk like a person" sheet in the system
           message (expected to do little; cheap to test).
"""

import math
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from .memory_view import style_note

MAX_LINES = 3

SHEET = (
    "### How to talk (this is a chat between friends, not a help desk)\n"
    "- Match their length: a one-liner gets a line or two back.\n"
    "- React first, like a person would; do not open by validating or summarising what they said.\n"
    "- Most replies should NOT end with a question. Stop when you have said it.\n"
    "- Venting wants to be heard, not fixed. A joke wants play, not analysis. A question wants the answer first.\n"
    "- Have opinions and say them. Tease a little. Disagree when you disagree.\n"
    "- Plain sentences: no bold, no bullet lists, no headers, no quoting their words back at them.\n"
    "- Vary your shape from turn to turn: short, long, a single word, a story — not the same three-part reply every time."
)

VALIDATION_OPENERS = re.compile(
    r"^\W*(that sounds|that's (a|an|so|such|really|great|awesome|wonderful|fantastic|amazing|totally|completely)|"
    r"that is (a|an|so|such|really|great)|it sounds like|sounds like|oh,? that|i love (that|how)|what a|"
    r"i('m| am) so (glad|sorry|happy)|i hear you|i (totally|completely) (get|understand)|"
    r"congratulations|congrats|ah,? the)\b",
    flags=re.IGNORECASE,
)
# What a blind judge kept citing against the baseline: "forced praise",
# "inflates", "canned reassurance".
HYPE = re.compile(
    r"\b(huge|massive|amazing|incredible|fantastic|wonderful|awesome|absolutely|victory|milestone|"
    r"so proud|you've earned|you deserve|well[- ]deserved|a (big|major|real) win)\b",
    flags=re.IGNORECASE,
)
MARKDOWN = re.compile(r"\*\*[^*]+\*\*|^\s*([-*•]|\d+\.)\s+\S|^#{1,4}\s", flags=re.MULTILINE)
QUOTED = re.compile(r"[\"“']([^\"”']{6,60})[\"”']")
# Scare quotes: a short phrase in double quotes mid-sentence ('that "pre-work"
# exhaustion'). The most persistent tell in a live 40-turn chat (15+ turns) —
# and mostly NOT a literal echo of the user, so the echo check never saw it.
SCARE_QUOTES = re.compile(r"(?<![\w])[\"“]([^\"”\n]{2,40})[\"”](?![\w])")
SIGN_OFFS = re.compile(
    r"(you('ve| have) got this|go (crush|get) (it|'em|some (sleep|rest))|you('ve| have) earned (it|this)|"
    r"you should be proud|i('m| am) (here|around) (whenever|if)|i'll be here|good ?night,? \w+|"
    r"you deserve (it|this))[.!]*\s*$",
    flags=re.IGNORECASE,
)


def _words(text: str) -> int:
    return len(text.split())


def observe(user_turns: list[str], replies: list[str]) -> list[str]:
    """Deterministic surface reads -> observation + action. Most pressing first."""
    lines: list[str] = []
    if not user_turns or len(replies) < 2:
        return lines
    latest = user_turns[-1]
    recent = replies[-5:]
    mean_len = statistics.mean(_words(reply) for reply in recent)

    # Length mirroring: people answer a one-liner with a line, not an essay.
    if _words(latest) <= 12 and mean_len >= 60:
        lines.append(
            f"They wrote one short line ({_words(latest)} words); your recent replies run about "
            f"{mean_len:.0f} words. Answer in one or two sentences."
        )
    elif len(recent) >= 4:
        lengths = [_words(reply) for reply in recent]
        spread = statistics.pstdev(lengths) / max(1.0, statistics.mean(lengths))
        if spread < 0.18 and mean_len >= 50:
            lines.append(
                f"Your last {len(recent)} replies were all about {mean_len:.0f} words — the same shape "
                "every time. Make this one clearly shorter."
            )

    window = replies[-4:]
    if sum(bool(VALIDATION_OPENERS.match(reply.strip())) for reply in window) >= 2:
        lines.append(
            "You keep opening by validating or summarising what they said. Start with your own "
            "reaction, an opinion, or the answer."
        )
    if sum(len(HYPE.findall(reply)) >= 2 for reply in window) >= 2:
        lines.append(
            "You have been cheerleading (\"huge\", \"massive win\", \"you've earned it\"). React at the "
            "size the thing actually is; skip the praise."
        )
    if sum(bool(MARKDOWN.search(reply)) for reply in window) >= 2 and not MARKDOWN.search(latest):
        lines.append("You have been using bold text or lists. This is a chat — plain sentences only.")
    # These notes work like a thermostat: obeyed every time they fire (10 of 10
    # on a replay), and the habit is back as soon as they stop. For a habit we
    # never want, the note therefore turns STICKY once the habit is established
    # — and it names no examples: quoted samples in the note primed more quoting.
    quoted_recently = sum(bool(SCARE_QUOTES.search(reply)) for reply in replies[-12:])
    if quoted_recently >= 3 or sum(bool(SCARE_QUOTES.search(reply)) for reply in window) >= 2:
        lines.append(
            "You have a habit of putting phrases in quotation marks. Write this reply with no "
            "quotation marks at all — say things in your own words."
        )
    if sum(bool(SIGN_OFFS.search(reply.strip())) for reply in window) >= 2:
        lines.append(
            "You keep closing on a pep-talk line (\"you've got this\", \"you earned it\", \"I'll be here\"). "
            "End on something about THEM or the thing itself, or just stop."
        )
    return lines


# -- typed System-1 read (D3) ----------------------------------------------------

REGISTER_QUESTION = (
    "[Private check — not part of the conversation. Reply with exactly ONE capital letter.]\n"
    "What does the user's last message mostly call for?\n"
    "A) to be heard — venting or sharing a feeling\n"
    "B) a concrete answer or practical help\n"
    "C) play — they are joking or bantering\n"
    "D) a reaction — they are sharing news or an update\n"
    "E) thinking a choice through together\n"
    "F) checking what you remember"
)
REGISTER_LINES = {
    "A": "They want to be heard. React to the feeling; no advice, no fixes, no silver linings.",
    "B": "They want the answer. Lead with it; skip the preamble and the pep talk.",
    "C": "They are playing. Be quick and playful back; do not analyse the joke or turn it practical.",
    "D": "They are sharing news. React the way a friend would — one genuine reaction — before anything else.",
    "E": "They are weighing a choice. Give YOUR opinion and why, not a balanced list of pros and cons.",
    "F": "They are checking your memory. Say plainly what you remember; if unsure, say that.",
}
TYPED_MIN_CONFIDENCE = 0.6


@dataclass
class ThoughtTrace:
    """Telemetry for one turn's thoughts."""

    modes: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    calls: int = 0
    seconds: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)


def _question_messages(messages: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    """The assembled conversation with the private question appended to the
    LAST user message — everything before it is the prefix the backend has
    already cached for the real reply."""
    out = list(messages)
    for index in range(len(out) - 1, -1, -1):
        if out[index].get("role") == "user":
            content = out[index].get("content")
            if isinstance(content, str):
                out[index] = {**out[index], "content": f"{content}\n\n{question}"}
            elif isinstance(content, list):
                out[index] = {**out[index], "content": content + [{"type": "text", "text": question}]}
            return out[: index + 1]
    return out + [{"role": "user", "content": question}]


async def typed_read(provider: Any, model: str, messages: list[dict[str, Any]], trace: ThoughtTrace) -> str | None:
    payload = {
        "model": model,
        "messages": _question_messages(messages, REGISTER_QUESTION),
        "temperature": 0,
        # Not 1: gemma under LM Studio spends its first token on a stripped
        # channel marker and returns nothing at all (live: 0 reads in 26 turns).
        "max_tokens": 6,
        "logprobs": True,
        "top_logprobs": 10,
        "stream": False,
    }
    response = await provider.chat_completions(payload)
    trace.calls += 1
    choice = response["choices"][0]
    probabilities: dict[str, float] = {}
    tokens = (choice.get("logprobs") or {}).get("content") or []
    # The answer is the first emitted token that IS one of the letters.
    answer = next((t for t in tokens if str(t.get("token", "")).strip().upper()[:1] in REGISTER_LINES), None)
    for token in [answer] if answer else []:
        for candidate in token.get("top_logprobs") or []:
            letter = str(candidate.get("token", "")).strip().upper()[:1]
            if letter in REGISTER_LINES:
                probabilities[letter] = probabilities.get(letter, 0.0) + math.exp(candidate["logprob"])
    if not probabilities:
        # Backend without logprobs: take the letter, trust it less.
        letter = str(choice["message"].get("content") or "").strip().upper()[:1]
        if letter in REGISTER_LINES:
            probabilities[letter] = TYPED_MIN_CONFIDENCE
    if not probabilities:
        return None
    best = max(probabilities, key=probabilities.get)
    total = sum(probabilities.values()) or 1.0
    confidence = probabilities[best] / total
    trace.detail["typed"] = {"read": best, "confidence": round(confidence, 3)}
    return REGISTER_LINES[best] if confidence >= TYPED_MIN_CONFIDENCE else None


# -- path sketch (D2) -------------------------------------------------------------

SKETCH_QUESTION = (
    "[Private check — not part of the conversation.]\n"
    "Before replying, list three genuinely DIFFERENT ways you could respond to the user's last "
    "message, at most 8 words each (for example: just react; tease; give my opinion; answer "
    "straight; ask one thing; say less). Then choose the one a good friend would pick.\n"
    "Format exactly:\n1. ...\n2. ...\n3. ...\nPICK: <number>"
)


SKETCH_LITE_QUESTION = (
    "[Private check — not part of the conversation.]\n"
    "Two DIFFERENT ways a good friend might respond to that, at most 6 words each, then pick.\n"
    "Format exactly:\n1. ...\n2. ...\nPICK: <number>"
)


async def sketch_lite(provider: Any, model: str, messages: list[dict[str, Any]], trace: ThoughtTrace) -> str | None:
    return await sketch_path(provider, model, messages, trace, question=SKETCH_LITE_QUESTION, max_tokens=34)


async def sketch_path(
    provider: Any, model: str, messages: list[dict[str, Any]], trace: ThoughtTrace,
    question: str = SKETCH_QUESTION, max_tokens: int = 70,
) -> str | None:
    payload = {
        "model": model,
        "messages": _question_messages(messages, question),
        "temperature": 0.8,
        "max_tokens": max_tokens,
        "stream": False,
    }
    response = await provider.chat_completions(payload)
    trace.calls += 1
    text = re.sub(r"<think>.*?</think>", "", response["choices"][0]["message"].get("content") or "", flags=re.DOTALL)
    options = dict(re.findall(r"^\s*([123])[.)]\s*(.+?)\s*$", text, flags=re.MULTILINE))
    picked = re.search(r"PICK:\s*([123])", text, flags=re.IGNORECASE)
    trace.detail["sketch"] = {"options": options, "pick": picked.group(1) if picked else None}
    if not picked or picked.group(1) not in options:
        return None
    path = options[picked.group(1)].strip().rstrip(".")
    return f"Your move this turn: {path[:90]}. Do that — and only that."


async def think(
    modes: list[str],
    provider: Any,
    model: str,
    messages: list[dict[str, Any]],
    user_turns: list[str],
    replies: list[str],
) -> ThoughtTrace:
    """Run the enabled designs; at most MAX_LINES lines come back. Model-backed
    designs fail open: a thought that errors is simply not had."""
    trace = ThoughtTrace(modes=list(modes))
    started = time.monotonic()
    lines: list[str] = []
    for mode, call in (("typed", typed_read), ("sketch", sketch_path), ("sketchlite", sketch_lite)):
        if mode in modes and user_turns and provider is not None:
            try:
                line = await call(provider, model, messages, trace)
                if line:
                    lines.append(line)
            except Exception as error:
                trace.detail[f"{mode}_error"] = repr(error)[:200]
    if "observe" in modes:
        lines += observe(user_turns, replies)
    if "rhythm" in modes or "observe" in modes:
        note = style_note(replies)
        if note:
            lines.append(note)
    trace.lines = lines[:MAX_LINES]
    trace.seconds = round(time.monotonic() - started, 2)
    return trace
