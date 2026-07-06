"""Steward (v1): structured extraction into the semantic ledger + episodes.

Runs at eviction time (same trigger as the v0 summarizer). One LLM call
proposes the COMPLETE updated ledger — facts, decisions with status and
reason, commitments with status — plus one episode line for the span being
folded. The runtime applies it deterministically via generation versioning.

The LLM proposes; the runtime disposes (CRS §19). If the proposal doesn't
parse, the caller falls back to the v0 prose summarizer — a degraded mind
must degrade to the previous working mind, never to nothing.
"""

import json
import re
from typing import Any

from .assembler import estimate_tokens
from .config import MindConfig
from .store import Event, MindStore

STEWARD_SYSTEM = """You maintain the structured memory of a conversation. Given the current \
memory and new conversation turns, output the COMPLETE UPDATED memory as JSON, nothing else:

{
  "threads": [
    {"key": "<stable-kebab-slug>", "kind": "topic" | "aside" | "inquiry", "summary": "<1-2 sentences: what this line of conversation is about and where it stands>", "anchors": ["<distinctive concrete words: names, tools, dishes, places>"], "open_questions": ["<questions raised in this thread and not yet answered>"]}
  ],
  "facts": [
    {"subject": "<short key>", "thread": "<key of the thread this belongs to>", "claim": "<current truth; if corrected, only the new value, noting the old as superseded>"}
  ],
  "decisions": [
    {"topic": "<short key>", "status": "decided" | "leaning" | "open", "choice": "<what>", "reason": "<why, if given>"}
  ],
  "commitments": [
    {"actor": "user" | "assistant", "statement": "<what was promised>", "trigger": "<when it should fire>", "status": "open" | "done" | "dropped"}
  ],
  "episode": "<2-3 sentence narrative of what happened in the new turns, including distinctive one-off details and asides>"
}

Rules:
- Carry forward every still-relevant entry from the current memory; drop nothing that is still open or true.
- Decisions belong to the USER. Record what the user settled or is leaning toward — NEVER the assistant's recommendations or comparisons. If the user says they are still thinking, the status is "leaning" or "open" no matter what the assistant argued for.
- "decided" only when explicitly settled. NEVER upgrade a leaning to decided on your own.
- Commitments are promises to act LATER (with a trigger or deadline), or standing requests to track something. A request the assistant fulfilled immediately in the same turn is NOT a commitment — do not record it.
- When the user says "remind me to X", "don't let me forget X", or "before we ship, X" — that is an OPEN commitment; record it verbatim with its trigger.
- Corrections replace the fact's claim; note the superseded value inside the claim (e.g. "9090 (was 8080, port collision)").
- Record distinctive one-off details and personal asides as facts, even if they seem irrelevant.
- Keep numbers, names, and file paths exact. Never invent entries.
- Threads partition the conversation: the main line of work is ONE "topic" thread; a personal aside or one-off tangent gets its own small "aside" thread. Reuse existing thread keys — never rename them. Every fact's "thread" must be one of the listed thread keys."""


class StewardParseError(Exception):
    pass


async def run_steward(
    config: MindConfig,
    store: MindStore,
    session_id: str,
    events: list[Event],
    provider: Any,
    model: str,
    upto_seq: int,
) -> None:
    """Fold events up to upto_seq into the ledger and an episode line."""
    episodes = store.live_episodes(session_id)
    folded_upto = max((span_to for _, span_to, _ in episodes), default=0)
    fold = [event for event in events if folded_upto < event.seq <= upto_seq]
    if not fold:
        return

    ledger = store.live_records(session_id)
    current_memory = json.dumps(
        {
            "threads": store.live_threads(session_id),
            "facts": ledger.get("fact", []),
            "decisions": ledger.get("decision", []),
            "commitments": ledger.get("commitment", []),
        },
        ensure_ascii=False,
        indent=1,
    )
    transcript = "\n".join(f"[{event.role}] {_flatten(event.message)}" for event in fold)

    payload = {
        "model": config.extraction_model or model,
        "messages": [
            {"role": "system", "content": STEWARD_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Current memory:\n{current_memory}\n\n"
                    f"New turns:\n{transcript}\n\n"
                    "Complete updated memory (JSON only):"
                ),
            },
        ],
        "temperature": 0.1,
        "stream": False,
    }
    response = await provider.chat_completions(payload)
    content = response["choices"][0]["message"].get("content") or ""
    proposal = _parse_proposal(content)

    store.replace_records(
        session_id,
        {
            "fact": proposal.get("facts", []),
            "decision": proposal.get("decisions", []),
            "commitment": proposal.get("commitments", []),
        },
        provenance_seq=upto_seq,
    )
    threads = [item for item in proposal.get("threads", []) if isinstance(item, dict)]
    if threads:
        store.replace_threads(session_id, threads, provenance_seq=upto_seq)
    episode = str(proposal.get("episode") or "").strip()
    if episode:
        store.append_episode(session_id, fold[0].seq, upto_seq, episode)
    # Keep the summaries watermark in sync so the assembler's coverage logic
    # (which reads the summary upto_seq) treats the folded span as covered.
    store.append_summary(session_id, upto_seq, "")


def _parse_proposal(content: str) -> dict[str, Any]:
    # Reasoning extraction models wrap output in think blocks whose braces
    # would garbage the greedy JSON match; deliberation is not the proposal.
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        raise StewardParseError(f"no JSON object in steward output: {content[:150]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise StewardParseError(f"steward JSON invalid: {error}") from error
    if not isinstance(data, dict):
        raise StewardParseError("steward output is not an object")
    for key in ("facts", "decisions", "commitments", "threads"):
        if not isinstance(data.get(key, []), list):
            raise StewardParseError(f"steward field {key} is not a list")
    return data


def _flatten(message: dict[str, Any]) -> str:
    parts = []
    content = message.get("content")
    if isinstance(content, str) and content:
        parts.append(content)
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        parts.append(f"(called tool {function.get('name')} with {function.get('arguments')})")
    text = " ".join(parts)
    return text if estimate_tokens(text) < 1200 else text[: 1200 * 4] + " …[truncated]"
