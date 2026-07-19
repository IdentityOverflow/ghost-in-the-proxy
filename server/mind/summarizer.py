"""Running-summary maintenance (v0 steward-lite).

Runs before assembly when enough live content would otherwise fall out of
texture uncoverd. The LLM proposes prose; the runtime decides when and what
it covers. The prompt front-loads exactly the record kinds the eval probes:
commitments, decisions with status, corrections, and distinctive asides.
"""

from typing import Any

from .assembler import estimate_tokens
from .config import MindConfig
from .store import Event, MindStore, content_text

SUMMARY_SYSTEM = (
    "You maintain the running memory of a conversation. Merge the existing "
    "memory with the new turns into one updated memory, under {budget} words. "
    "You MUST preserve, with highest priority:\n"
    "1. Commitments and promises (who promised what, and what should trigger it)\n"
    "2. Decisions and their status — clearly distinguish 'decided' from "
    "'leaning/considering', and keep the reason for each decision\n"
    "3. Corrections: when a fact was corrected, keep ONLY the new value as "
    "current, but note the old value as superseded\n"
    "4. Numbers, names, file paths, and distinctive one-off details (even "
    "seemingly irrelevant personal asides — note them briefly)\n"
    "5. Open questions and unfinished work\n"
    "Drop pleasantries and phrasing. Never invent content. Write compact "
    "prose or bullets, third person."
)


async def update_summary(
    config: MindConfig,
    store: MindStore,
    session_id: str,
    events: list[Event],
    provider: Any,
    model: str,
    upto_seq: int,
) -> None:
    """Fold live events with seq <= upto_seq into a new running summary."""
    previous = store.latest_summary(session_id)
    previous_upto, previous_text = previous or (0, "")
    fold = [event for event in events if previous_upto < event.seq <= upto_seq]
    if not fold:
        return

    transcript = "\n".join(
        f"[{event.role}] {_flatten(event.message)}" for event in fold
    )
    word_budget = max(120, int(config.summary_budget_tokens * 0.75))
    payload = {
        "model": config.extraction_model or model,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM.format(budget=word_budget)},
            {
                "role": "user",
                "content": (
                    f"Existing memory:\n{previous_text or '(none yet)'}\n\n"
                    f"New turns to fold in:\n{transcript}\n\n"
                    "Updated memory:"
                ),
            },
        ],
        "temperature": 0.1,
        "stream": False,
        "max_tokens": config.extraction_max_tokens,
    }
    response = await provider.chat_completions(payload)
    content = (response["choices"][0]["message"].get("content") or "").strip()
    if content:
        store.append_summary(session_id, upto_seq, content)


def _flatten(message: dict[str, Any]) -> str:
    parts = []
    content = content_text(message)
    if content:
        parts.append(content)
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        parts.append(f"(called tool {function.get('name')} with {function.get('arguments')})")
    text = " ".join(parts)
    # Tool payloads can be huge; the summary only needs their gist scale.
    return text if estimate_tokens(text) < 1200 else text[: 1200 * 4] + " …[truncated]"
