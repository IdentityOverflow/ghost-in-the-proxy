"""The recall tool (v3): the provenance escape hatch.

Because the assembler REPLACES the transcript, a workspace miss must be
recoverable: the mind offers the model a `recall` tool that searches the
raw event store and returns matching spans VERBATIM with seq provenance
(docs/architecture.md). Distilled memory answers "what is true"; recall
answers "what exactly was said" — s9's contract.

The recall exchange happens proxy-side and never enters the event store:
the client's transcript will not contain it, so recording it would desync
reconciliation. It is deliberation, not conversation truth.
"""

import json
from typing import Any

from .mem import LexicalMem, MemBackend, MemQuery
from .store import Event

RECALL_TOOL = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": (
            "Search your own verbatim memory of THIS conversation for earlier "
            "material that is no longer in view: exact quotes, pasted logs or "
            "tracebacks, code, commands, numbers, names, one-off things the "
            "user mentioned in passing. Use it whenever the user asks for "
            "exact wording, or refers to something from earlier that you "
            "cannot see right now — try a few different plausible words if "
            "the first search finds nothing. Returns the matching earlier "
            "messages word for word."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "what to look for (distinctive words from the material)",
                }
            },
            "required": ["query"],
        },
    },
}

MAX_RESULTS = 3
SPAN_CHAR_CAP = 4000  # ~1000 tokens per returned span


def is_recall_call(call: dict[str, Any]) -> bool:
    return call.get("function", {}).get("name") == "recall"


async def resolve_recall(
    events: list[Event],
    arguments_json: str,
    backend: MemBackend | None = None,
    session_id: str = "",
    trajectory: list[Event] | None = None,
    char_budget: int | None = None,
) -> str:
    """Search raw memory via the Mem backend; verbatim spans, best first.

    Default backend is lexical — v3 behavior unchanged. The runtime passes
    its configured backend plus the recent-events trajectory stub.
    """
    try:
        query = str(json.loads(arguments_json or "{}").get("query") or "")
    except json.JSONDecodeError:
        query = arguments_json or ""
    if not query.strip():
        return "recall error: empty query"

    mem = backend if backend is not None else LexicalMem()
    spans = await mem.query(
        session_id,
        MemQuery(text=query, trajectory=trajectory or [], k=MAX_RESULTS),
        events,
    )
    if not spans:
        return f"recall: nothing found for {query!r}"

    # The recall exchange rides on top of an already-assembled workspace:
    # unbounded, three 4000-char spans over three hops overflow an 8k window
    # on their own. The best span gets first call on the budget.
    remaining = char_budget if char_budget is not None else SPAN_CHAR_CAP * MAX_RESULTS
    marker = " …[truncated]"
    lines = []
    for span in spans:
        header = f"[seq {span.seq}, {span.role}, verbatim]\n"
        room = remaining - len(header) - len(marker) - (2 if lines else 0)
        if room < 120:
            break
        cap = min(SPAN_CHAR_CAP, room)
        text = span.text
        snippet = text if len(text) <= cap else text[:cap] + marker
        entry = header + snippet
        remaining -= len(entry) + (2 if lines else 0)
        lines.append(entry)
    if not lines:
        return f"recall: a match exists for {query!r} but no budget is left to show it"
    return "\n\n".join(lines)
