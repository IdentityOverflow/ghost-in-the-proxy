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
            "tracebacks, code, commands, numbers, names. Use it whenever the "
            "user asks for exact wording or a detail you cannot see verbatim "
            "right now. Returns the matching earlier messages word for word."
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

    lines = []
    for span in spans:
        text = span.text
        snippet = text if len(text) <= SPAN_CHAR_CAP else text[:SPAN_CHAR_CAP] + " …[truncated]"
        lines.append(f"[seq {span.seq}, {span.role}, verbatim]\n{snippet}")
    return "\n\n".join(lines)
