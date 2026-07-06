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

from .dynamics import coverage, tokenize
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


def resolve_recall(events: list[Event], arguments_json: str) -> str:
    """Lexical search over live events; verbatim spans, best first."""
    try:
        query = str(json.loads(arguments_json or "{}").get("query") or "")
    except json.JSONDecodeError:
        query = arguments_json or ""
    query_tokens = tokenize(query)
    if not query_tokens:
        return "recall error: empty query"

    scored: list[tuple[float, Event, str]] = []
    for event in events:
        text = _event_text(event)
        if not text:
            continue
        cov, hits = coverage(query_tokens, tokenize(text))
        if hits == 0:
            continue
        # Coverage ranks; an exact-phrase hit dominates (verbatim requests
        # usually carry a distinctive fragment of the sought material).
        score = cov + (1.0 if query.strip() and query.strip().lower() in text.lower() else 0.0)
        scored.append((score, event, text))
    if not scored:
        return f"recall: nothing found for {query!r}"

    scored.sort(key=lambda item: item[0], reverse=True)
    lines = []
    for score, event, text in scored[:MAX_RESULTS]:
        snippet = text if len(text) <= SPAN_CHAR_CAP else text[:SPAN_CHAR_CAP] + " …[truncated]"
        lines.append(f"[seq {event.seq}, {event.role}, verbatim]\n{snippet}")
    return "\n\n".join(lines)


def _event_text(event: Event) -> str:
    content = event.message.get("content")
    return content if isinstance(content, str) else ""
