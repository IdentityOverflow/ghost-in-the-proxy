"""Tool router (v3): scoped tool packs.

Real clients send the full tool belt on every request; schema bulk is pure
context tax on turns that need none (s10). Deterministic first, per the
architecture doc — an LLM oracle only if metrics later justify it:

- forward the full pack when the turn LOOKS tool-needy (imperative probe
  verbs, or an explicit tool-name mention), when tool traffic is already
  in flight (last event is a tool result or tool call), or when a tool was
  used in the recent past (the model may follow up);
- otherwise forward none of the client's tools.

The gate is binary by design: positive per-tool selection against
descriptions is brittle ("what's eating the space" shares no token with
disk_usage's description). Mis-pruning degrades to the baseline's armchair
answer — visible in evals, never a crash.
"""

import re
from typing import Any

from .store import Event

# Imperative probe verbs: the turn asks the assistant to LOOK at something
# live rather than reason from what it knows.
TOOL_NEED = re.compile(
    r"\b(check|find out|look up|look at|run|execute|query|fetch|inspect|"
    r"diagnose|measure|scan|ping|resolve|read the|show me|list the|"
    r"what does .{0,40}(say|show|report)|go see|pull up)\b",
    flags=re.IGNORECASE,
)
RECENT_TOOL_WINDOW = 6  # events


def scope_tools(
    tools: list[dict[str, Any]] | None,
    events: list[Event],
    user_text: str,
) -> list[dict[str, Any]] | None:
    """Return the subset of the client's tools to forward this turn."""
    if not tools or len(tools) <= 2:
        return tools  # nothing worth pruning

    recent = events[-RECENT_TOOL_WINDOW:]
    tool_in_flight = any(
        event.role == "tool" or event.message.get("tool_calls") for event in recent
    )
    if tool_in_flight:
        return tools
    if TOOL_NEED.search(user_text):
        return tools

    lowered = user_text.lower()
    named = [
        tool
        for tool in tools
        if _tool_name(tool) and _tool_name(tool).replace("_", " ") in lowered
    ]
    if named:
        return named
    return []


def _tool_name(tool: dict[str, Any]) -> str:
    return str(tool.get("function", {}).get("name", ""))
