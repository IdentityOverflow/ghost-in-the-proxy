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
disk_usage's description). The error costs are asymmetric: a false
positive is schema tax; a false NEGATIVE hands the model a conversation
full of tool talk with no tools, and a small model then play-acts the
call as text and confabulates its result (live xwiki repro: "use the
xwiki mcp tools to tell me..." matched no verb, belt stripped, model
emitted pseudo-tool-call syntax and invented page content). Lean
permissive.
"""

import re
from typing import Any

from .store import Event

# Imperative probe verbs: the turn asks the assistant to LOOK at something
# live rather than reason from what it knows.
TOOL_NEED = re.compile(
    r"\b(check|find out|look up|look at|run|execute|query|fetch|inspect|"
    r"diagnose|measure|scan|ping|resolve|read the|show me|list the|"
    r"what does .{0,40}(say|show|report)|go see|pull up|use|tell me)\b",
    flags=re.IGNORECASE,
)
# The user talking ABOUT tooling is always tool-needy.
TOOL_TALK = re.compile(r"\b(tools?|mcp|plugins?|functions?|integrations?)\b", flags=re.IGNORECASE)
RECENT_TOOL_WINDOW = 6  # events
# Tool-name words shorter than this (or in the stoplist) are too common to
# signal anything on their own.
NAME_WORD_MIN = 4
NAME_WORD_STOP = {"list", "info", "data", "with", "from", "into", "this", "that"}


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
    if TOOL_NEED.search(user_text) or TOOL_TALK.search(user_text):
        return tools

    text_words = set(re.findall(r"[a-z0-9]+", user_text.lower()))
    if any(word in text_words for word in _name_words(tools)):
        return tools
    return []


def _tool_name(tool: dict[str, Any]) -> str:
    return str(tool.get("function", {}).get("name", ""))


def _name_words(tools: list[dict[str, Any]]) -> set[str]:
    """Distinctive words from the tool names themselves ('xwiki' out of
    xwiki_get_document): the user naming the domain is a tool-needy turn."""
    words: set[str] = set()
    for tool in tools:
        for word in re.split(r"[_\-]", _tool_name(tool).lower()):
            if len(word) >= NAME_WORD_MIN and word not in NAME_WORD_STOP:
                words.add(word)
    return words
