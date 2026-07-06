"""Scenario data model for the cognitive-architecture eval harness.

A scenario is a scripted multi-turn conversation played by the harness in the
role of the *client*: full message history is resent every turn, exactly like
real clients (OpenClaw, hermes-agent, raw scripts) do. The system under test
is any OpenAI-compatible endpoint — the bare proxy gives the baseline; the
cognitive middleware is graded against the same script later.

Grading model:
- A turn may carry checks. `must_mention` / `must_not_mention` are
  deterministic regex groups; `judge` defers to an optional judge model.
- A turn passes if all of its evaluated checks pass. Judge checks without a
  configured judge are reported as skipped and excluded from scoring.
- Context load is observed via `usage.prompt_tokens` from the endpoint.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Check:
    """One graded assertion about the assistant's reply for a turn."""

    kind: str  # "must_mention" | "must_not_mention" | "judge"
    desc: str
    # For must_mention: reply passes if ANY pattern matches (an any-of group).
    # Several must_mention checks on one turn therefore AND together.
    # For must_not_mention: reply passes if NO pattern matches.
    patterns: list[str] = field(default_factory=list)
    # For judge: the rubric question, answered pass/fail by the judge model.
    rubric: str = ""


@dataclass
class CannedResult:
    """A canned tool output, selected by substring match on the call args.

    `match=None` is a wildcard. Entries are reusable, so reading the same
    file twice returns the same content.
    """

    match: str | None
    content: str


@dataclass
class ToolDef:
    """An OpenAI-format tool the harness offers and stubs out."""

    name: str
    description: str
    parameters: dict[str, Any]
    results: list[CannedResult] = field(default_factory=list)

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def resolve(self, arguments_json: str) -> str:
        for entry in self.results:
            if entry.match is None or entry.match in arguments_json:
                return entry.content
        return f"error: no result available for {self.name}({arguments_json})"


@dataclass
class Turn:
    """One scripted user turn plus expectations about the reply."""

    user: str
    checks: list[Check] = field(default_factory=list)
    # When True the turn is expected to trigger a tool call. If the model
    # answers without calling any tool, the harness sends `fallback_tool`'s
    # canned output as an immediate "I ran it myself" user message so the
    # information still enters the stream, and records tool_used=False.
    expects_tool: bool = False
    fallback_tool: str | None = None
    # Client-side history edits — the behaviors real chat UIs perform that a
    # middleware must survive (fork/regenerate/interrupt, s7/s8):
    # rewind_user_turns=N cuts the transcript back to just BEFORE the Nth-
    # most-recent user message before sending this turn's user text. With
    # different text that is an edit (fork); with identical text, a
    # regenerate. N=1 replaces the latest exchange.
    rewind_user_turns: int = 0
    # truncate_reply_chars=N keeps only the first N characters of this
    # turn's reply in the transcript afterwards — the client stop button.
    # The full reply is still recorded and graded; only history is cut.
    truncate_reply_chars: int | None = None
    # Optional canned assistant reply used by `--mock` plumbing self-tests.
    mock_reply: str | None = None
    note: str = ""


@dataclass
class Scenario:
    id: str
    title: str
    description: str
    turns: list[Turn]
    system_prompt: str | None = None
    tools: list[ToolDef] = field(default_factory=list)

    def tool(self, name: str) -> ToolDef:
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise KeyError(f"scenario {self.id} has no tool named {name}")
