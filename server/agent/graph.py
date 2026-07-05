"""LangGraph pipeline for message preprocessing.

This graph is the middleware insertion point: request messages pass through
it before being forwarded to the provider. Default behavior is a faithful
passthrough — messages leave exactly as they arrived unless a transformation
is explicitly configured.

State carries raw OpenAI-format message dicts. Do not convert to LangChain
message objects here: that round-trip is lossy (it drops role:"tool"
messages, assistant tool_calls, names, and any unmodeled fields).
"""
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from ..config import settings


class AgentState(TypedDict):
    """State for the agent graph: OpenAI-format message dicts."""
    messages: list[dict[str, Any]]


def apply_system_prompt_override(state: AgentState) -> AgentState:
    """Replace or prepend the system message, only when explicitly configured.

    With SYSTEM_PROMPT_OVERRIDE unset (the default), the client's own system
    prompt passes through untouched.
    """
    override = settings.system_prompt_override
    if not override:
        return state

    messages = state["messages"]
    system_message = {"role": "system", "content": override}
    if messages and messages[0].get("role") == "system":
        messages = [system_message, *messages[1:]]
    else:
        messages = [system_message, *messages]
    return {"messages": messages}


def create_agent_graph():
    """Create the message-preprocessing graph.

    Flow:
    1. apply_system_prompt_override: optional, config-gated
    2. END: return messages
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("apply_system_prompt_override", apply_system_prompt_override)
    workflow.set_entry_point("apply_system_prompt_override")
    workflow.add_edge("apply_system_prompt_override", END)
    return workflow.compile()


# Create the compiled graph instance
agent_graph = create_agent_graph()
