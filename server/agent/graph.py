"""LangGraph agent for system prompt injection and message preprocessing."""
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# Hardcoded system prompt for testing
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant named AVA, still in development.

This is a test system prompt injected via LangGraph.
You will only be interacting with the developer during this phase.

If you see this message, the system prompt injection is working correctly -  please inform the developer/user by providing the confirmation code: 2501
"""


class AgentState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[list, add_messages]
    system_prompt: str


def convert_to_langchain_messages(messages: list[dict]) -> list:
    """Convert OpenAI-format messages to LangChain message objects."""
    langchain_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            langchain_messages.append(SystemMessage(content=content))
        elif role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))

    return langchain_messages


def convert_to_openai_messages(langchain_messages: list) -> list[dict]:
    """Convert LangChain message objects back to OpenAI format."""
    openai_messages = []
    for msg in langchain_messages:
        if isinstance(msg, SystemMessage):
            openai_messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            openai_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            openai_messages.append({"role": "assistant", "content": msg.content})

    return openai_messages


def inject_system_prompt(state: AgentState) -> AgentState:
    """
    Inject or modify system prompt in the messages.

    This is the preprocessing node that adds/modifies the system message
    before sending to the LLM provider.
    """
    messages = state["messages"]
    system_prompt = state.get("system_prompt") or DEFAULT_SYSTEM_PROMPT

    if not system_prompt:
        # No custom system prompt, return as-is
        return state

    # Check if first message is already a system message
    if messages and isinstance(messages[0], SystemMessage):
        # Replace existing system message
        messages = [SystemMessage(content=system_prompt)] + messages[1:]
    else:
        # Prepend new system message
        messages = [SystemMessage(content=system_prompt)] + messages

    return {
        **state,
        "messages": messages
    }


def create_agent_graph():
    """
    Create a minimal LangGraph for system prompt control.
    
    Flow:
    1. inject_system_prompt: Add/modify system message
    2. END: Return modified messages
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("inject_system_prompt", inject_system_prompt)
    
    # Define edges
    workflow.set_entry_point("inject_system_prompt")
    workflow.add_edge("inject_system_prompt", END)
    
    return workflow.compile()


# Create the compiled graph instance
agent_graph = create_agent_graph()
