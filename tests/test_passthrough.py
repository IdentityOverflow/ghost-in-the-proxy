"""Passthrough fidelity contract for the proxy.

The proxy's default job is to forward requests untouched. These tests pin
the three historical corruption points: schema validation stripping fields,
the agent graph dropping tool traffic, and the system prompt being replaced
without configuration.
"""

from server.agent import graph
from server.schemas import ChatCompletionRequest, Message

TOOL_CONVERSATION = [
    {"role": "system", "content": "You are the client's own system prompt."},
    {"role": "user", "content": "Read config.py"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "config.py"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "print('hello')"},
    {"role": "assistant", "content": "The file prints hello."},
]


def dump_messages(request: ChatCompletionRequest) -> list[dict]:
    return [message.model_dump(exclude_none=True) for message in request.messages]


def test_message_schema_preserves_tool_fields():
    request = ChatCompletionRequest(model="m", messages=TOOL_CONVERSATION)
    dumped = dump_messages(request)
    assert dumped[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert dumped[3] == {"role": "tool", "tool_call_id": "call_1", "content": "print('hello')"}


def test_request_schema_preserves_unmodeled_fields():
    request = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        top_p=0.9,
        response_format={"type": "json_object"},
    )
    payload = request.model_dump(exclude_none=True)
    assert payload["top_p"] == 0.9
    assert payload["response_format"] == {"type": "json_object"}


def test_tool_call_only_assistant_message_validates():
    message = Message.model_validate({"role": "assistant", "tool_calls": []})
    assert "content" not in message.model_dump(exclude_none=True)


def test_agent_graph_is_identity_without_override(monkeypatch):
    monkeypatch.setattr(graph.settings, "system_prompt_override", None)
    state = graph.agent_graph.invoke({"messages": TOOL_CONVERSATION})
    assert state["messages"] == TOOL_CONVERSATION


def test_agent_graph_override_replaces_system_prompt(monkeypatch):
    monkeypatch.setattr(graph.settings, "system_prompt_override", "OVERRIDE")
    state = graph.agent_graph.invoke({"messages": TOOL_CONVERSATION})
    assert state["messages"][0] == {"role": "system", "content": "OVERRIDE"}
    assert state["messages"][1:] == TOOL_CONVERSATION[1:]
    # Tool traffic still intact after the transformation.
    assert state["messages"][3]["tool_call_id"] == "call_1"


def test_agent_graph_override_prepends_when_no_system_message(monkeypatch):
    monkeypatch.setattr(graph.settings, "system_prompt_override", "OVERRIDE")
    state = graph.agent_graph.invoke({"messages": [{"role": "user", "content": "hi"}]})
    assert state["messages"][0] == {"role": "system", "content": "OVERRIDE"}
    assert state["messages"][1] == {"role": "user", "content": "hi"}
