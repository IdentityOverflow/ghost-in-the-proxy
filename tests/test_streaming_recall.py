"""Streaming recall (v4.1): hold-and-decide interception + tool-call deltas.

The non-stream path intercepts pure-recall tool calls proxy-side (v3); the
streaming path could not, because a forwarded chunk cannot be unsent. These
tests pin the streaming contract: a pure recall reply is held back, resolved,
and re-queried invisibly; anything else flushes verbatim (wire truthfulness);
and a streamed tool-call reply is recorded in the event store in the same
shape as a non-stream one.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from server.endpoints import chat as chat_endpoint
from server.endpoints.chat import _DeltaCollector, chat_completions
from server.mind.config import MindConfig
from server.mind.perception import reconcile
from server.mind.runtime import MindRuntime
from server.schemas import ChatCompletionRequest


def sse_chunk(delta: dict) -> bytes:
    return f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode()


DONE = b"data: [DONE]\n\n"


def content_chunks(*pieces: str) -> list[bytes]:
    chunks = [sse_chunk({"role": "assistant"})]
    chunks += [sse_chunk({"content": piece}) for piece in pieces]
    return chunks + [DONE]


def recall_call_chunks(query: str, call_id: str = "call_r1") -> list[bytes]:
    """A recall tool call streamed OpenAI-style: name whole in the first
    fragment for its index, arguments fragmented across chunks."""
    arguments = json.dumps({"query": query})
    mid = len(arguments) // 2
    return [
        sse_chunk({"role": "assistant"}),
        sse_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "recall", "arguments": ""},
                    }
                ]
            }
        ),
        sse_chunk({"tool_calls": [{"index": 0, "function": {"arguments": arguments[:mid]}}]}),
        sse_chunk({"tool_calls": [{"index": 0, "function": {"arguments": arguments[mid:]}}]}),
        DONE,
    ]


class FakeStreamProvider:
    def __init__(self, streams: list[list[bytes]]):
        self.streams = list(streams)
        self.requests: list[dict] = []

    async def chat_completions_stream(self, payload):
        self.requests.append(payload)
        for chunk in self.streams.pop(0):
            yield chunk


class FakeRequest:
    headers: dict = {}

    async def is_disconnected(self) -> bool:
        return False


async def _drain(req_body: dict, provider) -> bytes:
    response = await chat_completions(ChatCompletionRequest(**req_body), FakeRequest())
    out = b""
    async for chunk in response.body_iterator:
        out += chunk if isinstance(chunk, bytes) else chunk.encode()
    return out


def run_stream(provider, transcript) -> bytes:
    return asyncio.run(_drain({"model": "m", "messages": transcript, "stream": True}, provider))


@pytest.fixture
def proxy(monkeypatch, tmp_path):
    """Endpoint harness: fake streaming backend, real mind on a tmp store."""
    runtime = MindRuntime(MindConfig(enabled=True, db_dir=str(tmp_path / "minds"), window=8192))
    holder = SimpleNamespace(provider=None, runtime=runtime)
    monkeypatch.setattr(
        chat_endpoint, "resolve_provider_and_model", lambda model: (holder.provider, None)
    )
    monkeypatch.setattr(chat_endpoint, "agent_graph", SimpleNamespace(invoke=lambda state: state))
    monkeypatch.setattr(chat_endpoint, "get_mind_runtime", lambda: runtime)
    monkeypatch.setattr(chat_endpoint.mind_config, "fail_mode", "strict")
    monkeypatch.setattr(chat_endpoint.mind_config, "recall_enabled", True)
    monkeypatch.setattr(chat_endpoint.mind_config, "recall_max_hops", 3)
    return holder


TRACEBACK_TURN = (
    "my app broke: sqlite3.OperationalError: no such column: "
    "photos.review_flag_v2 [ref: ingest-9f4c2e71]"
)


def folded_transcript(runtime) -> tuple[list[dict], str]:
    """A session with material folded out of view: recall gets offered."""
    transcript = [
        {"role": "user", "content": TRACEBACK_TURN},
        {"role": "assistant", "content": "the migration did not apply"},
        {"role": "user", "content": "what was the exact error line?"},
    ]
    recon = reconcile(runtime.store, transcript)
    runtime.store.append_summary(recon.session_id, 2, "user hit a database error")
    return transcript, recon.session_id


def last_assistant_event(runtime, session_id):
    events = [e for e in runtime.store.live_events(session_id) if e.role == "assistant"]
    return events[-1]


# ── collector ──────────────────────────────────────────────────────────────────


def test_collector_accumulates_fragmented_tool_calls():
    collector = _DeltaCollector()
    for chunk in [
        sse_chunk({"role": "assistant"}),
        sse_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "read_", "arguments": ""},
                    }
                ]
            }
        ),
        sse_chunk({"tool_calls": [{"index": 0, "function": {"name": "file", "arguments": '{"path":'}}]}),
        sse_chunk(
            {
                "tool_calls": [
                    {"index": 0, "function": {"arguments": ' "a.py"}'}},
                    {
                        "index": 1,
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "recall", "arguments": '{"query": "x"}'},
                    },
                ]
            }
        ),
        DONE,
    ]:
        collector.feed(chunk)
    assert collector.message() == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_a",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
            },
            {
                "id": "call_b",
                "type": "function",
                "function": {"name": "recall", "arguments": '{"query": "x"}'},
            },
        ],
    }
    assert collector.tool_names() == ["read_file", "recall"]
    assert not collector.has_content()


def test_collector_content_shape_unchanged():
    collector = _DeltaCollector()
    for chunk in content_chunks("Hel", "lo"):
        collector.feed(chunk)
    assert collector.message() == {"role": "assistant", "content": "Hello"}
    assert collector.message()["content"] == "Hello"
    assert _DeltaCollector().message() is None


# ── streaming recall interception ──────────────────────────────────────────────


def test_pure_recall_stream_intercepted_and_requeried(proxy):
    transcript, session_id = folded_transcript(proxy.runtime)
    proxy.provider = FakeStreamProvider(
        [
            recall_call_chunks("review_flag_v2 error"),
            content_chunks("The failing column was photos.review_flag_v2"),
        ]
    )
    out = run_stream(proxy.provider, transcript)

    assert b"tool_calls" not in out  # the exchange never reached the client
    assert b"review_flag_v2" in out  # the final content did
    assert len(proxy.provider.requests) == 2
    followup = proxy.provider.requests[1]["messages"]
    assert followup[-1]["role"] == "tool"
    assert followup[-1]["tool_call_id"] == "call_r1"
    assert "ingest-9f4c2e71" in followup[-1]["content"]  # verbatim span resolved
    # Only the final reply entered the event store; deliberation did not.
    recorded = last_assistant_event(proxy.runtime, session_id)
    assert "review_flag_v2" in recorded.message["content"]
    assert "tool_calls" not in recorded.message
    assert recorded.complete


def test_content_stream_flushes_hold_verbatim(proxy):
    transcript, session_id = folded_transcript(proxy.runtime)
    chunks = content_chunks("nothing ", "to recall here")
    proxy.provider = FakeStreamProvider([chunks])
    out = run_stream(proxy.provider, transcript)

    assert out == b"".join(chunks)  # byte-for-byte passthrough
    assert len(proxy.provider.requests) == 1
    recorded = last_assistant_event(proxy.runtime, session_id)
    assert recorded.message["content"] == "nothing to recall here"
    assert recorded.complete


def test_mixed_batch_passes_through_untouched(proxy):
    transcript, session_id = folded_transcript(proxy.runtime)
    chunks = [
        sse_chunk({"role": "assistant"}),
        sse_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_r1",
                        "type": "function",
                        "function": {"name": "recall", "arguments": '{"query": "x"}'},
                    }
                ]
            }
        ),
        sse_chunk(
            {
                "tool_calls": [
                    {
                        "index": 1,
                        "id": "call_w1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ]
            }
        ),
        DONE,
    ]
    proxy.provider = FakeStreamProvider([chunks])
    out = run_stream(proxy.provider, transcript)

    # Symmetric with non-stream: a mixed batch is the client's to execute.
    assert out == b"".join(chunks)
    assert len(proxy.provider.requests) == 1
    recorded = last_assistant_event(proxy.runtime, session_id)
    names = [call["function"]["name"] for call in recorded.message["tool_calls"]]
    assert names == ["recall", "get_weather"]


def test_streamed_client_tool_call_is_recorded(proxy):
    # No fold, no recall offered: transparent from the first chunk. This is
    # the old blind spot — a streamed tool-call reply recorded nothing and
    # reconciliation had to repair it next turn.
    transcript = [{"role": "user", "content": "use the xwiki tool to fetch the home page"}]
    session_id = reconcile(proxy.runtime.store, transcript).session_id
    chunks = [
        sse_chunk({"role": "assistant"}),
        sse_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_x1",
                        "type": "function",
                        "function": {"name": "xwiki_get_document", "arguments": ""},
                    }
                ]
            }
        ),
        sse_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"page": "home"}'}}]}),
        DONE,
    ]
    proxy.provider = FakeStreamProvider([chunks])
    out = run_stream(proxy.provider, transcript)

    assert out == b"".join(chunks)
    recorded = last_assistant_event(proxy.runtime, session_id)
    assert recorded.message.get("content") is None or "content" not in recorded.message
    assert recorded.message["tool_calls"][0]["function"]["name"] == "xwiki_get_document"
    assert recorded.message["tool_calls"][0]["function"]["arguments"] == '{"page": "home"}'
    assert recorded.complete


def test_hop_limit_flushes_recall_call_verbatim(proxy, monkeypatch):
    monkeypatch.setattr(chat_endpoint.mind_config, "recall_max_hops", 0)
    transcript, session_id = folded_transcript(proxy.runtime)
    chunks = recall_call_chunks("review_flag_v2 error")
    proxy.provider = FakeStreamProvider([chunks])
    out = run_stream(proxy.provider, transcript)

    # Symmetric with non-stream at the limit: pass through, stay truthful.
    assert out == b"".join(chunks)
    assert len(proxy.provider.requests) == 1
    recorded = last_assistant_event(proxy.runtime, session_id)
    assert recorded.message["tool_calls"][0]["function"]["name"] == "recall"
