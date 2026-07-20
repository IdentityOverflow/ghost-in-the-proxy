"""Chat completions endpoint."""
import asyncio
import json
from typing import AsyncIterator

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..schemas import ChatCompletionRequest
from ..routing.router import resolve_provider_and_model
from ..agent.graph import agent_graph
from ..mind import get_mind_runtime
from ..mind.config import mind_config


async def chat_completions(req: ChatCompletionRequest, request: Request):
    """Handle chat completion requests with streaming support and disconnect detection."""
    provider, target_model = resolve_provider_and_model(req.model)
    if not provider:
        raise HTTPException(400, f"No provider for model '{req.model}'")

    # Raw OpenAI-format dicts in and out of the agent graph; exclude_none so
    # optional fields we defaulted (e.g. content on tool-call messages) are
    # omitted rather than sent as explicit nulls.
    messages_dict = [msg.model_dump(exclude_none=True) for msg in req.messages]
    agent_state = agent_graph.invoke({"messages": messages_dict})
    outgoing_messages = agent_state["messages"]

    payload = req.model_dump(exclude_none=True)

    # Mind middleware: replace the transcript with an assembled workspace,
    # scope the client's tool pack, and offer the recall tool (v3).
    mind = get_mind_runtime()
    session_id: str | None = None
    recall_offered = False
    # Fake clock (v4 eval harness): trusted only when explicitly enabled,
    # otherwise clients could spoof the mind's sense of time.
    clock: float | None = None
    if mind_config.fake_clock:
        raw_clock = request.headers.get("x-mind-clock")
        if raw_clock:
            try:
                clock = float(raw_clock)
            except ValueError:
                pass
    if mind is not None:
        try:
            prepared = await mind.prepare(
                outgoing_messages, provider, req.model, tools=payload.get("tools"), now=clock
            )
            outgoing_messages = prepared.messages
            session_id = prepared.session_id
            recall_offered = prepared.recall_offered
            if prepared.tools_scoped:
                tools_out = prepared.tools or []
                if tools_out:
                    payload["tools"] = tools_out
                else:
                    payload.pop("tools", None)
        except Exception as error:
            if mind_config.fail_mode == "strict":
                raise HTTPException(500, f"mind failure (strict mode): {error}") from error
            print(f"[mind] ERROR, falling back to passthrough: {error!r}", flush=True)

    payload["messages"] = outgoing_messages
    if target_model:
        payload["model"] = target_model

    if req.stream:
        recall_active = (
            mind is not None
            and session_id is not None
            and mind_config.recall_enabled
            and recall_offered
        )

        async def sse() -> AsyncIterator[bytes]:
            # Streaming recall (v4.1): a tool call already forwarded cannot be
            # intercepted, so while the reply could still turn out to be a
            # pure `recall` call we HOLD chunks instead of yielding them. The
            # hold ends the moment the stream shows content or a non-recall
            # tool name (flush held bytes, then transparent passthrough —
            # mixed batches pass through untouched, symmetric with the
            # non-stream path). A stream that ends while still held is a pure
            # recall reply: resolve proxy-side, re-query streaming, repeat.
            # The client never sees the exchange and it never enters the
            # event store (deliberation, not conversation truth). Cost for
            # ordinary content replies: delayed by roughly one role-only
            # chunk before the hold resolves.
            stream_payload = payload
            hops = 0
            complete = False
            # Collector of the bytes the client actually saw; recall
            # exchanges are held back, never forwarded, never recorded.
            forwarded: _DeltaCollector | None = None
            try:
                while True:
                    collector = _DeltaCollector()
                    held: list[bytes] | None = [] if recall_active else None
                    if held is None:
                        forwarded = collector
                    complete = False
                    async for raw in provider.chat_completions_stream(stream_payload):
                        collector.feed(raw)
                        # Check for client disconnect
                        if await request.is_disconnected():
                            break
                        if held is None:
                            yield raw
                            continue
                        # A name still streaming in fragments ("re", "call")
                        # must not read as a foreign tool: only a name that
                        # can no longer become "recall" ends the hold.
                        if collector.has_content() or any(
                            not (name == "recall" or "recall".startswith(name))
                            for name in collector.tool_names()
                        ):
                            for held_raw in held:
                                yield held_raw
                            held = None
                            forwarded = collector
                            yield raw
                        else:
                            held.append(raw)
                    else:
                        complete = True
                    if not complete or held is None:
                        break  # disconnect, or a fully-forwarded stream
                    # Stream ended while held: pure-recall reply (or nothing).
                    message = collector.message()
                    calls = (message or {}).get("tool_calls") or []
                    all_recall = calls and all(
                        call["function"]["name"] == "recall" for call in calls
                    )
                    if all_recall and hops < mind_config.recall_max_hops:
                        hops += 1
                        followup = list(stream_payload["messages"]) + [message]
                        for call in calls:
                            content = mind.resolve_recall(
                                session_id, call.get("function", {}).get("arguments") or "{}"
                            )
                            followup.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call.get("id") or f"recall-{hops}",
                                    "content": content,
                                }
                            )
                        print(
                            f"[mind] recall hop {hops} (stream): {len(calls)} call(s)",
                            flush=True,
                        )
                        stream_payload = {**stream_payload, "messages": followup}
                        continue
                    if all_recall:
                        print("[mind] recall hop limit reached; passing through", flush=True)
                    # Held but not interceptable (hop limit, or an empty
                    # reply): flush verbatim so the wire stays truthful.
                    for held_raw in held:
                        yield held_raw
                    forwarded = collector
                    break
            except asyncio.CancelledError:
                # Client disconnected during streaming
                raise
            finally:
                # We are the stream: record exactly what went out, even on
                # interrupt (complete=False -> provisional truncated reply).
                if mind is not None and session_id is not None and forwarded is not None:
                    message = forwarded.message()
                    if message is not None:
                        mind.observe_reply(session_id, message, complete=complete, ts=clock)
        return StreamingResponse(sse(), media_type="text/event-stream")

    try:
        resp = await provider.chat_completions(payload)
        # Recall interception (v3): recall is OUR tool, invisible to the
        # client. Resolve it proxy-side and re-query; the exchange never
        # enters the event store (the client's next transcript won't contain
        # it — recording it would desync reconciliation). Mixed calls
        # (recall + client tools) pass through untouched: partially executing
        # a tool batch would leave the client with dangling call ids.
        hops = 0
        while mind is not None and session_id is not None and mind_config.recall_enabled:
            message = resp["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            recall_calls = [c for c in calls if c.get("function", {}).get("name") == "recall"]
            if not calls or len(recall_calls) != len(calls):
                if recall_calls:
                    print("[mind] mixed recall+client tool_calls; passing through", flush=True)
                break
            if hops >= mind_config.recall_max_hops:
                print("[mind] recall hop limit reached; passing through", flush=True)
                break
            hops += 1
            followup = list(payload["messages"]) + [message]
            for call in recall_calls:
                content = mind.resolve_recall(
                    session_id, call.get("function", {}).get("arguments") or "{}"
                )
                followup.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", f"recall-{hops}"),
                        "content": content,
                    }
                )
            print(f"[mind] recall hop {hops}: {len(recall_calls)} call(s)", flush=True)
            payload = {**payload, "messages": followup}
            resp = await provider.chat_completions(payload)
    except httpx.HTTPStatusError as error:
        # Mirror backend errors (status + body) instead of collapsing them
        # into an opaque proxy 500; clients rely on e.g. context-length 400s.
        return JSONResponse(status_code=error.response.status_code, content=_error_body(error))
    if mind is not None and session_id is not None:
        try:
            mind.observe_reply(session_id, resp["choices"][0]["message"], complete=True, ts=clock)
        except Exception as error:
            if mind_config.fail_mode == "strict":
                raise HTTPException(500, f"mind failure (strict mode): {error}") from error
            print(f"[mind] ERROR recording reply: {error!r}", flush=True)
    return JSONResponse(resp)


class _DeltaCollector:
    """Accumulate an assistant message from raw SSE chunks (tolerant parser).

    Collects content AND tool_call deltas so (a) a streamed tool-call reply
    is recorded in the event store in the same shape as a non-stream one,
    and (b) the recall interceptor can see what the model is doing before
    any of it is forwarded.
    """

    def __init__(self):
        self._buffer = b""
        self._content: list[str] = []
        self._tool_calls: dict[int, dict] = {}

    def feed(self, raw: bytes) -> None:
        self._buffer += raw
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            text = line.decode("utf-8", errors="replace").strip()
            if not text.startswith("data:"):
                continue
            data = text[5:].strip()
            if data == "[DONE]":
                continue
            try:
                delta = json.loads(data)["choices"][0].get("delta", {})
            except (ValueError, KeyError, IndexError):
                continue
            piece = delta.get("content")
            if piece:
                self._content.append(piece)
            for fragment in delta.get("tool_calls") or []:
                if not isinstance(fragment, dict):
                    continue
                call = self._tool_calls.setdefault(
                    fragment.get("index", 0),
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if fragment.get("id"):
                    call["id"] = fragment["id"]
                if fragment.get("type"):
                    call["type"] = fragment["type"]
                function = fragment.get("function") or {}
                if function.get("name"):
                    call["function"]["name"] += function["name"]
                if function.get("arguments"):
                    call["function"]["arguments"] += function["arguments"]

    def has_content(self) -> bool:
        return bool(self._content)

    def tool_names(self) -> list[str]:
        return [
            call["function"]["name"]
            for _, call in sorted(self._tool_calls.items())
            if call["function"]["name"]
        ]

    def message(self) -> dict | None:
        if not self._content and not self._tool_calls:
            return None
        message: dict = {"role": "assistant", "content": "".join(self._content) or None}
        if self._tool_calls:
            message["tool_calls"] = [
                {**call, "id": call["id"] or f"call_{index}"}
                for index, call in sorted(self._tool_calls.items())
            ]
        return message


def _error_body(error: httpx.HTTPStatusError):
    try:
        return error.response.json()
    except ValueError:
        return {"error": error.response.text[:2000]}
