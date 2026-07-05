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

    # Mind middleware: replace the transcript with an assembled workspace.
    mind = get_mind_runtime()
    session_id: str | None = None
    if mind is not None:
        try:
            prepared = await mind.prepare(outgoing_messages, provider, req.model)
            outgoing_messages = prepared.messages
            session_id = prepared.session_id
        except Exception as error:
            if mind_config.fail_mode == "strict":
                raise HTTPException(500, f"mind failure (strict mode): {error}") from error
            print(f"[mind] ERROR, falling back to passthrough: {error!r}", flush=True)

    payload = req.model_dump(exclude_none=True)
    payload["messages"] = outgoing_messages
    if target_model:
        payload["model"] = target_model

    if req.stream:
        async def sse() -> AsyncIterator[bytes]:
            collector = _DeltaCollector()
            complete = False
            try:
                async for raw in provider.chat_completions_stream(payload):
                    collector.feed(raw)
                    # Check for client disconnect
                    if await request.is_disconnected():
                        break
                    yield raw
                else:
                    complete = True
            except asyncio.CancelledError:
                # Client disconnected during streaming
                raise
            finally:
                # We are the stream: record exactly what went out, even on
                # interrupt (complete=False -> provisional truncated reply).
                if mind is not None and session_id is not None and collector.message():
                    mind.observe_reply(session_id, collector.message(), complete=complete)
        return StreamingResponse(sse(), media_type="text/event-stream")

    try:
        resp = await provider.chat_completions(payload)
    except httpx.HTTPStatusError as error:
        # Mirror backend errors (status + body) instead of collapsing them
        # into an opaque proxy 500; clients rely on e.g. context-length 400s.
        return JSONResponse(status_code=error.response.status_code, content=_error_body(error))
    if mind is not None and session_id is not None:
        try:
            mind.observe_reply(session_id, resp["choices"][0]["message"], complete=True)
        except Exception as error:
            if mind_config.fail_mode == "strict":
                raise HTTPException(500, f"mind failure (strict mode): {error}") from error
            print(f"[mind] ERROR recording reply: {error!r}", flush=True)
    return JSONResponse(resp)


class _DeltaCollector:
    """Accumulate assistant content from raw SSE chunks (tolerant parser)."""

    def __init__(self):
        self._buffer = b""
        self._content: list[str] = []

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

    def message(self) -> dict | None:
        if not self._content:
            return None
        return {"role": "assistant", "content": "".join(self._content)}


def _error_body(error: httpx.HTTPStatusError):
    try:
        return error.response.json()
    except ValueError:
        return {"error": error.response.text[:2000]}
