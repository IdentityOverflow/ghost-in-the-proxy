"""Chat completions endpoint."""
import asyncio
from typing import AsyncIterator

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..schemas import ChatCompletionRequest
from ..routing.router import resolve_provider_and_model
from ..agent.graph import agent_graph


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

    payload = req.model_dump(exclude_none=True)
    payload["messages"] = agent_state["messages"]
    if target_model:
        payload["model"] = target_model

    if req.stream:
        async def sse() -> AsyncIterator[bytes]:
            try:
                async for raw in provider.chat_completions_stream(payload):
                    # Check for client disconnect
                    if await request.is_disconnected():
                        break
                    yield raw
            except asyncio.CancelledError:
                # Client disconnected during streaming
                raise
        return StreamingResponse(sse(), media_type="text/event-stream")

    try:
        resp = await provider.chat_completions(payload)
    except httpx.HTTPStatusError as error:
        # Mirror backend errors (status + body) instead of collapsing them
        # into an opaque proxy 500; clients rely on e.g. context-length 400s.
        return JSONResponse(status_code=error.response.status_code, content=_error_body(error))
    return JSONResponse(resp)


def _error_body(error: httpx.HTTPStatusError):
    try:
        return error.response.json()
    except ValueError:
        return {"error": error.response.text[:2000]}
