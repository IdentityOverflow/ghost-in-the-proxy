"""Chat completions endpoint."""
import asyncio
from typing import AsyncIterator

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..schemas import ChatCompletionRequest
from ..routing.router import resolve_provider_and_model
from ..agent.graph import (
    agent_graph,
    convert_to_langchain_messages,
    convert_to_openai_messages
)


async def chat_completions(req: ChatCompletionRequest, request: Request):
    """Handle chat completion requests with streaming support and disconnect detection."""
    provider, target_model = resolve_provider_and_model(req.model)
    if not provider:
        raise HTTPException(400, f"No provider for model '{req.model}'")

    # Convert messages to LangChain format
    messages_dict = [msg.model_dump() for msg in req.messages]
    langchain_messages = convert_to_langchain_messages(messages_dict)

    # Process messages through LangGraph agent
    agent_state = agent_graph.invoke({
        "messages": langchain_messages,
        "system_prompt": ""  # Empty string will trigger DEFAULT_SYSTEM_PROMPT via `or`
    })

    # Convert back to OpenAI format
    modified_messages = convert_to_openai_messages(agent_state["messages"])

    # Use the modified messages from the agent
    payload = req.model_dump(exclude_none=True)
    payload["messages"] = modified_messages
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

    resp = await provider.chat_completions(payload)
    return JSONResponse(resp)
