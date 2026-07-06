"""Minimal OpenAI-compatible chat client for the eval harness.

Non-streaming on purpose: `usage` reporting is reliable and the harness is
sequential. When a backend omits `usage`, prompt tokens are estimated at
chars/4 and flagged so reports never silently mix measured and estimated
numbers.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class ChatResult:
    message: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    usage_estimated: bool
    latency_s: float


def estimate_tokens(payload: Any) -> int:
    return max(1, len(json.dumps(payload, ensure_ascii=False)) // 4)


@dataclass
class ChatClient:
    base_url: str
    model: str
    api_key: str = "not-needed"
    temperature: float = 0.2
    timeout_s: float = 300.0
    _client: httpx.AsyncClient | None = field(default=None, repr=False)

    async def __aenter__(self) -> "ChatClient":
        self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        assert self._client is not None, "use `async with ChatClient(...)`"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        if tools:
            body["tools"] = tools
        started = time.monotonic()
        response = await self._client.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()
        latency = time.monotonic() - started
        data = response.json()
        message = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        estimated = prompt_tokens is None
        if prompt_tokens is None:
            prompt_tokens = estimate_tokens(messages)
        if completion_tokens is None:
            completion_tokens = estimate_tokens(message)
        return ChatResult(
            message=message,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usage_estimated=estimated,
            latency_s=latency,
        )


@dataclass
class MockClient:
    """Plumbing self-test backend that runs the harness without a model.

    The runner sets `next_reply` (from the turn's `mock_reply`) before each
    call; unset turns get a fixed acknowledgement.
    """

    model: str = "mock"
    next_reply: str | None = None

    async def __aenter__(self) -> "MockClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        # Keep next_reply for the whole turn: the tool-fallback path calls
        # complete() twice per turn, and the runner resets it every turn.
        reply = self.next_reply or "Acknowledged."
        return ChatResult(
            message={"role": "assistant", "content": reply},
            prompt_tokens=estimate_tokens(messages),
            completion_tokens=estimate_tokens(reply),
            usage_estimated=True,
            latency_s=0.0,
        )
