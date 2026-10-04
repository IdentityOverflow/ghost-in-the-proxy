import asyncio
import os

import httpx
from typing import Optional, Dict, Any

# Opt-in retry of transient upstream failures on NON-streaming calls (shared
# routes like OpenRouter rate-limit per upstream; a 429 mid-conversation should
# not kill a long run or a background fold). 0 keeps the faithful-passthrough
# default: the client sees exactly what the backend said, first time.
PROVIDER_RETRIES = int(os.getenv("PROVIDER_RETRIES", "0"))
RETRY_STATUSES = {429, 500, 502, 503, 504, 529}


class OpenAILikeProvider:
    def __init__(self, name: str, base_url: str, api_key: Optional[str] = None, extra_headers: Dict[str,str] | None = None,
                 extra_body: Dict[str, Any] | None = None):
        # extra_body: provider-specific request fields merged UNDER the
        # client's payload (the client wins) — e.g. OpenRouter's
        # {"reasoning": {"enabled": false}} for thinking models.
        self.extra_body = extra_body or {}
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.extra_headers = extra_headers or {}


    def _headers(self, passthrough_extra: Dict[str,str] | None = None) -> Dict[str,str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if passthrough_extra:
            h.update(passthrough_extra)
        h.update(self.extra_headers)
        return h


    async def list_models(self) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/models"
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url, headers=self._headers())
            r.raise_for_status()
            return r.json()


    async def chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        async with httpx.AsyncClient(timeout=300) as client:
            for attempt in range(PROVIDER_RETRIES + 1):
                r = await client.post(url, headers=self._headers(), json={**self.extra_body, **payload})
                if r.status_code in RETRY_STATUSES and attempt < PROVIDER_RETRIES:
                    try:
                        delay = float(r.headers.get("retry-after", ""))
                    except ValueError:
                        delay = 0.0
                    await asyncio.sleep(min(max(delay, 3.0 * 2**attempt), 90.0))
                    continue
                break
            r.raise_for_status()
            return r.json()


    async def chat_completions_stream(self, payload: Dict[str, Any]):
        url = f"{self.base_url}/v1/chat/completions"
        headers = self._headers()
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, headers=headers, json={**self.extra_body, **payload}) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_raw():
                    yield chunk