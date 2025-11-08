import httpx
from typing import Optional, Dict, Any


class OpenAILikeProvider:
    def __init__(self, name: str, base_url: str, api_key: Optional[str] = None, extra_headers: Dict[str,str] | None = None):
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
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(url, headers=self._headers(), json=payload)
            r.raise_for_status()
            return r.json()


    async def chat_completions_stream(self, payload: Dict[str, Any]):
        url = f"{self.base_url}/v1/chat/completions"
        headers = self._headers()
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_raw():
                    yield chunk