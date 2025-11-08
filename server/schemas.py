from pydantic import BaseModel
from typing import Any, List, Optional


class Message(BaseModel):
    role: str
    content: Any

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    # pass-through fields (ignored by proxy but forwarded to provider)
    temperature: float | None = None
    max_tokens: int | None = None
    tools: Any | None = None
    tool_choice: Any | None = None
    # ... add other OpenAI fields as needed