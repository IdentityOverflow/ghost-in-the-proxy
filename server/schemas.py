from pydantic import BaseModel, ConfigDict
from typing import Any, List, Optional


class Message(BaseModel):
    # Faithful passthrough: keep fields we don't model (tool_calls,
    # tool_call_id, name, ...) instead of silently stripping them.
    model_config = ConfigDict(extra="allow")

    role: str
    # Assistant tool-call messages legally omit content.
    content: Any = None


class ChatCompletionRequest(BaseModel):
    # Forward unmodeled OpenAI fields (top_p, stop, response_format,
    # stream_options, ...) instead of dropping them at validation.
    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: float | None = None
    max_tokens: int | None = None
    tools: Any | None = None
    tool_choice: Any | None = None
