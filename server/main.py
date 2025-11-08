"""
Project-2501 - AI Passthrough Proxy

Minimal entry point that registers routes and starts the FastAPI application.
All endpoint logic lives in the endpoints/ package.
"""
from fastapi import FastAPI

from .schemas import ChatCompletionRequest
from .endpoints import chat, models, health

# Create FastAPI application
app = FastAPI(
    title="Project-2501",
    version="0.3.0",
    description="Lightweight passthrough proxy for OpenAI-compatible chat models"
)

# Register routes
app.get("/")(health.root)
app.get("/v1/models")(models.list_models)
app.post("/v1/chat/completions")(chat.chat_completions)