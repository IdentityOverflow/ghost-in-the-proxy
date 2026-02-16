# Experimental base

The point of this project is to server as an experimental base.
You can expanding on the agent/graph.py file to insert additional structure and capabilities to your AI model before you forward it to your application.

It provides a lightweight FastAPI application that acts as a passthrough proxy for OpenAI‑compatible chat models. It supports multiple back‑end providers (OpenAI, OpenRouter, Azure OpenAI, LM Studio, and Ollama) and allows you to map alias model names to provider-specific model identifiers. Streaming responses are forwarded as Server‑Sent Events.

In short, it allows you to inject additional capability to your model at inference time. You feed and OpenAI API endpoint in, passes the model inferece output through additional scaffolding (default langchain/langgraph processing) and outputs the same OpenAI API which you can pass to your application.

## Features
- **Multi‑provider support** – Configure any of the supported AI providers via environment variables.
- **Model mapping** – Define aliases that map to provider‐specific names using the `MODEL_MAP` environment variable.
- **Streaming** – `POST /v1/chat/completions?stream=true` returns an SSE stream of raw data.
- **Simple API surface** – Mirrors the OpenAI Chat Completions API.

## Installing
```bash
cd server
pip install -r requirements.txt
```

## Environment variables
| Variable | Description | Example |
|---|---|---|
| `DEFAULT_PROVIDER` | Default provider key (e.g., `ollama`) | `ollama` |
| `MODEL_MAP` | JSON mapping of alias → `provider:model` | `{"gpt4all": "ollama:gpt4all-falcon"}` |
| `OPENAI_BASE_URL` | Base URL for OpenAI compatible endpoint | `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | API key for OpenAI | `sk-...` |
| `OPENROUTER_BASE_URL` | Base URL for OpenRouter | `https://openrouter.ai/api/v1` |
| `OPENROUTER_API_KEY` | API key for OpenRouter | `or-...` |
| `AZURE_OPENAI_BASE_URL` | Azure OpenAI endpoint | `https://YOUR_RESOURCE.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | Azure key | `...` |
| `AZURE_OPENAI_API_VERSION` | Azure API version | `2024-02-01` |
| `LMSTUDIO_BASE_URL` | LM Studio URL | `http://localhost:1234` |
| `OLLAMA_BASE_URL` | Ollama URL | `http://localhost:11434` |

## Running
Start the development server:
```bash
# From project root
PYTHONPATH=. uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```
Or run with Docker:
```bash
docker-compose up --build
```

## Usage examples
```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "gpt4all",
  "messages": [{"role": "user", "content": "Say hi"}],
  "stream": true
}
```
The response will be sent as an SSE stream.

## License
MIT © 2025
