# Agent Guidelines for ghost-in-the-proxy

## Build/Run/Test
- **Run dev server**: `uvicorn server.main:app --reload --port 8000` (from project root; config comes from `ghost.env`, local overrides from `.env`)
- **Run with Docker**: `docker-compose up --build` (from project root)
- **Tests**: `pytest tests/` (conda env `ghost`); long-conversation soak: `python -m evals.run --scenario s14-soak` + `python -m evals.soak_report`

## Project Structure
```
ghost-in-the-proxy/
├── server/              # Backend API
│   ├── main.py         # Entry point
│   ├── endpoints/      # Route handlers
│   ├── routing/        # Provider routing logic
│   ├── providers/      # Provider implementations
│   └── mind/           # Cognitive middleware (docs/architecture.md, docs/memory-v6.md)
└── client/             # (Future) Frontend
```

## Code Style
- **Python version**: 3.10+ (use modern type hints: `str | None`, not `Optional[str]`)
- **Imports**: Standard library first, then third-party (fastapi, pydantic, httpx), then local (relative imports with `.`)
- **Formatting**: Follow PEP 8; prefer async/await patterns throughout
- **Types**: Use Pydantic models for all request/response schemas; type hint all function signatures
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPERCASE for constants/globals
- **Error handling**: Use FastAPI `HTTPException` with status codes; let httpx errors propagate with `raise_for_status()`
- **Config**: `ghost.env` (committed defaults) < `.env` (local) < shell env, loaded by `server/env.py`; env vars surface via `pydantic.BaseModel` in `config.py` / `mind/config.py`. Tests set `GHOST_NO_CONFIG=1`.
- **Providers**: New providers extend `OpenAILikeProvider` pattern; register in `routing/router.py:PROVIDERS`
- **Streaming**: Use `AsyncIterator[bytes]` for SSE; pass raw chunks through without parsing
- **Models**: Support model mapping via `MODEL_MAP` env var (format: `{"alias":"provider:actual_model"}`)
