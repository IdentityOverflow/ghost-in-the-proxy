# Agent Guidelines for Project-2501

## Build/Run/Test
- **Run dev server**: `PYTHONPATH=. uvicorn server.main:app --reload --host 0.0.0.0 --port 8000` (from project root)
- **Run with Docker**: `docker-compose up --build` (from project root)
- **No test suite configured yet** - follow FastAPI/pytest patterns if adding tests

## Project Structure
```
Project-2501/
├── server/              # Backend API
│   ├── main.py         # Entry point
│   ├── endpoints/      # Route handlers
│   ├── routing/        # Provider routing logic
│   └── providers/      # Provider implementations
└── client/             # (Future) Frontend
```

## Code Style
- **Python version**: 3.10+ (use modern type hints: `str | None`, not `Optional[str]`)
- **Imports**: Standard library first, then third-party (fastapi, pydantic, httpx), then local (relative imports with `.`)
- **Formatting**: Follow PEP 8; prefer async/await patterns throughout
- **Types**: Use Pydantic models for all request/response schemas; type hint all function signatures
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPERCASE for constants/globals
- **Error handling**: Use FastAPI `HTTPException` with status codes; let httpx errors propagate with `raise_for_status()`
- **Config**: All env vars loaded via `pydantic.BaseModel` in `config.py`; use `settings` singleton
- **Providers**: New providers extend `OpenAILikeProvider` pattern; register in `routing/router.py:PROVIDERS`
- **Streaming**: Use `AsyncIterator[bytes]` for SSE; pass raw chunks through without parsing
- **Models**: Support model mapping via `MODEL_MAP` env var (format: `{"alias":"provider:actual_model"}`)
